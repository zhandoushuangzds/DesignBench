#!/usr/bin/env python3
"""
将靶点 PDB 裁剪为仅保留 hotspot 周围指定距离（默认 20Å）内的残基，以加快 RFantibody 运行。

使用 Biopython PDB 模块加载结构，根据 hotspot 残基的原子坐标做距离筛选，
保留原始 Chain ID、Residue Name、Residue ID、Insertion Code，与 RFantibody 流程兼容。

关于 "PDBConstructionException: (' ', 18, ' ') defined twice"：
  Biopython 的 Entity（Chain/Model）内部用 child_list + child_dict 存子对象，add() 时
  用 entity.get_id() 作为 key 写入 child_dict。若用 chain.copy() 再 child_list.clear()，
  只清空了列表，child_dict 仍保留复制来的残基 id，导致第一次 add(残基) 时 has_id()
  为真而报错。解决：用空壳 Chain(id)/Model(id) 新建对象再逐个 add，且同链内同一 res_id 只 add 一次。
"""

import re
import sys
from pathlib import Path
from typing import Optional

# Hotspot 格式：逗号分隔的 "ChainResid" 或 "ChainResidIcode"，如 "A56,A115,A123" 或 "A56A"
HOTSPOT_PATTERN = re.compile(r"^([A-Za-z])(\d+)([A-Za-z]?)$")


def parse_hotspot(s: str) -> Optional[tuple[str, int, str]]:
    """
    解析单个 hotspot 字符串为 (chain_id, resseq, icode)。
    例如 "A56" -> ("A", 56, " "), "A56A" -> ("A", 56, "A")。
    """
    s = s.strip()
    if not s:
        return None
    m = HOTSPOT_PATTERN.match(s)
    if not m:
        return None
    chain_id, resseq_str, icode = m.groups()
    return (chain_id, int(resseq_str), icode if icode else " ")


def get_hotspot_atom_coords(struct, hotspot_specs: list[tuple[str, int, str]]) -> list[tuple[float, float, float]]:
    """
    从结构中收集所有 hotspot 残基中的原子坐标。
    hotspot_specs: [(chain_id, resseq, icode), ...]，icode 为 " " 或单字符。
    """
    coords: list[tuple[float, float, float]] = []
    for model in struct.get_models():
        for chain in model.get_chains():
            cid = chain.get_id()
            for res in chain:
                rid = res.get_id()
                # rid 为 (hetflag, resseq, icode)
                if (cid, rid[1], rid[2]) in hotspot_specs:
                    for atom in res.get_atoms():
                        coords.append(atom.get_coord().tolist())
    return coords


def crop_pdb_around_hotspots(
    input_pdb: Path,
    output_pdb: Path,
    hotspots_csv: str,
    distance_threshold: float = 20.0,
) -> tuple[int, int]:
    """
    保留与任意 hotspot 原子距离 <= distance_threshold 的残基，其余丢弃；
    保留原始 Chain ID、Residue Name、Residue ID、Insertion Code。
    返回 (原始残基数, 裁剪后残基数)。
    """
    try:
        from Bio.PDB import PDBIO, PDBParser
        from Bio.PDB.Residue import Residue
    except ImportError:
        raise RuntimeError("裁剪脚本需要 biopython: pip install biopython")

    # 解析 hotspot 列表，如 "A56,A115,A123"
    hotspot_specs: list[tuple[str, int, str]] = []
    for part in hotspots_csv.split(","):
        spec = parse_hotspot(part)
        if spec is not None:
            # Biopython icode 为 str，单字符或 " "
            hotspot_specs.append(spec)

    if not hotspot_specs:
        raise ValueError("未解析到有效的 hotspot，请检查 hotspots_csv 格式（如 A56,A115,A123）")

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("antigen", str(input_pdb))

    # 收集所有 hotspot 原子坐标
    hotspot_coords = get_hotspot_atom_coords(struct, hotspot_specs)
    if not hotspot_coords:
        raise ValueError(
            f"在结构中未找到任何 hotspot 残基: {hotspot_specs}。请确认 PDB 的 chain/residue 与配置一致。"
        )

    def dist_sq(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2

    def canonical_rid(rid: tuple) -> tuple:
        """归一化 res_id，使 (' ', 18, '') 与 (' ', 18, ' ') 视为同一残基，避免同链重复添加。"""
        return (rid[0], rid[1], rid[2].strip() or " ")

    threshold_sq = distance_threshold * distance_threshold

    # 遍历所有残基，判断是否保留：该残基任意原子与任意 hotspot 原子距离 <= threshold 则保留
    residues_to_keep: set[tuple[str, tuple]] = set()
    total_residue_count = 0

    for model in struct:
        for chain in model:
            chain_id = chain.get_id()
            for res in chain:
                total_residue_count += 1
                res_id = res.get_id()
                key_canon = (chain_id, canonical_rid(res_id))
                for atom in res.get_atoms():
                    ac = atom.get_coord()
                    ac_t = (float(ac[0]), float(ac[1]), float(ac[2]))
                    for hc in hotspot_coords:
                        if dist_sq(ac_t, tuple(hc)) <= threshold_sq:
                            residues_to_keep.add(key_canon)
                            break
                    else:
                        continue
                    break

    # 构建新结构：只包含保留的残基，且保持原始 chain/residue 信息
    #
    # 【为何会报 "defined twice"】
    # Biopython 的 Entity（Chain/Model）内部同时维护 child_list 和 child_dict，add() 时用
    # entity.get_id() 作为 key 写入 child_dict。若用 chain.copy() 再 child_list.clear()，
    # 只清空了列表，child_dict 仍保留复制来的所有残基 id，所以第一次 add(残基) 时
    # has_id(残基_id) 为 True，会直接抛 PDBConstructionException。
    # 正确做法：用空壳 Chain(id)、Model(id) 新建对象，再逐个 add(deepcopy(res))，保证
    # child_dict 初始为空，且同链内同一 res_id 只 add 一次。
    from copy import deepcopy

    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure

    new_struct = Structure(struct.get_id())
    for model in struct:
        new_model = Model(model.get_id())
        for chain in model:
            new_chain = Chain(chain.get_id())
            seen_rid_in_chain: set[tuple] = set()
            for res in chain:
                key_canon = (chain.get_id(), canonical_rid(res.get_id()))
                if key_canon not in residues_to_keep:
                    continue
                rid = res.get_id()
                if rid in seen_rid_in_chain:
                    continue
                seen_rid_in_chain.add(rid)
                new_chain.add(deepcopy(res))
            if len(new_chain.child_list) > 0:
                new_model.add(new_chain)
        if len(new_model.child_list) > 0:
            new_struct.add(new_model)

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(new_struct)
    io.save(str(output_pdb))

    return total_residue_count, len(residues_to_keep)


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: python crop_antigen_around_hotspots.py <input.pdb> <output.pdb> <hotspots_csv> [distance_threshold=20]",
            file=sys.stderr,
        )
        print("  hotspots_csv 例如: A56,A115,A123", file=sys.stderr)
        sys.exit(1)

    input_pdb = Path(sys.argv[1])
    output_pdb = Path(sys.argv[2])
    hotspots_csv = sys.argv[3]
    distance_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0

    if not input_pdb.exists():
        print(f"输入文件不存在: {input_pdb}", file=sys.stderr)
        sys.exit(1)

    n_orig, n_kept = crop_pdb_around_hotspots(
        input_pdb, output_pdb, hotspots_csv, distance_threshold
    )
    ratio = (n_kept / n_orig * 100) if n_orig else 0
    print(f"原始残基数: {n_orig}")
    print(f"裁剪后残基数: {n_kept}")
    print(f"裁剪比例: {ratio:.1f}% 保留, {100 - ratio:.1f}% 移除")
    print(f"已写入: {output_pdb}")


if __name__ == "__main__":
    main()
