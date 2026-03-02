#!/usr/bin/env python3
"""
Benchmark 抗原裁剪：按 target_config 生成 antigens_cropped，供所有模型（RFantibody、BoltzGen 等）统一使用。

裁剪逻辑：
1) 按 antigen_chains（第二列）只保留指定链与残基范围（auth 编号）。例如 A17-132 表示保留 A 链 17–132 残基；
   仅写 "A" 或 "B" 表示保留该链全部残基（Part2 靶点 12–20）。配置中的范围均为 auth 编号。
2) 对非 12–20 靶点且配置了 target_hotspots 的，再按与 hotspot 距离 ≤ 20Å 进一步裁剪。
3) 靶点 12–20 共 9 个：只做步骤 1，不做 20Å 裁剪。

输出：assets/antibody_nanobody/antigens_cropped/{target_id}.pdb
依赖：biopython。从 designbench 根目录运行：python assets/antibody_nanobody/scripts/crop_antigens_for_benchmark.py
说明：当前按 CIF 解析后的残基号（resseq）与 antigen_chains 比较；若 CIF 使用 label 编号，需在 CIF 中提供 auth 或先转换为 auth 再裁剪。
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Optional

# antigen_chains 一段：A17-132 或 A（整链）
_CHAIN_RANGE = re.compile(r"^([A-Za-z])(?:(\d+)-(\d+))?$")
# hotspot：A56 或 A56A
_HOTSPOT = re.compile(r"^([A-Za-z])(\d+)([A-Za-z]?)$")


def _parse_antigen_chains(s: str) -> list[tuple[str, Optional[int], Optional[int]]]:
    """解析 antigen_chains，返回 [(chain_id, start, end), ...]，整链时 start,end 为 None。"""
    out: list[tuple[str, Optional[int], Optional[int]]] = []
    s = (s or "").strip().strip('"')
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        m = _CHAIN_RANGE.match(part)
        if not m:
            continue
        cid, start_s, end_s = m.groups()
        if start_s and end_s:
            out.append((cid, int(start_s), int(end_s)))
        else:
            out.append((cid, None, None))
    return out


def _parse_hotspots(s: str) -> list[tuple[str, int, str]]:
    """解析 target_hotspots 为 [(chain_id, resseq, icode), ...]。"""
    out: list[tuple[str, int, str]] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        m = _HOTSPOT.match(part)
        if not m:
            continue
        cid, num_s, icode = m.groups()
        out.append((cid, int(num_s), icode if icode else " "))
    return out


def _target_index(target_id: str) -> int:
    """01_7UXQ -> 1, 12_1BI7 -> 12。"""
    try:
        return int(target_id.split("_")[0])
    except (ValueError, IndexError):
        return -1


def _is_part2_free(target_id: str) -> bool:
    """靶点 12–20 为 Part2 自由设计，只做链/范围裁剪，不做 20Å crop。"""
    return 12 <= _target_index(target_id) <= 20


def _cif_path(antigens_dir: Path, target_id: str) -> Optional[Path]:
    """优先 target_id.cif，再 6COB→6C0B；Part2 靶点 12–20 的 CIF 可能为 12_3QKG 等，按编号前缀匹配。"""
    p = antigens_dir / f"{target_id}.cif"
    if p.exists():
        return p
    alt = target_id.replace("6COB", "6C0B")
    if alt != target_id:
        p = antigens_dir / f"{alt}.cif"
        if p.exists():
            return p
    # Part2 等：config 中 target_id 与 CIF 文件名 PDB 部分可能不同（如 12_1BI7 vs 12_3QKG.cif），按编号前缀找
    num = _target_index(target_id)
    if num >= 0:
        candidates = sorted(antigens_dir.glob(f"{num:02d}_*.cif"))
        if candidates:
            return candidates[0]
    return None


def _canon_rid(rid: tuple) -> tuple:
    return (rid[0], rid[1], (rid[2].strip() or " ") if isinstance(rid[2], str) else rid[2])


def run_crop(
    designbench_root: Path,
    distance_angstrom: float = 20.0,
    dry_run: bool = False,
) -> None:
    try:
        from Bio.PDB import MMCIFParser, PDBIO
        from Bio.PDB.Chain import Chain
        from Bio.PDB.Model import Model
        from Bio.PDB.Structure import Structure
        from copy import deepcopy
    except ImportError:
        raise RuntimeError("需要 biopython: pip install biopython")

    assets = designbench_root / "assets" / "antibody_nanobody"
    config_path = assets / "config" / "target_config.csv"
    antigens_dir = assets / "antigens"
    out_dir = assets / "antigens_cropped"
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置: {config_path}")
    if not antigens_dir.is_dir():
        raise FileNotFoundError(f"未找到抗原目录: {antigens_dir}")

    rows: list[dict] = []
    with open(config_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parser = MMCIFParser(QUIET=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        target_id = (row.get("target_id") or "").strip() or (list(row.values())[0] if row else "").strip()
        if not target_id or target_id == "target_id":
            continue
        antigen_chains_raw = (row.get("antigen_chains") or "").strip().strip('"')
        chains_spec = _parse_antigen_chains(antigen_chains_raw)
        if not chains_spec:
            print(f"[skip] {target_id}: 未解析到 antigen_chains", file=sys.stderr)
            continue
        hotspots_spec = _parse_hotspots(row.get("target_hotspots") or "")
        do_distance_crop = not _is_part2_free(target_id) and bool(hotspots_spec)
        cif = _cif_path(antigens_dir, target_id)
        if not cif or not cif.exists():
            print(f"[skip] {target_id}: 未找到 CIF", file=sys.stderr)
            continue

        struct = parser.get_structure("antigen", str(cif))
        # 步骤 1：只保留指定链与 auth 残基范围
        keep_by_chain_range: set[tuple[str, tuple]] = set()
        for model in struct.get_models():
            for chain in model.get_chains():
                cid = chain.get_id()
                spec_for_chain = [(c, lo, hi) for c, lo, hi in chains_spec if c == cid]
                if not spec_for_chain:
                    continue
                for res in chain:
                    rid = res.get_id()
                    resseq = rid[1]
                    for _, lo, hi in spec_for_chain:
                        if lo is None:
                            keep_by_chain_range.add((cid, _canon_rid(rid)))
                            break
                        if lo <= resseq <= hi:
                            keep_by_chain_range.add((cid, _canon_rid(rid)))
                            break

        # 步骤 2：对非 12–20 且有 hotspot 的，再按距离 ≤ distance_angstrom 裁剪
        if do_distance_crop and hotspots_spec:
            hotspot_coords: list[list[float]] = []
            hset = {(h[0], h[1], h[2]) for h in hotspots_spec}
            for model in struct.get_models():
                for chain in model.get_chains():
                    cid = chain.get_id()
                    for res in chain:
                        rid = res.get_id()
                        if (cid, rid[1], (rid[2].strip() or " ") if isinstance(rid[2], str) else rid[2]) not in hset:
                            continue
                        for atom in res.get_atoms():
                            hotspot_coords.append(atom.get_coord().tolist())
            if hotspot_coords:
                threshold_sq = distance_angstrom * distance_angstrom
                keep_near_hotspot: set[tuple[str, tuple]] = set()
                for model in struct.get_models():
                    for chain in model.get_chains():
                        for res in chain:
                            key = (chain.get_id(), _canon_rid(res.get_id()))
                            if key not in keep_by_chain_range:
                                continue
                            for atom in res.get_atoms():
                                ac = atom.get_coord()
                                for hc in hotspot_coords:
                                    d2 = (ac[0] - hc[0]) ** 2 + (ac[1] - hc[1]) ** 2 + (ac[2] - hc[2]) ** 2
                                    if d2 <= threshold_sq:
                                        keep_near_hotspot.add(key)
                                        break
                                else:
                                    continue
                                break
                keep_by_chain_range = keep_near_hotspot

        # 写出：空壳 Model/Chain，逐个 add(deepcopy(res))，同链同 res_id 只加一次
        new_struct = Structure(struct.get_id())
        for model in struct.get_models():
            new_model = Model(model.get_id())
            for chain in model.get_chains():
                new_chain = Chain(chain.get_id())
                seen: set[tuple] = set()
                for res in chain:
                    key = (chain.get_id(), _canon_rid(res.get_id()))
                    if key not in keep_by_chain_range:
                        continue
                    rid = res.get_id()
                    if rid in seen:
                        continue
                    seen.add(rid)
                    new_chain.add(deepcopy(res))
                if new_chain.child_list:
                    new_model.add(new_chain)
            if new_model.child_list:
                new_struct.add(new_model)

        out_pdb = out_dir / f"{target_id}.pdb"
        if dry_run:
            print(f"[dry-run] 将写入 {out_pdb} (保留 {len(keep_by_chain_range)} 残基)")
            continue
        out_pdb.parent.mkdir(parents=True, exist_ok=True)
        io = PDBIO()
        io.set_structure(new_struct)
        io.save(str(out_pdb))
        print(f"  {target_id} -> {out_pdb.name} ({len(keep_by_chain_range)} residues)")

    print(f"已写入目录: {out_dir}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="按 target_config 裁剪抗原，输出到 antigens_cropped/")
    ap.add_argument("--designbench_root", type=Path, default=Path(__file__).resolve().parent.parent.parent.parent)
    ap.add_argument("--distance", type=float, default=20.0, help="hotspot 距离阈值 (Å)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    run_crop(args.designbench_root, distance_angstrom=args.distance, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
