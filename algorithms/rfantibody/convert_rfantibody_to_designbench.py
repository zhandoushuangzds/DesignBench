#!/usr/bin/env python3
"""
Convert RFantibody per-target output to DesignBench format.

RFantibody per-target .qv files (when write_trajectory=True):
- rfdiffusion.qv         : 主输出，每个 design 一个 tag（如 rfdiffusion_0），存的是扩散结束的最终结构 X0。
                            用 qvextract 提出到 extracted/*.pdb 后，本脚本据此生成 DesignBench design_dir。
- rfdiffusion_Xt-1_traj.qv : 轨迹，每 design 一 tag（如 rfdiffusion_0_Xt-1），每 tag 内为多帧去噪坐标 Xt-1。
- rfdiffusion_pX0_traj.qv  : 轨迹，每 design 一 tag（如 rfdiffusion_0_pX0），每 tag 内为多帧模型预测 p(X0|Xt)。

转换流程：对 rfdiffusion.qv 运行 qvextract -o extracted，再运行本脚本（--extracted_subdir extracted）。

- Copies/renames PDBs to design_dir as {target_id}_0.pdb ... {target_id}_99.pdb.
- Writes cdr_info.csv with 1-based inclusive CDR only (no sequences).

Usage:
  python convert_rfantibody_to_designbench.py \
    --run_dirs output/antibody \
    --design_dir designbench_designs/antibody \
    --cdr_info_csv designbench_designs/antibody/cdr_info.csv \
    --task antibody

  python convert_rfantibody_to_designbench.py \
    --run_dirs output/nanobody \
    --design_dir designbench_designs/nanobody \
    --cdr_info_csv designbench_designs/nanobody/cdr_info.csv \
    --task nanobody
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

# DesignBench CDR: 1-based inclusive. Default antibody (Fab) from README.
DEFAULT_AB_CDR = {
    "h_cdr1_start": 30, "h_cdr1_end": 35,
    "h_cdr2_start": 50, "h_cdr2_end": 65,
    "h_cdr3_start": 95, "h_cdr3_end": 102,
    "l_cdr1_start": 24, "l_cdr1_end": 34,
    "l_cdr2_start": 50, "l_cdr2_end": 56,
    "l_cdr3_start": 89, "l_cdr3_end": 97,
}
DEFAULT_NANO_CDR = {
    "h_cdr1_start": 26, "h_cdr1_end": 35,
    "h_cdr2_start": 51, "h_cdr2_end": 57,
    "h_cdr3_start": 95, "h_cdr3_end": 118,
    "l_cdr1_start": "", "l_cdr1_end": "",
    "l_cdr2_start": "", "l_cdr2_end": "",
    "l_cdr3_start": "", "l_cdr3_end": "",
}


def get_chain_ids_from_pdb(pdb_path: Path) -> list:
    """Return ordered list of chain IDs in the PDB. First = H, second = L (if antibody)."""
    seen = set()
    order = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM ") or line.startswith("HETATM"):
                ch = line[21:22].strip()
                if ch and ch not in seen:
                    seen.add(ch)
                    order.append(ch)
    return order


def main():
    ap = argparse.ArgumentParser(description="Convert RFantibody output to DesignBench design_dir + cdr_info.csv")
    ap.add_argument("--run_dirs", type=Path, required=True,
                    help="Directory of per-target run dirs (e.g. output/antibody with 01_7UXQ, ...)")
    ap.add_argument("--design_dir", type=Path, required=True)
    ap.add_argument("--cdr_info_csv", type=Path, required=True)
    ap.add_argument("--task", choices=["antibody", "nanobody"], required=True)
    ap.add_argument("--max_per_target", type=int, default=100)
    ap.add_argument("--extracted_subdir", type=str, default="extracted",
                    help="Subdir under each run containing PDBs (default: extracted)")
    args = ap.parse_args()

    run_base = args.run_dirs.resolve()
    if not run_base.is_dir():
        raise SystemExit(f"Not a directory: {run_base}")

    design_dir = args.design_dir.resolve()
    design_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id", "h_chain", "l_chain",
        "h_cdr1_start", "h_cdr1_end", "h_cdr2_start", "h_cdr2_end", "h_cdr3_start", "h_cdr3_end",
        "l_cdr1_start", "l_cdr1_end", "l_cdr2_start", "l_cdr2_end", "l_cdr3_start", "l_cdr3_end",
    ]
    cdr_rows = []
    default_cdr = DEFAULT_AB_CDR if args.task == "antibody" else DEFAULT_NANO_CDR

    for target_dir in sorted(run_base.iterdir()):
        if not target_dir.is_dir():
            continue
        target_id = target_dir.name
        if not (len(target_id) >= 6 and target_id[:2].isdigit() and target_id[2] == "_"):
            continue
        # Prefer extracted/ then fallback to run root for *.pdb
        pdb_dir = target_dir / args.extracted_subdir
        if not pdb_dir.is_dir():
            pdb_dir = target_dir
        pdbs = sorted(pdb_dir.glob("*.pdb"))
        pdbs = pdbs[: args.max_per_target]
        if len(pdbs) < args.max_per_target:
            print(f"Warning: {target_id} has only {len(pdbs)} PDBs", file=sys.stderr)
        for i, src in enumerate(pdbs):
            dest = design_dir / f"{target_id}_{i}.pdb"
            shutil.copy2(src, dest)
        print(f"Copied {len(pdbs)} PDBs for {target_id} -> {design_dir}")
        chain_ids = get_chain_ids_from_pdb(pdbs[0]) if pdbs else []
        h_chain = chain_ids[0] if len(chain_ids) >= 1 else "A"
        l_chain = chain_ids[1] if args.task == "antibody" and len(chain_ids) >= 2 else ""
        row = {"id": target_id, "h_chain": h_chain, "l_chain": l_chain, **default_cdr}
        cdr_rows.append(row)

    cdr_path = args.cdr_info_csv.resolve()
    cdr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cdr_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(cdr_rows)
    print(f"Wrote {len(cdr_rows)} rows to {cdr_path}")
    print(f"DesignBench design_dir: {design_dir} ({sum(1 for _ in design_dir.glob('*.pdb'))} PDBs)")


if __name__ == "__main__":
    main()
