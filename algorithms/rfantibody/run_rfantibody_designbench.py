#!/usr/bin/env python3
"""
Run RFantibody for all 22 DesignBench antibody/nanobody targets (100 designs per target),
then convert output to DesignBench design_dir + cdr_info.csv.

Prerequisites:
- RFantibody installed (see models/RFantibody/README.md). Activate env or use uv run.
- DesignBench assets: antigens (CIF), scaffolds (hu-4D5-8_Fv.pdb, h-NbBCII10.pdb).

Steps:
1. For each target: convert antigen CIF→PDB (if needed), run rfdiffusion → proteinmpnn → qvextract.
2. Convert per-target PDBs to DesignBench format (design_dir + cdr_info.csv).

Usage:
  # From designbench root; RFantibody in ../models/RFantibody
  python algorithms/rfantibody/run_rfantibody_designbench.py \
    --designbench_root /path/to/designbench \
    --rfantibody_root /path/to/models/RFantibody \
    --output_dir /path/to/rfantibody_designbench_output \
    --task both \
    --num_designs 100

  # Only convert existing runs
  python algorithms/rfantibody/run_rfantibody_designbench.py \
    --designbench_root . --output_dir out --convert_only --task both
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def load_target_config(designbench_root: Path) -> list[dict]:
    cfg = designbench_root / "assets" / "antibody_nanobody" / "config" / "target_config.csv"
    if not cfg.exists():
        raise FileNotFoundError(f"Target config not found: {cfg}")
    rows = []
    with open(cfg) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def antigen_path_for_target(designbench_root: Path, target_id: str) -> Path | None:
    antigens_dir = designbench_root / "assets" / "antibody_nanobody" / "antigens"
    p = antigens_dir / f"{target_id}.cif"
    if p.exists():
        return p
    alt = target_id.replace("6COB", "6C0B")
    if alt != target_id:
        p = antigens_dir / f"{alt}.cif"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Run RFantibody for 22 DesignBench targets and convert to DesignBench format")
    ap.add_argument("--designbench_root", type=Path, default=Path(__file__).resolve().parent.parent.parent)
    ap.add_argument("--rfantibody_root", type=Path, default=None,
                    help="RFantibody repo root (for uv run / rfdiffusion). If unset, use PATH.")
    ap.add_argument("--output_dir", type=Path, default=Path("rfantibody_designbench_output"))
    ap.add_argument("--task", choices=["antibody", "nanobody", "both"], default="both")
    ap.add_argument("--num_designs", type=int, default=100)
    ap.add_argument("--convert_only", action="store_true", help="Only convert existing runs to DesignBench format")
    ap.add_argument("--rfdiffusion_cmd", type=str, default="rfdiffusion")
    ap.add_argument("--proteinmpnn_cmd", type=str, default="proteinmpnn")
    ap.add_argument("--qvextract_cmd", type=str, default="qvextract")
    args = ap.parse_args()

    designbench_root = args.designbench_root.resolve()
    if not designbench_root.is_dir():
        raise SystemExit(f"DesignBench root not found: {designbench_root}")

    algo_dir = designbench_root / "algorithms" / "rfantibody"
    scaffolds_ab = designbench_root / "assets" / "antibody_nanobody" / "scaffolds" / "antibody" / "hu-4D5-8_Fv.pdb"
    scaffolds_nano = designbench_root / "assets" / "antibody_nanobody" / "scaffolds" / "nanobody" / "h-NbBCII10.pdb"
    if not scaffolds_ab.exists():
        raise SystemExit(f"Antibody scaffold not found: {scaffolds_ab}")
    if not scaffolds_nano.exists():
        raise SystemExit(f"Nanobody scaffold not found: {scaffolds_nano}")

    targets = load_target_config(designbench_root)
    target_ids = [r.get("target_id", (list(r.values())[0] if r else "") or "").strip() for r in targets]
    target_ids = [t for t in target_ids if t and t != "target_id"]

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    work_dir = out_root / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # CIF→PDB helper
    def ensure_target_pdb(target_id: str) -> Path | None:
        cif = antigen_path_for_target(designbench_root, target_id)
        if not cif:
            return None
        pdb_dir = work_dir / "antigen_pdb"
        pdb_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = pdb_dir / f"{target_id}.pdb"
        if pdb_path.exists() and pdb_path.stat().st_mtime >= cif.stat().st_mtime:
            return pdb_path
        try:
            subprocess.run(
                [sys.executable, str(algo_dir / "cif_to_pdb.py"), str(cif), str(pdb_path)],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"Skip {target_id}: CIF→PDB failed: {e}", file=sys.stderr)
            return None
        return pdb_path

    tasks = ["antibody", "nanobody"] if args.task == "both" else [args.task]
    framework_map = {"antibody": str(scaffolds_ab), "nanobody": str(scaffolds_nano)}
    loops_ab = "H1:7,H2:6,H3:5-13,L1:8-13,L2:7,L3:9-11"
    loops_nano = "H1:7,H2:6,H3:5-13"

    if not args.convert_only:
        rf_env = None
        if args.rfantibody_root:
            rf_root = Path(args.rfantibody_root).resolve()
            venv_bin = rf_root / ".venv" / "bin"
            rf_env = {**os.environ, "PATH": f"{venv_bin}:{rf_root}:{os.environ.get('PATH', '')}"}
        for task in tasks:
            run_dir = out_root / task
            run_dir.mkdir(parents=True, exist_ok=True)
            framework = framework_map[task]
            loops = loops_ab if task == "antibody" else loops_nano
            for row in targets:
                target_id = row.get("target_id", "").strip() or (list(row.values())[0] if row else "")
                if not target_id or target_id == "target_id":
                    continue
                target_pdb = ensure_target_pdb(target_id)
                if not target_pdb:
                    print(f"Skip {target_id}: no antigen PDB", file=sys.stderr)
                    continue
                hotspots_raw = (row.get("target_hotspots") or "").strip()
                hotspots = ",".join(h.strip() for h in hotspots_raw.split(",") if h.strip()) if hotspots_raw else None
                target_out = run_dir / target_id
                target_out.mkdir(parents=True, exist_ok=True)
                qv1 = target_out / "rfdiffusion.qv"
                qv2 = target_out / "proteinmpnn.qv"
                extracted_dir = target_out / "extracted"
                target_pdb_str = str(target_pdb.resolve())
                framework_str = str(Path(framework).resolve())
                rf_cwd = (args.rfantibody_root or designbench_root).resolve()
                # 1. rfdiffusion
                cmd_rf = [
                    args.rfdiffusion_cmd,
                    "-t", target_pdb_str,
                    "-f", framework_str,
                    "-q", str(qv1.resolve()),
                    "-n", str(args.num_designs),
                    "-l", loops,
                ]
                if hotspots:
                    cmd_rf += ["-h", hotspots]
                print(f"Run: {' '.join(cmd_rf)}")
                ret = subprocess.run(cmd_rf, cwd=str(rf_cwd), env=rf_env)
                if ret.returncode != 0:
                    print(f"rfdiffusion failed for {target_id}", file=sys.stderr)
                    continue
                # 2. proteinmpnn (1 seq per struct → 100 designs)
                cmd_pm = [
                    args.proteinmpnn_cmd,
                    "-q", str(qv1.resolve()),
                    "--output-quiver", str(qv2.resolve()),
                    "-n", "1",
                ]
                print(f"Run: {' '.join(cmd_pm)}")
                ret = subprocess.run(cmd_pm, cwd=str(rf_cwd), env=rf_env)
                if ret.returncode != 0:
                    print(f"proteinmpnn failed for {target_id}", file=sys.stderr)
                    continue
                # 3. qvextract
                extracted_dir.mkdir(parents=True, exist_ok=True)
                ret = subprocess.run(
                    [args.qvextract_cmd, str(qv2.resolve()), "-o", str(extracted_dir.resolve())],
                    cwd=str(rf_cwd),
                    env=rf_env,
                )
                if ret.returncode != 0:
                    print(f"qvextract failed for {target_id}", file=sys.stderr)

    # Convert to DesignBench format
    conv_script = algo_dir / "convert_rfantibody_to_designbench.py"
    if not conv_script.exists():
        raise SystemExit(f"Converter not found: {conv_script}")
    for task in tasks:
        run_dirs = out_root / task
        design_dir = out_root / "designbench_format" / task
        cdr_csv = design_dir / "cdr_info.csv"
        design_dir.mkdir(parents=True, exist_ok=True)
        # Converter expects PDBs in run_dirs/<target_id>/extracted/ or run_dirs/<target_id>/
        subprocess.run(
            [
                sys.executable, str(conv_script),
                "--run_dirs", str(run_dirs),
                "--design_dir", str(design_dir),
                "--cdr_info_csv", str(cdr_csv),
                "--task", task,
                "--max_per_target", str(args.num_designs),
                "--extracted_subdir", "extracted",
            ],
            check=True,
            cwd=str(algo_dir),
        )
    print("DesignBench-ready outputs under", out_root / "designbench_format")
    print("Antibody: design_dir =", out_root / "designbench_format" / "antibody", ", cdr_info_csv =", out_root / "designbench_format" / "antibody" / "cdr_info.csv")
    print("Nanobody: design_dir =", out_root / "designbench_format" / "nanobody", ", cdr_info_csv =", out_root / "designbench_format" / "nanobody" / "cdr_info.csv")


if __name__ == "__main__":
    main()
