#!/usr/bin/env python3
"""
Run BoltzGen for all 22 DesignBench antibody/nanobody targets (100 designs per target),
then convert output to DesignBench design_dir + cdr_info.csv.

Prerequisites:
- BoltzGen installed (pip install boltzgen) and `boltzgen` on PATH.
- DesignBench repo with assets/antibody_nanobody/antigens/*.cif and configs.

Steps:
1. Generate BoltzGen spec YAMLs per target (antibody and nanobody) if missing.
2. For each target: boltzgen run <spec.yaml> --output <out>/<task>/<target_id> \\
     --protocol antibody-anything|nanobody-anything --num_designs 100 [--reuse]
3. Convert each task's per-target runs to one design_dir + cdr_info.csv.

Usage:
  # From designbench repo root (or set --designbench_root)
  python algorithms/boltzgen/run_boltzgen_designbench.py \
    --designbench_root /path/to/designbench \
    --output_dir /path/to/boltzgen_runs \
    --task both \
    --num_designs 100

  # Only generate specs (no run)
  python algorithms/boltzgen/run_boltzgen_designbench.py --designbench_root . --output_dir out --gen_specs_only

  # Only convert existing runs (skip BoltzGen)
  python algorithms/boltzgen/run_boltzgen_designbench.py --designbench_root . --output_dir out --convert_only --task both
"""

import argparse
import subprocess
import sys
from pathlib import Path


def load_target_ids(designbench_root: Path) -> list[str]:
    cfg = designbench_root / "assets" / "antibody_nanobody" / "config" / "target_config.csv"
    if not cfg.exists():
        raise FileNotFoundError(f"Target config not found: {cfg}")
    import csv
    ids = []
    with open(cfg) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = (row.get("target_id") or (list(row.values())[0] if row else "") or "").strip()
            if tid and tid != "target_id":
                ids.append(tid)
    return ids


def main():
    ap = argparse.ArgumentParser(description="Run BoltzGen for 22 DesignBench targets and convert to DesignBench format")
    ap.add_argument("--designbench_root", type=Path, default=Path(__file__).resolve().parent.parent.parent,
                    help="DesignBench repo root")
    ap.add_argument("--output_dir", type=Path, default=Path("boltzgen_designbench_output"),
                    help="Base output dir: <output_dir>/antibody/<target_id>, <output_dir>/nanobody/<target_id>")
    ap.add_argument("--task", choices=["antibody", "nanobody", "both"], default="both")
    ap.add_argument("--num_designs", type=int, default=100)
    ap.add_argument("--reuse", action="store_true", help="Pass --reuse to boltzgen run")
    ap.add_argument("--gen_specs_only", action="store_true", help="Only generate YAML specs, do not run BoltzGen")
    ap.add_argument("--convert_only", action="store_true", help="Only convert existing runs to DesignBench format")
    ap.add_argument("--boltzgen_cmd", type=str, default="boltzgen", help="boltzgen CLI command")
    args = ap.parse_args()

    designbench_root = args.designbench_root.resolve()
    if not designbench_root.is_dir():
        raise SystemExit(f"DesignBench root not found: {designbench_root}")

    algo_boltz = designbench_root / "algorithms" / "boltzgen"
    configs_dir = algo_boltz / "configs"
    target_ids = load_target_ids(designbench_root)

    # 1. Generate specs if needed
    if not args.convert_only:
        if not (configs_dir / "antibody").exists() or not (configs_dir / "nanobody").exists():
            spec_script = algo_boltz / "generate_designbench_specs.py"
            if not spec_script.exists():
                raise SystemExit(f"Spec generator not found: {spec_script}")
            subprocess.run(
                [sys.executable, str(spec_script), "--designbench_root", str(designbench_root), "--out_dir", str(configs_dir)],
                check=True,
                cwd=str(designbench_root),
            )
        if args.gen_specs_only:
            print("Specs generated (--gen_specs_only). Exiting.")
            return

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = ["antibody", "nanobody"] if args.task == "both" else [args.task]
    protocol_map = {"antibody": "antibody-anything", "nanobody": "nanobody-anything"}

    # 2. Run BoltzGen per target
    if not args.convert_only:
        for task in tasks:
            spec_dir = configs_dir / task
            run_dir = out_root / task
            run_dir.mkdir(parents=True, exist_ok=True)
            for target_id in target_ids:
                yaml_path = spec_dir / f"{target_id}.yaml"
                if not yaml_path.exists():
                    print(f"Skip {target_id}: spec not found {yaml_path}", file=sys.stderr)
                    continue
                target_out = run_dir / target_id
                cmd = [
                    args.boltzgen_cmd, "run", str(yaml_path),
                    "--output", str(target_out),
                    "--protocol", protocol_map[task],
                    "--num_designs", str(args.num_designs),
                ]
                if args.reuse:
                    cmd.append("--reuse")
                print(f"Run: {' '.join(cmd)}")
                subprocess.run(cmd, cwd=str(designbench_root), check=False)

    # 3. Convert to DesignBench format
    conv_script = algo_boltz / "convert_boltzgen_to_designbench.py"
    if not conv_script.exists():
        raise SystemExit(f"Converter not found: {conv_script}")
    for task in tasks:
        run_dirs = out_root / task
        design_dir = out_root / "designbench_format" / task
        cdr_csv = design_dir / "cdr_info.csv"
        design_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable, str(conv_script),
                "--run_dirs", str(run_dirs),
                "--design_dir", str(design_dir),
                "--cdr_info_csv", str(cdr_csv),
                "--task", task,
                "--max_per_target", str(args.num_designs),
            ],
            check=True,
            cwd=str(algo_boltz),
        )
    print(f"DesignBench-ready outputs under {out_root / 'designbench_format'}")
    print("Antibody pipeline: design_dir=", out_root / "designbench_format" / "antibody", "cdr_info_csv=", out_root / "designbench_format" / "antibody" / "cdr_info.csv")
    print("Nanobody pipeline: design_dir=", out_root / "designbench_format" / "nanobody", "cdr_info_csv=", out_root / "designbench_format" / "nanobody" / "cdr_info.csv")


if __name__ == "__main__":
    main()
