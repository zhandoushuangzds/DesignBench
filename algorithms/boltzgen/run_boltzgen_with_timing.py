#!/usr/bin/env python3
"""
Run BoltzGen for DesignBench targets with accurate timing records.

Usage:
  python run_boltzgen_with_timing.py \
    --designbench_root /path/to/designbench \
    --output_dir /path/to/boltzgen_output \
    --boltzgen_env /DATA/disk0/qtfeng/miniforge3/envs/boltzgen \
    --task both \
    --num_designs 100 \
    --gpu 0
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def load_target_ids(designbench_root: Path) -> list[str]:
    cfg = designbench_root / "assets" / "antibody_nanobody" / "config" / "target_config.csv"
    if not cfg.exists():
        raise FileNotFoundError(f"Target config not found: {cfg}")
    import csv as csv_module
    ids = []
    with open(cfg) as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            tid = (row.get("target_id") or (list(row.values())[0] if row else "") or "").strip()
            if tid and tid != "target_id":
                ids.append(tid)
    return ids


def run_boltzgen_with_timing(
    boltzgen_cmd: str,
    yaml_path: Path,
    output_dir: Path,
    protocol: str,
    num_designs: int,
    gpu: int,
    reuse: bool = False,
    hf_cache_dir: Path = None,
    steps: str = None,
) -> tuple[bool, float, str]:
    """
    Run boltzgen for one target and return (success, elapsed_time, error_msg).
    """
    # Set CUDA_VISIBLE_DEVICES to use single GPU
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    # Set HuggingFace cache directory if provided
    # HF cache structure: hf_cache/models--boltzgen--boltzgen-1/...
    #                     hf_cache/datasets--boltzgen--inference-data/...
    # HF_HOME should point to the directory containing these cache dirs
    if hf_cache_dir and hf_cache_dir.exists():
        # HF_HOME points to the cache directory itself
        env["HF_HOME"] = str(hf_cache_dir)
        # HUGGINGFACE_HUB_CACHE also points to the cache directory
        env["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir)
        # HF_DATASETS_CACHE for datasets
        env["HF_DATASETS_CACHE"] = str(hf_cache_dir)
        # Also set TRANSFORMERS_CACHE if needed
        env["TRANSFORMERS_CACHE"] = str(hf_cache_dir)
    
    # Parse command (could be a single binary or "python -m boltzgen")
    if " " in boltzgen_cmd:
        # Command with spaces (e.g., "python -m boltzgen")
        cmd = boltzgen_cmd.split() + ["run", str(yaml_path),
                                       "--output", str(output_dir),
                                       "--protocol", protocol,
                                       "--num_designs", str(num_designs)]
    else:
        # Single binary command
        cmd = [
            boltzgen_cmd, "run", str(yaml_path),
            "--output", str(output_dir),
            "--protocol", protocol,
            "--num_designs", str(num_designs),
        ]
    
    if reuse:
        cmd.append("--reuse")
    
    # Add --steps parameter if specified (e.g., "--steps design" to only run design)
    if steps:
        cmd.extend(["--steps", steps])
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            return True, elapsed, ""
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            return False, elapsed, error_msg[:500]  # Limit error message length
    except Exception as e:
        elapsed = time.time() - start_time
        return False, elapsed, str(e)[:500]


def main():
    ap = argparse.ArgumentParser(description="Run BoltzGen with timing records")
    ap.add_argument("--designbench_root", type=Path, default=Path.cwd(),
                    help="DesignBench repo root")
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="Base output directory")
    ap.add_argument("--boltzgen_env", type=Path, required=True,
                    help="Path to boltzgen conda environment (e.g., /DATA/disk0/qtfeng/miniforge3/envs/boltzgen)")
    ap.add_argument("--task", choices=["antibody", "nanobody", "both"], default="both")
    ap.add_argument("--num_designs", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0, help="GPU ID to use (single GPU)")
    ap.add_argument("--reuse", action="store_true", help="Pass --reuse to boltzgen")
    ap.add_argument("--steps", type=str, default="design",
                    help="BoltzGen steps to run (default: 'design' to only run design, "
                         "use 'all' or omit for full pipeline)")
    ap.add_argument("--hf_cache_dir", type=Path, default=None,
                    help="HuggingFace cache directory (e.g., /home/qiantai/zihan/hf_cache). "
                         "If not specified, will try to use HF_HOME environment variable.")
    args = ap.parse_args()

    designbench_root = args.designbench_root.resolve()
    if not designbench_root.is_dir():
        raise SystemExit(f"DesignBench root not found: {designbench_root}")

    # Construct boltzgen command from environment path
    boltzgen_env = Path(args.boltzgen_env).resolve()
    if not boltzgen_env.exists():
        raise SystemExit(f"BoltzGen environment not found: {boltzgen_env}")
    
    # Determine HF cache directory
    hf_cache_dir = args.hf_cache_dir
    if hf_cache_dir is None:
        # Try to auto-detect from common locations
        possible_paths = [
            Path.home() / "hf_cache",
            Path("/home/qiantai/zihan/hf_cache"),
            Path(os.environ.get("HF_HOME", "")),
        ]
        for path in possible_paths:
            if path and path.exists() and (path / "models--boltzgen--boltzgen-1").exists():
                hf_cache_dir = path
                print(f"Auto-detected HF cache: {hf_cache_dir}", file=sys.stderr)
                break
    
    if hf_cache_dir:
        hf_cache_dir = Path(hf_cache_dir).resolve()
        if not hf_cache_dir.exists():
            print(f"Warning: HF cache directory not found: {hf_cache_dir}", file=sys.stderr)
            hf_cache_dir = None
        else:
            print(f"Using HF cache directory: {hf_cache_dir}", file=sys.stderr)
    
    # Use python from the environment to run boltzgen
    boltzgen_python = boltzgen_env / "bin" / "python"
    if not boltzgen_python.exists():
        raise SystemExit(f"Python not found in environment: {boltzgen_python}")
    
    # Check if boltzgen is available in the environment
    boltzgen_bin = boltzgen_env / "bin" / "boltzgen"
    if boltzgen_bin.exists():
        boltzgen_cmd = str(boltzgen_bin)
    else:
        # Use python -m boltzgen if boltzgen binary not found
        boltzgen_cmd = f"{boltzgen_python} -m boltzgen"
        print(f"Note: Using '{boltzgen_cmd}' (boltzgen binary not found)", file=sys.stderr)

    algo_boltz = designbench_root / "algorithms" / "boltzgen"
    configs_dir = algo_boltz / "configs"
    target_ids = load_target_ids(designbench_root)

    tasks = ["antibody", "nanobody"] if args.task == "both" else [args.task]
    protocol_map = {"antibody": "antibody-anything", "nanobody": "nanobody-anything"}

    # Create output directory
    out_root = args.output_dir.resolve()
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise SystemExit(f"Permission denied: Cannot create output directory {out_root}. Please check:\n"
                        f"  1. The path is correct and writable\n"
                        f"  2. You have write permissions to the parent directory\n"
                        f"  3. The path is not a placeholder (e.g., '/path/to/...')\n"
                        f"Error: {e}")
    except Exception as e:
        raise SystemExit(f"Failed to create output directory {out_root}: {e}")

    # Create timing log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_log = out_root / f"boltzgen_timing_{timestamp}.csv"
    
    # Prepare timing log
    timing_records = []
    total_start = time.time()

    print(f"Starting BoltzGen runs with timing...")
    print(f"Environment: {boltzgen_env}")
    print(f"GPU: {args.gpu}")
    print(f"Tasks: {tasks}")
    print(f"Targets: {len(target_ids)}")
    print(f"Timing log: {timing_log}")
    print("=" * 80)

    # Run for each task and target
    for task in tasks:
        spec_dir = configs_dir / task
        run_dir = out_root / task
        run_dir.mkdir(parents=True, exist_ok=True)
        protocol = protocol_map[task]

        print(f"\n{'='*80}")
        print(f"Task: {task.upper()}")
        print(f"{'='*80}\n")

        for idx, target_id in enumerate(target_ids, 1):
            yaml_path = spec_dir / f"{target_id}.yaml"
            if not yaml_path.exists():
                print(f"[{idx}/{len(target_ids)}] Skip {target_id}: spec not found {yaml_path}", file=sys.stderr)
                timing_records.append({
                    "task": task,
                    "target_id": target_id,
                    "status": "skipped",
                    "elapsed_seconds": 0.0,
                    "error": "Config not found",
                    "timestamp": datetime.now().isoformat(),
                })
                continue

            target_out = run_dir / target_id
            print(f"[{idx}/{len(target_ids)}] Running {target_id}...", end=" ", flush=True)
            
            success, elapsed, error_msg = run_boltzgen_with_timing(
                str(boltzgen_cmd),
                yaml_path,
                target_out,
                protocol,
                args.num_designs,
                args.gpu,
                args.reuse,
                hf_cache_dir,
                args.steps,
            )

            if success:
                print(f"✓ Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
                status = "success"
            else:
                print(f"✗ Failed in {elapsed:.1f}s: {error_msg[:100]}")
                status = "failed"

            timing_records.append({
                "task": task,
                "target_id": target_id,
                "status": status,
                "elapsed_seconds": round(elapsed, 2),
                "elapsed_minutes": round(elapsed / 60, 2),
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
            })

    total_elapsed = time.time() - total_start

    # Write timing log
    with open(timing_log, "w", newline="") as f:
        fieldnames = ["task", "target_id", "status", "elapsed_seconds", "elapsed_minutes", "error", "timestamp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_records)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min, {total_elapsed/3600:.2f} hours)")
    print(f"Timing log saved to: {timing_log}")
    
    # Statistics by task
    for task in tasks:
        task_records = [r for r in timing_records if r["task"] == task]
        success_count = sum(1 for r in task_records if r["status"] == "success")
        failed_count = sum(1 for r in task_records if r["status"] == "failed")
        skipped_count = sum(1 for r in task_records if r["status"] == "skipped")
        total_time = sum(r["elapsed_seconds"] for r in task_records)
        avg_time = total_time / len(task_records) if task_records else 0
        
        print(f"\n{task.upper()}:")
        print(f"  Success: {success_count}/{len(task_records)}")
        print(f"  Failed: {failed_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Average time per target: {avg_time:.1f}s ({avg_time/60:.1f} min)")


if __name__ == "__main__":
    main()
