#!/usr/bin/env python3
"""
Run RFantibody for BenchCore antibody + nanobody tasks with accurate timing (single GPU).

- By default skips proteinmpnn: only rfdiffusion -> qvextract (backbone PDBs). Use --run_proteinmpnn to run inverse fold.
- Pass --rfdiffusion_extra to add Hydra overrides (e.g. diffuser.T=25 for fewer steps, or any inference.* key).
Uses CUDA_VISIBLE_DEVICES to pin to one GPU. Records per-step and per-target elapsed time, writes CSV.

Usage:
  # From benchcore root; use GPU 1
  python algorithms/rfantibody/run_rfantibody_with_timing.py \
    --benchcore_root /path/to/benchcore \
    --rfantibody_root /path/to/models/RFantibody \
    --output_dir /path/to/rfantibody_output \
    --task both \
    --num_designs 100 \
    --gpu 1
"""

import argparse
import csv
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def load_target_config(benchcore_root: Path) -> list[dict]:
    cfg = benchcore_root / "assets" / "antibody_nanobody" / "config" / "target_config.csv"
    if not cfg.exists():
        raise FileNotFoundError(f"Target config not found: {cfg}")
    rows = []
    with open(cfg) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def antigen_path_for_target(benchcore_root: Path, target_id: str) -> Path | None:
    """CIF 路径（用于 CIF→PDB 回退）。"""
    antigens_dir = benchcore_root / "assets" / "antibody_nanobody" / "antigens"
    p = antigens_dir / f"{target_id}.cif"
    if p.exists():
        return p
    alt = target_id.replace("6COB", "6C0B")
    if alt != target_id:
        p = antigens_dir / f"{alt}.cif"
        if p.exists():
            return p
    return None


def antigen_pdb_for_target(benchcore_root: Path, target_id: str, work_dir: Path, ensure_target_pdb_fn) -> Path | None:
    """优先使用 assets 下已裁剪的 antigens_cropped/{target_id}.pdb，否则用 ensure_target_pdb 从 CIF 转 PDB。"""
    cropped = benchcore_root / "assets" / "antibody_nanobody" / "antigens_cropped" / f"{target_id}.pdb"
    if cropped.exists():
        return cropped
    return ensure_target_pdb_fn(target_id)


def run_cmd_with_timing(
    cmd: list,
    cwd: str,
    env: dict | None,
    step_name: str,
) -> tuple[bool, float, str]:
    """Run command; return (success, elapsed_seconds, error_msg). On Ctrl+C, kill subprocess and exit cleanly."""
    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print("\n[Ctrl+C] Interrupted; subprocess killed. Exiting.", file=sys.stderr)
            sys.exit(130)
        elapsed = time.time() - start
        if proc.returncode == 0:
            return True, elapsed, ""
        err = (stderr or stdout or "Unknown error").strip()[:500]
        return False, elapsed, err
    except KeyboardInterrupt:
        raise
    except Exception as e:
        elapsed = time.time() - start
        return False, elapsed, str(e)[:500]


def main():
    ap = argparse.ArgumentParser(description="Run RFantibody with timing (single GPU)")
    ap.add_argument("--benchcore_root", type=Path, default=Path(__file__).resolve().parent.parent.parent)
    ap.add_argument("--rfantibody_root", type=Path, default=None,
                    help="RFantibody repo root. If unset, rfdiffusion/proteinmpnn/qvextract must be on PATH.")
    ap.add_argument("--output_dir", type=Path, default=Path("rfantibody_benchcore_output"))
    ap.add_argument("--task", choices=["antibody", "nanobody", "both"], default="both")
    ap.add_argument("--num_designs", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=1, help="GPU ID to use (e.g. 1 for GPU1)")
    ap.add_argument("--weights_dir", type=Path, default=None,
                    help="RFantibody weights directory (e.g. /path/to/RFantibody/weights). "
                         "Must contain RFdiffusion_Ab.pt and ProteinMPNN_v48_noise_0.2.pt. "
                         "If not set, RFantibody uses its default or RFANTIBODY_WEIGHTS env.")
    ap.add_argument("--skip_proteinmpnn", action="store_true", default=True,
                    help="Skip proteinmpnn (default: True). Only rfdiffusion -> qvextract; output is backbone PDBs.")
    ap.add_argument("--run_proteinmpnn", action="store_true",
                    help="Run proteinmpnn after rfdiffusion; default is to skip it.")
    ap.add_argument("--rfdiffusion_extra", type=str, action="append", default=None,
                    help="Extra Hydra overrides for rfdiffusion, e.g. --rfdiffusion_extra diffuser.T=25. Can repeat.")
    ap.add_argument("--rfdiffusion_cmd", type=str, default="rfdiffusion")
    ap.add_argument("--proteinmpnn_cmd", type=str, default="proteinmpnn")
    ap.add_argument("--qvextract_cmd", type=str, default="qvextract")
    ap.add_argument("--skip_convert", action="store_true", help="Do not run BenchCore format conversion at end")
    ap.add_argument(
        "--target_slice",
        type=str,
        default=None,
        help="Optional subset of targets as 'start-end' (1-based, inclusive), e.g. 1-11, to split work across GPUs.",
    )
    ap.add_argument(
        "--max_concurrent_targets",
        type=int,
        default=1,
        help="Maximum number of targets to run in parallel on this GPU (per task). "
             "Use >1 to let one GPU handle multiple targets simultaneously.",
    )
    args = ap.parse_args()

    benchcore_root = args.benchcore_root.resolve()
    if not benchcore_root.is_dir():
        raise SystemExit(f"BenchCore root not found: {benchcore_root}")

    algo_dir = benchcore_root / "algorithms" / "rfantibody"
    scaffolds_ab = benchcore_root / "assets" / "antibody_nanobody" / "scaffolds" / "antibody" / "hu-4D5-8_Fv.pdb"
    scaffolds_nano = benchcore_root / "assets" / "antibody_nanobody" / "scaffolds" / "nanobody" / "h-NbBCII10.pdb"
    if not scaffolds_ab.exists():
        raise SystemExit(f"Antibody scaffold not found: {scaffolds_ab}")
    if not scaffolds_nano.exists():
        raise SystemExit(f"Nanobody scaffold not found: {scaffolds_nano}")

    targets = load_target_config(benchcore_root)
    target_ids = [r.get("target_id", (list(r.values())[0] if r else "") or "").strip() for r in targets]
    target_ids = [t for t in target_ids if t and t != "target_id"]

    selected_target_ids: set[str] | None = None
    if args.target_slice:
        try:
            start_s, end_s = args.target_slice.split("-", 1)
            start_i = int(start_s)
            end_i = int(end_s)
            if start_i < 1 or end_i < start_i or end_i > len(target_ids):
                raise ValueError
        except Exception:
            raise SystemExit(
                f"Invalid --target_slice '{args.target_slice}'. Use 'start-end' with 1 <= start <= end <= {len(target_ids)}."
            )
        selected_list = target_ids[start_i - 1 : end_i]
        selected_target_ids = set(selected_list)
        print(
            f"Using target slice {start_i}-{end_i}: {len(selected_list)} targets "
            f"out of {len(target_ids)} total."
        )

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    work_dir = out_root / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Env: single GPU + optional weights
    base_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    if args.rfantibody_root:
        rf_root = Path(args.rfantibody_root).resolve()
        venv_bin = rf_root / ".venv" / "bin"
        base_env["PATH"] = f"{venv_bin}:{rf_root}:{os.environ.get('PATH', '')}"
    weights_rf = None
    weights_pm = None
    if args.weights_dir:
        wdir = Path(args.weights_dir).resolve()
        base_env["RFANTIBODY_WEIGHTS"] = str(wdir)
        if (wdir / "RFdiffusion_Ab.pt").exists():
            weights_rf = str(wdir / "RFdiffusion_Ab.pt")
        if (wdir / "ProteinMPNN_v48_noise_0.2.pt").exists():
            weights_pm = str(wdir / "ProteinMPNN_v48_noise_0.2.pt")
    rf_cwd = str((args.rfantibody_root or benchcore_root).resolve())

    # Early check: can we get antigen PDB for first target? (avoid skipping all 22 silently)
    first_id = target_ids[0] if target_ids else None
    if first_id:
        ac = antigen_path_for_target(benchcore_root, first_id)
        if not ac:
            raise SystemExit(
                f"Antigen not found for first target {first_id}. "
                f"Expected CIF at: {benchcore_root / 'assets' / 'antibody_nanobody' / 'antigens' / f'{first_id}.cif'}"
            )
        pdb_dir = work_dir / "antigen_pdb"
        pdb_dir.mkdir(parents=True, exist_ok=True)
        test_pdb = pdb_dir / f"{first_id}.pdb"
        try:
            subprocess.run(
                [sys.executable, str(algo_dir / "cif_to_pdb.py"), str(ac), str(test_pdb)],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(algo_dir),
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or e.stdout or "").strip()[:500]
            raise SystemExit(
                f"CIF→PDB conversion failed for {first_id}. {stderr}\n"
                "Install biopython: pip install biopython"
            )
        except FileNotFoundError:
            raise SystemExit("cif_to_pdb.py not found. Run from benchcore root.")

    # CIF→PDB (fail once with clear error so user sees e.g. "pip install biopython")
    _cif_to_pdb_error_logged = []

    def ensure_target_pdb(target_id: str) -> Path | None:
        cif = antigen_path_for_target(benchcore_root, target_id)
        if not cif:
            if target_id not in _cif_to_pdb_error_logged:
                print(f"Antigen CIF not found for {target_id} (expected {benchcore_root / 'assets' / 'antibody_nanobody' / 'antigens' / f'{target_id}.cif'})", file=sys.stderr)
                _cif_to_pdb_error_logged.append(target_id)
            return None
        pdb_dir = work_dir / "antigen_pdb"
        pdb_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = pdb_dir / f"{target_id}.pdb"
        if pdb_path.exists() and pdb_path.stat().st_mtime >= cif.stat().st_mtime:
            return pdb_path
        try:
            r = subprocess.run(
                [sys.executable, str(algo_dir / "cif_to_pdb.py"), str(cif), str(pdb_path)],
                capture_output=True,
                text=True,
                cwd=str(algo_dir),
            )
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "").strip()[:400]
                if "cif_to_pdb" not in str(_cif_to_pdb_error_logged):
                    print(f"CIF→PDB failed for {target_id}: {err}", file=sys.stderr)
                    print("Tip: install biopython: pip install biopython", file=sys.stderr)
                    _cif_to_pdb_error_logged.append("cif_to_pdb")
                return None
        except Exception as e:
            if "cif_to_pdb" not in str(_cif_to_pdb_error_logged):
                print(f"CIF→PDB error for {target_id}: {e}", file=sys.stderr)
                print("Tip: install biopython: pip install biopython", file=sys.stderr)
                _cif_to_pdb_error_logged.append("cif_to_pdb")
            return None
        return pdb_path

    tasks = ["antibody", "nanobody"] if args.task == "both" else [args.task]
    framework_map = {"antibody": str(scaffolds_ab.resolve()), "nanobody": str(scaffolds_nano.resolve())}
    loops_ab = "H1:7,H2:6,H3:5-13,L1:8-13,L2:7,L3:9-11"
    loops_nano = "H1:7,H2:6,H3:5-13"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_log = out_root / f"rfantibody_timing_{timestamp}.csv"
    timing_records = []
    fieldnames = [
        "task",
        "target_id",
        "status",
        "rfdiffusion_seconds",
        "proteinmpnn_seconds",
        "qvextract_seconds",
        "total_seconds",
        "error",
        "timestamp",
    ]
    # Initialize log file with header so that even if interrupted, partial timing is recorded.
    with open(timing_log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
    write_lock = threading.Lock()
    total_wall_start = time.time()

    print(f"RFantibody with timing — GPU: {args.gpu}, Tasks: {tasks}, Targets: {len(target_ids)}, Num designs: {args.num_designs}")
    print(f"Timing log: {timing_log}")
    print("=" * 80)

    def _append_timing_row(rec: dict) -> None:
        """Append a single timing record to CSV immediately (thread-safe) and keep in-memory copy."""
        if not rec:
            return
        timing_records.append(rec)
        with write_lock:
            with open(timing_log, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writerow(rec)

    def _run_single_target(task: str, row_idx: int, row: dict) -> dict:
        nonlocal benchcore_root, work_dir, framework_map, loops_ab, loops_nano

        target_id = row.get("target_id", "").strip() or (list(row.values())[0] if row else "")
        if not target_id or target_id == "target_id":
            return {}
        if selected_target_ids is not None and target_id not in selected_target_ids:
            return {}

        run_dir = out_root / task
        run_dir.mkdir(parents=True, exist_ok=True)
        framework = framework_map[task]
        loops = loops_ab if task == "antibody" else loops_nano

        target_pdb = antigen_pdb_for_target(benchcore_root, target_id, work_dir, ensure_target_pdb)
        if not target_pdb:
            print(f"[{row_idx}/{len(targets)}] Skip {target_id}: no antigen PDB", file=sys.stderr)
            return {
                "task": task,
                "target_id": target_id,
                "status": "skipped",
                "rfdiffusion_seconds": 0.0,
                "proteinmpnn_seconds": 0.0,
                "qvextract_seconds": 0.0,
                "total_seconds": 0.0,
                "error": "No antigen PDB",
                "timestamp": datetime.now().isoformat(),
            }

        hotspots_raw = (row.get("target_hotspots") or "").strip()
        hotspots = ",".join(h.strip() for h in hotspots_raw.split(",") if h.strip()) if hotspots_raw else None
        target_out = run_dir / target_id
        target_out.mkdir(parents=True, exist_ok=True)
        qv1 = target_out / "rfdiffusion.qv"
        qv2 = target_out / "proteinmpnn.qv"
        extracted_dir = target_out / "extracted"

        t_rf, t_pm, t_qv = 0.0, 0.0, 0.0
        status = "success"
        err_msg = ""

        # 1. rfdiffusion
        cmd_rf = [
            args.rfdiffusion_cmd,
            "-t", str(target_pdb.resolve()),
            "-f", framework,
            "-q", str(qv1.resolve()),
            "-n", str(args.num_designs),
            "-l", loops,
        ]
        if weights_rf:
            cmd_rf += ["-w", weights_rf]
        if hotspots:
            cmd_rf += ["-h", hotspots]
        if args.rfdiffusion_extra:
            for e in args.rfdiffusion_extra:
                cmd_rf += ["-e", e]
        print(f"[{row_idx}/{len(targets)}] {task} {target_id} rfdiffusion...", flush=True)
        ok_rf, t_rf, err_rf = run_cmd_with_timing(cmd_rf, rf_cwd, base_env, "rfdiffusion")
        if not ok_rf:
            status = "failed"
            err_msg = f"rfdiffusion: {err_rf}"
            print(f"[{row_idx}/{len(targets)}] {task} {target_id} rfdiffusion FAILED ({t_rf:.1f}s)")
            return {
                "task": task,
                "target_id": target_id,
                "status": status,
                "rfdiffusion_seconds": round(t_rf, 2),
                "proteinmpnn_seconds": 0.0,
                "qvextract_seconds": 0.0,
                "total_seconds": round(t_rf, 2),
                "error": err_msg,
                "timestamp": datetime.now().isoformat(),
            }
        print(f"[{row_idx}/{len(targets)}] {task} {target_id} rfdiffusion {t_rf:.1f}s")

        # 2. proteinmpnn (optional; default skip)
        run_pm = getattr(args, "run_proteinmpnn", False)
        qv_for_extract = qv1  # default: extract from rfdiffusion output (backbone only)
        if run_pm:
            cmd_pm = [
                args.proteinmpnn_cmd,
                "-q", str(qv1.resolve()),
                "--output-quiver", str(qv2.resolve()),
                "-n", "1",
            ]
            if weights_pm:
                cmd_pm += ["-checkpoint_path", weights_pm]
            print(f"[{row_idx}/{len(targets)}] {task} {target_id} proteinmpnn...", flush=True)
            ok_pm, t_pm, err_pm = run_cmd_with_timing(cmd_pm, rf_cwd, base_env, "proteinmpnn")
            if not ok_pm:
                status = "failed"
                err_msg = f"proteinmpnn: {err_pm}"
                print(f"[{row_idx}/{len(targets)}] {task} {target_id} proteinmpnn FAILED ({t_pm:.1f}s)")
                return {
                    "task": task,
                    "target_id": target_id,
                    "status": status,
                    "rfdiffusion_seconds": round(t_rf, 2),
                    "proteinmpnn_seconds": round(t_pm, 2),
                    "qvextract_seconds": 0.0,
                    "total_seconds": round(t_rf + t_pm, 2),
                    "error": err_msg,
                    "timestamp": datetime.now().isoformat(),
                }
            print(f"[{row_idx}/{len(targets)}] {task} {target_id} proteinmpnn {t_pm:.1f}s")
            qv_for_extract = qv2
        else:
            t_pm = 0.0

        # 3. qvextract (from rfdiffusion.qv or proteinmpnn.qv)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        cmd_qv = [args.qvextract_cmd, str(qv_for_extract.resolve()), "-o", str(extracted_dir.resolve())]
        print(f"[{row_idx}/{len(targets)}] {task} {target_id} qvextract...", flush=True)
        ok_qv, t_qv, err_qv = run_cmd_with_timing(cmd_qv, rf_cwd, base_env, "qvextract")
        if not ok_qv:
            err_msg = f"qvextract: {err_qv}"
            status_final = "qvextract_failed"
        else:
            status_final = status
        print(
            f"[{row_idx}/{len(targets)}] {task} {target_id} qvextract {t_qv:.1f}s  "
            f"total={t_rf + t_pm + t_qv:.1f}s"
        )

        return {
            "task": task,
            "target_id": target_id,
            "status": status_final,
            "rfdiffusion_seconds": round(t_rf, 2),
            "proteinmpnn_seconds": round(t_pm, 2),
            "qvextract_seconds": round(t_qv, 2),
            "total_seconds": round(t_rf + t_pm + t_qv, 2),
            "error": err_msg,
            "timestamp": datetime.now().isoformat(),
        }

    for task in tasks:
        print(f"\n{'='*80}\nTask: {task.upper()}\n{'='*80}\n")

        # 预先过滤好本 task 要跑的 rows
        task_rows = []
        for idx, row in enumerate(targets, 1):
            target_id = row.get("target_id", "").strip() or (list(row.values())[0] if row else "")
            if not target_id or target_id == "target_id":
                continue
            if selected_target_ids is not None and target_id not in selected_target_ids:
                continue
            task_rows.append((idx, row))

        if not task_rows:
            continue

        max_workers = max(1, int(args.max_concurrent_targets))
        print(f"{task.upper()}: running {len(task_rows)} targets with up to {max_workers} concurrent targets.")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {
                ex.submit(_run_single_target, task, idx, row): idx for idx, row in task_rows
            }
            for fut in as_completed(future_to_idx):
                rec = fut.result()
                if rec:
                    _append_timing_row(rec)

    total_wall = time.time() - total_wall_start

    # Summary
    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    print(f"Wall-clock total: {total_wall:.1f}s ({total_wall/60:.1f} min, {total_wall/3600:.2f} h)")
    print(f"Timing log: {timing_log}")

    for task in tasks:
        recs = [r for r in timing_records if r["task"] == task]
        n_ok = sum(1 for r in recs if r["status"] == "success")
        n_fail = sum(1 for r in recs if r["status"] in ("failed", "qvextract_failed"))
        n_skip = sum(1 for r in recs if r["status"] == "skipped")
        tot = sum(r["total_seconds"] for r in recs)
        tot_rf = sum(r["rfdiffusion_seconds"] for r in recs)
        tot_pm = sum(r["proteinmpnn_seconds"] for r in recs)
        tot_qv = sum(r["qvextract_seconds"] for r in recs)
        print(f"\n{task.upper()}: success={n_ok} failed={n_fail} skipped={n_skip}")
        print(f"  Total (sum of targets): {tot:.1f}s ({tot/60:.1f} min)")
        print(f"  rfdiffusion: {tot_rf:.1f}s  proteinmpnn: {tot_pm:.1f}s  qvextract: {tot_qv:.1f}s")
        if recs:
            avg = tot / len(recs)
            print(f"  Avg per target: {avg:.1f}s ({avg/60:.1f} min)")

    # BenchCore format conversion
    if not args.skip_convert:
        conv_script = algo_dir / "convert_rfantibody_to_benchcore.py"
        if conv_script.exists():
            print("\nRunning BenchCore format conversion...")
            for task in tasks:
                subprocess.run(
                    [
                        sys.executable, str(conv_script),
                        "--run_dirs", str(out_root / task),
                        "--design_dir", str(out_root / "benchcore_format" / task),
                        "--cdr_info_csv", str(out_root / "benchcore_format" / task / "cdr_info.csv"),
                        "--task", task,
                        "--max_per_target", str(args.num_designs),
                        "--extracted_subdir", "extracted",
                    ],
                    check=True,
                    cwd=str(algo_dir),
                )
            print("BenchCore format:", out_root / "benchcore_format")
    else:
        print("\nSkipped conversion (--skip_convert).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Interrupted. Exiting.", file=sys.stderr)
        sys.exit(130)
