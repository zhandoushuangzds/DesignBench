#!/usr/bin/env python3
"""
Convert BoltzGen per-target output to BenchCore antibody/nanobody format.

- Copies/renames CIFs to design_dir as {target_id}_0.cif ... {target_id}_99.cif.
- Writes cdr_info.csv with one row per design (id = 01_7UXQ_0, ...), 1-based inclusive CDR only (no sequences).

Usage:
  # After running BoltzGen per target (e.g. output/antibody/01_7UXQ, ...)
  python convert_boltzgen_to_benchcore.py \
    --run_dirs output/antibody \
    --design_dir benchcore_designs/antibody \
    --cdr_info_csv benchcore_designs/antibody/cdr_info.csv \
    --task antibody

  python convert_boltzgen_to_benchcore.py \
    --run_dirs output/nanobody \
    --design_dir benchcore_designs/nanobody \
    --cdr_info_csv benchcore_designs/nanobody/cdr_info.csv \
    --task nanobody
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

# --- Inline CIF/NPZ CDR helpers (nanobody) so this script is self-contained ---
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def _parse_atom_site_columns(cif_path: Path):
    idx = {}
    in_loop = False
    col = 0
    with open(cif_path) as f:
        for line in f:
            line = line.strip()
            if line == "loop_":
                in_loop = True
                col = 0
                continue
            if in_loop and line.startswith("_atom_site."):
                key = line.replace("_atom_site.", "").strip()
                idx[key] = col
                col += 1
            elif in_loop and line and not line.startswith("_"):
                break
    return idx


def get_chain_residue_counts_from_cif(cif_path: Path):
    idx_map = _parse_atom_site_columns(cif_path)
    idx_asym = idx_map.get("label_asym_id", idx_map.get("auth_asym_id", 6))
    idx_seq = idx_map.get("auth_seq_id", idx_map.get("label_seq_id", 16))
    idx_label_seq = idx_map.get("label_seq_id", 8)
    max_idx = max(idx_asym, idx_seq, idx_label_seq)
    past_atom_site_header = False
    chain_residues = {}
    with open(cif_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("_atom_site."):
                past_atom_site_header = True
                continue
            if not past_atom_site_header:
                continue
            if line.startswith("#") or (line.startswith("_") and not line.startswith("_atom_site")) or line == "loop_":
                continue
            parts = line.split()
            if len(parts) <= max_idx:
                continue
            asym = parts[idx_asym]
            try:
                seq_val = parts[idx_seq]
                seq_id = int(seq_val) if seq_val not in ("?", ".") else int(parts[idx_label_seq])
            except (ValueError, IndexError):
                continue
            if asym not in chain_residues:
                chain_residues[asym] = set()
            chain_residues[asym].add(seq_id)
    chain_order = sorted(chain_residues.keys())
    return [(c, len(chain_residues[c])) for c in chain_order]


def design_mask_to_cdr_ranges(design_mask: np.ndarray, nanobody_start: int, nanobody_len: int):
    sl = design_mask[nanobody_start : nanobody_start + nanobody_len]
    n = len(sl)
    if n == 0:
        return None
    runs = []
    i = 0
    while i < n:
        if sl[i]:
            start = i
            while i < n and sl[i]:
                i += 1
            runs.append((start, i))
        else:
            i += 1
    if len(runs) < 3:
        return None
    (h1_s, h1_e), (h2_s, h2_e), (h3_s, h3_e) = runs[0], runs[1], runs[2]
    return (h1_s, h1_e, h2_s, h2_e, h3_s, h3_e)


def get_nanobody_sequence_and_cys_positions(cif_path: Path, nanobody_chain_id: str):
    idx_map = _parse_atom_site_columns(cif_path)
    idx_asym = idx_map.get("label_asym_id", idx_map.get("auth_asym_id", 6))
    idx_comp = idx_map.get("label_comp_id", 5)
    idx_label_seq = idx_map.get("label_seq_id", 8)
    idx_auth_seq = idx_map.get("auth_seq_id", 16)
    max_idx = max(idx_asym, idx_comp, idx_label_seq, idx_auth_seq)
    seq_list = []
    seen = set()
    cys_positions = []
    past_atom_site_header = False
    with open(cif_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("_atom_site."):
                past_atom_site_header = True
                continue
            if not past_atom_site_header:
                continue
            if line.startswith("#") or (line.startswith("_") and not line.startswith("_atom_site")) or line == "loop_":
                continue
            parts = line.split()
            if len(parts) <= max_idx:
                continue
            asym = parts[idx_asym]
            if asym != nanobody_chain_id:
                continue
            try:
                seq_val = parts[idx_auth_seq]
                seq_id = int(seq_val) if seq_val not in ("?", ".") else int(parts[idx_label_seq])
            except (ValueError, IndexError):
                continue
            if seq_id in seen:
                continue
            seen.add(seq_id)
            res_three = parts[idx_comp].strip()
            if res_three in AA_3TO1:
                seq_list.append((seq_id, AA_3TO1[res_three]))
                if res_three == "CYS":
                    cys_positions.append(seq_id)
    if not seq_list:
        return None, None, None
    seq_list.sort(key=lambda x: x[0])
    res_ids = [s[0] for s in seq_list]
    return "".join(s[1] for s in seq_list), res_ids, sorted(cys_positions)


def infer_cdr_vhh(res_ids: list, cys_positions: list):
    if not res_ids or len(res_ids) < 60:
        return None
    n = len(res_ids)
    res_id_to_idx = {r: i for i, r in enumerate(res_ids)}
    cys1 = cys_positions[0] if len(cys_positions) >= 1 else 22
    cys2 = cys_positions[1] if len(cys_positions) >= 2 else (res_ids[-1] if res_ids else 97)

    def clamp_1based_to_idx(start_1, end_1_inclusive):
        start_0 = res_id_to_idx.get(start_1)
        end_0 = res_id_to_idx.get(end_1_inclusive)
        if start_0 is None or end_0 is None:
            return None, None
        return start_0, end_0 + 1

    h1_s, h1_e = clamp_1based_to_idx(26, 35)
    h2_s, h2_e = clamp_1based_to_idx(51, 57)
    h3_s, _ = clamp_1based_to_idx(95, min(95 + 20, cys2 - 1 if cys2 > 95 else 95 + 10))
    if h3_s is not None:
        idx_cys2 = res_id_to_idx.get(cys2)
        h3_e = idx_cys2 if idx_cys2 is not None else min(h3_s + 15, n)
    else:
        h3_s, h3_e = max(0, n - 20), n
    if h1_s is None:
        h1_s, h1_e = 25, min(36, n)
    if h2_s is None:
        h2_s, h2_e = 50, min(58, n)
    if h3_s is None or h3_e is None:
        h3_s, h3_e = max(0, n - 25), n
    return (h1_s, h1_e, h2_s, h2_e, h3_s, h3_e)

# BenchCore CDR CSV: 1-based inclusive. Default antibody (Fab) CDR from README.
DEFAULT_AB_CDR = {
    "h_cdr1_start": 30, "h_cdr1_end": 35,
    "h_cdr2_start": 50, "h_cdr2_end": 65,
    "h_cdr3_start": 95, "h_cdr3_end": 102,
    "l_cdr1_start": 24, "l_cdr1_end": 34,
    "l_cdr2_start": 50, "l_cdr2_end": 56,
    "l_cdr3_start": 89, "l_cdr3_end": 97,
}


def _get_sequence_for_chain(cif_path: Path, chain_id: str) -> str | None:
    """Get one-letter sequence for a chain from CIF. Returns None if chain not found."""
    seq, _, _ = get_nanobody_sequence_and_cys_positions(cif_path, chain_id)
    return seq


def get_antibody_sequences_from_cif(cif_path: Path) -> tuple[str | None, str | None]:
    """Get (heavy_fv_seq, light_fv_seq) from CIF using last two chains (longer=heavy). Returns (None, None) if not enough chains."""
    chain_counts = get_chain_residue_counts_from_cif(cif_path)
    if len(chain_counts) < 2:
        return None, None
    last_two = list(chain_counts[-2:])
    if last_two[0][1] >= last_two[1][1]:
        heavy_id, light_id = last_two[0][0], last_two[1][0]
    else:
        light_id, heavy_id = last_two[0][0], last_two[1][0]
    return _get_sequence_for_chain(cif_path, heavy_id), _get_sequence_for_chain(cif_path, light_id)


def _design_mask_to_three_runs(design_mask: np.ndarray, start: int, length: int) -> tuple | None:
    """Get (run1, run2, run3) as (s1,e1,s2,e2,s3,e3) 0-based end-exclusive, or None."""
    sl = design_mask[start : start + length]
    n = len(sl)
    if n == 0:
        return None
    runs = []
    i = 0
    while i < n:
        if sl[i]:
            s = i
            while i < n and sl[i]:
                i += 1
            runs.append((s, i))
        else:
            i += 1
    if len(runs) < 3:
        return None
    (s1, e1), (s2, e2), (s3, e3) = runs[0], runs[1], runs[2]
    return (s1, e1, s2, e2, s3, e3)


def get_antibody_cdr_row(cif_path: Path, npz_path: Path | None) -> dict | None:
    """
    Get one CDR row for antibody (Fab) from CIF + optional NPZ.
    Uses design_mask from NPZ to get per-design CDR; falls back to DEFAULT_AB_CDR if NPZ missing.
    Returns dict with 1-based inclusive CDR only (no sequences).
    """
    chain_counts = get_chain_residue_counts_from_cif(cif_path)
    if len(chain_counts) < 3:
        return None  # need at least target + heavy + light
    # Last two chains = antibody (heavy + light). Assume longer = heavy (VH ~121, VL ~107).
    ab_chains = list(chain_counts[-2:])
    if ab_chains[0][1] >= ab_chains[1][1]:
        heavy_id, heavy_len = ab_chains[0][0], ab_chains[0][1]
        light_id, light_len = ab_chains[1][0], ab_chains[1][1]
    else:
        light_id, light_len = ab_chains[0][0], ab_chains[0][1]
        heavy_id, heavy_len = ab_chains[1][0], ab_chains[1][1]
    heavy_start = sum(c for _, c in chain_counts[:-2])
    light_start = heavy_start + heavy_len
    heavy_seq = _get_sequence_for_chain(cif_path, heavy_id)
    light_seq = _get_sequence_for_chain(cif_path, light_id)
    if not heavy_seq:
        return None

    if npz_path and npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            design_mask = data.get("design_mask")
            if design_mask is not None:
                design_mask = np.asarray(design_mask).ravel().astype(bool)
                if heavy_start + heavy_len + light_len <= len(design_mask):
                    h_runs = _design_mask_to_three_runs(design_mask, heavy_start, heavy_len)
                    l_runs = _design_mask_to_three_runs(design_mask, light_start, light_len)
                    # 1-based inclusive: start = 0-based start+1, end = 0-based exclusive end (= last 1-based)
                    if h_runs and l_runs:
                        (h1_s, h1_e, h2_s, h2_e, h3_s, h3_e) = h_runs
                        (l1_s, l1_e, l2_s, l2_e, l3_s, l3_e) = l_runs
                        return {
                            "h_cdr1_start": h1_s + 1, "h_cdr1_end": h1_e,
                            "h_cdr2_start": h2_s + 1, "h_cdr2_end": h2_e,
                            "h_cdr3_start": h3_s + 1, "h_cdr3_end": h3_e,
                            "l_cdr1_start": l1_s + 1, "l_cdr1_end": l1_e,
                            "l_cdr2_start": l2_s + 1, "l_cdr2_end": l2_e,
                            "l_cdr3_start": l3_s + 1, "l_cdr3_end": l3_e,
                        }
                    # Only heavy has 3 runs: use NPZ for heavy, default for light
                    if h_runs:
                        (h1_s, h1_e, h2_s, h2_e, h3_s, h3_e) = h_runs
                        return {
                            "h_cdr1_start": h1_s + 1, "h_cdr1_end": h1_e,
                            "h_cdr2_start": h2_s + 1, "h_cdr2_end": h2_e,
                            "h_cdr3_start": h3_s + 1, "h_cdr3_end": h3_e,
                            "l_cdr1_start": DEFAULT_AB_CDR["l_cdr1_start"],
                            "l_cdr1_end": DEFAULT_AB_CDR["l_cdr1_end"],
                            "l_cdr2_start": DEFAULT_AB_CDR["l_cdr2_start"],
                            "l_cdr2_end": DEFAULT_AB_CDR["l_cdr2_end"],
                            "l_cdr3_start": DEFAULT_AB_CDR["l_cdr3_start"],
                            "l_cdr3_end": DEFAULT_AB_CDR["l_cdr3_end"],
                        }
        except Exception:
            pass
    # Fallback: default scaffold CDR
    return {**DEFAULT_AB_CDR}


def _zero_to_one_based_inclusive(start_0: int, end_0_exclusive: int) -> tuple:
    """Convert 0-based [start, end) to 1-based inclusive [start, end]."""
    if end_0_exclusive <= start_0:
        return start_0 + 1, start_0 + 1
    return start_0 + 1, end_0_exclusive  # last 0-based index = end_0_exclusive - 1 -> 1-based end = end_0_exclusive


def get_nanobody_cdr_row(cif_path: Path, npz_path: Path | None, nanobody_chain: str = "B"):
    """Get one CDR row for nanobody from CIF + optional NPZ. Returns dict with 1-based inclusive CDR."""
    seq, res_ids, cys_pos = get_nanobody_sequence_and_cys_positions(cif_path, nanobody_chain)
    if seq is None or not seq:
        return None
    cdrs = None
    if npz_path and npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            design_mask = data.get("design_mask")
            if design_mask is not None:
                design_mask = np.asarray(design_mask).ravel().astype(bool)
                chain_counts = get_chain_residue_counts_from_cif(cif_path)
                nb_start, nb_len = 0, 0
                for ch_id, count in chain_counts:
                    if ch_id == nanobody_chain:
                        nb_len = count
                        break
                    nb_start += count
                if nb_len > 0 and nb_start + nb_len <= len(design_mask):
                    cdrs = design_mask_to_cdr_ranges(design_mask, nb_start, nb_len)
        except Exception:
            pass
    if cdrs is None:
        cdrs = infer_cdr_vhh(res_ids, cys_pos)
    if cdrs is None:
        return None
    h1_s, h1_e, h2_s, h2_e, h3_s, h3_e = cdrs
    # Convert 0-based end-exclusive to 1-based inclusive
    h1_1, h1_2 = _zero_to_one_based_inclusive(h1_s, h1_e)
    h2_1, h2_2 = _zero_to_one_based_inclusive(h2_s, h2_e)
    h3_1, h3_2 = _zero_to_one_based_inclusive(h3_s, h3_e)
    return {
        "h_cdr1_start": h1_1, "h_cdr1_end": h1_2,
        "h_cdr2_start": h2_1, "h_cdr2_end": h2_2,
        "h_cdr3_start": h3_1, "h_cdr3_end": h3_2,
        "l_cdr1_start": "", "l_cdr1_end": "",
        "l_cdr2_start": "", "l_cdr2_end": "",
        "l_cdr3_start": "", "l_cdr3_end": "",
    }


def main():
    ap = argparse.ArgumentParser(description="Convert BoltzGen output to BenchCore design_dir + cdr_info.csv")
    ap.add_argument("--run_dirs", type=Path, required=True,
                    help="Directory containing per-target run dirs (e.g. output/antibody with 01_7UXQ, 02_1TNF, ...)")
    ap.add_argument("--design_dir", type=Path, required=True, help="Output design directory for BenchCore")
    ap.add_argument("--cdr_info_csv", type=Path, required=True, help="Output cdr_info.csv path")
    ap.add_argument("--task", choices=["antibody", "nanobody"], required=True)
    ap.add_argument("--max_per_target", type=int, default=100)
    ap.add_argument("--nanobody_chain", type=str, default="B")
    ap.add_argument("--intermediate_subdir", type=str, default="intermediate_designs",
                    help="Subdir under each run (intermediate_designs or intermediate_designs_inverse_folded)")
    args = ap.parse_args()

    run_base = args.run_dirs.resolve()
    if not run_base.is_dir():
        raise SystemExit(f"Not a directory: {run_base}")

    design_dir = args.design_dir.resolve()
    design_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "h_cdr1_start", "h_cdr1_end", "h_cdr2_start", "h_cdr2_end", "h_cdr3_start", "h_cdr3_end",
        "l_cdr1_start", "l_cdr1_end", "l_cdr2_start", "l_cdr2_end", "l_cdr3_start", "l_cdr3_end",
    ]
    cdr_rows = []

    for target_dir in sorted(run_base.iterdir()):
        if not target_dir.is_dir():
            continue
        target_id = target_dir.name
        # Skip non-target dirs
        if not (len(target_id) >= 6 and target_id[:2].isdigit() and target_id[2] == "_"):
            continue
        interm = target_dir / args.intermediate_subdir
        if not interm.is_dir():
            print(f"Skip {target_id}: no {args.intermediate_subdir}", file=sys.stderr)
            continue
        cif_files = sorted(f for f in interm.glob("*.cif") if not f.name.startswith("lightning"))
        # Take first max_per_target
        cif_files = cif_files[: args.max_per_target]
        if len(cif_files) < args.max_per_target:
            print(f"Warning: {target_id} has only {len(cif_files)} designs", file=sys.stderr)
        for i, src in enumerate(cif_files):
            dest = design_dir / f"{target_id}_{i}.cif"
            shutil.copy2(src, dest)
        print(f"Copied {len(cif_files)} designs for {target_id} -> {design_dir}")

        # One CDR row per design (BenchCore: id = 01_7UXQ_0, 01_7UXQ_1, ...)
        for i, cif_path in enumerate(cif_files):
            design_id = f"{target_id}_{i}"
            if args.task == "antibody":
                npz_path = interm / f"{cif_path.stem}.npz"
                cdr = get_antibody_cdr_row(cif_path, npz_path)
                if cdr is None:
                    cdr = {**DEFAULT_AB_CDR}
                row = {"id": design_id, **cdr}
            else:
                npz_path = interm / f"{cif_path.stem}.npz"
                cdr = get_nanobody_cdr_row(cif_path, npz_path, args.nanobody_chain)
                if cdr is None:
                    cdr = {
                        "h_cdr1_start": 26, "h_cdr1_end": 35, "h_cdr2_start": 51, "h_cdr2_end": 57,
                        "h_cdr3_start": 95, "h_cdr3_end": 118,
                        "l_cdr1_start": "", "l_cdr1_end": "", "l_cdr2_start": "", "l_cdr2_end": "",
                        "l_cdr3_start": "", "l_cdr3_end": "",
                    }
                row = {"id": design_id, **cdr}
            cdr_rows.append(row)

    cdr_path = args.cdr_info_csv.resolve()
    cdr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cdr_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(cdr_rows)
    print(f"Wrote {len(cdr_rows)} rows to {cdr_path}")
    print(f"BenchCore design_dir: {design_dir} ({sum(1 for _ in design_dir.glob('*.cif'))} CIFs)")


if __name__ == "__main__":
    main()
