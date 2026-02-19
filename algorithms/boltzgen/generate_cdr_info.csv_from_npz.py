#!/usr/bin/env python3
"""
Generate BenchCore cdr_info.csv from BoltzGen nanobody CIF + NPZ outputs.

When .npz exists next to .cif, uses design_mask from NPZ (BoltzGen's actual
redesign regions). Otherwise falls back to sequence-based CDR inference.
Indices are 0-based; for TNP mode BenchCore uses end EXCLUSIVE (range(start, end)).

Usage:
    python boltzgen_to_benchcore_cdr.py \
        --cif_dir intermediate_designs \
        --out cdr_info.csv \
        [--nanobody_chain B]
"""

import argparse
import csv
import numpy as np
from pathlib import Path

# 3-letter to 1-letter
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def _parse_atom_site_columns(cif_path: Path):
    """Parse _atom_site loop header and return column indices (0-based)."""
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
                break  # data row, stop
    return idx


def get_chain_residue_counts_from_cif(cif_path: Path):
    """
    Return list of (chain_id, count) in CIF order (target first, then nanobody typically).
    Used to slice design_mask: tokens are in same order as chains.
    """
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
                seq_id = int(seq_val) if seq_val != "?" and seq_val != "." else int(parts[idx_label_seq])
            except (ValueError, IndexError):
                continue
            if asym not in chain_residues:
                chain_residues[asym] = set()
            chain_residues[asym].add(seq_id)
    chain_order = sorted(chain_residues.keys())
    return [(c, len(chain_residues[c])) for c in chain_order]


def design_mask_to_cdr_ranges(design_mask: np.ndarray, nanobody_start: int, nanobody_len: int):
    """
    Extract three CDR runs (CDR1, CDR2, CDR3) from design_mask for the nanobody slice.
    design_mask: full token-level mask (True = designed residue).
    Returns (h_cdr1_start, h_cdr1_end, h_cdr2_start, h_cdr2_end, h_cdr3_start, h_cdr3_end)
    in 0-based indices within the nanobody chain; end is EXCLUSIVE.
    """
    sl = design_mask[nanobody_start : nanobody_start + nanobody_len]
    n = len(sl)
    if n == 0:
        return None
    # Find contiguous True runs
    runs = []
    i = 0
    while i < n:
        if sl[i]:
            start = i
            while i < n and sl[i]:
                i += 1
            runs.append((start, i))  # end exclusive
        else:
            i += 1
    if len(runs) < 3:
        return None
    # First three runs = CDR1, CDR2, CDR3 (order in sequence)
    (h1_s, h1_e), (h2_s, h2_e), (h3_s, h3_e) = runs[0], runs[1], runs[2]
    return (h1_s, h1_e, h2_s, h2_e, h3_s, h3_e)


def get_nanobody_sequence_and_cys_positions(cif_path: Path, nanobody_chain_id: str):
    """
    Parse CIF and return (sequence, list of residue indices for chain, cys_positions).
    Residue indices are 1-based as in the CIF auth_seq_id.
    No Biopython required; parses _atom_site loop.
    """
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
                seq_id = int(seq_val) if seq_val != "?" and seq_val != "." else int(parts[idx_label_seq])
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
    sequence = "".join(s[1] for s in seq_list)
    res_ids = [s[0] for s in seq_list]
    return sequence, res_ids, sorted(cys_positions)


def infer_cdr_vhh(res_ids: list, cys_positions: list):
    """
    Infer CDR boundaries for VHH (nanobody) using convention.
    - CDR1: IMGT 26-35 -> 1-based inclusive; we use first CYS+4 to first CYS+13 (approx).
    - CDR2: 51-57
    - CDR3: 95 to second CYS - 1 (inclusive).

    Returns (h_cdr1_start, h_cdr1_end, h_cdr2_start, h_cdr2_end, h_cdr3_start, h_cdr3_end)
    as 0-based indices; end is EXCLUSIVE (one past last) for BenchCore TNP.
    """
    if not res_ids or len(res_ids) < 60:
        return None

    n = len(res_ids)
    # Map 1-based res_id -> 0-based index
    res_id_to_idx = {r: i for i, r in enumerate(res_ids)}

    # First CYS usually ~22, second CYS variable (disulfide partner)
    cys1 = cys_positions[0] if len(cys_positions) >= 1 else 22
    cys2 = cys_positions[1] if len(cys_positions) >= 2 else (res_ids[-1] if res_ids else 97)

    # CDR1: convention 26-35 (1-based). Use positions that exist in res_ids.
    def clamp_1based_to_idx(start_1, end_1_inclusive):
        start_0 = res_id_to_idx.get(start_1)
        end_0 = res_id_to_idx.get(end_1_inclusive)
        if start_0 is None or end_0 is None:
            return None, None
        # BenchCore TNP: end exclusive
        return start_0, end_0 + 1

    # Typical VHH: CDR1 26-35, CDR2 51-57, CDR3 95 to (second CYS - 1)
    h1_s, h1_e = clamp_1based_to_idx(26, 35)
    h2_s, h2_e = clamp_1based_to_idx(51, 57)
    # CDR3: from 95 to second CYS (exclusive), i.e. 1-based 95..(cys2-1); end exclusive = index of cys2
    h3_s, _ = clamp_1based_to_idx(95, min(95 + 20, cys2 - 1 if cys2 > 95 else 95 + 10))
    if h3_s is not None:
        idx_cys2 = res_id_to_idx.get(cys2)
        if idx_cys2 is not None:
            h3_e = idx_cys2  # end exclusive: last CDR3 residue is the one before CYS
        else:
            h3_e = min(h3_s + 15, n)
    else:
        h3_s, h3_e = max(0, n - 20), n

    if h1_s is None:
        h1_s, h1_e = 25, min(36, n)
    if h2_s is None:
        h2_s, h2_e = 50, min(58, n)
    if h3_s is None or h3_e is None:
        h3_s, h3_e = max(0, n - 25), n

    return (h1_s, h1_e, h2_s, h2_e, h3_s, h3_e)


def main():
    ap = argparse.ArgumentParser(description="Generate BenchCore cdr_info.csv from BoltzGen nanobody CIFs")
    ap.add_argument("--cif_dir", type=Path, default=Path("intermediate_designs"), help="Directory of CIF files")
    ap.add_argument("--out", type=Path, default=Path("cdr_info.csv"), help="Output CSV path")
    ap.add_argument("--nanobody_chain", type=str, default="B", help="Chain ID of nanobody in CIF (default B)")
    ap.add_argument("--pattern", type=str, default="*.cif", help="Glob for CIF files (default *.cif)")
    args = ap.parse_args()

    cif_dir = args.cif_dir.resolve()
    if not cif_dir.is_dir():
        raise SystemExit(f"Not a directory: {cif_dir}")

    cif_files = sorted(cif_dir.glob(args.pattern))
    # Exclude non-design CIFs if any
    cif_files = [f for f in cif_files if f.name != "penguinpox.cif" and not f.name.startswith("lightning")]

    rows = []
    for cif_path in cif_files:
        sample_id = cif_path.stem
        npz_path = cif_dir / f"{sample_id}.npz"
        seq, res_ids, cys_pos = get_nanobody_sequence_and_cys_positions(cif_path, args.nanobody_chain)
        if seq is None or not seq:
            print(f"Skip (no nanobody chain): {cif_path.name}")
            continue

        cdrs = None
        if npz_path.exists():
            try:
                data = np.load(npz_path, allow_pickle=True)
                design_mask = data.get("design_mask")
                if design_mask is not None:
                    design_mask = np.asarray(design_mask).ravel().astype(bool)
                    chain_counts = get_chain_residue_counts_from_cif(cif_path)
                    # Token order = chain order in CIF; find nanobody chain index
                    nanobody_chain_id = args.nanobody_chain
                    nanobody_start = 0
                    nanobody_len = 0
                    for ch_id, count in chain_counts:
                        if ch_id == nanobody_chain_id:
                            nanobody_len = count
                            break
                        nanobody_start += count
                    if nanobody_len > 0 and nanobody_start + nanobody_len <= len(design_mask):
                        cdrs = design_mask_to_cdr_ranges(design_mask, nanobody_start, nanobody_len)
            except Exception as e:
                print(f"NPZ read failed for {sample_id}: {e}, using sequence heuristic")
        if cdrs is None:
            cdrs = infer_cdr_vhh(res_ids, cys_pos)
        if cdrs is None:
            print(f"Skip (CDR infer failed): {cif_path.name}")
            continue

        h1_s, h1_e, h2_s, h2_e, h3_s, h3_e = cdrs
        # BenchCore CSV: id, heavy_fv, light_fv, h_cdr1_start, h_cdr1_end, ...
        # For TNP, end is exclusive. heavy_fv: use full sequence so DevelopabilityScorer builds full index_map.
        rows.append({
            "id": sample_id,
            "heavy_fv": seq,
            "light_fv": "",
            "h_cdr1_start": h1_s,
            "h_cdr1_end": h1_e,
            "h_cdr2_start": h2_s,
            "h_cdr2_end": h2_e,
            "h_cdr3_start": h3_s,
            "h_cdr3_end": h3_e,
            "l_cdr1_start": "",
            "l_cdr1_end": "",
            "l_cdr2_start": "",
            "l_cdr2_end": "",
            "l_cdr3_start": "",
            "l_cdr3_end": "",
        })

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "heavy_fv", "light_fv",
        "h_cdr1_start", "h_cdr1_end", "h_cdr2_start", "h_cdr2_end", "h_cdr3_start", "h_cdr3_end",
        "l_cdr1_start", "l_cdr1_end", "l_cdr2_start", "l_cdr2_end", "l_cdr3_start", "l_cdr3_end",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print("BenchCore TNP uses 0-based indices; CDR end is exclusive.")


if __name__ == "__main__":
    main()
