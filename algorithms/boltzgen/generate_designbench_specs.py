#!/usr/bin/env python3
"""
Generate BoltzGen design spec YAMLs for all 22 DesignBench antibody/nanobody targets.

Each target gets one YAML for antibody (Fab) and one for nanobody (VHH). Outputs go to
configs/antibody/ and configs/nanobody/. Antigen CIFs are read from DesignBench assets;
scaffolds from DesignBench scaffolds (nanobody) or a mix of DesignBench + BoltzGen (antibody).

Usage:
  python generate_designbench_specs.py --designbench_root /path/to/designbench [--boltzgen_example /path/to/boltzgen/example]
  Default designbench_root: parent of this file's parent (designbench repo root).
"""

import argparse
import re
import sys
from pathlib import Path

# DesignBench target config is under designbench_root/assets/antibody_nanobody/config/target_config.csv
# Antigens: designbench_root/assets/antibody_nanobody/antigens/{target_id}.cif
# Nanobody scaffolds: designbench_root/assets/antibody_nanobody/scaffolds/nanobody/*.yaml (7eow, 7xl0, 8coh, 8z8v)
# Antibody Part 1: hu-4D5-8_Fv.pdb (DesignBench); Part 2: multiple Fab YAMLs (DesignBench or BoltzGen example)


def parse_antigen_chains(antigen_chains: str):
    """Parse antigen_chains e.g. 'A17-132' or 'A12-157,B12-157,C12-157' -> list of chain ids (A, B, C)."""
    if not antigen_chains or (isinstance(antigen_chains, float) and str(antigen_chains) == "nan"):
        return []
    chain_ids = []
    for part in str(antigen_chains).strip().split(","):
        part = part.strip()
        # Chain id is the part before the first digit (e.g. A17-132 -> A, B12 -> B)
        m = re.match(r"^([A-Za-z]+)", part)
        if m:
            chain_ids.append(m.group(1))
    return list(dict.fromkeys(chain_ids))  # preserve order, no dupes


def load_target_config(designbench_root: Path):
    cfg_path = designbench_root / "assets" / "antibody_nanobody" / "config" / "target_config.csv"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Target config not found: {cfg_path}")
    import csv
    rows = []
    with open(cfg_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def is_part1(target_row: dict) -> bool:
    h = target_row.get("target_hotspots", "") or ""
    return bool(str(h).strip())


def yaml_escape_path(p: Path) -> str:
    s = str(p)
    if " " in s or ":" in s:
        return repr(s)
    return s


def write_nanobody_yaml(
    target_id: str,
    antigen_path: Path,
    chain_ids: list,
    scaffold_paths: list,
    out_path: Path,
):
    # entities: [ target (antigen), scaffold(s) ]
    include_chains = "\n".join(f"            - chain:\n                id: {c}" for c in chain_ids)
    if not include_chains:
        include_chains = "            - chain:\n                id: A"
    if len(scaffold_paths) == 1:
        scaffold_yaml = f"        path: {yaml_escape_path(scaffold_paths[0])}"
    else:
        scaffold_yaml = "        path:\n" + "\n".join(f"            - {yaml_escape_path(p)}" for p in scaffold_paths)
    content = f"""# DesignBench nanobody target {target_id}
# Generated for BoltzGen; 100 designs per target.
entities:
  - file:
      path: {yaml_escape_path(antigen_path)}
      include:
{include_chains}
  - file:
{scaffold_yaml}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


def write_antibody_yaml(
    target_id: str,
    antigen_path: Path,
    chain_ids: list,
    scaffold_paths: list,
    out_path: Path,
):
    include_chains = "\n".join(f"            - chain:\n                id: {c}" for c in chain_ids)
    if not include_chains:
        include_chains = "            - chain:\n                id: A"
    if len(scaffold_paths) == 1:
        scaffold_yaml = f"        path: {yaml_escape_path(scaffold_paths[0])}"
    else:
        scaffold_yaml = "        path:\n" + "\n".join(f"            - {yaml_escape_path(p)}" for p in scaffold_paths)
    content = f"""# DesignBench antibody (Fab) target {target_id}
# Generated for BoltzGen; 100 designs per target.
entities:
  - file:
      path: {yaml_escape_path(antigen_path)}
      include:
{include_chains}
  - file:
{scaffold_yaml}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Generate BoltzGen design specs for DesignBench 22 targets")
    ap.add_argument(
        "--designbench_root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="DesignBench repo root (default: parent of algorithms/boltzgen)",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "configs",
        help="Output directory for configs (default: algorithms/boltzgen/configs)",
    )
    args = ap.parse_args()
    designbench_root = args.designbench_root.resolve()
    out_dir = args.out_dir.resolve()
    antigens_dir = designbench_root / "assets" / "antibody_nanobody" / "antigens"
    nano_scaffolds_dir = designbench_root / "assets" / "antibody_nanobody" / "scaffolds" / "nanobody"
    ab_scaffolds_dir = designbench_root / "assets" / "antibody_nanobody" / "scaffolds" / "antibody"

    targets = load_target_config(designbench_root)
    # Nanobody scaffolds: Part 1 fixed h-NbBCII10 (3EAK), Part 2 uses whitelist (7eow, 7xl0, 8coh, 8z8v)
    nano_part1_fixed = nano_scaffolds_dir / "h-NbBCII10.yaml"  # Part 1 fixed scaffold
    nano_scaffold_yamls = []
    for name in ["7eow", "7xl0", "8coh", "8z8v"]:
        p = nano_scaffolds_dir / f"{name}.yaml"
        if p.exists():
            nano_scaffold_yamls.append(p)
    if not nano_scaffold_yamls:
        nano_scaffold_yamls = [nano_scaffolds_dir / "7eow.yaml"]  # placeholder

    # Antibody: Part 1 fixed hu-4D5-8_Fv (1FVC), Part 2 uses whitelist
    ab_part1_fixed = ab_scaffolds_dir / "hu-4D5-8_Fv.yaml"  # Part 1 fixed scaffold
    ab_single = ab_part1_fixed if ab_part1_fixed.exists() else (ab_scaffolds_dir / "adalimumab.6cr1.yaml")
    if not ab_single.exists():
        for cand in ["adalimumab.6cr1.yaml", "belimumab.5y9k.yaml", "secukinumab.6wio.yaml"]:
            if (ab_scaffolds_dir / cand).exists():
                ab_single = ab_scaffolds_dir / cand
                break
    ab_multi = [ab_scaffolds_dir / f for f in [
        "adalimumab.6cr1.yaml", "belimumab.5y9k.yaml", "dupilumab.6wgb.yaml", "golimumab.5yoy.yaml",
        "guselkumab.4m6m.yaml", "nirsevimab.5udc.yaml", "sarilumab.8iow.yaml", "secukinumab.6wio.yaml",
        "tezepelumab.5j13.yaml", "tralokinumab.5l6y.yaml", "ustekinumab.3hmw.yaml", "mab1.3h42.yaml",
        "necitumumab.6b3s.yaml", "crenezumab.5vzy.yaml",
    ] if (ab_scaffolds_dir / f).exists()]
    if not ab_multi:
        ab_multi = [ab_single]

    for row in targets:
        target_id = row["target_id"]
        antigen_path = antigens_dir / f"{target_id}.cif"
        if not antigen_path.exists():
            # Fallback: try O->0 (21_6COB -> 21_6C0B) or same seq number (12_1BI7 -> 12_3QKG.cif)
            alt_id = target_id.replace("6COB", "6C0B").replace("6COb", "6C0B")
            if alt_id != target_id:
                antigen_path = antigens_dir / f"{alt_id}.cif"
        if not antigen_path.exists():
            seq_num = target_id.split("_")[0] if "_" in target_id else ""
            fallback = list(antigens_dir.glob(f"{seq_num}_*.cif")) if seq_num else []
            if fallback:
                antigen_path = sorted(fallback)[0]
        if not antigen_path.exists():
            print(f"Skip {target_id}: antigen not found", file=sys.stderr)
            continue
        chain_ids = parse_antigen_chains(row.get("antigen_chains", "A"))
        if not chain_ids:
            chain_ids = ["A"]

        # Nanobody
        nano_specs_dir = out_dir / "nanobody"
        if is_part1(row):
            # Part 1: use fixed scaffold h-NbBCII10 (3EAK)
            nano_scaffolds = [nano_part1_fixed] if nano_part1_fixed.exists() else (nano_scaffold_yamls[:1] if nano_scaffold_yamls else [])
        else:
            # Part 2: use whitelist scaffolds
            nano_scaffolds = nano_scaffold_yamls
        if nano_scaffolds:
            write_nanobody_yaml(
                target_id, antigen_path, chain_ids, nano_scaffolds,
                nano_specs_dir / f"{target_id}.yaml",
            )
            print(f"Wrote nanobody spec: {nano_specs_dir / f'{target_id}.yaml'}")

        # Antibody
        ab_specs_dir = out_dir / "antibody"
        if is_part1(row):
            # Part 1: use fixed scaffold hu-4D5-8_Fv (1FVC)
            ab_scaffolds = [ab_part1_fixed] if ab_part1_fixed.exists() else ([ab_single] if ab_single.exists() else ab_multi[:1])
        else:
            # Part 2: use whitelist scaffolds
            ab_scaffolds = ab_multi
        if ab_scaffolds:
            write_antibody_yaml(
                target_id, antigen_path, chain_ids, ab_scaffolds,
                ab_specs_dir / f"{target_id}.yaml",
            )
            print(f"Wrote antibody spec: {ab_specs_dir / f'{target_id}.yaml'}")

    print(f"Done. Configs under {out_dir}")


if __name__ == "__main__":
    main()
