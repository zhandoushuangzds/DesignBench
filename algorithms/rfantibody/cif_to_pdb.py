#!/usr/bin/env python3
"""Convert CIF to PDB for RFantibody (expects PDB). Uses Bio.PDB. Optional dependency: biopython."""

import sys
from pathlib import Path


def cif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    try:
        from Bio.PDB import MMCIFParser, PDBIO
    except ImportError:
        raise RuntimeError("CIF→PDB requires biopython: pip install biopython")
    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure("cif", str(cif_path))
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(pdb_path))


def main():
    if len(sys.argv) != 3:
        print("Usage: python cif_to_pdb.py <input.cif> <output.pdb>", file=sys.stderr)
        sys.exit(1)
    cif_path = Path(sys.argv[1])
    pdb_path = Path(sys.argv[2])
    if not cif_path.exists():
        print(f"Not found: {cif_path}", file=sys.stderr)
        sys.exit(1)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    cif_to_pdb(cif_path, pdb_path)
    print(f"Wrote {pdb_path}")


if __name__ == "__main__":
    main()
