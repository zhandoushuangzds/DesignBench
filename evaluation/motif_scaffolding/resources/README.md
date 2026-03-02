# Motif Scaffolding Resources

This directory contains resources needed for motif scaffolding evaluation.

## Directory Structure

- `motif_pdbs/`: Directory containing motif PDB files
  - Each motif should be named as `{motif_name}.pdb` (e.g., `01_1LDB.pdb`)
  - These are the reference motif structures used for RMSD calculation

## Configuration

You can configure the motif PDBs directory in your config file:

```yaml
motif_scaffolding:
  motif_pdbs_dir: /path/to/your/motif_pdbs  # Optional: defaults to internal resources/motif_pdbs
```

If not configured, the system will use the default internal location:
`evaluation/motif_scaffolding/resources/motif_pdbs/`

## Obtaining Motif PDBs

The motif PDB files can be obtained from the original MotifBench dataset.
You can copy them from `MotifBench/motif_pdbs/` to this directory, or configure
a custom path in your config file.
