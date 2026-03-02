# BoltzGen × DesignBench: 22 targets × 100 designs

Use [BoltzGen](https://github.com/boltzgen/boltzgen) to generate 100 designs for each of DesignBench's 22 antibody/nanobody targets, and convert to DesignBench's required directory and CDR CSV format for running `run_antibody_pipeline.py` evaluation.

## Requirements

- **BoltzGen** installed: `pip install boltzgen`, and `boltzgen` in PATH
- **DesignBench** assets complete: `assets/antibody_nanobody/antigens/*.cif`, `config/target_config.csv`, scaffolds

## Quick Start: Generate Configs + Run BoltzGen + Convert to DesignBench Format

From **designbench repository root**:

```bash
# Default: antibody + nanobody, 22 targets × 100 designs each, output to boltzgen_designbench_output/
python algorithms/boltzgen/run_boltzgen_designbench.py \
  --designbench_root /path/to/designbench \
  --output_dir /path/to/boltzgen_designbench_output \
  --task both \
  --num_designs 100
```

- `--task both`: Run both antibody and nanobody; can be changed to `antibody` or `nanobody`
- `--num_designs 100`: 100 designs per target (DesignBench requires max 100 per target)
- Add `--reuse` to resume after interruption

After completion:

- **DesignBench-ready directories**:
  - antibody: `boltzgen_designbench_output/designbench_format/antibody/` (contains `01_7UXQ_0.cif` ... `22_7WPC_99.cif` and `cdr_info.csv`)
  - nanobody: `boltzgen_designbench_output/designbench_format/nanobody/` (same format)

## Step-by-Step Execution

### 1. Generate BoltzGen YAML Configs Only (No Design)

```bash
python algorithms/boltzgen/run_boltzgen_designbench.py \
  --designbench_root /path/to/designbench \
  --output_dir out \
  --gen_specs_only
```

Generated YAMLs are in `algorithms/boltzgen/configs/antibody/` and `configs/nanobody/` (e.g., `01_7UXQ.yaml`, etc.).

### 2. Run BoltzGen Manually (Adjust batch/GPU as needed)

For each target, run separately, for example:

```bash
boltzgen run algorithms/boltzgen/configs/antibody/01_7UXQ.yaml \
  --output boltzgen_designbench_output/antibody/01_7UXQ \
  --protocol antibody-anything \
  --num_designs 100
```

For nanobody, change `antibody` to `nanobody` and `antibody-anything` to `nanobody-anything`.

### 3. Format Conversion Only (When BoltzGen Output Already Exists)

If you already have `output/antibody/<target_id>`, `output/nanobody/<target_id>` (each containing `intermediate_designs/`), convert to DesignBench format:

```bash
python algorithms/boltzgen/run_boltzgen_designbench.py \
  --designbench_root /path/to/designbench \
  --output_dir /path/to/your_boltzgen_output \
  --convert_only \
  --task both
```

Or call the converter directly (nanobody example):

```bash
python algorithms/boltzgen/convert_boltzgen_to_designbench.py \
  --run_dirs /path/to/output/nanobody \
  --design_dir /path/to/designbench_designs/nanobody \
  --cdr_info_csv /path/to/designbench_designs/nanobody/cdr_info.csv \
  --task nanobody \
  --max_per_target 100
```

## Run DesignBench Antibody/Nanobody Evaluation

After obtaining a DesignBench-ready directory (contains `design_dir/` with `01_7UXQ_0.cif` ... and `cdr_info.csv`), run the pipeline from **designbench repository root**:

- `design_dir` must point to the **folder that contains the CIF/PDB files** (e.g. `.../antibody_designbench_test/design_dir`).
- `cdr_info_csv` must point to the CSV next to it (e.g. `.../antibody_designbench_test/cdr_info.csv`).

### If using `antibody_designbench_test` / `nanobody_designbench_test` (under `boltzgen_output/`)

Use **list syntax** for `gpus` to avoid Hydra parsing commas as separate values (e.g. `gpus=[5,6,7]`):

```bash
# From designbench repo root
cd /path/to/designbench

# Antibody (scFv/Fab)
python scripts/run_antibody_pipeline.py \
  design_dir=boltzgen_output/antibody_designbench_test/design_dir \
  antibody_type=antibody \
  cdr_info_csv=boltzgen_output/antibody_designbench_test/cdr_info.csv \
  target_config_path=assets/antibody_nanobody/config/target_config.csv \
  max_designs_per_target=100 \
  gpus=[5,6,7]

# Nanobody (VHH)
python scripts/run_antibody_pipeline.py \
  design_dir=boltzgen_output/nanobody_designbench_test/design_dir \
  antibody_type=nanobody \
  cdr_info_csv=boltzgen_output/nanobody_designbench_test/cdr_info.csv \
  target_config_path=assets/antibody_nanobody/config/target_config.csv \
  max_designs_per_target=100 \
  gpus=[5,6,7]
```

### Generic paths (replace with your output base)

```bash
# antibody (scFv/Fab)
python scripts/run_antibody_pipeline.py \
  design_dir=/path/to/boltzgen_designbench_output/designbench_format/antibody \
  antibody_type=antibody \
  cdr_info_csv=/path/to/boltzgen_designbench_output/designbench_format/antibody/cdr_info.csv \
  target_config_path=assets/antibody_nanobody/config/target_config.csv \
  max_designs_per_target=100 \
  gpus=0,1

# nanobody (VHH)
python scripts/run_antibody_pipeline.py \
  design_dir=/path/to/boltzgen_designbench_output/designbench_format/nanobody \
  antibody_type=nanobody \
  cdr_info_csv=/path/to/boltzgen_designbench_output/designbench_format/nanobody/cdr_info.csv \
  target_config_path=assets/antibody_nanobody/config/target_config.csv \
  max_designs_per_target=100 \
  gpus=0,1
```

## File Descriptions

| File | Purpose |
|------|---------|
| `generate_designbench_specs.py` | Generate BoltzGen YAML for each target from `target_config.csv` and assets |
| `run_boltzgen_designbench.py` | Generate specs → Call BoltzGen → Call converter |
| `convert_boltzgen_to_designbench.py` | Rename `intermediate_designs/*.cif` to `{target_id}_0.cif`...`_99.cif` and generate DesignBench-required `cdr_info.csv` (1-based inclusive CDR) |
| `generate_cdr_info.csv_from_npz.py` | Generate CDR info from CIF+NPZ files (used by converter) |
| `run_boltzgen_designbench.sh` | Shell script to run BoltzGen for all targets (alternative to Python script) |

## Notes

- **Part 1 vs Part 2**: Spec generator uses Part 1 (single scaffold) or Part 2 (multiple scaffolds) from `target_config.csv` (non-empty `target_hotspots` ⇒ Part 1).
- **Part 1 Fixed Scaffolds**: 
  - Antibody: `hu-4D5-8_Fv.pdb` (scaffold ID: 1FVC)
  - Nanobody: `h-NbBCII10.pdb` (scaffold ID: 3EAK)
- **CDR Indices**: CDR info CSV uses 1-based inclusive indices (PDB numbering) as required by DesignBench.
- **File Naming**: Design files must follow `{TargetName}_{Index}.cif` format (e.g., `01_7UXQ_0.cif`, `01_7UXQ_1.cif`, ..., `01_7UXQ_99.cif`).

## About RFantibody Evaluation

To evaluate **RFantibody** similarly, use RFantibody's pipeline (rfdiffusion → proteinmpnn → rf2) to generate 100 designs for the same 22 targets, then organize them according to DesignBench's naming and CDR CSV requirements into `design_dir` + `cdr_info.csv`, and use the above `run_antibody_pipeline.py` for evaluation. Scripts in this directory only handle BoltzGen generation and format conversion.
