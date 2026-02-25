# BenchCore Input Contract

## Overview

BenchCore is a **multi-task evaluation framework** supporting various protein design tasks. Each task type has specific input format requirements. All model-specific preprocessing (e.g., PPIFlow's Poly-Ala fix) must be done **before** passing data to BenchCore.

## General Principles

1. **Standardized Format**: BenchCore expects standardized input formats, not raw model outputs
2. **Preprocessing Responsibility**: Users/external scripts handle model-specific data conversion
3. **Task-Specific Requirements**: Different tasks may require different metadata files

## Task-Specific Input Formats

### 1. Protein Design (`run_protein_pipeline.py`)

**Input Format:**
- Directory containing PDB files (backbone structures)
- Files are preprocessed via `Preprocess.format_output_pdb()`

**Directory Structure:**
```
design_dir/
├── sample_0.pdb
├── sample_1.pdb
└── sample_2.pdb
```

**Requirements:**
- ✅ PDB format files
- ✅ Backbone atoms (CA, C, N, O)
- ✅ Real residue names (not Poly-Ala)
- ✅ Standard PDB format

**Pipeline:**
1. Preprocess: `format_output_pdb()` → `formatted_designs/`
2. Inverse Fold: ProteinMPNN → `inverse_fold/`
3. Refold: ESMFold → `refold/esmfold_out/`
4. Evaluate: `run_protein_evaluation()`

---

### 2. Protein Binding Protein (PBP) (`run_pbp_pipeline.py`)

**Input Format:**
- Directory containing PDB files (protein-protein complexes)
- Similar to protein design, but for complexes

**Directory Structure:**
```
design_dir/
├── complex_0.pdb
├── complex_1.pdb
└── complex_2.pdb
```

**Requirements:**
- ✅ PDB format files
- ✅ Multi-chain structures (protein-protein complexes)
- ✅ Real residue names

**Pipeline:**
1. Preprocess: `format_output_pdb()` → `formatted_designs/`
2. Inverse Fold: LigandMPNN → `inverse_fold/`
3. Refold: AlphaFold3 → `refold/af3_out/`
4. Evaluate: `run_protein_binding_protein_evaluation()`

---

### 3. Antibody Design (`run_antibody_pipeline.py`)

**Input Format:**
- Directory containing PDB/CIF files (antibody structures)
- **Required**: CDR information CSV for scaffold fixing and developability evaluation

**Directory Structure:**
```
design_dir/
├── 01_PDL1_0.pdb
├── 01_PDL1_1.pdb
├── ...
├── 01_PDL1_99.pdb
├── 02_TNFA_0.pdb
├── ...
└── 22_FHAB_99.pdb
```

**Strict Naming Convention (Benchmark Requirement):**
- **Format**: `{TargetName}_{Index}.pdb` or `{TargetName}_{Index}.cif`
- **TargetName Format**: `{Numerical_Order}_{Four_Capital_Letters_ID}` (e.g., `01_PDL1`, `12_AMBP`)
- **Index**: Design number starting from 0 (0, 1, 2, ..., 99)
- **Examples**: 
  - `01_7UXQ_0.pdb` (Part 1, target 7UXQ, design 0)
  - `12_1BI7_42.pdb` (Part 2, target 1BI7, design 42)
  - `22_7WPC_99.pdb` (Part 1, target 7WPC, design 99)

**Target List:**
Target configuration is loaded from `assets/antibody_nanobody/config/target_config.csv`.

- **Part 1 (01-11, 21, 22)**: Benchmark validation group with binding hotspots (fixed scaffold requirement)
  - Part 1 targets have non-empty `target_hotspots` in the config CSV
  - Examples: `01_7UXQ`, `02_1TNF`, `03_3MJG`, `04_3DI3`, `05_4ZXB`, `06_5VLI`, `07_7LVW`, `08_7LUC`, `09_4G8A`, `10_6X93`, `11_4ZAM`, `21_6COB`, `22_7WPC`
- **Part 2 (12-20)**: Challenge group without binding hotspots (allows scaffold diversity)
  - Part 2 targets have empty `target_hotspots` in the config CSV
  - Examples: `12_1BI7`, `13_1G1D`, `14_1LCE`, `15_1I9A`, `16_2O25`, `17_3BX7`, `18_2A1X`, `19_1WAK`, `20_1P4M`

**Quota Limits:**
- **Maximum 100 designs per target** (indices 0-99)
- If more than 100 designs are found for a target, pipeline will raise an error
- Designs are sorted by index and only the first 100 are processed

**Requirements:**
- ✅ PDB or CIF format files
- ✅ Antibody structures (Heavy + Light chains for scFv/Fab, or Heavy only for VHH/Nanobody)
- ✅ Real residue names
- ✅ **Required**: CDR information CSV
- ✅ **Required**: Strict filename format compliance
- ✅ **Required**: Target name validation

**Module Selection:**
- **AntibodyDesignModule** (scFv/Fab): Full antibodies with heavy and light chains
- **NanobodyDesignModule** (VHH): Nanobodies with heavy chain only (no light chain)
- Set `antibody_type: 'antibody'` or `antibody_type: 'nanobody'` in config

**Required Metadata: `cdr_info_csv`**

**Must provide** a CSV with CDR indices for each antibody. This CSV is used for:
1. **Input Validation**: Matching PDB files to CDR information (pipeline stops if match fails)
2. **Inverse Folding**: Fixing scaffold residues (all residues EXCEPT CDR loops)
3. **Developability Evaluation**: Calculating developability metrics

**CSV Format:**
```csv
id,heavy_fv,light_fv,h_cdr1_start,h_cdr1_end,h_cdr2_start,h_cdr2_end,h_cdr3_start,h_cdr3_end,l_cdr1_start,l_cdr1_end,l_cdr2_start,l_cdr2_end,l_cdr3_start,l_cdr3_end
01_PDL1,H,L,30,35,50,65,95,102,24,34,50,56,89,97
02_TNFA,H,L,30,35,50,65,95,102,24,34,50,56,89,97
12_AMBP,H,,30,35,50,65,95,102,,,,,,
```

**Column Descriptions:**
- `id`: **Must match target name** (e.g., `01_PDL1`, `12_AMBP`) - used for PDB file matching
- `heavy_fv`: Heavy chain ID (typically 'H')
- `light_fv`: Light chain ID (typically 'L', empty for nanobodies/VHH)
- `h_cdr1_start`, `h_cdr1_end`: Heavy chain CDR1 range (1-based, inclusive)
- `h_cdr2_start`, `h_cdr2_end`: Heavy chain CDR2 range (1-based, inclusive)
- `h_cdr3_start`, `h_cdr3_end`: Heavy chain CDR3 range (1-based, inclusive)
- `l_cdr1_start`, `l_cdr1_end`: Light chain CDR1 range (optional, for nanobodies must be empty)
- `l_cdr2_start`, `l_cdr2_end`: Light chain CDR2 range (optional, for nanobodies must be empty)
- `l_cdr3_start`, `l_cdr3_end`: Light chain CDR3 range (optional, for nanobodies must be empty)

**CDR-Based Scaffold Fixing:**
- During inverse folding, **all residues EXCEPT the 3 CDR loops are fixed**
- This ensures only CDR regions are redesigned, keeping the scaffold constant
- Matching is done by extracting target name from PDB filename (e.g., `01_PDL1_0.pdb` → `01_PDL1`)

**Scaffold Whitelist & Compliance:**
- **Antibody Whitelist** (15 scaffolds): 1FVC (hu-4D5-8_Fv.pdb), 6CR1, 5Y9K, 6WGB, 5YOY, 4M6M, 5UDC, 8IOW, 6WIO, 5J13, 5L6Y, 3HMW, 3H42, 6B3S, 5VZY
  - Scaffold files located in: `assets/antibody_nanobody/scaffolds/antibody/`
- **Nanobody Whitelist** (5 scaffolds): 3EAK, 7EOW, 7XL0, 8COH, 8Z8V
  - Scaffold files located in: `assets/antibody_nanobody/scaffolds/nanobody/`
  - Part 1 fixed scaffold: `h-NbBCII10.pdb` (mapped to scaffold ID in whitelist)
- **Part 1 Compliance**: Each target must use a single fixed scaffold
  - **Antibody**: `hu-4D5-8_Fv.pdb` (scaffold ID: 1FVC)
  - **Nanobody**: `h-NbBCII10.pdb` (scaffold ID determined from file)
- **Part 2 Compliance**: Scaffolds must be selected from whitelist, diversity ≥ 3 recommended
- Pipeline performs pre-run compliance audit and generates a report

**Pre-Run Compliance Report:**
The pipeline automatically generates a compliance report before execution:
```
Sequence | Target | Scaffold | Count | Status
01        | 01_7UXQ | 1FVC    | 100   | Pass
12        | 12_1BI7 | 3EAK,7EOW,8COH | 100 | Pass
21        | 21_6COB | 1FVC    | 100   | Pass
```
- **Status**: Pass (compliant) or Warning (non-compliant)
- Warnings are listed separately with detailed explanations
- Pipeline can be configured to stop on warnings (`proceed_with_warnings: false`) or continue (`proceed_with_warnings: true`)

**Pipeline:**
1. **Input Audit**: Validate filenames, target names, CDR matching, quotas, scaffold compliance
2. **Compliance Report**: Generate and display pre-run compliance report
3. Preprocess: Format designs → `formatted_designs/`
4. Inverse Fold: LigandMPNN (with CDR-based scaffold fixing) → `inverse_fold/`
5. Refold: AlphaFold3 → `refold/af3_out/`
6. Evaluate: `run_protein_binding_protein_evaluation()`
7. Developability: `run_antibody_developability_evaluation()` (uses CDR info from CSV)

---

### 4. Protein Binding Ligand (PBL) (`run_pbl_pipeline.py`)

**Input Format:**
- Directory containing CIF files (protein-ligand complexes)

**Directory Structure:**
```
design_dir/
├── complex_0.cif
├── complex_1.cif
└── complex_2.cif
```

**Requirements:**
- ✅ CIF format files (mmCIF)
- ✅ Protein-ligand complexes
- ✅ Ligand coordinates included

**Pipeline:**
1. Preprocess: Format designs → `formatted_designs/`
2. Inverse Fold: LigandMPNN → `inverse_fold/`
3. Refold: Chai-1 → `refold/chai1_out/`
4. Evaluate: `run_ligand_binding_protein_evaluation()`

---

### 5. Nucleotide Design (NUC) (`run_nuc_pipeline.py`)

**Input Format:**
- Directory containing CIF files (RNA/DNA structures)

**Directory Structure:**
```
design_dir/
├── rna_0.cif
├── rna_1.cif
└── rna_2.cif
```

**Requirements:**
- ✅ CIF format files
- ✅ RNA or DNA structures
- ✅ Nucleotide coordinates

**Pipeline:**
1. Preprocess: Format designs → `formatted_designs/`
2. Inverse Fold: ODesign → `inverse_fold/`
3. Refold: AlphaFold3 → `refold/af3_out/`
4. Evaluate: `run_nuc_evaluation()`

---

### 6. Nucleotide Binding Ligand (NBL) (`run_nbl_pipeline.py`)

**Input Format:**
- Directory containing CIF files (nucleic acid-ligand complexes)

**Directory Structure:**
```
design_dir/
├── complex_0.cif
├── complex_1.cif
└── complex_2.cif
```

**Requirements:**
- ✅ CIF format files
- ✅ Nucleic acid-ligand complexes

**Pipeline:**
1. Preprocess: Format designs → `formatted_designs/`
2. Inverse Fold: ODesign → `inverse_fold/`
3. Refold: AlphaFold3 → `refold/af3_out/`
4. Evaluate: `run_nuc_binding_ligand_evaluation()`

---

### 7. Protein Binding Nucleotide (PBN) (`run_pbn_pipeline.py`)

**Input Format:**
- Directory containing CIF files (protein-nucleic acid complexes)

**Directory Structure:**
```
design_dir/
├── complex_0.cif
├── complex_1.cif
└── complex_2.cif
```

**Requirements:**
- ✅ CIF format files
- ✅ Protein-nucleic acid complexes

**Pipeline:**
1. Preprocess: Format designs → `formatted_designs/`
2. Inverse Fold: ODesign → `inverse_fold/`
3. Refold: AlphaFold3 → `refold/af3_out/`
4. Evaluate: `run_protein_binding_nuc_evaluation()`

---

### 8. Motif Scaffolding (`run_motif_scaffolding_pipeline.py`)

**Input Format:**
- Directory containing PDB files (scaffold structures)
- **Required**: `scaffold_info.csv` with motif placement information

**Directory Structure:**
```
input_dir/
├── sample_0.pdb
├── sample_1.pdb
├── sample_2.pdb
└── scaffold_info.csv
```

**Requirements:**
- ✅ PDB format files
- ✅ Real residues (not Poly-Ala)
- ✅ Backbone atoms present

**Required Metadata: `scaffold_info.csv`**

```csv
sample_num,motif_placements
0,34/A/70
1,30/A/25/B/30
2,100/A/50
```

**Column Descriptions:**
- `sample_num`: Sample index (0, 1, 2, ...)
- `motif_placements`: Motif position specification
  - Format: `"scaffold_before/chain/scaffold_after"`
  - Example: `"34/A/70"` (34 residues before motif chain A, 70 after)
  - Multi-chain: `"30/A/25/B/30"` (30 before A, 25 between A and B, 30 after B)

**Additional Requirements:**
- Motif PDB file must exist in MotifBench directory
- MotifBench configuration in config file

**Pipeline:**
1. Load inputs (PDB + scaffold_info.csv)
2. Inverse Fold: ProteinMPNN (with motif constraints) → `inverse_fold/`
3. Refold: ESMFold → `refold/esmfold_out/`
4. Evaluate: MotifBench metrics (scRMSD, motifRMSD, Novelty, Diversity)

---

### 9. Antigen-Antibody Interface Analysis (`run_interface_pipeline.py`)

**Input Format:**
- Directory containing CIF files (antigen-antibody complexes)

**Directory Structure:**
```
design_dir/
├── complex_0.cif
├── complex_1.cif
└── complex_2.cif
```

**Requirements:**
- ✅ CIF format files (mmCIF)
- ✅ Antigen-antibody complexes
- ✅ Chain IDs: Typically 'H'/'L' for antibody, others for antigen
- ✅ Can specify chain IDs via config

**Pipeline:**
1. Preprocess: Format designs → `formatted_designs/`
2. Inverse Fold: LigandMPNN → `inverse_fold/`
3. Refold: AlphaFold3 → `refold/af3_out/`
4. Evaluate: `run_protein_binding_protein_evaluation()`
5. Interface Analysis: `InterfaceAnalyzer.analyze_interface()` (if enabled)

**Interface Analysis Metrics:**
- Geometry: Paratope/Epitope size, BSA, Epitope patches
- Interactions: Hydrophobic clusters, Hydrogen bonds
- Composition: Paratope composition, Charge complementarity
- Structure: Epitope secondary structure, Segmentation

---

## Common Input Requirements

### File Formats

**PDB Files:**
- Standard PDB format
- Real residue names (not Poly-Ala)
- Backbone atoms (CA, C, N, O) required
- Optional: Side chain atoms

**CIF Files:**
- mmCIF format
- Standard mmCIF conventions
- All required atoms present

### Preprocessing

**User/External Script Responsibility:**
- Convert model outputs to standard PDB/CIF format
- Fix Poly-Ala sequences (if applicable)
- Generate required metadata files (CSV)
- Ensure correct file naming conventions
- Validate structure integrity

**BenchCore Responsibility:**
- Load standardized inputs
- Run Inverse Folding
- Run Refolding
- Calculate metrics

---

## Validation

BenchCore will validate:
- ✅ Input directory exists
- ✅ Required files found (PDB/CIF)
- ✅ Metadata file format (if required)
- ✅ Required columns present (for task-specific requirements)
- ✅ File format compatibility

If validation fails, BenchCore will raise clear error messages indicating what is missing.

---

## Quick Reference

| Task Type | Input Format | Metadata | Inverse Fold | Refold |
|-----------|-------------|----------|--------------|--------|
| Protein | PDB | None | ProteinMPNN | ESMFold |
| PBP | PDB | None | LigandMPNN | AF3 |
| Antibody | PDB/CIF | CDR CSV (req) | LigandMPNN | AF3 |
| PBL | CIF | None | LigandMPNN | Chai-1 |
| NUC | CIF | None | ODesign | AF3 |
| NBL | CIF | None | ODesign | AF3 |
| PBN | CIF | None | ODesign | AF3 |
| Motif Scaffolding | PDB | scaffold_info.csv | ProteinMPNN | ESMFold |
| Interface Analysis | CIF | None | LigandMPNN | AF3 |

---

## Examples

### Example 1: Protein Design
```bash
python scripts/run_protein_pipeline.py \
    design_dir=/path/to/backbones \
    inversefold=ProteinMPNN \
    refold=esmfold \
    gpus=0
```

### Example 2: Antibody Design with Developability
```bash
# scFv/Fab (full antibody)
python scripts/run_antibody_pipeline.py \
    design_dir=/path/to/antibodies \
    antibody_type=antibody \
    inversefold=LigandMPNN \
    refold=af3 \
    cdr_info_csv=/path/to/cdr_info.csv \
    target_config_path=assets/antibody_nanobody/config/target_config.csv \
    max_designs_per_target=100 \
    proceed_with_warnings=false \
    gpus=0,1

# VHH (nanobody)
python scripts/run_antibody_pipeline.py \
    design_dir=/path/to/nanobodies \
    antibody_type=nanobody \
    inversefold=LigandMPNN \
    refold=af3 \
    cdr_info_csv=/path/to/cdr_info.csv \
    target_config_path=assets/antibody_nanobody/config/target_config.csv \
    max_designs_per_target=100 \
    proceed_with_warnings=false \
    gpus=0,1
```

**Input File Naming Examples:**
- Valid: `01_7UXQ_0.pdb`, `01_7UXQ_1.pdb`, ..., `01_7UXQ_99.pdb`
- Valid: `12_1BI7_0.pdb`, `12_1BI7_1.pdb`, ..., `12_1BI7_99.pdb`
- Invalid: `7UXQ_0.pdb` (missing sequence number)
- Invalid: `01_7uxq_0.pdb` (target ID must be uppercase)
- Invalid: `01_7UXQ_100.pdb` (exceeds quota of 100)

**CDR CSV Example:**
```csv
id,heavy_fv,light_fv,h_cdr1_start,h_cdr1_end,h_cdr2_start,h_cdr2_end,h_cdr3_start,h_cdr3_end,l_cdr1_start,l_cdr1_end,l_cdr2_start,l_cdr2_end,l_cdr3_start,l_cdr3_end
01_7UXQ,H,L,30,35,50,65,95,102,24,34,50,56,89,97
02_1TNF,H,L,30,35,50,65,95,102,24,34,50,56,89,97
12_1BI7,H,,30,35,50,65,95,102,,,,,,
21_6COB,H,L,30,35,50,65,95,102,24,34,50,56,89,97
22_7WPC,H,L,30,35,50,65,95,102,24,34,50,56,89,97
```

**Target Configuration:**
The pipeline automatically loads target configuration from `assets/antibody_nanobody/config/target_config.csv`. This file defines:
- Target IDs and their corresponding PDB IDs
- Antigen chain information
- Binding hotspots (for Part 1 targets)
- Epitope descriptions

You can override the default path by setting `target_config_path` in the config.

### Example 3: Motif Scaffolding
```bash
python scripts/run_motif_scaffolding_pipeline.py \
    input_dir=/path/to/scaffolds \
    metadata_file=/path/to/scaffold_info.csv \
    gpus=0
```

### Example 4: Interface Analysis
```bash
python scripts/run_interface_pipeline.py \
    design_dir=/path/to/complexes \
    ab_chain_ids=['H','L'] \
    ag_chain_ids=['A'] \
    gpus=0
``` 

# BenchCore Usage Example

## 1. Protein Evaluation 

### Usage

```python
from evaluation.evaluation_api import Evaluation
from omegaconf import DictConfig

config = DictConfig({...})
evaluator = Evaluation(config)

# run_protein evaluation
evaluator.run_protein_evaluation(
    pipeline_dir="/path/to/pipeline",
    output_csv="/path/to/results.csv",
    ca_rmsd_threshold=2.0
)
```

### Pipeline script

```bash
python scripts/run_protein_pipeline.py \
    design_dir=/path/to/designs \
    inversefold=ProteinMPNN \
    refold=esmfold
```

## 2. Protein Binding Protein (PBP) Evaluation

### Usage

```python
evaluator.run_protein_binding_protein_evaluation(
    pipeline_dir="/path/to/pipeline",
    output_csv="/path/to/results.csv"
)
```

### Pipeline Script

```bash
python scripts/run_pbp_pipeline.py \
    design_dir=/path/to/designs \
    inversefold=ProteinMPNN \
    refold=esmfold
```

## 3. Protein Binding Ligand (PBL) Evaluation

### Usage

```python
evaluator.run_protein_binding_ligand_evaluation(
    input_dir="/path/to/input",
    output_dir="/path/to/output",
    dist_cutoff=10.0,
    exhaustiveness=16
)
```

### Pipeline Script

```bash
python scripts/run_pbl_pipeline.py \
    design_dir=/path/to/designs \
    inversefold=LigandMPNN \
    refold=esmfold
```

## 4. Nucleotide (NUC) Evaluation

### Usage

```python
evaluator.run_nuc_evaluation(
    pipeline_dir="/path/to/pipeline",
    output_csv="/path/to/results.csv"
)
```

### Pipeline Script

```bash
python scripts/run_nuc_pipeline.py \
    design_dir=/path/to/designs \
    inversefold=ODesign \
    refold=af3
```

## 5. Motif Scaffolding Evaluation

### Architecture Overview

Motif Scaffolding uses Generator + Evaluator architecture:
- **Generator Layer** (`generators/`): Handles model-specific data conversion
  - `PPIFlowGenerator`: Fixes all-ALA sequences, structure alignment
  - `RFD3Generator`: Parses JSON files, extracts contig information
- **Evaluator Layer** (`evaluation/motif_evaluator.py`): Model-agnostic evaluation logic

### Usage Method 1: Via evaluation_api

```python
from evaluation.evaluation_api import Evaluation
from omegaconf import DictConfig

config = DictConfig({
    'motif_scaffolding': {
        'model_name': 'PPIFlow',  # or 'RFD3'
        'motifbench_dir': '/path/to/MotifBench',
        'foldseek_database': '/path/to/foldseek/db'
    },
    'gpus': '0'
})

evaluator = Evaluation(config)

# Run motif scaffolding evaluation
results = evaluator.run_motif_scaffolding_evaluation(
    design_dir="/path/to/design/outputs",
    pipeline_dir="/path/to/pipeline/results",
    model_name="PPIFlow",  # Optional, uses value from config if not provided
    motif_list=["01_1LDB", "02_1ITU"]  # Optional, None means evaluate all
)

# results is a dictionary: {motif_name: result_dir}
```

### Usage Method 2: Direct use of MotifScaffoldingEvaluation

```python
from evaluation.motif_scaffolding_evaluation import MotifScaffoldingEvaluation
from omegaconf import DictConfig

config = DictConfig({...})

# Initialize (automatically selects Generator)
motif_evaluation = MotifScaffoldingEvaluation(config, model_name='PPIFlow')

# Run evaluation
results = motif_evaluation.run_motif_scaffolding_evaluation(
    design_dir="/path/to/design/outputs",
    pipeline_dir="/path/to/pipeline/results",
    motif_list=["01_1LDB", "02_1ITU"]
)
```

### Pipeline Script

```bash
python scripts/run_motif_scaffolding_pipeline.py \
    design_dir=/path/to/design/outputs \
    model_name=PPIFlow \
    motif_scaffolding.motif_list=[01_1LDB,02_1ITU] \
    inversefold=ProteinMPNN \
    refold=esmfold \
    gpus=0
```

### Evaluation Workflow

1. **Generator.run()**: Standardizes model outputs
   - PPIFlow: Fixes all-ALA sequences → Real residue names
   - RFD3: Parses JSON → Generates scaffold_info.csv
   - Output: Standardized PDB files and scaffold_info.csv

2. **MotifEvaluator.run_evaluation()**: Model-agnostic evaluation
   - InverseFold (ProteinMPNN): Sequence design
   - ReFold (ESMFold): Structure prediction
   - Metrics: RMSD, Novelty, Diversity (alpha=5 clustering)

### Output Results

Evaluation results for each motif are saved in:
```
{pipeline_dir}/motif_scaffolding/{motif_name}/evaluation_results/
├── esm_complete_results.csv      # Complete evaluation results
├── esm_novelty_results.csv       # Novelty results
├── summary.txt                   # Summary (JSON format)
└── successful_backbones/         # Successful backbone structures
```

## 6. Other Evaluation Tasks

### Ligand Binding Protein (LBP)

```python
evaluator.run_ligand_binding_protein_evaluation(
    pipeline_dir="/path/to/pipeline",
    cands="/path/to/candidates",
    output_csv="/path/to/results.csv"
)
```

### Nucleotide Binding Ligand (NBL)

```python
evaluator.run_nuc_binding_ligand_evaluation(
    pipeline_dir="/path/to/pipeline",
    output_csv="/path/to/results.csv"
)
```

### Protein Binding Nucleotide (PBN)

```python
evaluator.run_protein_binding_nuc_evaluation(
    pipeline_dir="/path/to/pipeline",
    output_csv="/path/to/results.csv"
)
```

### Therapeutic Antibody/Nanobody Profile

```
# basic usage
python benchcore/scripts/run_developability_evaluation.py \
    --csv_file antibodies_fv.csv \
    --results_dir ./af3_results \
    --output developability_metrics.csv

# test single antibody
python benchcore/scripts/run_developability_evaluation.py \
    --csv_file antibodies_fv.csv \
    --results_dir ./af3_results \
    --antibody-id AB001 \
    --output test_metrics.csv
```

## General Pattern

All evaluation tasks follow the same pattern:

1. **Initialize**: `evaluator = Evaluation(config)`
2. **Run Evaluation**: `evaluator.run_<task>_evaluation(...)`
3. **Get Results**: Results are saved in the specified output directory or CSV file

Special considerations for Motif Scaffolding:
- Requires specifying `model_name` (PPIFlow or RFD3)
- Uses Generator layer to handle model-specific data conversion
- Evaluation logic is model-agnostic

# nuc inverse folding
```bash
python scripts/run_nuc_pipeline.py inverse_fold.data_name=rna design_dir={ODesign backbone path} prefix={prefix}
python scripts/run_nuc_pipeline.py inverse_fold.data_name=dna design_dir={ODesign backbone path} prefix={prefix}
```
# ligand inverse folding
```bash
python scripts/run_nuc_pipeline.py inverse_fold.data_name=ligand design_dir={ODesign backbone path} prefix={prefix}
```