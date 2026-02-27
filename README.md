# DesignBench Input Contract

## Overview

DesignBench is a **multi-task evaluation framework** supporting various protein design tasks. Each task type has specific input format requirements. All model-specific preprocessing (e.g., PPIFlow's Poly-Ala fix) must be done **before** passing data to DesignBench.

## General Principles

1. **Standardized Format**: DesignBench expects standardized input formats, not raw model outputs
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
3. Refold: AlphaFold3 → `refold/af3_out/` (sequence-only prediction)
4. Evaluate: `run_protein_binding_protein_evaluation()`

**PBP 11-Target Benchmark**

Fixed set of 11 protein targets. For each target, the **design** is chain A (binder) and the **target** is chain B. Reference structures (one complex per target) are in `assets/pbp/target/` (e.g. `CD3d.pdb`, `EGFR.pdb`). Target list and required design lengths (chain A length) are in `assets/pbp/config/target_config.csv`.

| target_id | design_length | description |
|-----------|---------------|-------------|
| CD3d      | 63            | CD3 delta |
| EGFR      | 191           | EGFR |
| EGFR2     | 101           | FGFR2 (EGFR2) |
| H3        | 210           | Influenza H3 |
| IL7Ra     | 193           | IL-7 receptor alpha |
| InsulinR  | 150           | Insulin receptor |
| PDGFR     | 187           | PDGFR |
| TGFb      | 82            | TGF-beta |
| Tie2      | 141           | Tie2 |
| TrkA      | 101           | TrkA |
| VirB8     | 138           | VirB8 |

**Requirements:**
- **100 designs per target.** Submit exactly 100 design structures for each of the 11 targets (e.g. `CD3d_0.pdb` … `CD3d_99.pdb`).
- Each design is a **protein–protein complex** (PDB): **chain A** = designed binder (length must equal the `design_length` in the table above for that target); **chain B** = fixed target.
- Naming: `{target_id}_{index}.pdb` with `index` 0–99 (e.g. `CD3d_0.pdb`, `EGFR_42.pdb`).

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
  - **Scaffold**: Each Part 1 target must use **exactly one fixed scaffold** (see below).
  - Examples: `01_7UXQ`, `02_1TNF`, `03_3MJG`, `04_3DI3`, `05_4ZXB`, `06_5VLI`, `07_7LVW`, `08_7LUC`, `09_4G8A`, `10_6X93`, `11_4ZAM`, `21_6COB`, `22_7WPC`
- **Part 2 (12-20)**: Challenge group without binding hotspots (allows scaffold diversity)
  - Part 2 targets have empty `target_hotspots` in the config CSV
  - **Scaffold**: Part 2 targets may use **any scaffold from the whitelist** (antibody: 15 scaffolds, nanobody: 5 scaffolds); diversity ≥ 3 per target is recommended.
  - Examples: `12_1BI7`, `13_1G1D`, `14_1LCE`, `15_1I9A`, `16_2O25`, `17_3BX7`, `18_2A1X`, `19_1WAK`, `20_1P4M`

**Target configuration (`target_config.csv`)**

| Column | Description |
|--------|-------------|
| `target_id` | Target name, e.g. `01_7UXQ`, `12_1BI7`. Must match design filenames. |
| `antigen_chains` | **Antigen region the model should focus on** (chain + residue range in **auth** numbering). Format: `ChainStart-End` or `Chain1Start-End,Chain2Start-End`. Examples: `A17-132` = chain A, residues 17–132; `A12-109` = chain A, residues 12–109. For **Part 2** targets (12–20), this column may contain only a chain letter (e.g. `A` or `B`): **no specific region is specified**; the entire chain(s) are valid for design (free design). |
| `target_hotspots` | Comma-separated hotspot residues (e.g. `A56,A115,A123`) for Part 1; empty for Part 2. |
| `epitope_description` | Human-readable epitope description. |

**Antigen cropping (optional, shared by all models)**  
Cropped antigens live under `assets/antibody_nanobody/antigens_cropped/` and are produced by a **single assets-side script** (not inside any model runner). All benchmark runs (e.g. RFantibody, BoltzGen) should use these when present.

- **Step 1 — by `antigen_chains`**: Keep only the specified chain(s) and residue range(s) in **auth** numbering. Examples: `A17-132` → keep chain A residues 17–132; `A` or `B` → keep that entire chain (used for Part 2 targets 12–20).
- **Step 2 — by distance to hotspots**: For targets that are **not** 12–20 and have `target_hotspots`, further keep only residues within **20 Å** of any hotspot atom.
- **Exception**: Targets **12–20** (9 Part 2 targets): only Step 1 is applied; no distance-based crop.

Generate cropped PDBs (requires `biopython`):

```bash
# From designbench repo root
python assets/antibody_nanobody/scripts/crop_antigens_for_benchmark.py
# Optional: --distance 20 (default), --dry_run
```

Output: `assets/antibody_nanobody/antigens_cropped/{target_id}.pdb`. Model runners (e.g. RFantibody) will use these files when present instead of converting from CIF.

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
id,h_chain,l_chain,h_cdr1_start,h_cdr1_end,h_cdr2_start,h_cdr2_end,h_cdr3_start,h_cdr3_end,l_cdr1_start,l_cdr1_end,l_cdr2_start,l_cdr2_end,l_cdr3_start,l_cdr3_end
01_PDL1,A,B,30,35,50,65,95,102,24,34,50,56,89,97
02_TNFA,A,B,30,35,50,65,95,102,24,34,50,56,89,97
12_AMBP,A,,30,35,50,65,95,102,,,,,,
```

**Column Descriptions:**
- `id`: **Must match target name** (e.g., `01_PDL1`, `12_AMBP`) - used for PDB file matching
- `h_chain`: **Required**. Chain ID of heavy chain in the structure (e.g., A, H). Design models must provide this.
- `l_chain`: **Required for antibody**. Chain ID of light chain in the structure (e.g., B, L). Empty for nanobody/VHH.
- `h_cdr1_start`, `h_cdr1_end`: Heavy chain CDR1 range (1-based, inclusive, PDB numbering)
- `h_cdr2_start`, `h_cdr2_end`: Heavy chain CDR2 range (1-based, inclusive, PDB numbering)
- `h_cdr3_start`, `h_cdr3_end`: Heavy chain CDR3 range (1-based, inclusive, PDB numbering)
- `l_cdr1_start`, `l_cdr1_end`: Light chain CDR1 range (optional, for nanobodies must be empty)
- `l_cdr2_start`, `l_cdr2_end`: Light chain CDR2 range (optional, for nanobodies must be empty)
- `l_cdr3_start`, `l_cdr3_end`: Light chain CDR3 range (optional, for nanobodies must be empty)

**CDR-Based Scaffold Fixing (Inverse Fold):**
- **H chain**: Fix all residues EXCEPT CDR1, CDR2, CDR3
- **L chain**: Fix all residues EXCEPT CDR1, CDR2, CDR3 (if present; nanobody has no L chain)
- **All other chains** (antigen, etc.): Fix ALL residues
- Chain IDs (`h_chain`, `l_chain`) must match the actual chain IDs in the PDB/CIF. Fallback to 'H' and 'L' if columns are missing (legacy).
- Matching is done by extracting target name from PDB filename (e.g., `01_PDL1_0.pdb` → `01_PDL1`)

**Scaffold Whitelist & Compliance**

- **Part 1 — fixed scaffold (one per task)**  
  Each Part 1 target must use **exactly one** fixed scaffold. No mixing of scaffolds within a Part 1 target.
  - **Antibody**: `hu-4D5-8_Fv.pdb` (scaffold ID: **1FVC**)
    - Config: `assets/antibody_nanobody/scaffolds/antibody/hu-4D5-8_Fv.yaml`
    - Heavy chain CDRs (1-based): CDR1 (26-33), CDR2 (51-58), CDR3 (97-105)
    - Light chain CDRs (1-based): CDR1 (143-148), CDR2 (166-168), CDR3 (205-213)
  - **Nanobody**: `h-NbBCII10.pdb` (scaffold ID: **3EAK**)
    - Config: `assets/antibody_nanobody/scaffolds/nanobody/h-NbBCII10.yaml`
    - Heavy chain CDRs (1-based): CDR1 (26-36), CDR2 (54-61), CDR3 (100-117)
- **Part 2 — whitelist, diversity encouraged**  
  Part 2 targets may use **any** scaffold from the whitelist; using **at least 3 different scaffolds** per target is recommended.
  - **Antibody whitelist** (15 scaffolds): 1FVC (hu-4D5-8_Fv.pdb), 6CR1, 5Y9K, 6WGB, 5YOY, 4M6M, 5UDC, 8IOW, 6WIO, 5J13, 5L6Y, 3HMW, 3H42, 6B3S, 5VZY  
    - Files: `assets/antibody_nanobody/scaffolds/antibody/` (PDB/CIF + YAML per scaffold)
  - **Nanobody whitelist** (5 scaffolds): 3EAK, 7EOW, 7XL0, 8COH, 8Z8V  
    - Files: `assets/antibody_nanobody/scaffolds/nanobody/` (PDB/CIF + YAML per scaffold)
- The pipeline loads scaffold and CDR information from these YAML files. You can override the scaffolds directory with the `scaffolds_dir` parameter in `run_antibody_pipeline.py`.
- **Scaffold Configuration Format**: YAML files in scaffolds directory
  - **Extended Format** (recommended for Part 1 scaffolds): Includes explicit CDR information
    ```yaml
    path: hu-4D5-8_Fv.pdb
    chains:
      - id: H
        type: heavy
        cdr_regions:
          cdr1:
            start: 26
            end: 33
          cdr2:
            start: 51
            end: 58
          cdr3:
            start: 97
            end: 105
      - id: L
        type: light
        cdr_regions:
          cdr1:
            start: 143
            end: 148
          cdr2:
            start: 166
            end: 168
          cdr3:
            start: 205
            end: 213
    scaffold_id: 1FVC
    scaffold_name: hu4D5-8
    ```
  - **Boltzgen Format**: Compatible with existing boltzgen YAML files (used for Part 2 scaffolds)
    - Uses `design` sections with `res_index` ranges
    - Can be converted to extended format if needed
  - CDR regions are specified in PDB residue numbering (1-based, inclusive)
  - Configuration files: `hu-4D5-8_Fv.yaml`, `h-NbBCII10.yaml`, etc.
- Pipeline automatically loads scaffold configurations and extracts CDR information
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
5. Refold: AlphaFold3 → `refold/af3_out/` (sequence-only; see note below)
6. **Step 5 (Evaluation)** — all in `run_antibody_pipeline.py`:
   - **PBP evaluation**: `run_protein_binding_protein_evaluation()` → `raw_data.csv`
   - **Interface analysis**: `run_antibody_interface_analysis()` → `interface_metrics.csv` (optional; disable with `enable_interface_analysis: false`)
   - **Developability**: `run_antibody_developability_evaluation()` → `developability_metrics.csv` (requires `cdr_info_csv`)

**Start from a specific step (`start_step`):**  
Set `start_step=5` to re-run only Step 5 (PBP evaluation + interface analysis + developability). Use the same config `root` (and Hydra `run.dir`) as the run that produced `refold/af3_out/` and `inverse_fold/backbones/`.

- **With `cdr_info_csv`:** runs both PBP evaluation and developability.
- **Without `cdr_info_csv`:** runs only PBP evaluation (writes `raw_data.csv`); developability is skipped.

Example:
```bash
# From project root; pipeline_dir = e.g. output/rfantibody/results
python scripts/run_antibody_pipeline.py start_step=5
# With developability (provide CDR CSV):
python scripts/run_antibody_pipeline.py start_step=5 cdr_info_csv=/path/to/cdr_info.csv
```

**Refold (AlphaFold3) note:** The AF3 refold step uses **sequence-only** input (no custom templates). The inverse-fold step fixes non-CDR scaffold; AF3 then predicts structure from the designed sequences without constraining non-CDR regions. CDR-only refold (fixing scaffold + antigen in AF3) is not supported by the current AF3 container format.

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

#### Antigen-Antibody Interface Analysis (Antibody Step 5)

Interface analysis is **part of** `run_antibody_pipeline.py` Step 5: it runs after PBP evaluation, on the same `refold/af3_out/` CIFs, and writes `interface_metrics.csv`. Disable with `enable_interface_analysis: false` in config.

**Metrics:** Paratope/epitope size, BSA, epitope patches, hydrophobic clusters, hydrogen bonds, paratope composition, charge complementarity, epitope secondary structure. Chain IDs: configurable via `ab_chain_ids` / `ag_chain_ids` (default antibody B,C and antigen A).

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

**DesignBench Responsibility:**
- Load standardized inputs
- Run Inverse Folding
- Run Refolding (AF3 refold is sequence-only; no custom template/region fixing)
- Calculate metrics

---

## Validation

DesignBench will validate:
- ✅ Input directory exists
- ✅ Required files found (PDB/CIF)
- ✅ Metadata file format (if required)
- ✅ Required columns present (for task-specific requirements)
- ✅ File format compatibility

If validation fails, DesignBench will raise clear error messages indicating what is missing.

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
```

**Quick validation (first target, first design only):**
```bash
python scripts/run_antibody_pipeline.py \
    design_dir=/path/to/boltzgen_output/antibody_benchcore_1per/design_dir \
    antibody_type=antibody \
    cdr_info_csv=/path/to/boltzgen_output/antibody_benchcore_1per/cdr_info.csv \
    target_config_path=assets/antibody_nanobody/config/target_config.csv \
    max_designs_per_target=1 \
    max_targets=1 \
    gpus=0
```

# VHH (nanobody)
```bash
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
id,h_chain,l_chain,h_cdr1_start,h_cdr1_end,h_cdr2_start,h_cdr2_end,h_cdr3_start,h_cdr3_end,l_cdr1_start,l_cdr1_end,l_cdr2_start,l_cdr2_end,l_cdr3_start,l_cdr3_end
01_7UXQ,A,B,30,35,50,65,95,102,24,34,50,56,89,97
02_1TNF,A,B,30,35,50,65,95,102,24,34,50,56,89,97
12_1BI7,A,,30,35,50,65,95,102,,,,,,
21_6COB,A,B,30,35,50,65,95,102,24,34,50,56,89,97
22_7WPC,A,B,30,35,50,65,95,102,24,34,50,56,89,97
```

**Target Configuration:**
The pipeline automatically loads target configuration from `assets/antibody_nanobody/config/target_config.csv`. This file defines:
- **target_id**: Target name (must match design filenames).
- **antigen_chains**: Antigen region to focus on (auth numbering). Format `ChainStart-End` (e.g. `A12-109` = chain A, residues 12–109). Part 2 targets (12–20) may list only a chain (e.g. `A` or `B`) with no residue range (free design).
- **target_hotspots**: Binding hotspots for Part 1; empty for Part 2.
- **epitope_description**: Human-readable description.

You can override the config path with `target_config_path` and the scaffolds directory (for Part 1/Part 2 YAML and PDB/CIF) with `scaffolds_dir` (default: `assets/antibody_nanobody/scaffolds`).

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

# DesignBench Usage Example

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

### Therapeutic Antibody/Nanobody Profile (Developability)

Developability is run inside **Step 5** of `run_antibody_pipeline.py`. To run the same evaluation standalone (e.g. on an existing `refold/af3_out/` directory):

```
# Same API as pipeline Step 5; results_dir = pipeline root (contains refold/af3_out)
python scripts/run_developability_evaluation.py cdr_info.csv ./output/rfantibody/results --output developability_metrics.csv

# Single design
python scripts/run_developability_evaluation.py cdr_info.csv ./output/rfantibody/results --antibody-id 01_7UXQ --num-seeds 8
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