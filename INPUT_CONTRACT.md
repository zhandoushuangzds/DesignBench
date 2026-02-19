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
├── target1-scaffold1-1.pdb  (or .cif)
├── target1-scaffold1-2.pdb
├── ...
├── target1-scaffold1-100.pdb
├── target2-scaffold1-1.pdb
└── ...
```

**Naming Convention:**
- Format: `{target}:{scaffold}-{design_number}.pdb`
- Example: `target1:scaffold1-1.pdb`, `target1:scaffold1-2.pdb`
- Supports both `:` and `-` as separators

**Requirements:**
- ✅ PDB or CIF format files
- ✅ Antibody structures (Heavy + Light chains, or Nanobody)
- ✅ Real residue names
- ✅ **Required**: CDR information CSV

**Required Metadata: `cdr_info_csv`**

**Must provide** a CSV with CDR indices for each antibody. This CSV is used for:
1. **Inverse Folding**: Fixing scaffold residues (all residues EXCEPT CDR loops)
2. **Developability Evaluation**: Calculating developability metrics

**CSV Format:**
```csv
id,heavy_fv,light_fv,h_cdr1_start,h_cdr1_end,h_cdr2_start,h_cdr2_end,h_cdr3_start,h_cdr3_end,l_cdr1_start,l_cdr1_end,l_cdr2_start,l_cdr2_end,l_cdr3_start,l_cdr3_end
target1:scaffold1,H,L,30,35,50,65,95,102,24,34,50,56,89,97
target2:scaffold1,H,,30,35,50,65,95,102,,,,,,
target1:scaffold2,H,L,30,35,50,65,95,102,24,34,50,56,89,97
```

**Column Descriptions:**
- `id`: Identifier matching PDB filename (supports "target:scaffold" format)
- `heavy_fv`: Heavy chain ID (typically 'H')
- `light_fv`: Light chain ID (typically 'L', empty for nanobodies)
- `h_cdr1_start`, `h_cdr1_end`: Heavy chain CDR1 range (1-based, inclusive)
- `h_cdr2_start`, `h_cdr2_end`: Heavy chain CDR2 range (1-based, inclusive)
- `h_cdr3_start`, `h_cdr3_end`: Heavy chain CDR3 range (1-based, inclusive)
- `l_cdr1_start`, `l_cdr1_end`: Light chain CDR1 range (optional, for nanobodies)
- `l_cdr2_start`, `l_cdr2_end`: Light chain CDR2 range (optional, for nanobodies)
- `l_cdr3_start`, `l_cdr3_end`: Light chain CDR3 range (optional, for nanobodies)

**CDR-Based Scaffold Fixing:**
- During inverse folding, **all residues EXCEPT the 3 CDR loops are fixed**
- This ensures only CDR regions are redesigned, keeping the scaffold constant
- Matching is done by PDB filename (supports "target:scaffold" format)

**Pipeline:**
1. Preprocess: Format designs → `formatted_designs/`
2. Inverse Fold: LigandMPNN → `inverse_fold/`
3. Refold: AlphaFold3 → `refold/af3_out/`
4. Evaluate: `run_protein_binding_protein_evaluation()`
5. Developability: `run_antibody_developability_evaluation()` (if `cdr_info_csv` provided)

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
| Antibody | PDB/CIF | CDR CSV (opt) | LigandMPNN | AF3 |
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
python scripts/run_antibody_pipeline.py \
    design_dir=/path/to/antibodies \
    inversefold=LigandMPNN \
    refold=af3 \
    cdr_info_csv=/path/to/cdr_info.csv \
    gpus=0,1
```

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
