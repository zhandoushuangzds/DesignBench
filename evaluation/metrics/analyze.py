import numpy as np
from Bio.PDB import PDBParser, NeighborSearch, is_aa
import biotite.structure as struc
from biotite.structure.io import pdb, pdbx
from pathlib import Path

# ==============================================================================
# 1. Hydrophobicity Calculation Constants (Thermofisher / Jeremie Alexander Constants)
# ==============================================================================

hydrophobicity_info = {
    "W": {"Rc": 12.25, "Rc1": 11.1, "Rc2": 11.8, "Rn": 12.25, "Rn1": 12.1},
    "F": {"Rc": 10.90, "Rc1": 7.5, "Rc2": 9.5, "Rn": 10.90, "Rn1": 10.3},
    "L": {"Rc": 9.30, "Rc1": 5.55, "Rc2": 7.4, "Rn": 9.30, "Rn1": 9.3},
    "I": {"Rc": 8.00, "Rc1": 5.2, "Rc2": 6.6, "Rn": 8.00, "Rn1": 7.7},
    "M": {"Rc": 6.20, "Rc1": 4.4, "Rc2": 5.7, "Rn": 6.20, "Rn1": 6.0},
    "V": {"Rc": 5.00, "Rc1": 2.9, "Rc2": 3.4, "Rn": 5.00, "Rn1": 4.2},
    "Y": {"Rc": 4.85, "Rc1": 3.7, "Rc2": 4.5, "Rn": 4.85, "Rn1": 4.4},
    "C": {
        "Rc": 0.45,
        "Rc1": 0.9,
        "Rc2": 0.2,
        "Rn": 0.45,
        "Rn1": -0.5,
    },  # carbamidomethylated Cys
    "P": {"Rc": 2.10, "Rc1": 2.1, "Rc2": 2.1, "Rn": 2.10, "Rn1": 2.1},
    "A": {"Rc": 1.10, "Rc1": 0.35, "Rc2": 0.5, "Rn": 1.10, "Rn1": -0.1},
    "E": {"Rc": 0.95, "Rc1": 1.0, "Rc2": 0.0, "Rn": 0.95, "Rn1": -0.1},
    "T": {"Rc": 0.65, "Rc1": 0.8, "Rc2": 0.6, "Rn": 0.65, "Rn1": 0.0},
    "D": {"Rc": 0.15, "Rc1": 0.5, "Rc2": 0.4, "Rn": 0.15, "Rn1": -0.5},
    "Q": {"Rc": -0.40, "Rc1": -0.7, "Rc2": -0.2, "Rn": -0.40, "Rn1": -1.1},
    "S": {"Rc": -0.15, "Rc1": 0.8, "Rc2": -0.1, "Rn": -0.15, "Rn1": -1.2},
    "G": {"Rc": -0.35, "Rc1": 0.2, "Rc2": 0.15, "Rn": -0.35, "Rn1": -0.7},
    "R": {"Rc": -1.40, "Rc1": 0.5, "Rc2": -1.1, "Rn": -1.30, "Rn1": -1.1},
    "N": {"Rc": -0.85, "Rc1": 0.2, "Rc2": -0.2, "Rn": -0.85, "Rn1": -1.1},
    "H": {"Rc": -1.45, "Rc1": -0.1, "Rc2": -0.2, "Rn": -1.45, "Rn1": -1.7},
    "K": {"Rc": -2.05, "Rc1": -0.6, "Rc2": -1.5, "Rn": -1.90, "Rn1": -1.45},
}

# Nearest-neighbor penalty for hydrophobics adjacent to H/R/K
nn_penalty = {"W": 0.15, "F": 0.10, "L": 0.30, "I": 0.15, "V": 0.20, "Y": 0.05}


# ==============================================================================
# 2. SSRCalc Hydrophobicity Algorithm Core Implementation
# ==============================================================================

def calc_base_h(seq: str) -> float:
    """
    Calculate base hydrophobicity score (Base Hydrophobicity), including position effects and neighbor penalties.
    """
    s = seq.upper()
    n = len(s)
    if n == 0:
        return 0.0
    H = 0.0

    # 1. Position-specific coefficients
    for i, aa in enumerate(s):
        if aa not in hydrophobicity_info:
            # Skip non-standard amino acids to prevent errors
            continue 
            
        if i == 0:
            key = "Rc1"
        elif i == 1:
            key = "Rc2"
        elif i == n - 1:
            key = "Rn"
        elif i == n - 2:
            key = "Rn1"
        else:
            key = "Rc"
        H += hydrophobicity_info[aa][key]

    # 2. Nearest-neighbor penalties
    for i, aa in enumerate(s):
        if aa in ("H", "R", "K"):
            for j in (i - 1, i + 1):
                if 0 <= j < n and s[j] in nn_penalty:
                    H -= nn_penalty[s[j]]

    # 3. Proline run penalties
    i = 0
    while i < n:
        if s[i] == "P":
            j = i
            while j < n and s[j] == "P":
                j += 1
            run = j - i
            if run >= 4:
                H -= 5.0
            elif run == 3:
                H -= 3.5
            elif run == 2:
                H -= 1.2
            i = j
        else:
            i += 1
    return H


def apply_length_weight(H: float, n: int) -> float:
    """
    Length correction factor.
    Short peptides (n < 8) are more hydrophilic, and hydrophobicity growth for very long peptides (n > 20) 
    shows diminishing returns due to folding and burial.
    """
    if n < 8:
        KL = 1.0 - 0.055 * (8 - n)
    elif n > 20:
        KL = 1.0 / (1.0 + 0.027 * (n - 20))
    else:
        KL = 1.0
    return H * KL


def overall_penalty(H: float) -> float:
    """
    Nonlinear normalization/penalty.
    Compresses extremely high hydrophobicity values to simulate saturation effects in actual retention time experiments.
    """
    if H <= 20:
        return H
    if H <= 30:
        return H - 0.27 * (H - 18.0)
    if H <= 40:
        return H - 0.33 * (H - 18.0)
    if H <= 50:
        return H - 0.38 * (H - 18.0)
    return H - 0.447 * (H - 18.0)


def calc_hydrophobicity(seq: str) -> float:
    """
    [Main function] Calculate final hydrophobicity score for a sequence.
    Pipeline: Base H -> Length Weight -> Overall Penalty
    """
    s = (seq or "").strip().upper()
    if not s or "X" in s:
        return float("nan")
        
    base = calc_base_h(s)
    base = apply_length_weight(base, len(s))
    return round(overall_penalty(base), 4)


# ==============================================================================
# 3. Structure Metrics Calculation (Hydrogen Bonds and Salt Bridges)
# ==============================================================================

def count_noncovalents(structure_path: str):
    """
    Calculate non-covalent interactions: number of hydrogen bonds and salt bridges.
    Supports PDB and CIF formats.
    
    Args:
        structure_path: Path to PDB or CIF file
        
    Returns:
        dict: Dictionary containing 'num_hydrogen_bonds' and 'num_salt_bridges'
    """
    try:
        # Use biotite to read structure file (supports PDB and CIF)
        path = Path(structure_path)
        suffix = path.suffix.lower()
        
        if suffix in ['.cif', '.mmcif']:
            cif_file = pdbx.CIFFile.read(str(path))
            atom_array = pdbx.get_structure(cif_file, model=1)
        elif suffix in ['.pdb', '.ent']:
            pdb_file = pdb.PDBFile.read(str(path))
            atom_array = pdb.get_structure(pdb_file, model=1)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")
        
        # Only process protein atoms (exclude ligands, water, etc.)
        protein_mask = ~atom_array.hetero
        atom_array = atom_array[protein_mask]
        
        if len(atom_array) == 0:
            return {'num_hydrogen_bonds': 0, 'num_salt_bridges': 0}
        
        # Use biotite's hydrogen bond detection
        try:
            hbonds = struc.hbond(atom_array)
            # hbonds is an (N, 3) array, each row is [donor_idx, hydrogen_idx, acceptor_idx]
            h_bonds_count = len(hbonds)
        except Exception as e:
            print(f"Warning: Failed to compute hydrogen bonds for {structure_path}: {e}")
            h_bonds_count = 0
        
        # Calculate salt bridges
        # Get charged residues
        acidic_residues = {'ASP', 'GLU'}
        basic_residues = {'LYS', 'ARG', 'HIS'}
        
        # Atoms from acidic residues (negatively charged)
        acidic_atom_names = {'OD1', 'OD2', 'OE1', 'OE2'}
        # Atoms from basic residues (positively charged)
        basic_atom_names = {'NZ', 'NH1', 'NH2', 'ND1', 'NE2'}
        
        # Filter relevant atoms
        neg_atoms = []
        pos_atoms = []
        
        for i, atom in enumerate(atom_array):
            res_name = str(atom.res_name).strip()
            atom_name = str(atom.atom_name).strip()
            
            if res_name in acidic_residues and atom_name in acidic_atom_names:
                neg_atoms.append(i)
            elif res_name in basic_residues and atom_name in basic_atom_names:
                pos_atoms.append(i)
        
        # Calculate salt bridges (distance threshold 4.0-5.5 Å)
        salt_bridge_count = 0
        if len(neg_atoms) > 0 and len(pos_atoms) > 0:
            neg_coords = atom_array.coord[neg_atoms]
            pos_coords = atom_array.coord[pos_atoms]
            
            # Calculate all distances
            distances = np.sqrt(((neg_coords[:, None, :] - pos_coords[None, :, :]) ** 2).sum(axis=2))
            
            # Find pairs with distances between 4.0-5.5 Å
            valid_pairs = np.where((distances >= 4.0) & (distances <= 5.5))
            
            # Check if from different residues
            unique_sb_pairs = set()
            for neg_idx, pos_idx in zip(valid_pairs[0], valid_pairs[1]):
                neg_atom_idx = neg_atoms[neg_idx]
                pos_atom_idx = pos_atoms[pos_idx]
                
                # Check if from different residues
                neg_res_id = (atom_array.chain_id[neg_atom_idx], atom_array.res_id[neg_atom_idx])
                pos_res_id = (atom_array.chain_id[pos_atom_idx], atom_array.res_id[pos_atom_idx])
                
                if neg_res_id != pos_res_id:
                    pair_id = tuple(sorted((neg_atom_idx, pos_atom_idx)))
                    unique_sb_pairs.add(pair_id)
            
            salt_bridge_count = len(unique_sb_pairs)
        
        return {
            'num_hydrogen_bonds': h_bonds_count,
            'num_salt_bridges': salt_bridge_count
        }
        
    except Exception as e:
        print(f"Error parsing structure {structure_path}: {e}")
        return {'num_hydrogen_bonds': 0, 'num_salt_bridges': 0}