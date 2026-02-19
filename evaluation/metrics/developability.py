#!/usr/bin/env python3
"""
Antibody Developability Metrics Module

Automatically detects antibody type (TAP mAb vs TNP Nanobody) and calculates
appropriate developability metrics with threshold-based categorization.

Metrics:
- Common: PSH (Hydrophobicity), PPC (Positive Charge), PNC (Negative Charge)
- TAP Only: SFvCSP (Charge Symmetry)
- TNP Only: CDR3_Length, CDR3_Compactness
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import warnings
import tempfile
warnings.filterwarnings('ignore')

try:
    from Bio.PDB import MMCIFParser, PDBParser, PDBIO
    import freesasa
    from scipy.spatial import cKDTree
except ImportError as e:
    print(f"❌ Missing library: {e}")
    sys.exit(1)

# ============================================================================
# Constants
# ============================================================================

# Theoretical MaxASA (Tien et al., 2013)
MAX_ASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0, 
    'GLU': 223.0, 'GLN': 225.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0, 
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0, 
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
}

HYDROPHOBIC = {'PHE', 'ILE', 'LEU', 'MET', 'VAL', 'TRP'}
POSITIVE = {'ARG', 'LYS'}
NEGATIVE = {'ASP', 'GLU'}
SURFACE_THRESHOLD = 7.5  # RSA > 7.5%
CDR_VICINITY_DISTANCE = 5.0  # Å (heavy atom distance)
PATCH_CONNECTIVITY_DISTANCE = 5.0  # Å (Cβ-Cβ distance)

# Amino acid 3-to-1 mapping
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# TNP Thresholds (Nanobody) - 1%/5% Percentiles
TNP_THRESHOLDS = {
    'Total_CDR_Length': {
        'red_lower': 21.40, 'amber_lower': 24.00,
        'amber_upper': 37.25, 'red_upper': 39.95
    },
    'CDR3_Length': {
        'red_lower': 6.05, 'amber_lower': 8.00,
        'amber_upper': 21.25, 'red_upper': 24.60
    },
    'PSH': {
        'red_lower': 83.94, 'amber_lower': 90.62,
        'amber_upper': 203.26, 'red_upper': 211.42
    },
    'CDR3_Compactness': {
        'red_lower': 0.557, 'amber_lower': 0.609,
        'amber_upper': 1.069, 'red_upper': 1.164
    },
    'PPC': {
        'amber_upper': 1.0, 'red_upper': 1.0
    },
    'PNC': {
        'amber_upper': 2.0, 'red_upper': 2.0
    }
}

# TAP Thresholds (mAb) - Standard
TAP_THRESHOLDS = {
    'Total_CDR_Length': {
        'amber_lower': 43.0, 'red_lower': 40.0,
        'amber_upper': 55.0, 'red_upper': 60.0
    },
    'PSH': {
        'amber_lower': 70.81, 'red_lower': 58.70,
        'amber_upper': 205.65, 'red_upper': 239.39
    },
    'SFvCSP': {
        'amber_upper': -7.00, 'red_upper': -12.00
    },
    'PPC': {
        'amber_upper': 1.25, 'red_upper': 2.00
    },
    'PNC': {
        'amber_upper': 2.00, 'red_upper': 2.00
    }
}


class DevelopabilityScorer:
    """
    Antibody Developability Metrics Scorer
    
    Automatically detects antibody type (TAP mAb vs TNP Nanobody) and calculates
    appropriate developability metrics with threshold-based categorization.
    """
    
    def __init__(self):
        """Initialize the scorer"""
        pass
    
    def detect_mode(self, row: pd.Series) -> str:
        """
        Detect antibody type based on CDR indices
        
        Args:
            row: CSV row with CDR indices
            
        Returns:
            'TAP' if light chain CDRs present, 'TNP' if only heavy chain CDRs
        """
        # Check if light chain CDRs are present
        has_light_cdrs = (
            not pd.isna(row.get('l_cdr1_start', np.nan)) and
            not pd.isna(row.get('l_cdr1_end', np.nan)) and
            not pd.isna(row.get('l_cdr2_start', np.nan)) and
            not pd.isna(row.get('l_cdr2_end', np.nan)) and
            not pd.isna(row.get('l_cdr3_start', np.nan)) and
            not pd.isna(row.get('l_cdr3_end', np.nan))
        )
        
        if has_light_cdrs:
            return 'TAP'
        else:
            return 'TNP'
    
    def parse_structure_and_map(self, cif_path: str, heavy_seq: str, light_seq: Optional[str] = None):
        """
        Parse structure and map PDB residues to input sequence indices
        
        Args:
            cif_path: CIF file path
            heavy_seq: Heavy chain Fv sequence
            light_seq: Light chain Fv sequence (optional, for TAP mode)
            
        Returns:
            (structure, atom_coords, index_map, sasa_dict, chain_h_id, chain_l_id)
        """
        parser = MMCIFParser(QUIET=True)
        try:
            structure = parser.get_structure('ab', cif_path)
        except Exception as e:
            return None, None, None, None, None, None

        # 1. Identify Chains
        chains = list(structure.get_chains())
        chain_h = None
        chain_l = None
        
        # Strategy A: Look for ID 'H' and 'L'
        for c in chains:
            if c.id == 'H':
                chain_h = c
            if c.id == 'L':
                chain_l = c
        
        # Strategy B: If not found, try 'A' and 'B'
        if not chain_h or not chain_l:
            for c in chains:
                if c.id == 'A' and not chain_h:
                    chain_h = c
                if c.id == 'B' and not chain_l:
                    chain_l = c
        
        # Strategy C: If still not found, use longest chains
        if not chain_h or not chain_l:
            candidates = sorted(chains, key=lambda c: len(list(c.get_residues())), reverse=True)
            if len(candidates) >= 2:
                if not chain_h:
                    chain_h = candidates[0]
                if not chain_l and len(candidates) > 1:
                    chain_l = candidates[1]
            elif len(candidates) == 1:
                # Nanobody case: only one chain
                if not chain_h:
                    chain_h = candidates[0]

        if not chain_h:
            return None, None, None, None, None, None
        
        chain_h_id = chain_h.id
        chain_l_id = chain_l.id if chain_l else None

        # 2. Build Sequence Map
        index_map = {}  # Keys: ('heavy', 0), ('light', 0)... Values: (chain_id, res_id_tuple)
        
        def map_chain(chain_obj, ref_seq, chain_type):
            if ref_seq is None or len(ref_seq) == 0:
                return True
                
            pdb_residues = list(chain_obj.get_residues())
            pdb_seq = ""
            valid_pdb_res = []
            for r in pdb_residues:
                if r.resname in AA_3TO1:
                    pdb_seq += AA_3TO1[r.resname]
                    valid_pdb_res.append(r)
            
            # Simple exact match check first
            if pdb_seq == ref_seq:
                for i, r in enumerate(valid_pdb_res):
                    index_map[(chain_type, i)] = (chain_obj.id, r.get_id())
                return True
            
            # Fallback: map 0..min_len
            limit = min(len(pdb_seq), len(ref_seq))
            for i in range(limit):
                index_map[(chain_type, i)] = (chain_obj.id, valid_pdb_res[i].get_id())
            return True

        map_chain(chain_h, heavy_seq, 'heavy')
        if chain_l and light_seq:
            map_chain(chain_l, light_seq, 'light')

        # 3. Calculate SASA
        sasa_dict = {}  # {(chain_id, res_id): {'total': X, 'sidechain': Y, 'resname': Z}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            io = PDBIO()
            io.set_structure(structure)
            io.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            struct_fs = freesasa.Structure(tmp_path)
            res_fs = freesasa.calc(struct_fs)
            
            idx = 0
            for chain in structure.get_chains():
                for res in chain.get_residues():
                    r_key = (chain.id, res.get_id())
                    if r_key not in sasa_dict:
                        sasa_dict[r_key] = {'total': 0.0, 'sidechain': 0.0, 'resname': res.resname}
                    
                    for atom in res.get_atoms():
                        if idx < struct_fs.nAtoms():
                            area = res_fs.atomArea(idx)
                            sasa_dict[r_key]['total'] += area
                            if atom.name not in ['N', 'CA', 'C', 'O']:
                                sasa_dict[r_key]['sidechain'] += area
                        idx += 1
                        
        except Exception as e:
            return None, None, None, None, None, None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # 4. Extract Atom Coords
        atom_coords = {}
        for chain in structure.get_chains():
            for res in chain.get_residues():
                r_key = (chain.id, res.get_id())
                for atom in res.get_atoms():
                    atom_coords[(r_key, atom.name)] = atom.get_coord()

        return structure, atom_coords, index_map, sasa_dict, chain_h_id, chain_l_id
    
    def get_surface_residues(self, sasa_dict):
        """Identify surface residues (RSA > 7.5%)"""
        surface = set()
        for r_key, data in sasa_dict.items():
            resname = data['resname']
            sc_sasa = data['sidechain']
            max_asa = MAX_ASA.get(resname, 200.0)
            rsa = (sc_sasa / max_asa) * 100.0
            if rsa > SURFACE_THRESHOLD:
                surface.add(r_key)
        return surface
    
    def calculate_common_metrics(self, structure, atom_coords, index_map, sasa_dict, 
                                  surface_residues, cdr_residues, chain_h_id, chain_l_id):
        """
        Calculate common metrics: PSH, PPC, PNC
        
        Returns:
            Dict with 'PSH', 'PPC', 'PNC'
        """
        # Build KDTree for vicinity queries
        all_heavy_atoms = []
        all_heavy_keys = []
        
        for (r_key, atom_name), coord in atom_coords.items():
            if not atom_name.startswith('H'):  # Heavy atoms only
                all_heavy_atoms.append(coord)
                all_heavy_keys.append(r_key)
        
        if not all_heavy_atoms:
            return {'PSH': 0.0, 'PPC': 0, 'PNC': 0}
        
        tree = cKDTree(all_heavy_atoms)
        
        # Identify CDR atom indices
        cdr_atom_indices = []
        for i, r_key in enumerate(all_heavy_keys):
            if r_key in cdr_residues:
                cdr_atom_indices.append(i)
        
        if not cdr_atom_indices:
            return {'PSH': 0.0, 'PPC': 0, 'PNC': 0}
        
        # Build CDR Tree
        cdr_coords = [all_heavy_atoms[i] for i in cdr_atom_indices]
        cdr_tree = cKDTree(cdr_coords)
        
        # Find vicinity residues
        vicinity_residues = set(cdr_residues)
        candidate_surface = surface_residues - cdr_residues
        
        for r_key in candidate_surface:
            try:
                chain_obj = structure[0][r_key[0]]
                res_obj = chain_obj[r_key[1]]
            except (KeyError, IndexError):
                continue
            
            for atom in res_obj:
                if atom.element == 'H':
                    continue
                dists, _ = cdr_tree.query(atom.get_coord(), k=1, distance_upper_bound=CDR_VICINITY_DISTANCE)
                if dists <= CDR_VICINITY_DISTANCE:
                    vicinity_residues.add(r_key)
                    break

        # Calculate Patches
        vicinity_list = list(vicinity_residues)
        
        # Get C-beta coords
        cb_coords = []
        valid_indices = []
        
        for i, r_key in enumerate(vicinity_list):
            try:
                chain_obj = structure[0][r_key[0]]
                res_obj = chain_obj[r_key[1]]
            except (KeyError, IndexError):
                continue
            atom = None
            if 'CB' in res_obj:
                atom = res_obj['CB']
            elif 'CA' in res_obj:
                atom = res_obj['CA']
            
            if atom:
                cb_coords.append(atom.get_coord())
                valid_indices.append(i)
        
        if not cb_coords:
            return {'PSH': 0.0, 'PPC': 0, 'PNC': 0}
        
        patch_tree = cKDTree(cb_coords)
        pairs = patch_tree.query_pairs(PATCH_CONNECTIVITY_DISTANCE)
        
        # Build graph
        neighbors = defaultdict(set)
        for i, j in pairs:
            idx_i, idx_j = valid_indices[i], valid_indices[j]
            r_i, r_j = vicinity_list[idx_i], vicinity_list[idx_j]
            neighbors[r_i].add(r_j)
            neighbors[r_j].add(r_i)
        
        def get_largest_patch_sasa(res_type_set):
            """Calculate largest patch SASA"""
            nodes = [r for r in vicinity_list if sasa_dict[r]['resname'] in res_type_set]
            node_set = set(nodes)
            
            visited = set()
            max_sasa = 0.0
            
            for node in nodes:
                if node not in visited:
                    patch_sasa = 0.0
                    stack = [node]
                    visited.add(node)
                    while stack:
                        curr = stack.pop()
                        patch_sasa += sasa_dict[curr]['total']
                        
                        for n in neighbors[curr]:
                            if n in node_set and n not in visited:
                                visited.add(n)
                                stack.append(n)
                    
                    if patch_sasa > max_sasa:
                        max_sasa = patch_sasa
            return max_sasa
        
        def get_largest_patch_count(res_type_set):
            """Calculate largest patch residue count"""
            nodes = [r for r in vicinity_list if sasa_dict[r]['resname'] in res_type_set]
            node_set = set(nodes)
            
            visited = set()
            max_count = 0
            
            for node in nodes:
                if node not in visited:
                    count = 0
                    stack = [node]
                    visited.add(node)
                    while stack:
                        curr = stack.pop()
                        count += 1
                        
                        for n in neighbors[curr]:
                            if n in node_set and n not in visited:
                                visited.add(n)
                                stack.append(n)
                    
                    if count > max_count:
                        max_count = count
            return float(max_count)
        
        psh = get_largest_patch_sasa(HYDROPHOBIC)
        ppc = get_largest_patch_count(POSITIVE)
        pnc = get_largest_patch_count(NEGATIVE)
        
        return {'PSH': psh, 'PPC': ppc, 'PNC': pnc}
    
    def calculate_sfvcsp(self, surface_residues, sasa_dict, chain_h_id, chain_l_id):
        """
        Calculate SFvCSP (Charge Symmetry Product) for TAP mode
        
        SFvCSP = Net_Charge(Heavy) * Net_Charge(Light)
        """
        def get_net_charge(chain_id):
            q = 0
            for r_key in surface_residues:
                if r_key[0] == chain_id:
                    resname = sasa_dict[r_key]['resname']
                    if resname in POSITIVE:
                        q += 1
                    if resname in NEGATIVE:
                        q -= 1
            return q
        
        if chain_h_id and chain_l_id:
            q_h = get_net_charge(chain_h_id)
            q_l = get_net_charge(chain_l_id)
            return q_h * q_l
        else:
            return np.nan
    
    def calculate_cdr3_compactness(self, structure, atom_coords, cdr3_residues, chain_h_id):
        """
        Calculate CDR3 Compactness for TNP mode
        
        Formula: Compactness = CDR3_Length / Reach
        Reach = Maximum distance from anchor center to any CDR3 loop atom
        """
        if not cdr3_residues or len(cdr3_residues) < 2:
            return np.nan
        
        try:
            # Get anchor residues (first and last)
            first_res_key = cdr3_residues[0]
            last_res_key = cdr3_residues[-1]
            anchor_set = {first_res_key, last_res_key}
            
            if first_res_key[0] != last_res_key[0]:
                return np.nan
            
            chain_obj = structure[0][first_res_key[0]]
            first_res = chain_obj[first_res_key[1]]
            last_res = chain_obj[last_res_key[1]]
            
            # Calculate anchor center
            anchor_atoms = []
            for res in [first_res, last_res]:
                for atom in res.get_atoms():
                    anchor_atoms.append(atom.get_coord())
            
            if not anchor_atoms:
                return np.nan
            
            anchor_center = np.mean(anchor_atoms, axis=0)
            
            # Collect loop atoms (excluding anchors)
            loop_atoms = []
            for res_key in cdr3_residues:
                if res_key not in anchor_set:
                    res_obj = chain_obj[res_key[1]]
                    for atom in res_obj.get_atoms():
                        loop_atoms.append(atom.get_coord())
            
            if not loop_atoms:
                return np.nan
            
            # Calculate Reach
            distances = [np.linalg.norm(atom_coord - anchor_center) 
                        for atom_coord in loop_atoms]
            reach = max(distances)
            
            # Calculate Compactness
            cdr3_length = len(cdr3_residues)
            if reach > 0:
                return cdr3_length / reach
            else:
                return np.nan
                
        except (KeyError, IndexError, AttributeError, ValueError):
            return np.nan
    
    def calculate_metrics(self, cif_path: str, row: pd.Series):
        """
        Calculate developability metrics based on detected antibody type
        
        Args:
            cif_path: CIF file path
            row: CSV row with CDR indices and sequences
            
        Returns:
            Dict with metrics and categorization
        """
        # Detect mode
        mode = self.detect_mode(row)
        
        # Parse structure
        heavy_seq = row.get('heavy_fv', '')
        light_seq = row.get('light_fv', None) if mode == 'TAP' else None
        
        structure, atom_coords, index_map, sasa_dict, chain_h_id, chain_l_id = \
            self.parse_structure_and_map(cif_path, heavy_seq, light_seq)
        
        if not structure:
            return None
        
        # Get surface residues
        surface_residues = self.get_surface_residues(sasa_dict)
        
        # Identify CDR residues
        cdr_residues = set()
        cdr3_residues = []
        
        def add_cdr(chain_type, start, end, is_cdr3=False):
            if pd.isna(start) or pd.isna(end):
                return
            start_idx = int(start)
            end_idx = int(end)  # Note: end is exclusive (not included) for CDR3 Length calculation
            for i in range(start_idx, end_idx):
                key = (chain_type, i)
                if key in index_map:
                    r_key = index_map[key]
                    cdr_residues.add(r_key)
                    if is_cdr3:
                        cdr3_residues.append(r_key)
        
        # Heavy chain CDRs
        # For TAP: use inclusive end (end+1), for TNP: use exclusive end (for CDR3 Length calculation)
        if mode == 'TAP':
            # TAP mode: end is inclusive (matching calculate_tap_metrics_v2.py)
            def add_cdr_tap(chain_type, start, end):
                if pd.isna(start) or pd.isna(end):
                    return
                start_idx = int(start)
                end_idx = int(end) + 1  # Inclusive end for TAP
                for i in range(start_idx, end_idx):
                    key = (chain_type, i)
                    if key in index_map:
                        cdr_residues.add(index_map[key])
            
            add_cdr_tap('heavy', row.get('h_cdr1_start'), row.get('h_cdr1_end'))
            add_cdr_tap('heavy', row.get('h_cdr2_start'), row.get('h_cdr2_end'))
            add_cdr_tap('heavy', row.get('h_cdr3_start'), row.get('h_cdr3_end'))
            add_cdr_tap('light', row.get('l_cdr1_start'), row.get('l_cdr1_end'))
            add_cdr_tap('light', row.get('l_cdr2_start'), row.get('l_cdr2_end'))
            add_cdr_tap('light', row.get('l_cdr3_start'), row.get('l_cdr3_end'))
        else:
            # TNP mode: end is exclusive (for correct CDR3 Length = end - start)
            add_cdr('heavy', row.get('h_cdr1_start'), row.get('h_cdr1_end'))
            add_cdr('heavy', row.get('h_cdr2_start'), row.get('h_cdr2_end'))
            add_cdr('heavy', row.get('h_cdr3_start'), row.get('h_cdr3_end'), is_cdr3=True)
        
        if not cdr_residues:
            return None
        
        # Calculate common metrics
        common_metrics = self.calculate_common_metrics(
            structure, atom_coords, index_map, sasa_dict,
            surface_residues, cdr_residues, chain_h_id, chain_l_id
        )
        
        # Calculate mode-specific metrics
        results = common_metrics.copy()
        results['Mode'] = mode
        
        # Calculate total CDR length
        total_cdr_length = len(cdr_residues)
        results['Total_CDR_Length'] = total_cdr_length
        
        if mode == 'TAP':
            # TAP-specific: SFvCSP
            results['SFvCSP'] = self.calculate_sfvcsp(
                surface_residues, sasa_dict, chain_h_id, chain_l_id
            )
        else:
            # TNP-specific: CDR3 Length and Compactness
            cdr3_length = len(cdr3_residues)
            results['CDR3_Length'] = cdr3_length
            results['CDR3_Compactness'] = self.calculate_cdr3_compactness(
                structure, atom_coords, cdr3_residues, chain_h_id
            )
        
        # Apply thresholds and categorize
        results.update(self.categorize_metrics(results, mode))
        
        return results
    
    def categorize_metrics(self, metrics: Dict, mode: str) -> Dict:
        """
        Categorize metrics into Green/Amber/Red based on thresholds
        
        Returns:
            Dict with '_Category' suffix for each metric
        """
        thresholds = TAP_THRESHOLDS if mode == 'TAP' else TNP_THRESHOLDS
        categories = {}
        
        for metric_name, value in metrics.items():
            if metric_name in ['Mode'] or metric_name.endswith('_Category'):
                continue
            
            if metric_name not in thresholds:
                continue
            
            if pd.isna(value) or np.isnan(value):
                categories[f'{metric_name}_Category'] = 'Unknown'
                continue
            
            thresh = thresholds[metric_name]
            category = 'Green'  # Default
            
            # Check red thresholds first
            if 'red_lower' in thresh and value < thresh['red_lower']:
                category = 'Red'
            elif 'red_upper' in thresh and value > thresh['red_upper']:
                category = 'Red'
            # Check amber thresholds
            elif 'amber_lower' in thresh and value < thresh['amber_lower']:
                category = 'Amber'
            elif 'amber_upper' in thresh and value > thresh['amber_upper']:
                category = 'Amber'
            
            categories[f'{metric_name}_Category'] = category
        
        return categories
