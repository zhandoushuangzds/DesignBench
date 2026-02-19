#!/usr/bin/env python3
"""
Antigen-Antibody Interface Analysis Module

Provides comprehensive analysis of antigen-antibody binding interfaces,
including geometry, interactions, composition, and structure metrics.
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
    from Bio.PDB.DSSP import DSSP
    import freesasa
    from scipy.spatial import cKDTree
    import networkx as nx
except ImportError as e:
    print(f"❌ Missing library: {e}")
    sys.exit(1)

# ============================================================================
# Constants
# ============================================================================

# Interface distance threshold
INTERFACE_DISTANCE = 5.0  # Å

# Epitope patch connectivity distance
EPITOPE_PATCH_DISTANCE = 6.0  # Å

# Hydrophobic cluster distance
HYDROPHOBIC_CLUSTER_DISTANCE = 4.5  # Å

# Hydrogen bond distance
HBOND_DISTANCE = 3.5  # Å

# Hydrophobic residues (non-polar carbons)
HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}

# Paratope composition residues
PARATOPE_COMPOSITION_RESIDUES = {'TYR', 'SER', 'GLY', 'TRP'}

# Charged residues (pH 7)
POSITIVE_CHARGED = {'ARG', 'LYS'}  # +1
NEGATIVE_CHARGED = {'ASP', 'GLU'}  # -1
HISTIDINE = {'HIS'}  # +0.1 (approximate)

# Amino acid 3-to-1 mapping
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


class InterfaceAnalyzer:
    """
    Antigen-Antibody Interface Analyzer
    
    Analyzes binding interfaces between antibodies and antigens, providing
    comprehensive metrics on geometry, interactions, composition, and structure.
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        pass
    
    def parse_structure(self, cif_path: str, ab_chain_ids: Optional[List[str]] = None, ag_chain_ids: Optional[List[str]] = None):
        """
        Parse structure file and return structure object and atom arrays.
        
        Args:
            cif_path: Path to CIF or PDB file
            ab_chain_ids: Optional list of antibody chain IDs
            ag_chain_ids: Optional list of antigen chain IDs
            
        Returns:
            (structure, ab_atoms, ag_atoms, ab_coords, ag_coords, ab_residues, ag_residues, ab_chains, ag_chains)
        """
        parser = MMCIFParser(QUIET=True)
        try:
            structure = parser.get_structure('complex', cif_path)
        except Exception:
            # Try PDB parser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('complex', cif_path)
        
        # Identify chains
        chains = list(structure.get_chains())
        
        ab_chains = []
        ag_chains = []
        
        # Strategy: Use provided chain IDs or auto-detect
        if ab_chain_ids and ag_chain_ids:
            # Use specified chain IDs
            for chain in chains:
                if chain.id in ab_chain_ids:
                    ab_chains.append(chain)
                elif chain.id in ag_chain_ids:
                    ag_chains.append(chain)
        else:
            # Auto-detect: Look for H/L chains as antibody, others as antigen
            for chain in chains:
                if chain.id in ['H', 'L']:
                    ab_chains.append(chain)
                else:
                    ag_chains.append(chain)
            
            # Fallback: if no H/L found, use first chain(s) as antibody
            if not ab_chains:
                if len(chains) >= 2:
                    ab_chains = chains[:2]  # First 2 chains as antibody
                    ag_chains = chains[2:]  # Rest as antigen
                elif len(chains) == 1:
                    # Single chain - cannot determine interface
                    return None, None, None, None, None, None, None, None, None
        
        if not ag_chains:
            # No antigen chain found
            return None, None, None, None, None, None, None, None, None
        
        # Collect antibody atoms and coordinates
        ab_atoms = []
        ab_coords = []
        ab_residues = []
        
        for chain in ab_chains:
            for res in chain.get_residues():
                res_key = (chain.id, res.get_id())
                ab_residues.append(res_key)
                for atom in res.get_atoms():
                    if not atom.element == 'H':  # Heavy atoms only
                        ab_atoms.append((res_key, atom.name, atom))
                        ab_coords.append(atom.get_coord())
        
        # Collect antigen atoms and coordinates
        ag_atoms = []
        ag_coords = []
        ag_residues = []
        
        for chain in ag_chains:
            for res in chain.get_residues():
                res_key = (chain.id, res.get_id())
                ag_residues.append(res_key)
                for atom in res.get_atoms():
                    if not atom.element == 'H':  # Heavy atoms only
                        ag_atoms.append((res_key, atom.name, atom))
                        ag_coords.append(atom.get_coord())
        
        if not ab_coords or not ag_coords:
            return None, None, None, None, None, None, None, None, None
        
        return structure, ab_atoms, ag_atoms, np.array(ab_coords), np.array(ag_coords), ab_residues, ag_residues, ab_chains, ag_chains
    
    def identify_interface_residues(self, ab_coords, ag_coords, ab_residues, ag_residues, ab_atoms, ag_atoms):
        """
        Identify interface residues using cKDTree.
        
        Args:
            ab_coords: Antibody atom coordinates
            ag_coords: Antigen atom coordinates
            ab_residues: Antibody residue keys
            ag_residues: Antigen residue keys
            ab_atoms: Antibody atom list
            ag_atoms: Antigen atom list
            
        Returns:
            (paratope_residues, epitope_residues)
        """
        # Build KDTree for antigen
        ag_tree = cKDTree(ag_coords)
        
        # Find antibody atoms within interface distance
        paratope_residues = set()
        for i, (res_key, atom_name, atom) in enumerate(ab_atoms):
            dists, _ = ag_tree.query(ab_coords[i], k=1, distance_upper_bound=INTERFACE_DISTANCE)
            if dists <= INTERFACE_DISTANCE:
                paratope_residues.add(res_key)
        
        # Build KDTree for antibody
        ab_tree = cKDTree(ab_coords)
        
        # Find antigen atoms within interface distance
        epitope_residues = set()
        for i, (res_key, atom_name, atom) in enumerate(ag_atoms):
            dists, _ = ab_tree.query(ag_coords[i], k=1, distance_upper_bound=INTERFACE_DISTANCE)
            if dists <= INTERFACE_DISTANCE:
                epitope_residues.add(res_key)
        
        return paratope_residues, epitope_residues
    
    def calculate_bsa(self, structure, ab_chains, ag_chains, ab_residues, ag_residues):
        """
        Calculate Buried Surface Area (BSA).
        
        BSA = SASA_Ab + SASA_Ag - SASA_Complex
        
        Args:
            structure: Bio.PDB structure object
            ab_chains: List of antibody chain objects
            ag_chains: List of antigen chain objects
            ab_residues: Set of antibody residue keys
            ag_residues: Set of antigen residue keys
            
        Returns:
            BSA value in Å²
        """
        # Write structure to temporary PDB file for freesasa
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            io = PDBIO()
            io.set_structure(structure)
            io.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Calculate SASA for complex
            struct_complex = freesasa.Structure(tmp_path)
            res_complex = freesasa.calc(struct_complex)
            sasa_complex = res_complex.totalArea()
            
            # Calculate SASA for antibody (extract coordinates)
            ab_coords_list = []
            ab_atom_names = []
            for chain in ab_chains:
                for res in chain.get_residues():
                    for atom in res.get_atoms():
                        if not atom.element == 'H':
                            ab_coords_list.append(atom.get_coord())
                            ab_atom_names.append(atom.name)
            
            # Calculate SASA for antigen (extract coordinates)
            ag_coords_list = []
            ag_atom_names = []
            for chain in ag_chains:
                for res in chain.get_residues():
                    for atom in res.get_atoms():
                        if not atom.element == 'H':
                            ag_coords_list.append(atom.get_coord())
                            ag_atom_names.append(atom.name)
            
            # Create separate structures for SASA calculation
            # Note: freesasa needs full structure, so we'll use a workaround
            # Calculate SASA for each component separately by creating temporary files
            
            # For antibody
            ab_structure = structure.copy()
            # Remove antigen chains
            for chain in ag_chains:
                ab_structure[0].detach_child(chain.id)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_ab:
                io_ab = PDBIO()
                io_ab.set_structure(ab_structure)
                io_ab.save(tmp_ab.name)
                tmp_ab_path = tmp_ab.name
            
            # For antigen
            ag_structure = structure.copy()
            # Remove antibody chains
            for chain in ab_chains:
                ag_structure[0].detach_child(chain.id)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_ag:
                io_ag = PDBIO()
                io_ag.set_structure(ag_structure)
                io_ag.save(tmp_ag.name)
                tmp_ag_path = tmp_ag.name
            
            try:
                # Calculate SASA for antibody
                struct_ab = freesasa.Structure(tmp_ab_path)
                res_ab = freesasa.calc(struct_ab)
                sasa_ab = res_ab.totalArea()
                
                # Calculate SASA for antigen
                struct_ag = freesasa.Structure(tmp_ag_path)
                res_ag = freesasa.calc(struct_ag)
                sasa_ag = res_ag.totalArea()
                
                # Calculate BSA
                bsa = sasa_ab + sasa_ag - sasa_complex
                
            finally:
                if os.path.exists(tmp_ab_path):
                    os.unlink(tmp_ab_path)
                if os.path.exists(tmp_ag_path):
                    os.unlink(tmp_ag_path)
            
            return bsa
            
        except Exception as e:
            print(f"Warning: Failed to calculate BSA: {e}")
            return np.nan
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def calculate_epitope_patches(self, structure, epitope_residues, ag_chains):
        """
        Calculate number of epitope patches using graph-based approach.
        
        Args:
            structure: Bio.PDB structure object
            epitope_residues: Set of epitope residue keys
            ag_chains: List of antigen chain objects
            
        Returns:
            Number of connected components (patches)
        """
        if not epitope_residues:
            return 0
        
        # Build graph
        G = nx.Graph()
        
        # Add nodes
        epitope_list = list(epitope_residues)
        for res_key in epitope_list:
            G.add_node(res_key)
        
        # Add edges based on distance
        for i, res_key1 in enumerate(epitope_list):
            try:
                chain1 = structure[0][res_key1[0]]
                res1 = chain1[res_key1[1]]
                # Get CA coordinate
                if 'CA' in res1:
                    coord1 = res1['CA'].get_coord()
                else:
                    continue
            except (KeyError, IndexError):
                continue
            
            for j, res_key2 in enumerate(epitope_list[i+1:], start=i+1):
                try:
                    chain2 = structure[0][res_key2[0]]
                    res2 = chain2[res_key2[1]]
                    # Get CA coordinate
                    if 'CA' in res2:
                        coord2 = res2['CA'].get_coord()
                    else:
                        continue
                except (KeyError, IndexError):
                    continue
                
                # Calculate distance
                dist = np.linalg.norm(coord1 - coord2)
                if dist < EPITOPE_PATCH_DISTANCE:
                    G.add_edge(res_key1, res_key2)
        
        # Count connected components
        return nx.number_connected_components(G)
    
    def calculate_hydrophobic_clusters(self, structure, paratope_residues, epitope_residues, ab_chains, ag_chains):
        """
        Calculate size of largest hydrophobic cluster at interface.
        
        Args:
            structure: Bio.PDB structure object
            paratope_residues: Set of paratope residue keys
            epitope_residues: Set of epitope residue keys
            ab_chains: List of antibody chain objects
            ag_chains: List of antigen chain objects
            
        Returns:
            Size (number of atoms) of largest cluster
        """
        # Collect carbon atoms from hydrophobic residues at interface
        carbon_atoms = []
        carbon_coords = []
        
        # From paratope
        for chain in ab_chains:
            for res in chain.get_residues():
                res_key = (chain.id, res.get_id())
                if res_key in paratope_residues:
                    resname = res.resname
                    if resname in HYDROPHOBIC_RESIDUES:
                        for atom in res.get_atoms():
                            if atom.element == 'C':  # Carbon atoms
                                carbon_atoms.append(('ab', res_key, atom.name))
                                carbon_coords.append(atom.get_coord())
        
        # From epitope
        for chain in ag_chains:
            for res in chain.get_residues():
                res_key = (chain.id, res.get_id())
                if res_key in epitope_residues:
                    resname = res.resname
                    if resname in HYDROPHOBIC_RESIDUES:
                        for atom in res.get_atoms():
                            if atom.element == 'C':  # Carbon atoms
                                carbon_atoms.append(('ag', res_key, atom.name))
                                carbon_coords.append(atom.get_coord())
        
        if len(carbon_atoms) < 2:
            return 0
        
        carbon_coords = np.array(carbon_coords)
        
        # Build graph for clustering
        G = nx.Graph()
        
        # Add nodes
        for i, atom_info in enumerate(carbon_atoms):
            G.add_node(i)
        
        # Build KDTree for efficient distance queries
        tree = cKDTree(carbon_coords)
        
        # Find pairs within cluster distance
        pairs = tree.query_pairs(HYDROPHOBIC_CLUSTER_DISTANCE)
        
        # Add edges
        for i, j in pairs:
            # Only connect atoms from different sides (ab-ag interaction)
            if carbon_atoms[i][0] != carbon_atoms[j][0]:
                G.add_edge(i, j)
        
        # Find largest connected component
        if G.number_of_nodes() == 0:
            return 0
        
        largest_cluster = max(nx.connected_components(G), key=len)
        return len(largest_cluster)
    
    def calculate_hydrogen_bonds(self, structure, paratope_residues, epitope_residues, ab_chains, ag_chains):
        """
        Calculate approximate hydrogen bonds at interface.
        
        Args:
            structure: Bio.PDB structure object
            paratope_residues: Set of paratope residue keys
            epitope_residues: Set of epitope residue keys
            ab_chains: List of antibody chain objects
            ag_chains: List of antigen chain objects
            
        Returns:
            Number of hydrogen bonds
        """
        # Collect donor and acceptor atoms
        ab_donors = []  # N, O atoms in paratope
        ab_acceptors = []  # N, O atoms in paratope
        ag_donors = []  # N, O atoms in epitope
        ag_acceptors = []  # N, O atoms in epitope
        
        # From paratope
        for chain in ab_chains:
            for res in chain.get_residues():
                res_key = (chain.id, res.get_id())
                if res_key in paratope_residues:
                    for atom in res.get_atoms():
                        if atom.element in ['N', 'O']:
                            coord = atom.get_coord()
                            if atom.element == 'N':
                                ab_donors.append(coord)
                            if atom.element == 'O':
                                ab_acceptors.append(coord)
        
        # From epitope
        for chain in ag_chains:
            for res in chain.get_residues():
                res_key = (chain.id, res.get_id())
                if res_key in epitope_residues:
                    for atom in res.get_atoms():
                        if atom.element in ['N', 'O']:
                            coord = atom.get_coord()
                            if atom.element == 'N':
                                ag_donors.append(coord)
                            if atom.element == 'O':
                                ag_acceptors.append(coord)
        
        if not (ab_donors or ab_acceptors) or not (ag_donors or ag_acceptors):
            return 0
        
        # Count hydrogen bonds
        # Donor from Ab to Acceptor from Ag
        hbonds = 0
        
        if ab_donors and ag_acceptors:
            ab_donor_tree = cKDTree(np.array(ab_donors))
            ag_acceptors_array = np.array(ag_acceptors)
            dists, _ = ab_donor_tree.query(ag_acceptors_array, k=1, distance_upper_bound=HBOND_DISTANCE)
            hbonds += np.sum(dists <= HBOND_DISTANCE)
        
        # Donor from Ag to Acceptor from Ab
        if ag_donors and ab_acceptors:
            ag_donor_tree = cKDTree(np.array(ag_donors))
            ab_acceptors_array = np.array(ab_acceptors)
            dists, _ = ag_donor_tree.query(ab_acceptors_array, k=1, distance_upper_bound=HBOND_DISTANCE)
            hbonds += np.sum(dists <= HBOND_DISTANCE)
        
        return int(hbonds)
    
    def calculate_paratope_composition(self, structure, paratope_residues, ab_chains):
        """
        Calculate percentage of Tyr, Ser, Gly, Trp in paratope.
        
        Args:
            structure: Bio.PDB structure object
            paratope_residues: Set of paratope residue keys
            ab_chains: List of antibody chain objects
            
        Returns:
            Percentage of composition residues
        """
        if not paratope_residues:
            return 0.0
        
        total_residues = len(paratope_residues)
        composition_count = 0
        
        for chain in ab_chains:
            for res in chain.get_residues():
                res_key = (chain.id, res.get_id())
                if res_key in paratope_residues:
                    if res.resname in PARATOPE_COMPOSITION_RESIDUES:
                        composition_count += 1
        
        return (composition_count / total_residues) * 100.0 if total_residues > 0 else 0.0
    
    def calculate_charge_complementarity(self, structure, paratope_residues, epitope_residues, ab_chains, ag_chains):
        """
        Calculate charge complementarity metrics.
        
        Args:
            structure: Bio.PDB structure object
            paratope_residues: Set of paratope residue keys
            epitope_residues: Set of epitope residue keys
            ab_chains: List of antibody chain objects
            ag_chains: List of antigen chain objects
            
        Returns:
            Dict with 'charge_product' and 'charge_sum'
        """
        def get_net_charge(residues, chains):
            charge = 0.0
            for chain in chains:
                for res in chain.get_residues():
                    res_key = (chain.id, res.get_id())
                    if res_key in residues:
                        resname = res.resname
                        if resname in POSITIVE_CHARGED:
                            charge += 1.0
                        elif resname in NEGATIVE_CHARGED:
                            charge -= 1.0
                        elif resname in HISTIDINE:
                            charge += 0.1  # Approximate
            return charge
        
        ab_charge = get_net_charge(paratope_residues, ab_chains)
        ag_charge = get_net_charge(epitope_residues, ag_chains)
        
        return {
            'charge_product': ab_charge * ag_charge,
            'charge_sum': ab_charge + ag_charge,
            'ab_charge': ab_charge,
            'ag_charge': ag_charge
        }
    
    def calculate_epitope_secondary_structure(self, structure, epitope_residues, ag_chains):
        """
        Calculate secondary structure composition of epitope using DSSP.
        
        Args:
            structure: Bio.PDB structure object
            epitope_residues: Set of epitope residue keys
            ag_chains: List of antigen chain objects
            
        Returns:
            Dict with 'helix_percent', 'strand_percent', 'coil_percent'
        """
        if not epitope_residues:
            return {
                'helix_percent': np.nan,
                'strand_percent': np.nan,
                'coil_percent': np.nan
            }
        
        try:
            # Write structure to temporary file for DSSP
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                io = PDBIO()
                io.set_structure(structure)
                io.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                dssp = DSSP(structure[0], tmp_path, dssp='mkdssp')
                
                helix_count = 0
                strand_count = 0
                coil_count = 0
                total_epitope = 0
                
                for chain in ag_chains:
                    for res in chain.get_residues():
                        res_key = (chain.id, res.get_id())
                        if res_key in epitope_residues:
                            try:
                                dssp_key = (chain.id, res.get_id())
                                if dssp_key in dssp:
                                    ss = dssp[dssp_key][2]  # Secondary structure code
                                    if ss in ['H', 'G', 'I']:  # Helix
                                        helix_count += 1
                                    elif ss in ['E', 'B']:  # Strand
                                        strand_count += 1
                                    else:  # Coil
                                        coil_count += 1
                                    total_epitope += 1
                            except (KeyError, IndexError):
                                continue
                
                if total_epitope > 0:
                    return {
                        'helix_percent': (helix_count / total_epitope) * 100.0,
                        'strand_percent': (strand_count / total_epitope) * 100.0,
                        'coil_percent': (coil_count / total_epitope) * 100.0
                    }
                else:
                    return {
                        'helix_percent': np.nan,
                        'strand_percent': np.nan,
                        'coil_percent': np.nan
                    }
                    
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            # DSSP not available or error
            return {
                'helix_percent': np.nan,
                'strand_percent': np.nan,
                'coil_percent': np.nan
            }
    
    def calculate_epitope_segmentation(self, structure, epitope_residues, ag_chains):
        """
        Calculate number of continuous segments in epitope.
        
        Args:
            structure: Bio.PDB structure object
            epitope_residues: Set of epitope residue keys
            ag_chains: List of antigen chain objects
            
        Returns:
            Number of continuous segments (allowing gaps ≤ 2 residues)
        """
        if not epitope_residues:
            return 0
        
        # Collect sequence indices for each chain
        chain_indices = defaultdict(list)
        
        for chain in ag_chains:
            chain_id = chain.id
            for res in chain.get_residues():
                res_key = (chain_id, res.get_id())
                if res_key in epitope_residues:
                    # Get sequence index (residue number)
                    try:
                        res_num = res.get_id()[1]  # res_id is typically (hetflag, resnum, icode)
                        chain_indices[chain_id].append(res_num)
                    except (IndexError, AttributeError):
                        continue
        
        # Count segments for each chain
        total_segments = 0
        
        for chain_id, indices in chain_indices.items():
            if not indices:
                continue
            
            indices = sorted(set(indices))
            if len(indices) == 0:
                continue
            
            # Group into segments (allowing gaps ≤ 2)
            segments = 1
            for i in range(1, len(indices)):
                gap = indices[i] - indices[i-1]
                if gap > 3:  # Gap > 2 means new segment (gap of 3 = 2 residues between)
                    segments += 1
            
            total_segments += segments
        
        return total_segments
    
    def analyze_interface(self, cif_path: str, ab_chain_ids: Optional[List[str]] = None, ag_chain_ids: Optional[List[str]] = None):
        """
        Complete interface analysis.
        
        Args:
            cif_path: Path to structure file (CIF or PDB)
            ab_chain_ids: Optional list of antibody chain IDs (e.g., ['H', 'L'])
            ag_chain_ids: Optional list of antigen chain IDs (e.g., ['A'])
            
        Returns:
            Dictionary with all interface metrics organized by category
        """
        # Parse structure
        result = self.parse_structure(cif_path, ab_chain_ids, ag_chain_ids)
        if result[0] is None:
            return None
        
        structure, ab_atoms, ag_atoms, ab_coords, ag_coords, ab_residues, ag_residues, ab_chains, ag_chains = result
        
        if not ab_chains or not ag_chains:
            return None
        
        # Identify interface residues
        paratope_residues, epitope_residues = self.identify_interface_residues(
            ab_coords, ag_coords, ab_residues, ag_residues, ab_atoms, ag_atoms
        )
        
        if not paratope_residues or not epitope_residues:
            return None
        
        # ========================================================================
        # 1. Geometry Metrics
        # ========================================================================
        bsa = self.calculate_bsa(structure, ab_chains, ag_chains, paratope_residues, epitope_residues)
        num_epitope_patches = self.calculate_epitope_patches(structure, epitope_residues, ag_chains)
        
        geometry_metrics = {
            'paratope_size': len(paratope_residues),
            'epitope_size': len(epitope_residues),
            'bsa': bsa,
            'num_epitope_patches': num_epitope_patches
        }
        
        # ========================================================================
        # 2. Interaction Metrics
        # ========================================================================
        hydrophobic_cluster_size = self.calculate_hydrophobic_clusters(
            structure, paratope_residues, epitope_residues, ab_chains, ag_chains
        )
        num_hydrogen_bonds = self.calculate_hydrogen_bonds(
            structure, paratope_residues, epitope_residues, ab_chains, ag_chains
        )
        
        interaction_metrics = {
            'hydrophobic_cluster_size': hydrophobic_cluster_size,
            'num_hydrogen_bonds': num_hydrogen_bonds
        }
        
        # ========================================================================
        # 3. Composition & Charge Metrics
        # ========================================================================
        paratope_composition = self.calculate_paratope_composition(
            structure, paratope_residues, ab_chains
        )
        charge_metrics = self.calculate_charge_complementarity(
            structure, paratope_residues, epitope_residues, ab_chains, ag_chains
        )
        
        composition_metrics = {
            'paratope_composition_percent': paratope_composition,
            'charge_product': charge_metrics['charge_product'],
            'charge_sum': charge_metrics['charge_sum'],
            'ab_charge': charge_metrics['ab_charge'],
            'ag_charge': charge_metrics['ag_charge']
        }
        
        # ========================================================================
        # 4. Structure Metrics
        # ========================================================================
        ss_metrics = self.calculate_epitope_secondary_structure(
            structure, epitope_residues, ag_chains
        )
        num_segments = self.calculate_epitope_segmentation(
            structure, epitope_residues, ag_chains
        )
        
        structure_metrics = {
            'epitope_helix_percent': ss_metrics['helix_percent'],
            'epitope_strand_percent': ss_metrics['strand_percent'],
            'epitope_coil_percent': ss_metrics['coil_percent'],
            'epitope_num_segments': num_segments
        }
        
        # Combine all metrics
        results = {
            'Geometry': geometry_metrics,
            'Interaction': interaction_metrics,
            'Composition': composition_metrics,
            'Structure': structure_metrics
        }
        
        return results
