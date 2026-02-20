"""
CDR Utilities for Antibody Inverse Folding

Provides functions to calculate fixed residues (scaffold) based on CDR information.
For antibody design, we fix all residues EXCEPT the 3 CDR loops.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from Bio.PDB import PDBParser, MMCIFParser
import warnings
warnings.filterwarnings('ignore')


def parse_cdr_ranges(cdr_info_row: pd.Series) -> Dict[str, Optional[Tuple[int, int]]]:
    """
    Parse CDR ranges from a row in CDR info CSV.
    
    Args:
        cdr_info_row: Pandas Series with CDR information
        
    Returns:
        Dictionary with keys: 'h_cdr1', 'h_cdr2', 'h_cdr3', 'l_cdr1', 'l_cdr2', 'l_cdr3'
        Values are (start, end) tuples (inclusive) or None if not present
    """
    cdr_ranges = {}
    
    # Heavy chain CDRs
    if pd.notna(cdr_info_row.get('h_cdr1_start')) and pd.notna(cdr_info_row.get('h_cdr1_end')):
        cdr_ranges['h_cdr1'] = (int(cdr_info_row['h_cdr1_start']), int(cdr_info_row['h_cdr1_end']))
    else:
        cdr_ranges['h_cdr1'] = None
    
    if pd.notna(cdr_info_row.get('h_cdr2_start')) and pd.notna(cdr_info_row.get('h_cdr2_end')):
        cdr_ranges['h_cdr2'] = (int(cdr_info_row['h_cdr2_start']), int(cdr_info_row['h_cdr2_end']))
    else:
        cdr_ranges['h_cdr2'] = None
    
    if pd.notna(cdr_info_row.get('h_cdr3_start')) and pd.notna(cdr_info_row.get('h_cdr3_end')):
        cdr_ranges['h_cdr3'] = (int(cdr_info_row['h_cdr3_start']), int(cdr_info_row['h_cdr3_end']))
    else:
        cdr_ranges['h_cdr3'] = None
    
    # Light chain CDRs (optional for nanobodies)
    if pd.notna(cdr_info_row.get('l_cdr1_start')) and pd.notna(cdr_info_row.get('l_cdr1_end')):
        cdr_ranges['l_cdr1'] = (int(cdr_info_row['l_cdr1_start']), int(cdr_info_row['l_cdr1_end']))
    else:
        cdr_ranges['l_cdr1'] = None
    
    if pd.notna(cdr_info_row.get('l_cdr2_start')) and pd.notna(cdr_info_row.get('l_cdr2_end')):
        cdr_ranges['l_cdr2'] = (int(cdr_info_row['l_cdr2_start']), int(cdr_info_row['l_cdr2_end']))
    else:
        cdr_ranges['l_cdr2'] = None
    
    if pd.notna(cdr_info_row.get('l_cdr3_start')) and pd.notna(cdr_info_row.get('l_cdr3_end')):
        cdr_ranges['l_cdr3'] = (int(cdr_info_row['l_cdr3_start']), int(cdr_info_row['l_cdr3_end']))
    else:
        cdr_ranges['l_cdr3'] = None
    
    return cdr_ranges


def get_cdr_residue_set(cdr_ranges: Dict[str, Optional[Tuple[int, int]]], 
                        chain_id: str, 
                        sequence_length: int) -> Set[int]:
    """
    Get set of residue indices (1-based) that are in CDR regions for a given chain.
    
    Args:
        cdr_ranges: Dictionary of CDR ranges from parse_cdr_ranges
        chain_id: Chain ID ('H' for heavy, 'L' for light)
        sequence_length: Length of the chain sequence
        
    Returns:
        Set of residue indices (1-based) that are in CDR regions
    """
    cdr_residues = set()
    
    if chain_id.upper() == 'H':
        # Heavy chain CDRs
        for cdr_name in ['h_cdr1', 'h_cdr2', 'h_cdr3']:
            if cdr_ranges.get(cdr_name) is not None:
                start, end = cdr_ranges[cdr_name]
                # CDR ranges are inclusive
                for res_idx in range(start, end + 1):
                    if 1 <= res_idx <= sequence_length:
                        cdr_residues.add(res_idx)
    elif chain_id.upper() == 'L':
        # Light chain CDRs
        for cdr_name in ['l_cdr1', 'l_cdr2', 'l_cdr3']:
            if cdr_ranges.get(cdr_name) is not None:
                start, end = cdr_ranges[cdr_name]
                # CDR ranges are inclusive
                for res_idx in range(start, end + 1):
                    if 1 <= res_idx <= sequence_length:
                        cdr_residues.add(res_idx)
    
    return cdr_residues


def calculate_fixed_residues_for_antibody(
    pdb_path: Path,
    cdr_info_row: pd.Series,
    heavy_chain_id: str = 'H',
    light_chain_id: Optional[str] = 'L'
) -> List[str]:
    """
    Calculate fixed residues (scaffold) for antibody inverse folding.
    
    Fixed residues = All residues EXCEPT CDR loops.
    Format: "H123", "L456" (chain_id + residue_number)
    
    Args:
        pdb_path: Path to PDB/CIF file
        cdr_info_row: Pandas Series with CDR information
        heavy_chain_id: Chain ID for heavy chain (default: 'H')
        light_chain_id: Chain ID for light chain (default: 'L', None for nanobodies)
        
    Returns:
        List of fixed residue identifiers in format "H123", "L456"
    """
    # Parse structure
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure('antibody', str(pdb_path))
    except Exception:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('antibody', str(pdb_path))
    
    # Parse CDR ranges
    cdr_ranges = parse_cdr_ranges(cdr_info_row)
    
    # Get chain objects
    chains = list(structure.get_chains())
    heavy_chain = None
    light_chain = None
    
    for chain in chains:
        if chain.id == heavy_chain_id:
            heavy_chain = chain
        elif light_chain_id and chain.id == light_chain_id:
            light_chain = chain
    
    if heavy_chain is None:
        raise ValueError(f"Heavy chain '{heavy_chain_id}' not found in structure")
    
    fixed_residues = []
    
    # Get CDR residue sets using PDB residue IDs
    heavy_cdr_set = get_cdr_residue_set_from_pdb(cdr_ranges, heavy_chain, 'H')
    
    # Process heavy chain
    for res in heavy_chain.get_residues():
        res_id = res.get_id()
        res_key = (heavy_chain_id, res_id)
        
        # Check if this residue is in CDR
        if res_key not in heavy_cdr_set:
            # This is a scaffold residue - add to fixed list
            if isinstance(res_id, tuple):
                pdb_res_num = res_id[1]  # (hetflag, resnum, icode)
            else:
                pdb_res_num = res_id
            fixed_residues.append(f"{heavy_chain_id}{pdb_res_num}")
    
    # Process light chain (if present)
    if light_chain is not None:
        light_cdr_set = get_cdr_residue_set_from_pdb(cdr_ranges, light_chain, 'L')
        
        for res in light_chain.get_residues():
            res_id = res.get_id()
            res_key = (light_chain_id, res_id)
            
            # Check if this residue is in CDR
            if res_key not in light_cdr_set:
                # This is a scaffold residue - add to fixed list
                if isinstance(res_id, tuple):
                    pdb_res_num = res_id[1]
                else:
                    pdb_res_num = res_id
                fixed_residues.append(f"{light_chain_id}{pdb_res_num}")
    
    return fixed_residues


def load_cdr_info_csv(cdr_info_csv: str) -> pd.DataFrame:
    """
    Load and validate CDR info CSV.
    
    Required columns:
    - id: Identifier for the antibody (should match PDB filename)
    - h_cdr1_start, h_cdr1_end, h_cdr2_start, h_cdr2_end, h_cdr3_start, h_cdr3_end
    - l_cdr1_start, l_cdr1_end, l_cdr2_start, l_cdr2_end, l_cdr3_start, l_cdr3_end (optional for nanobodies)
    
    Args:
        cdr_info_csv: Path to CDR info CSV file
        
    Returns:
        DataFrame with CDR information
    """
    df = pd.read_csv(cdr_info_csv)
    
    # Validate required columns
    required_cols = ['id', 'h_cdr1_start', 'h_cdr1_end', 'h_cdr2_start', 'h_cdr2_end', 
                     'h_cdr3_start', 'h_cdr3_end']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CDR info CSV: {missing_cols}")
    
    return df


def match_pdb_to_cdr_info(pdb_path: Path, cdr_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Match a PDB file to its CDR information row.
    
    Matching strategy:
    1. Extract target name from PDB path (e.g., "01_PDL1_0.pdb" -> "01_PDL1")
    2. Try to match with 'id' column in CDR CSV
    3. Support "target:scaffold" format in CSV
    
    Args:
        pdb_path: Path to PDB file
        cdr_df: DataFrame with CDR information
        
    Returns:
        Series with CDR information for this PDB, or None if not found
    """
    import re
    
    # Extract target name from PDB path
    # Format: "01_PDL1_0.pdb" or "01_PDL1_scaffold_0.pdb"
    pdb_stem = pdb_path.stem  # Without extension
    
    # Pattern: {序号}_{四位大写ID}_{可选scaffold}_{索引}
    # We want to extract: {序号}_{四位大写ID}
    pattern = r'^(\d{2}_[A-Z0-9]{4,})'
    match = re.match(pattern, pdb_stem)
    
    if match:
        target_name = match.group(1)
    else:
        # Fallback: try to extract base name
        base_name = re.sub(r'[-_]?\d+$', '', pdb_stem)
        target_name = base_name
    
    # Try exact match first
    matches = cdr_df[cdr_df['id'] == target_name]
    if len(matches) > 0:
        return matches.iloc[0]
    
    # Try with "target:scaffold" format
    # If CSV has "target:scaffold" format, try matching
    for idx, row in cdr_df.iterrows():
        csv_id = str(row['id'])
        # Check if CSV ID starts with target name
        if csv_id.startswith(target_name):
            return row
        # Check if CSV ID matches (with or without colon)
        if csv_id.replace(':', '-') == target_name or csv_id.replace(':', '_') == target_name:
            return row
        
        # Also try reverse: if target_name has colon, try matching
        if ':' in csv_id:
            csv_base = csv_id.split(':')[0] + '_' + csv_id.split(':')[1].split('_')[0] if '_' in csv_id.split(':')[1] else csv_id.split(':')[0]
            if csv_base == target_name:
                return row
    
    # If still no match, try partial match
    if target_name in cdr_df['id'].values:
        return cdr_df[cdr_df['id'] == target_name].iloc[0]
    
    return None
