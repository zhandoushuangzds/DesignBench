"""
Scaffold Configuration Loader

Loads scaffold information from YAML files in the scaffolds directory.
Supports both boltzgen format and our extended format with CDR information.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_scaffold_config(scaffold_file_path: str) -> Optional[Dict]:
    """
    Load scaffold configuration from YAML file.
    
    Supports both boltzgen format and extended format with CDR information.
    
    Args:
        scaffold_file_path: Path to scaffold YAML file
        
    Returns:
        Dictionary with scaffold configuration or None if not found
    """
    if not os.path.exists(scaffold_file_path):
        return None
    
    try:
        with open(scaffold_file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        warnings.warn(f"Failed to load scaffold config from {scaffold_file_path}: {e}")
        return None


def get_scaffold_cdr_info(scaffold_config: Dict, chain_id: str = 'H') -> Optional[Dict]:
    """
    Extract CDR information from scaffold configuration.
    
    Args:
        scaffold_config: Scaffold configuration dictionary
        chain_id: Chain ID ('H' for heavy, 'L' for light)
        
    Returns:
        Dictionary with CDR ranges: {'h_cdr1_start': int, 'h_cdr1_end': int, ...}
        or None if not found
    """
    if not scaffold_config:
        return None
    
    # Try extended format first (with chains and cdr_regions)
    if 'chains' in scaffold_config:
        for chain in scaffold_config['chains']:
            if chain.get('id') == chain_id and 'cdr_regions' in chain:
                cdr_info = {}
                cdr_regions = chain['cdr_regions']
                
                # Map to standard CDR naming
                prefix = 'h_' if chain_id == 'H' else 'l_'
                
                if 'cdr1' in cdr_regions:
                    cdr_info[f'{prefix}cdr1_start'] = cdr_regions['cdr1']['start']
                    cdr_info[f'{prefix}cdr1_end'] = cdr_regions['cdr1']['end']
                
                if 'cdr2' in cdr_regions:
                    cdr_info[f'{prefix}cdr2_start'] = cdr_regions['cdr2']['start']
                    cdr_info[f'{prefix}cdr2_end'] = cdr_regions['cdr2']['end']
                
                if 'cdr3' in cdr_regions:
                    cdr_info[f'{prefix}cdr3_start'] = cdr_regions['cdr3']['start']
                    cdr_info[f'{prefix}cdr3_end'] = cdr_regions['cdr3']['end']
                
                return cdr_info if cdr_info else None
    
    # Try boltzgen format (with design sections)
    if 'design' in scaffold_config:
        # Boltzgen format uses res_index ranges in design sections
        # Example: res_index: 26..32,52..57,99..110 (CDR1, CDR2, CDR3)
        cdr_info = {}
        prefix = 'h_' if chain_id == 'H' else 'l_'
        
        for design_section in scaffold_config['design']:
            # Boltzgen format: design is a list, each item has 'chain' with 'id' and 'res_index'
            design_chain = design_section.get('chain', {})
            if isinstance(design_chain, dict) and design_chain.get('id') == chain_id:
                res_index_str = design_chain.get('res_index', '')
                if res_index_str:
                    # Parse ranges like "26..32,52..57,99..110"
                    ranges = [r.strip() for r in res_index_str.split(',')]
                    cdr_ranges = []
                    for r in ranges:
                        if '..' in r:
                            start, end = r.split('..')
                            try:
                                cdr_ranges.append((int(start), int(end)))
                            except ValueError:
                                continue
                    
                    # Assign to CDR1, CDR2, CDR3 (assuming order)
                    if len(cdr_ranges) >= 1:
                        cdr_info[f'{prefix}cdr1_start'] = cdr_ranges[0][0]
                        cdr_info[f'{prefix}cdr1_end'] = cdr_ranges[0][1]
                    if len(cdr_ranges) >= 2:
                        cdr_info[f'{prefix}cdr2_start'] = cdr_ranges[1][0]
                        cdr_info[f'{prefix}cdr2_end'] = cdr_ranges[1][1]
                    if len(cdr_ranges) >= 3:
                        cdr_info[f'{prefix}cdr3_start'] = cdr_ranges[2][0]
                        cdr_info[f'{prefix}cdr3_end'] = cdr_ranges[2][1]
                    
                    return cdr_info if cdr_info else None
    
    return None


def find_scaffold_config_file(
    scaffold_name: str,
    scaffold_type: str = 'antibody',
    scaffolds_dir: Optional[str] = None
) -> Optional[str]:
    """
    Find scaffold configuration YAML file.
    
    Args:
        scaffold_name: Scaffold file name (e.g., 'hu-4D5-8_Fv.pdb', 'h-NbBCII10.pdb')
        scaffold_type: 'antibody' or 'nanobody'
        scaffolds_dir: Base directory for scaffolds (default: assets/antibody_nanobody/scaffolds)
        
    Returns:
        Path to YAML file or None if not found
    """
    if scaffolds_dir is None:
        # Default path relative to this file
        default_dir = Path(__file__).parent.parent.parent.parent / "assets" / "antibody_nanobody" / "scaffolds"
        scaffolds_dir = str(default_dir)
    
    scaffold_base = Path(scaffold_name).stem  # Remove extension
    yaml_path = Path(scaffolds_dir) / scaffold_type / f"{scaffold_base}.yaml"
    
    if yaml_path.exists():
        return str(yaml_path)
    
    return None


def get_part1_scaffold_info(
    scaffold_type: str = 'antibody',
    scaffolds_dir: Optional[str] = None
) -> Optional[Dict]:
    """
    Get Part 1 fixed scaffold information.
    
    Args:
        scaffold_type: 'antibody' or 'nanobody'
        scaffolds_dir: Base directory for scaffolds
        
    Returns:
        Dictionary with scaffold info including CDR regions
    """
    if scaffold_type == 'antibody':
        scaffold_file = 'hu-4D5-8_Fv.pdb'
    elif scaffold_type == 'nanobody':
        scaffold_file = 'h-NbBCII10.pdb'
    else:
        return None
    
    yaml_path = find_scaffold_config_file(scaffold_file, scaffold_type, scaffolds_dir)
    if not yaml_path:
        return None
    
    config = load_scaffold_config(yaml_path)
    if not config:
        return None
    
    # Extract CDR info for all chains
    scaffold_info = {
        'scaffold_file': scaffold_file,
        'scaffold_id': config.get('scaffold_id'),
        'scaffold_name': config.get('scaffold_name'),
        'path': config.get('path', scaffold_file),
        'cdr_info': {}
    }
    
    # Get CDR info for each chain
    if 'chains' in config:
        for chain in config['chains']:
            chain_id = chain.get('id')
            cdr_info = get_scaffold_cdr_info(config, chain_id)
            if cdr_info:
                scaffold_info['cdr_info'].update(cdr_info)
    
    return scaffold_info
