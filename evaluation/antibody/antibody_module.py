"""
Antibody Design Module (scFv/Fab)

Handles full antibody structures with both heavy and light chains.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re
from Bio.PDB import PDBParser, MMCIFParser
import warnings
warnings.filterwarnings('ignore')

from .target_config import load_target_config, get_part1_targets, get_part2_targets, is_part1_target


# Scaffold Whitelist for Antibodies (scFv/Fab)
# Maps scaffold PDB ID to display name
ANTIBODY_SCAFFOLD_WHITELIST = {
    '1FVC': 'hu4D5-8',  # hu-4D5-8_Fv.pdb
    '6CR1': 'Adalimumab',
    '5Y9K': 'Belimumab',
    '6WGB': 'Dupilumab',
    '5YOY': 'Golimumab',
    '4M6M': 'Guselkumab',
    '5UDC': 'Nirsevimab',
    '8IOW': 'Sarilumab',
    '6WIO': 'Secukinumab',
    '5J13': 'Tezepelumab',
    '5L6Y': 'Tralokinumab',
    '3HMW': 'Ustekinumab',
    '3H42': 'MAB1',
    '6B3S': 'Necitumumab',
    '5VZY': 'Crenezumab'
}

# Part 1 fixed scaffold (hu-4D5-8_Fv.pdb corresponds to 1FVC)
PART1_FIXED_SCAFFOLD = '1FVC'

# Scaffold file mapping (filename to scaffold ID)
ANTIBODY_SCAFFOLD_FILES = {
    'hu-4D5-8_Fv.pdb': '1FVC',
    'adalimumab.6cr1.cif': '6CR1',
    'belimumab.5y9k.cif': '5Y9K',
    'dupilumab.6wgb.cif': '6WGB',
    'golimumab.5yoy.cif': '5YOY',
    'guselkumab.4m6m.cif': '4M6M',
    'nirsevimab.5udc.cif': '5UDC',
    'sarilumab.8iow.cif': '8IOW',
    'secukinumab.6wio.cif': '6WIO',
    'tezepelumab.5j13.cif': '5J13',
    'tralokinumab.5l6y.cif': '5L6Y',
    'ustekinumab.3hmw.cif': '3HMW',
    'mab1.3h42.cif': '3H42',
    'necitumumab.6b3s.cif': '6B3S',
    'crenezumab.5vzy.cif': '5VZY'
}


class AntibodyDesignModule:
    """
    Antibody Design Module for scFv/Fab structures.
    
    Handles full antibodies with both heavy and light chains.
    """
    
    def __init__(self, config, target_config_path: Optional[str] = None):
        self.config = config
        self.scaffold_whitelist = ANTIBODY_SCAFFOLD_WHITELIST
        self.part1_fixed_scaffold = PART1_FIXED_SCAFFOLD
        
        # Load target configuration
        try:
            self.target_df = load_target_config(target_config_path)
            self.part1_targets = get_part1_targets(self.target_df)
            self.part2_targets = get_part2_targets(self.target_df)
        except Exception as e:
            print(f"Warning: Failed to load target config: {e}")
            print("Using default target lists")
            # Fallback to default if config not available
            self.target_df = None
            self.part1_targets = []
            self.part2_targets = []
    
    def validate_target_name(self, target_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate target name format: {sequence_number}_{target_id}
        
        Args:
            target_name: Target name to validate
            
        Returns:
            (is_valid, error_message)
        """
        pattern = r'^(\d{2})_([A-Z0-9]{4,})$'
        match = re.match(pattern, target_name)
        
        if not match:
            return False, f"Invalid target name format: {target_name}. Expected format: {{sequence_number}}_{{target_id}}"
        
        seq_num = int(match.group(1))
        target_id = match.group(2)
        
        # Check if sequence number is in valid range
        if seq_num < 1 or seq_num > 22:
            return False, f"Sequence number {seq_num} out of range (01-22)"
        
        # If target config is loaded, validate against it
        if self.target_df is not None:
            valid_targets = self.target_df['target_id'].tolist()
            if target_name not in valid_targets:
                return False, f"Target {target_name} not found in target configuration"
        
        return True, None
    
    def parse_pdb_filename(self, filename: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """
        Parse PDB filename: {TargetName}_{Index}.pdb
        
        Args:
            filename: PDB filename
            
        Returns:
            (target_name, index, scaffold_name) or (None, None, None) if invalid
        """
        stem = Path(filename).stem
        pattern = r'^(\d{2}_[A-Z0-9]{4,})_(\d+)\.(pdb|cif)$'
        match = re.match(pattern, stem + Path(filename).suffix, re.IGNORECASE)
        
        if match:
            target_name = match.group(1)
            index = int(match.group(2))
            return target_name, index, None
        
        # Try alternative format: {TargetName}_{Scaffold}_{Index}.pdb
        pattern2 = r'^(\d{2}_[A-Z0-9]{4,})_([A-Z0-9]+)_(\d+)\.(pdb|cif)$'
        match2 = re.match(pattern2, stem + Path(filename).suffix, re.IGNORECASE)
        if match2:
            target_name = match2.group(1)
            scaffold_name = match2.group(2)
            index = int(match2.group(3))
            return target_name, index, scaffold_name
        
        return None, None, None
    
    def extract_scaffold_from_pdb(self, pdb_path: Path) -> Optional[str]:
        """
        Extract scaffold identifier from PDB file (from header or filename).
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            Scaffold identifier (e.g., '1FVC') or None
        """
        # Try to extract from filename first
        filename = pdb_path.name
        _, _, scaffold = self.parse_pdb_filename(filename)
        if scaffold:
            return scaffold.upper()
        
        # Check if filename matches known scaffold files
        filename_lower = filename.lower()
        for scaffold_file, scaffold_id in ANTIBODY_SCAFFOLD_FILES.items():
            if filename_lower == scaffold_file.lower() or scaffold_file.lower() in filename_lower:
                return scaffold_id
        
        return None
    
    def audit_input_directory(
        self, 
        input_dir: str, 
        cdr_info_csv: str,
        max_designs_per_target: int = 100
    ) -> Dict:
        """
        Audit input directory for compliance.
        
        Args:
            input_dir: Input directory containing PDB files
            cdr_info_csv: Path to CDR info CSV
            max_designs_per_target: Maximum designs per target (default: 100)
            
        Returns:
            Dictionary with audit results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Load CDR info
        if not os.path.exists(cdr_info_csv):
            raise FileNotFoundError(f"CDR info CSV not found: {cdr_info_csv}")
        
        cdr_df = pd.read_csv(cdr_info_csv)
        required_cols = ['id', 'h_cdr1_start', 'h_cdr1_end', 'h_cdr2_start', 'h_cdr2_end', 
                         'h_cdr3_start', 'h_cdr3_end']
        missing_cols = [col for col in required_cols if col not in cdr_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CDR info CSV: {missing_cols}")
        
        # Collect all PDB files
        pdb_files = list(input_path.rglob("*.pdb")) + list(input_path.rglob("*.cif"))
        
        # Group by target
        target_files = defaultdict(list)
        target_scaffolds = defaultdict(set)
        validation_errors = []
        
        for pdb_path in pdb_files:
            target_name, index, scaffold = self.parse_pdb_filename(pdb_path.name)
            
            if target_name is None:
                validation_errors.append(f"Invalid filename format: {pdb_path.name}")
                continue
            
            # Validate target name
            is_valid, error_msg = self.validate_target_name(target_name)
            if not is_valid:
                validation_errors.append(f"{pdb_path.name}: {error_msg}")
                continue
            
            # Check CDR info match
            cdr_match = cdr_df[cdr_df['id'] == target_name]
            if len(cdr_match) == 0:
                validation_errors.append(f"{pdb_path.name}: No CDR info found for target {target_name}")
                continue
            
            # Extract scaffold
            scaffold_id = scaffold
            if scaffold_id is None:
                scaffold_id = self.extract_scaffold_from_pdb(pdb_path)
            
            target_files[target_name].append((pdb_path, index, scaffold_id))
            if scaffold_id:
                target_scaffolds[target_name].add(scaffold_id)
        
        if validation_errors:
            raise ValueError(f"Input validation errors:\n" + "\n".join(validation_errors))
        
        # Check quota per target
        quota_errors = []
        for target_name, files in target_files.items():
            if len(files) > max_designs_per_target:
                quota_errors.append(
                    f"Target {target_name}: {len(files)} designs exceeds limit of {max_designs_per_target}"
                )
        
        if quota_errors:
            raise ValueError(f"Quota violations:\n" + "\n".join(quota_errors))
        
        # Sort files by index and limit to max_designs_per_target
        for target_name in target_files:
            files = sorted(target_files[target_name], key=lambda x: x[1])
            target_files[target_name] = files[:max_designs_per_target]
        
        # Scaffold diversity audit
        audit_results = {}
        warnings = []
        
        for target_name in sorted(target_files.keys()):
            is_part1 = is_part1_target(self.target_df, target_name) if self.target_df is not None else False
            scaffolds = target_scaffolds[target_name]
            count = len(target_files[target_name])
            
            # Part 1 audit: Check for single fixed scaffold requirement
            if is_part1:
                expected_scaffold = self.part1_fixed_scaffold
                if len(scaffolds) > 1:
                    warnings.append(
                        f"Part 1 target {target_name}: Multiple scaffolds detected {scaffolds}. "
                        f"Expected single scaffold {expected_scaffold}. Non-compliant with benchmark requirements."
                    )
                    status = "Warning"
                elif len(scaffolds) == 1:
                    scaffold = list(scaffolds)[0]
                    if scaffold != expected_scaffold:
                        warnings.append(
                            f"Part 1 target {target_name}: Scaffold {scaffold} does not match "
                            f"expected {expected_scaffold}. Non-compliant with benchmark requirements."
                        )
                        status = "Warning"
                    elif scaffold not in self.scaffold_whitelist:
                        warnings.append(
                            f"Part 1 target {target_name}: Scaffold {scaffold} not in whitelist. "
                            f"Non-compliant with benchmark requirements."
                        )
                        status = "Warning"
                    else:
                        status = "Pass"
                else:
                    status = "Warning"
                    warnings.append(f"Part 1 target {target_name}: No scaffold detected")
            else:
                # Part 2 audit: Check whitelist and diversity
                invalid_scaffolds = scaffolds - set(self.scaffold_whitelist.keys())
                if invalid_scaffolds:
                    warnings.append(
                        f"Part 2 target {target_name}: Scaffolds {invalid_scaffolds} not in whitelist"
                    )
                    status = "Warning"
                elif len(scaffolds) < 3:
                    warnings.append(
                        f"Part 2 target {target_name}: Scaffold diversity too low ({len(scaffolds)} < 3). "
                        f"This will severely impact generalization performance scoring."
                    )
                    status = "Warning"
                else:
                    status = "Pass"
            
            scaffold_str = ", ".join(sorted(scaffolds)) if scaffolds else "Unknown"
            audit_results[target_name] = {
                'target': target_name,
                'scaffolds': scaffold_str,
                'count': count,
                'status': status
            }
        
        return {
            'target_files': target_files,
            'audit_results': audit_results,
            'warnings': warnings,
            'cdr_df': cdr_df
        }
    
    def generate_compliance_report(self, audit_results: Dict) -> str:
        """
        Generate pre-run compliance report table.
        
        Args:
            audit_results: Results from audit_input_directory
            
        Returns:
            Formatted report string
        """
        results = audit_results['audit_results']
        warnings = audit_results['warnings']
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ANTIBODY DESIGN BENCHMARK - PRE-RUN COMPLIANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"{'Sequence':<12} {'Target':<15} {'Scaffold':<30} {'Count':<8} {'Status':<10}")
        report_lines.append("-" * 80)
        
        for target_name in sorted(results.keys()):
            result = results[target_name]
            seq_num = target_name.split('_')[0]
            report_lines.append(
                f"{seq_num:<12} {result['target']:<15} {result['scaffolds']:<30} "
                f"{result['count']:<8} {result['status']:<10}"
            )
        
        report_lines.append("")
        report_lines.append("-" * 80)
        
        if warnings:
            report_lines.append("WARNINGS:")
            for i, warning in enumerate(warnings, 1):
                report_lines.append(f"  {i}. {warning}")
        else:
            report_lines.append("No warnings detected.")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def calculate_fixed_residues(self, pdb_path: Path, cdr_row: pd.Series) -> List[str]:
        """
        Calculate fixed residues for antibody inverse folding.
        Fixed residues = All residues EXCEPT CDR loops.
        
        Args:
            pdb_path: Path to PDB file
            cdr_row: CDR information row from CSV
            
        Returns:
            List of fixed residue identifiers (e.g., ["H1", "H2", "L1", ...])
        """
        from inversefold.cdr_utils import calculate_fixed_residues_for_antibody
        return calculate_fixed_residues_for_antibody(pdb_path, cdr_row)
