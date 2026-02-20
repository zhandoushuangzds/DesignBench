"""
Nanobody Design Module (VHH)

Handles nanobody structures with only heavy chain (no light chain).
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


# Scaffold Whitelist for Nanobodies (VHH)
NANOBODY_SCAFFOLD_WHITELIST = {
    '3EAK': 'VHH scaffold',
    '7EOW': 'VHH scaffold',
    '7XL0': 'VHH scaffold',
    '8COH': 'VHH scaffold',
    '8Z8V': 'VHH scaffold'
}

# Part 1 Benchmark Targets (01-11) - Fixed scaffold requirement
PART1_TARGETS = {
    '01_PDL1': '3EAK',  # Standard scaffold for Part 1 (VHH)
    '02_TNFA': '3EAK',
    '03_PDGF': '3EAK',
    '04_IL7R': '3EAK',
    '05_INSR': '3EAK',
    '06_H1HA': '3EAK',
    '07_RSV1': '3EAK',
    '08_RSV3': '3EAK',
    '09_RBDS': '3EAK',
    '10_IL10': '3EAK',
    '11_BLAC': '3EAK'
}

# Part 2 Challenge Targets (12-22) - Allow scaffold diversity
PART2_TARGETS = [
    '12_AMBP', '13_GM2A', '14_HNMT', '15_IDI2', '16_MZB1',
    '17_ORM2', '18_PHYH', '19_PMVK', '20_RFK', '21_PPOX', '22_FHAB'
]


class NanobodyDesignModule:
    """
    Nanobody Design Module for VHH structures.
    
    Handles nanobodies with only heavy chain (no light chain).
    """
    
    def __init__(self, config):
        self.config = config
        self.scaffold_whitelist = NANOBODY_SCAFFOLD_WHITELIST
        self.part1_targets = PART1_TARGETS
        self.part2_targets = PART2_TARGETS
    
    def validate_target_name(self, target_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate target name format: {序号}_{四位大写ID}
        
        Args:
            target_name: Target name to validate
            
        Returns:
            (is_valid, error_message)
        """
        pattern = r'^(\d{2})_([A-Z0-9]{4,})$'
        match = re.match(pattern, target_name)
        
        if not match:
            return False, f"Invalid target name format: {target_name}. Expected format: {{序号}}_{{四位大写ID}}"
        
        seq_num = int(match.group(1))
        target_id = match.group(2)
        
        # Check if sequence number is in valid range
        if seq_num < 1 or seq_num > 22:
            return False, f"Sequence number {seq_num} out of range (01-22)"
        
        # Check if target is in known list
        if seq_num <= 11:
            expected_id = list(self.part1_targets.keys())[seq_num - 1].split('_')[1]
            if target_id != expected_id:
                return False, f"Part 1 target {seq_num:02d} should be {expected_id}, got {target_id}"
        elif seq_num <= 22:
            expected_id = PART2_TARGETS[seq_num - 12].split('_')[1]
            if target_id != expected_id:
                return False, f"Part 2 target {seq_num:02d} should be {expected_id}, got {target_id}"
        
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
            Scaffold identifier (e.g., '3EAK') or None
        """
        # Try to extract from filename first
        filename = pdb_path.name
        _, _, scaffold = self.parse_pdb_filename(filename)
        if scaffold:
            return scaffold.upper()
        
        # Try to read from PDB header
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('nanobody', str(pdb_path))
            # Check header for scaffold info (if available)
            # For now, return None and rely on filename
        except Exception:
            pass
        
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
        
        # Verify no light chain CDR columns (VHH should not have light chain)
        light_chain_cols = ['l_cdr1_start', 'l_cdr1_end', 'l_cdr2_start', 'l_cdr2_end', 
                           'l_cdr3_start', 'l_cdr3_end']
        if any(col in cdr_df.columns for col in light_chain_cols):
            # Check if any row has light chain data
            has_light_chain = False
            for col in light_chain_cols:
                if col in cdr_df.columns:
                    if cdr_df[col].notna().any():
                        has_light_chain = True
                        break
            if has_light_chain:
                raise ValueError(
                    "Nanobody (VHH) module detected light chain CDR data in CSV. "
                    "VHH structures should only have heavy chain CDRs."
                )
        
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
            seq_num = int(target_name.split('_')[0])
            scaffolds = target_scaffolds[target_name]
            count = len(target_files[target_name])
            
            # Part 1 audit: Check for single scaffold requirement
            if seq_num <= 11:
                expected_scaffold = self.part1_targets.get(target_name)
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
        report_lines.append("NANOBODY DESIGN BENCHMARK - PRE-RUN COMPLIANCE REPORT")
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
        Calculate fixed residues for nanobody inverse folding.
        Fixed residues = All residues EXCEPT CDR loops.
        Note: VHH only has heavy chain, no light chain.
        
        Args:
            pdb_path: Path to PDB file
            cdr_row: CDR information row from CSV (should only have heavy chain CDRs)
            
        Returns:
            List of fixed residue identifiers (e.g., ["H1", "H2", ...])
        """
        from inversefold.cdr_utils import calculate_fixed_residues_for_antibody
        # For VHH, we only process heavy chain, light_chain_id should be None
        return calculate_fixed_residues_for_antibody(
            pdb_path, 
            cdr_row,
            heavy_chain_id='H',
            light_chain_id=None
        )
