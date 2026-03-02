"""
Motif Scaffolding Evaluation for designbench.

Assumes standardized input:
- PDB files with real residues (not Poly-Ala)
- scaffold_info.csv with motif residue ranges specified
- Format: {sample_num, motif_placements, ...}

Pipeline:
1. Load Inputs
2. Inverse Folding (with motif constraints)
3. Refolding
4. Metrics: scRMSD, motifRMSD, Novelty, Diversity
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import pandas as pd
import os
import shutil
import json

from evaluation.base import BaseEvaluator


class MotifBenchEvaluator(BaseEvaluator):
    """
    Evaluator for Motif Scaffolding task.
    
    Input Contract:
    - PDB files: Real residues, motif positions already identified
    - scaffold_info.csv: Must contain 'motif_placements' column
      Format: "scaffold_before/chain1/scaffold_middle/chain2/scaffold_after"
      Example: "34/A/70" or "30/A/25/B/30"
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Import analysis modules from motif_scaffolding package
        try:
            from evaluation.motif_scaffolding.analysis import utils as au
            from evaluation.motif_scaffolding.analysis import diversity as du
            from evaluation.motif_scaffolding.analysis import novelty as nu
            self.au = au
            self.du = du
            self.nu = nu
        except ImportError as e:
            self.logger.warning(f"Analysis modules not available: {e}")
            self.au = self.du = self.nu = None
        
        # Get internal paths for motif scaffolding resources
        # Get the directory where this file is located
        self._module_dir = Path(__file__).parent
        self._motif_scaffolding_dir = self._module_dir / "motif_scaffolding"
        
        # Scripts directory (internal)
        self._scripts_dir = self._motif_scaffolding_dir / "scripts"
        
        # Motif PDBs directory - can be configured or use default internal resources
        if hasattr(config, 'motif_scaffolding') and hasattr(config.motif_scaffolding, 'motif_pdbs_dir'):
            # Use configured path if provided
            self.motif_pdbs_dir = Path(config.motif_scaffolding.motif_pdbs_dir)
        else:
            # Use default internal resources directory
            self.motif_pdbs_dir = self._motif_scaffolding_dir / "resources" / "motif_pdbs"
        
        # Create resources directory if it doesn't exist
        self.motif_pdbs_dir.mkdir(parents=True, exist_ok=True)
    
    def load_inputs(
        self,
        input_dir: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Load inputs with motif-specific validation.
        
        Validates that scaffold_info.csv contains motif_placements.
        """
        inputs = super().load_inputs(input_dir, metadata_file)
        
        # Validate motif_placements column
        if 'motif_placements' not in inputs['metadata'].columns:
            raise ValueError(
                "scaffold_info.csv must contain 'motif_placements' column. "
                "Format: 'scaffold_before/chain/scaffold_after' (e.g., '34/A/70')"
            )
        
        return inputs
    
    def generate_motif_info(
        self,
        scaffold_info_path: Union[str, Path],
        motif_name: str,
        output_dir: Union[str, Path]
    ) -> Path:
        """
        Generate motif_info.csv from scaffold_info.csv.
        
        This uses internal script to convert scaffold_info to motif_info format.
        """
        import subprocess
        import sys
        
        scaffold_info_path = Path(scaffold_info_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        motif_pdb_path = self.motif_pdbs_dir / f"{motif_name}.pdb"
        if not motif_pdb_path.exists():
            raise FileNotFoundError(
                f"Motif PDB not found: {motif_pdb_path}\n"
                f"Please ensure motif PDB files are available in: {self.motif_pdbs_dir}\n"
                f"Or configure 'motif_pdbs_dir' in your config file."
            )
        
        motif_info_path = output_dir / "motif_info.csv"
        
        script_path = self._scripts_dir / "write_motifInfo_from_scaffoldInfo.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        python_path = None
        if hasattr(self.config, 'motif_scaffolding') and hasattr(self.config.motif_scaffolding, 'python_path'):
            python_path = self.config.motif_scaffolding.python_path
        if not python_path:
            python_path = sys.executable
        
        cmd = [
            python_path,
            str(script_path),
            str(scaffold_info_path),
            str(motif_pdb_path),
            str(motif_info_path)
        ]
        
        self.logger.info(f"Generating motif_info.csv for {motif_name}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if not motif_info_path.exists():
            raise RuntimeError(f"Failed to generate motif_info.csv")
        
        return motif_info_path
    
    def calculate_metrics(
        self,
        input_backbones: List[Path],
        refold_structures: List[Path],
        metadata: pd.DataFrame,
        output_dir: Union[str, Path],
        motif_name: Optional[str] = None,
        scaffold_info_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Calculate motif-specific metrics: scRMSD, motifRMSD, Novelty, Diversity.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.au is None or self.du is None or self.nu is None:
            self.logger.warning("MotifBench analysis modules not available. Using base metrics only.")
            return super().calculate_metrics(
                input_backbones, refold_structures, metadata, output_dir
            )
        
        # Get motif_name from metadata or use default
        if motif_name is None:
            col = metadata.get('motif_name', None)
            motif_name = col.iloc[0] if hasattr(col, 'iloc') and len(metadata) > 0 else 'default_motif'
        
        motif_pdb_path = self.motif_pdbs_dir / f"{motif_name}.pdb"
        if not motif_pdb_path.exists():
            raise FileNotFoundError(f"Motif PDB not found: {motif_pdb_path}")
        
        # Generate motif_info.csv from scaffold_info.csv (MotifBench format)
        scaffold_info_path = scaffold_info_path or metadata.get('scaffold_info_path', None)
        if scaffold_info_path and Path(scaffold_info_path).exists():
            motif_info_path = self.generate_motif_info(
                scaffold_info_path, motif_name, output_dir
            )
            motif_info_df = pd.read_csv(motif_info_path)
        else:
            # Create minimal motif_info from metadata
            motif_info_df = metadata.copy()
        
        results = []
        successful_backbones = []
        
        # Map design_base (backbone stem) -> (motif_info row, backbone path) for lookup
        design_base_to_row_and_backbone = {}
        for i in range(min(len(input_backbones), len(motif_info_df))):
            design_base = Path(input_backbones[i]).stem
            design_base_to_row_and_backbone[design_base] = (
                motif_info_df.iloc[i],
                input_backbones[i]
            )
        
        # Iterate over ALL refold structures (e.g. 5 samples x 8 seqs = 40)
        for refold_path in sorted(refold_structures):
            if not refold_path.exists() or refold_path.suffix.lower() != '.pdb':
                continue
            stem = refold_path.stem
            # Parse "01_1LDB_0-1" -> design_base="01_1LDB_0", seq_idx="1"
            if "-" not in stem:
                continue
            design_base, seq_idx = stem.rsplit("-", 1)
            lookup = design_base_to_row_and_backbone.get(design_base)
            if lookup is None:
                continue
            row, backbone_path = lookup
            sample_num = row['sample_num']
            contig = row.get('contig', '')
            
            # Calculate motif RMSD
            try:
                reference_contig = self.au.reference_contig_from_segments(
                    str(motif_pdb_path),
                    row.get('segment_order', '')
                )
                # Build motif-only contig for refolded structure (motif_extract expects
                # contig string like "A1-7/A28-79", not parse_contig output)
                contig_segments = self.au.parse_contig(contig)
                motif_parts = []
                for seg in contig_segments:
                    if seg[0] != "scaffold":
                        chain, start, end = seg
                        motif_parts.append(f"{chain}{start}-{end}" if start != end else f"{chain}{start}")
                sample_contig = "/".join(motif_parts) if motif_parts else contig
                
                ref_motif = self.au.motif_extract(
                    reference_contig,
                    str(motif_pdb_path),
                    atom_part="backbone"
                )
                refold_motif = self.au.motif_extract(
                    sample_contig,
                    str(refold_path),
                    atom_part="backbone"
                )
                
                motif_rmsd = self.au.rmsd(ref_motif, refold_motif)
                
                # Calculate scRMSD (self-consistency)
                from evaluation.metrics.rmsd import RMSDCalculator
                if backbone_path and Path(backbone_path).exists():
                    sc_rmsd = RMSDCalculator.compute_protein_ca_rmsd(
                        pred=str(refold_path),
                        refold=str(backbone_path)
                    )
                else:
                    sc_rmsd = None
                
                # Success threshold: motif RMSD < 2.0 Å
                success = motif_rmsd < 2.0
                
                results.append({
                    'sample_num': sample_num,
                    'seq_idx': seq_idx,
                    'motif_rmsd': motif_rmsd,
                    'sc_rmsd': sc_rmsd,
                    'success': success,
                    'refold_path': str(refold_path)
                })
                
                if success:
                    successful_backbones.append(str(refold_path))
                    
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.logger.error(f"Error calculating metrics for {refold_path.name}: {e}\n{tb}")
                continue
        
        results_df = pd.DataFrame(results)
        
        # Calculate diversity and novelty if there are successful backbones
        if successful_backbones:
            self._calculate_diversity_and_novelty(
                successful_backbones, motif_name, output_dir, results_df
            )
        
        results_df.to_csv(output_dir / "motif_metrics.csv", index=False)
        return results_df
    
    def _calculate_diversity_and_novelty(
        self,
        successful_backbones: List[str],
        motif_name: str,
        output_dir: Path,
        results_df: pd.DataFrame
    ):
        """Calculate diversity and novelty metrics."""
        successful_dir = output_dir / "successful_backbones"
        successful_dir.mkdir(parents=True, exist_ok=True)
        
        for backbone_path in successful_backbones:
            shutil.copy2(backbone_path, successful_dir / Path(backbone_path).name)
            # Replace UNK with GLY for foldseek compatibility
            pdb_file = successful_dir / Path(backbone_path).name
            with open(pdb_file, 'r') as f:
                content = f.read()
            content = content.replace('UNK', 'GLY')
            with open(pdb_file, 'w') as f:
                f.write(content)
        
        # Calculate diversity with alpha=5
        foldseek_db = self.config.motif_scaffolding.foldseek_database
        assist_protein = self.motif_pdbs_dir / f"{motif_name}.pdb"
        
        diversity_result = self._calculate_diversity_with_alpha5(
            successful_dir, assist_protein, foldseek_db
        )
        
        # Calculate novelty
        novelty_value = 0.0
        success_results = results_df[results_df['success'] == True].copy()
        if len(success_results) > 0:
            # Prepare dataframe for novelty calculation
            # Novelty calculation expects 'backbone_path' column
            novelty_input = success_results.copy()
            if 'backbone_path' not in novelty_input.columns:
                if 'refold_path' in novelty_input.columns:
                    novelty_input['backbone_path'] = novelty_input['refold_path']
                else:
                    self.logger.warning("Cannot calculate novelty: missing refold_path column")
                    novelty_input = None
            
            if novelty_input is not None and len(novelty_input) > 0:
                try:
                    novelty_results = self.nu.calculate_novelty(
                        input_csv=novelty_input,
                        foldseek_database_path=foldseek_db,
                        max_workers=4,
                        cpu_threshold=75.0
                    )
                    novelty_path = output_dir / "novelty_results.csv"
                    novelty_results.to_csv(novelty_path, index=False)
                    # Calculate mean novelty (pdbTM column)
                    if 'pdbTM' in novelty_results.columns:
                        # Filter out NaN values
                        pdbTM_values = novelty_results['pdbTM'].dropna()
                        if len(pdbTM_values) > 0:
                            novelty_value = float(pdbTM_values.mean())
                    elif 'novelty' in novelty_results.columns:
                        novelty_values = novelty_results['novelty'].dropna()
                        if len(novelty_values) > 0:
                            novelty_value = float(novelty_values.mean())
                except Exception as e:
                    self.logger.warning(f"Failed to calculate novelty: {e}")
        
        # Count unique solutions (unique successful backbones)
        num_unique_solutions = len(set(successful_backbones))
        
        # Calculate success rate
        success_rate = len(successful_backbones) / len(results_df) if len(results_df) > 0 else 0
        
        # Save summary (JSON format)
        summary = {
            'motif_name': motif_name,
            'total_samples': len(results_df),
            'successful_samples': len(successful_backbones),
            'success_rate': success_rate,
            'diversity': diversity_result.get('Diversity', 0),
            'num_clusters': diversity_result.get('Clusters', 0),
            'num_solutions': num_unique_solutions,
            'alpha5_clusters': diversity_result.get('Alpha5_Clusters', 0),
            'novelty': novelty_value
        }
        
        summary_path = output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate esm_summary.txt (MotifBench format)
        esm_summary_path = output_dir / "esm_summary.txt"
        with open(esm_summary_path, 'w') as f:
            f.write(f"Evaluated Protein | {motif_name}\n")
            f.write(f"Number of Unique Solutions (unique successful backbones) | {num_unique_solutions}\n")
            f.write(f"Novelty | {novelty_value:.4f}\n")
            f.write(f"Success Rate | {success_rate * 100:.2f}\n")
            f.write(f"Number of Scaffolds Evaluated | {len(results_df)}\n")
        
        self.logger.info(f"Evaluation complete: {summary}")
    
    def _calculate_diversity_with_alpha5(
        self,
        successful_dir: Path,
        assist_protein: Path,
        foldseek_db: str
    ) -> Dict:
        """Calculate diversity using alpha=5 saturation curve."""
        if self.du is None:
            return {'Diversity': 0, 'Clusters': 0, 'Samples': 0, 'Alpha5_Clusters': 0}
        
        target_clusters = 5
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_result = None
        best_diff = float('inf')
        
        for threshold in thresholds:
            try:
                result = self.du.foldseek_cluster(
                    input=str(successful_dir),
                    assist_protein_path=str(assist_protein),
                    tmscore_threshold=threshold,
                    alignment_type=1,
                    output_mode="DICT",
                    save_tmp=True,
                    foldseek_path=None
                )
                
                num_clusters = result.get('Clusters', 0)
                diff = abs(num_clusters - target_clusters)
                
                if diff < best_diff:
                    best_diff = diff
                    best_result = result
                    best_result['Alpha5_Clusters'] = num_clusters
                    best_result['Alpha5_Threshold'] = threshold
                
                if diff <= 1:
                    break
            except Exception as e:
                self.logger.warning(f"Error clustering with threshold {threshold}: {e}")
                continue
        
        if best_result is None:
            best_result = self.du.foldseek_cluster(
                input=str(successful_dir),
                assist_protein_path=str(assist_protein),
                tmscore_threshold=0.6,
                alignment_type=1,
                output_mode="DICT",
                save_tmp=True,
                foldseek_path=None
            )
            best_result['Alpha5_Clusters'] = best_result.get('Clusters', 0)
            best_result['Alpha5_Threshold'] = 0.6
        
        return best_result
