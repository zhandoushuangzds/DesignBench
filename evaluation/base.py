"""
Base Evaluator for benchcore.

This evaluator assumes standardized input:
- Input: Directory containing PDB files (real residues, not Poly-Ala)
- Metadata: scaffold_info.csv with backbone indices
- For Motif Scaffolding: Must specify motif residue ranges

The evaluator is task-agnostic and model-agnostic.
It only focuses on the evaluation pipeline starting from Inverse Folding.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import pandas as pd
import os
import shutil


class BaseEvaluator:
    """
    Base evaluator for all tasks.
    
    Assumes standardized input format:
    - PDB files with real residues (not Poly-Ala)
    - scaffold_info.csv with backbone indices
    - For motif tasks: motif residue ranges specified
    
    Pipeline:
    1. Load Inputs: Read PDBs and metadata
    2. Inverse Folding: Design sequences (with optional motif constraints)
    3. Refolding: Predict structures
    4. Metric Calculation: Compute evaluation metrics
    """
    
    def __init__(self, config):
        """
        Initialize the base evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_inputs(
        self,
        input_dir: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Load input PDBs and metadata.
        
        Input Contract:
        - input_dir: Directory containing PDB files
        - metadata_file: Path to scaffold_info.csv (optional, will search if not provided)
        
        Returns:
            Dictionary with:
            - 'pdbs': List of PDB file paths
            - 'metadata': DataFrame with scaffold_info
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all PDB files
        pdb_files = sorted(list(input_dir.glob("*.pdb")))
        if not pdb_files:
            raise ValueError(f"No PDB files found in {input_dir}")
        
        self.logger.info(f"Found {len(pdb_files)} PDB files in {input_dir}")
        
        # Load metadata
        if metadata_file is None:
            metadata_file = input_dir / "scaffold_info.csv"
        
        metadata_file = Path(metadata_file)
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file)
            self.logger.info(f"Loaded metadata from {metadata_file}")
        else:
            # Create minimal metadata from PDB filenames
            self.logger.warning(f"Metadata file not found: {metadata_file}. Creating minimal metadata.")
            metadata = pd.DataFrame({
                'sample_num': range(len(pdb_files)),
                'pdb_file': [f.name for f in pdb_files]
            })
        
        return {
            'pdbs': pdb_files,
            'metadata': metadata,
            'input_dir': input_dir
        }
    
    def run_inverse_folding(
        self,
        backbone_dir: Union[str, Path],
        output_dir: Union[str, Path],
        motif_constraints: Optional[Dict] = None
    ) -> Path:
        """
        Run inverse folding to design sequences for backbones.
        
        Args:
            backbone_dir: Directory containing backbone PDB files
            output_dir: Directory to save designed sequences
            motif_constraints: Optional dict with motif residue ranges to fix
                             Format: {sample_num: [(chain, start, end), ...]}
        
        Returns:
            Path to inverse folding output directory
        """
        from inversefold.inversefold_api import InverseFold
        
        backbone_dir = Path(backbone_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        inverse_fold_model = InverseFold(self.config)
        
        # If motif constraints provided, need to pass them to ProteinMPNN
        # This requires extending InverseFold API to support fixed positions
        gpu_list = str(self.config.gpus).split(',') if hasattr(self.config, 'gpus') else ['0']
        
        inverse_fold_model.run_proteinmpnn_distributed(
            input_dir=backbone_dir,
            output_dir=str(output_dir),
            gpu_list=gpu_list,
            origin_cwd=os.getcwd()
        )
        
        self.logger.info(f"Inverse folding completed. Output: {output_dir}")
        return output_dir
    
    def run_refolding(
        self,
        sequences_dir: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> Path:
        """
        Run refolding to predict structures from sequences.
        
        Args:
            sequences_dir: Directory containing designed sequences
            output_dir: Directory to save predicted structures
        
        Returns:
            Path to refolding output directory
        """
        from refold.refold_api import ReFold
        
        sequences_dir = Path(sequences_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        refold_model = ReFold(self.config)
        
        # Prepare input JSON for refolding model
        refold_input_json = output_dir / "refold_inputs.json"
        refold_model.make_esmfold_json_multi_process(
            backbone_dir=sequences_dir / "backbones",
            output_dir=str(refold_input_json)
        )
        
        # Run refolding
        refold_output = output_dir / "refold_output"
        refold_output.mkdir(parents=True, exist_ok=True)
        
        refold_model.run_esmfold(
            sequences_file_json=str(refold_input_json),
            output_dir=str(refold_output)
        )
        
        self.logger.info(f"Refolding completed. Output: {refold_output}")
        return refold_output
    
    def calculate_metrics(
        self,
        input_backbones: List[Path],
        refold_structures: List[Path],
        metadata: pd.DataFrame,
        output_dir: Union[str, Path]
    ) -> pd.DataFrame:
        """
        Calculate evaluation metrics.
        
        This is a base implementation. Subclasses should override for task-specific metrics.
        
        Args:
            input_backbones: List of input backbone PDB paths
            refold_structures: List of refolded structure PDB paths
            metadata: DataFrame with scaffold metadata
            output_dir: Directory to save metric results
        
        Returns:
            DataFrame with evaluation metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Base metrics: scRMSD (self-consistency RMSD)
        # This compares refolded structure to input backbone
        results = []
        
        for i, (backbone_path, refold_path) in enumerate(zip(input_backbones, refold_structures)):
            if not refold_path.exists():
                self.logger.warning(f"Refold structure not found: {refold_path}")
                continue
            
            # Calculate scRMSD
            from evaluation.metrics.rmsd import RMSDCalculator
            sc_rmsd = RMSDCalculator.compute_protein_ca_rmsd(
                pred=str(refold_path),
                refold=str(backbone_path)
            )
            
            results.append({
                'sample_num': i,
                'sc_rmsd': sc_rmsd,
                'backbone_path': str(backbone_path),
                'refold_path': str(refold_path)
            })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "metrics.csv", index=False)
        
        return results_df
    
    def run_evaluation(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Args:
            input_dir: Directory containing input PDB files
            output_dir: Directory to save evaluation results
            metadata_file: Path to scaffold_info.csv (optional)
        
        Returns:
            Dictionary with evaluation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load inputs
        self.logger.info("Step 1: Loading inputs...")
        inputs = self.load_inputs(input_dir, metadata_file)
        
        # Step 2: Inverse folding
        self.logger.info("Step 2: Running inverse folding...")
        inverse_fold_output = output_dir / "inverse_fold"
        self.run_inverse_folding(
            backbone_dir=input_dir,
            output_dir=inverse_fold_output
        )
        
        # Step 3: Refolding
        self.logger.info("Step 3: Running refolding...")
        refold_output = output_dir / "refold"
        self.run_refolding(
            sequences_dir=inverse_fold_output,
            output_dir=refold_output
        )
        
        # Step 4: Calculate metrics
        self.logger.info("Step 4: Calculating metrics...")
        metrics = self.calculate_metrics(
            input_backbones=inputs['pdbs'],
            refold_structures=list((refold_output / "refold_output").glob("*.pdb")),
            metadata=inputs['metadata'],
            output_dir=output_dir / "metrics"
        )
        
        return {
            'metrics': metrics,
            'output_dir': output_dir
        }
