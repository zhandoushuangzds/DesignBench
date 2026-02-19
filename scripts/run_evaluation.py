"""
Unified evaluation script for benchcore.

Assumes standardized input format:
- Input: Directory containing PDB files (real residues, not Poly-Ala)
- Metadata: scaffold_info.csv with backbone indices
- For Motif Scaffolding: Must specify motif residue ranges

Pipeline:
1. Load Inputs
2. Inverse Folding (with optional motif constraints)
3. Refolding
4. Metric Calculation
"""
import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation import get_evaluator


@hydra.main(config_path="../configs", config_name="config_evaluation", version_base=None)
def main(cfg: DictConfig):
    """
    Main evaluation pipeline.
    
    Args:
        cfg: Configuration with:
            - task_type: Type of evaluation (e.g., "motif_scaffolding")
            - input_dir: Directory containing input PDB files
            - output_dir: Directory to save results
            - metadata_file: Path to scaffold_info.csv (optional)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)
    
    # Get task type
    task_type = cfg.get("task_type", "motif_scaffolding")
    
    # Get input/output directories
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir) if cfg.get("output_dir") else Path(os.getcwd()) / "evaluation_results"
    metadata_file = Path(cfg.metadata_file) if cfg.get("metadata_file") else None
    
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get evaluator for task type
    evaluator = get_evaluator(task_type, cfg)
    
    # Run evaluation
    print(f"Running {task_type} evaluation...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    results = evaluator.run_evaluation(
        input_dir=input_dir,
        output_dir=output_dir,
        metadata_file=metadata_file
    )
    
    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"{'='*60}")
    print(f"Task: {task_type}")
    print(f"Results saved to: {results['output_dir']}")
    print(f"Metrics: {len(results['metrics'])} samples evaluated")
    
    return results


if __name__ == "__main__":
    main()
