"""
Pipeline script for motif scaffolding evaluation.
Follows benchcore's Generator + Evaluator architecture.

New Architecture Pipeline:
1. Generator.run() -> Standardize model outputs (model-specific fixes)
2. Evaluator.evaluate() -> Run evaluation (model-agnostic)

Usage:
    python scripts/run_motif_scaffolding_pipeline.py \
        design_dir=/path/to/designs \
        model_name=PPIFlow \
        motif_scaffolding.motif_list=[01_1LDB,02_1ITU] \
        inversefold=ProteinMPNN \
        refold=esmfold
"""
import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys

# Add benchcore to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.motif_scaffolding_evaluation import MotifScaffoldingEvaluation

@hydra.main(config_path="../configs", config_name="config_motif_scaffolding", version_base=None)
def main(cfg: DictConfig):
    """
    Main pipeline for motif scaffolding evaluation.
    
    New Architecture:
    1. Generator: Converts model outputs to standardized format (model-specific)
    2. Evaluator: Performs model-agnostic evaluation
    
    Pipeline:
    - Generator.run() -> Standardized PDBs and scaffold_info.csv
    - InverseFold (ProteinMPNN) -> Sequence design
    - ReFold (ESMFold) -> Structure prediction
    - MotifBench Analysis -> Metrics (RMSD, novelty, diversity with alpha=5)
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)

    # Initialize directories
    pipeline_dir = Path(os.getcwd()) / cfg.root
    design_dir = Path(cfg.design_dir)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    if not design_dir.exists():
        raise ValueError(f"Design directory not found: {design_dir}")

    # Get model name (required for Generator selection)
    model_name = cfg.get("model_name") or cfg.motif_scaffolding.get("model_name")
    if model_name is None:
        raise ValueError(
            "model_name must be specified in config. "
            "Available models: PPIFlow, RFD3"
        )

    # Initialize MotifScaffoldingEvaluation
    # This will use Generator + Evaluator internally
    motif_evaluation = MotifScaffoldingEvaluation(cfg, model_name=model_name)

    # Get motif list from config or use all
    motif_list = cfg.motif_scaffolding.get("motif_list", None)
    if motif_list is not None and isinstance(motif_list, list):
        motif_list = [str(m) for m in motif_list]

    # Run evaluation pipeline
    # This will:
    # 1. Generator.run() -> Standardize model outputs
    # 2. InverseFold (ProteinMPNN) -> Sequence design
    # 3. ReFold (ESMFold) -> Structure prediction
    # 4. Calculate motif metrics (RMSD, novelty, diversity with alpha=5)
    results = motif_evaluation.run_motif_scaffolding_evaluation(
        design_dir=design_dir,
        pipeline_dir=pipeline_dir,
        motif_list=motif_list
    )

    print(f"\n{'='*60}")
    print("Motif Scaffolding Evaluation Complete")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Pipeline: Generator -> InverseFold (ProteinMPNN) -> ReFold (ESMFold) -> MotifBench Analysis")
    print(f"Evaluated {len(results)} motifs:")
    for motif_name, result_dir in results.items():
        print(f"  {motif_name}: {result_dir}")
    print(f"\nResults saved to: {pipeline_dir}")

if __name__ == "__main__":
    main()
