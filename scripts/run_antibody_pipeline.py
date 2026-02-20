"""
Antibody Design Benchmark Pipeline

Refactored pipeline with strict input auditing, naming validation, and scaffold compliance checks.
Supports both scFv/Fab (AntibodyDesignModule) and VHH (NanobodyDesignModule).
"""

import os
import json
import hydra
from hydra.utils import to_absolute_path, get_original_cwd
import pandas as pd
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from preprocess.preprocess import Preprocess
from inversefold.inversefold_api import InverseFold
from refold.refold_api import ReFold
from evaluation.evaluation_api import Evaluation
from evaluation.antibody import AntibodyDesignModule, NanobodyDesignModule


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main pipeline function with strict input auditing.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)

    # Initialize directories
    hydra_cwd = os.getcwd()
    pipeline_dir = os.path.join(hydra_cwd, cfg.root)
    design_dir = cfg.design_dir
    os.makedirs(pipeline_dir, exist_ok=True)

    # Determine antibody type (scFv/Fab or VHH)
    antibody_type = cfg.get('antibody_type', 'antibody').lower()  # 'antibody' or 'nanobody'
    
    if antibody_type == 'nanobody':
        design_module = NanobodyDesignModule(cfg)
        print("=" * 80)
        print("NANOBODY (VHH) DESIGN MODULE")
        print("=" * 80)
    else:
        design_module = AntibodyDesignModule(cfg)
        print("=" * 80)
        print("ANTIBODY (scFv/Fab) DESIGN MODULE")
        print("=" * 80)

    # Check for required CDR info CSV
    cdr_info_csv = cfg.get('cdr_info_csv', None)
    if cdr_info_csv:
        cdr_info_csv = os.path.join(get_original_cwd(), cdr_info_csv) if not os.path.isabs(cdr_info_csv) else cdr_info_csv
        if not os.path.exists(cdr_info_csv):
            raise FileNotFoundError(f"CDR info CSV not found: {cdr_info_csv}")
        print(f"✓ Using CDR info CSV: {cdr_info_csv}")
    else:
        raise ValueError(
            "cdr_info_csv is REQUIRED for antibody design benchmark. "
            "Please provide path to CDR info CSV in config."
        )

    # Maximum designs per target
    max_designs_per_target = cfg.get('max_designs_per_target', 100)
    print(f"✓ Maximum designs per target: {max_designs_per_target}")

    # ============================================================================
    # STEP 1: INPUT AUDIT & COMPLIANCE CHECK
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 1: INPUT AUDIT & COMPLIANCE CHECK")
    print("=" * 80)
    
    try:
        audit_results = design_module.audit_input_directory(
            input_dir=design_dir,
            cdr_info_csv=cdr_info_csv,
            max_designs_per_target=max_designs_per_target
        )
    except Exception as e:
        print(f"\n❌ INPUT AUDIT FAILED:")
        print(f"   {str(e)}")
        print("\nPlease fix the input errors before proceeding.")
        raise

    # Generate and print compliance report
    compliance_report = design_module.generate_compliance_report(audit_results)
    print("\n" + compliance_report)
    
    # Check if we should proceed despite warnings
    if audit_results['warnings']:
        proceed = cfg.get('proceed_with_warnings', False)
        if not proceed:
            print("\n⚠️  Warnings detected. Set 'proceed_with_warnings: true' in config to continue.")
            raise ValueError("Input validation warnings detected. Fix issues or set proceed_with_warnings=true")
        else:
            print("\n⚠️  Proceeding with warnings (proceed_with_warnings=true)")

    # Prepare formatted designs directory
    formatted_designs_dir = os.path.join(pipeline_dir, "formatted_designs")
    os.makedirs(formatted_designs_dir, exist_ok=True)
    
    # Copy validated files to formatted_designs (or create symlinks)
    print("\n" + "=" * 80)
    print("STEP 2: PREPARING FORMATTED DESIGNS")
    print("=" * 80)
    
    target_files = audit_results['target_files']
    total_files = sum(len(files) for files in target_files.values())
    print(f"✓ Processing {total_files} validated design files across {len(target_files)} targets")
    
    # Copy files to formatted_designs directory
    import shutil
    for target_name, files in target_files.items():
        for pdb_path, index, scaffold in files:
            # Create target subdirectory
            target_subdir = os.path.join(formatted_designs_dir, target_name)
            os.makedirs(target_subdir, exist_ok=True)
            
            # Copy file
            dest_path = os.path.join(target_subdir, pdb_path.name)
            shutil.copy2(pdb_path, dest_path)
    
    print(f"✓ Copied {total_files} files to {formatted_designs_dir}")

    # Initialize models
    p = Preprocess(cfg)
    evaluation_model = Evaluation(cfg)
    inverse_fold_model = InverseFold(cfg)
    refold_model = ReFold(cfg)

    # ============================================================================
    # STEP 3: INVERSE FOLDING (LigandMPNN)
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: INVERSE FOLDING (LigandMPNN)")
    print("=" * 80)
    
    # Use module-specific fixed residues calculator
    fixed_residues_calculator = design_module.calculate_fixed_residues
    
    inverse_fold_model.run_ligandmpnn_distributed(
        input_dir=Path(formatted_designs_dir), 
        output_dir=os.path.join(pipeline_dir, "inverse_fold"), 
        gpu_list=str(cfg.gpus).split(','), 
        origin_cwd=get_original_cwd(),
        cdr_info_csv=cdr_info_csv,
        use_cdr_fix=True,
        fixed_residues_calculator=fixed_residues_calculator,
        cdr_df=audit_results['cdr_df']
    )
    
    # ============================================================================
    # STEP 4: REFOLDING (AlphaFold3)
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: REFOLDING (AlphaFold3)")
    print("=" * 80)
    
    # Make AF3 input JSON
    refold_model.make_af3_json_multi_process(
        backbone_dir=os.path.join(pipeline_dir, "inverse_fold", "backbones"), 
        output_path=os.path.join(pipeline_dir, "refold", "af3_input.json"), 
    )
    
    # Run AF3
    refold_model.run_alphafold3(
        input_json=os.path.join(pipeline_dir, "refold", "af3_input.json"), 
        output_dir=os.path.join(pipeline_dir, "refold", "af3_out")
    )
    
    # ============================================================================
    # STEP 5: EVALUATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 5: EVALUATION")
    print("=" * 80)
    
    # Gather confidence metrics
    evaluation_model.run_protein_binding_protein_evaluation(
        pipeline_dir=pipeline_dir, 
        output_csv=os.path.join(pipeline_dir, "raw_data.csv")
    )
    
    # Run developability evaluation
    print(f"\n🔬 Running Developability Evaluation...")
    print(f"   CDR info CSV: {cdr_info_csv}")
    evaluation_model.run_antibody_developability_evaluation(
        pipeline_dir=pipeline_dir,
        cdr_info_csv=cdr_info_csv,
        output_csv=os.path.join(pipeline_dir, "developability_metrics.csv"),
        num_seeds=cfg.get('developability_num_seeds', 4)
    )
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Results saved to: {pipeline_dir}")


if __name__ == "__main__":
    main()
