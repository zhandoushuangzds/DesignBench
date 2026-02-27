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


def _gpus_to_list(cfg):
    """Normalize gpus config to a list of GPU id strings (supports list or string '5,6,7' or \"['5','6','7']\"). Strips [ ] and quotes from each element."""
    def _clean(s):
        return str(s).strip().strip("[]'\"")
    g = cfg.gpus
    if isinstance(g, (list, tuple)):
        return [_clean(x) for x in g]
    s = str(g).strip()
    if s.startswith("[") and s.endswith("]"):
        import re
        s = re.sub(r"[\s\[\]]", "", s)
    return [_clean(x) for x in s.split(",") if x.strip()]


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main pipeline function with strict input auditing.
    """
    gpu_list = _gpus_to_list(cfg)
    cfg.gpus = gpu_list  # overwrite so refold/inversefold get clean list
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)

    # Initialize directories (resolve relative paths from repo root: Hydra may change cwd)
    hydra_cwd = os.getcwd()
    pipeline_dir = os.path.join(hydra_cwd, cfg.root)
    design_dir = cfg.design_dir
    if not os.path.isabs(design_dir):
        design_dir = os.path.join(get_original_cwd(), design_dir)
    design_dir = os.path.normpath(design_dir)
    os.makedirs(pipeline_dir, exist_ok=True)

    # Determine antibody type (scFv/Fab or VHH)
    antibody_type = cfg.get('antibody_type', 'antibody').lower()  # 'antibody' or 'nanobody'
    
    # Get target config path
    target_config_path = cfg.get('target_config_path', None)
    if target_config_path:
        target_config_path = os.path.join(get_original_cwd(), target_config_path) if not os.path.isabs(target_config_path) else target_config_path
    
    # Get scaffolds directory path
    scaffolds_dir = cfg.get('scaffolds_dir', None)
    if scaffolds_dir:
        scaffolds_dir = os.path.join(get_original_cwd(), scaffolds_dir) if not os.path.isabs(scaffolds_dir) else scaffolds_dir
    else:
        # Default scaffolds directory
        default_scaffolds_dir = os.path.join(get_original_cwd(), "assets", "antibody_nanobody", "scaffolds")
        scaffolds_dir = default_scaffolds_dir if os.path.exists(default_scaffolds_dir) else None
    
    if antibody_type == 'nanobody':
        design_module = NanobodyDesignModule(
            cfg, 
            target_config_path=target_config_path,
            scaffolds_dir=scaffolds_dir
        )
        print("=" * 80)
        print("NANOBODY (VHH) DESIGN MODULE")
        print("=" * 80)
    else:
        design_module = AntibodyDesignModule(
            cfg, 
            target_config_path=target_config_path,
            scaffolds_dir=scaffolds_dir
        )
        print("=" * 80)
        print("ANTIBODY (scFv/Fab) DESIGN MODULE")
        print("=" * 80)

    # Optional: start from step N (1-5). Used to decide if cdr_info_csv is required.
    start_step = int(cfg.get('start_step', 1))

    # Check for required CDR info CSV (required for steps 1-4 and for developability in step 5)
    cdr_info_csv = cfg.get('cdr_info_csv', None)
    if cdr_info_csv:
        cdr_info_csv = os.path.join(get_original_cwd(), cdr_info_csv) if not os.path.isabs(cdr_info_csv) else cdr_info_csv
        if not os.path.exists(cdr_info_csv):
            raise FileNotFoundError(f"CDR info CSV not found: {cdr_info_csv}")
        print(f"✓ Using CDR info CSV: {cdr_info_csv}")
    elif start_step <= 4:
        raise ValueError(
            "cdr_info_csv is REQUIRED for antibody design benchmark (steps 1-4). "
            "Please provide path to CDR info CSV in config, or use start_step=5 with cdr_info_csv to run only evaluation+developability."
        )
    else:
        print("✓ No cdr_info_csv: only PBP evaluation (raw_data.csv) will run; developability will be skipped.")

    # Maximum designs per target
    max_designs_per_target = cfg.get('max_designs_per_target', 100)
    # Optional: limit to first N targets (for quick validation, e.g. max_targets=1)
    max_targets = cfg.get('max_targets', None)
    print(f"✓ Maximum designs per target: {max_designs_per_target}")
    if max_targets is not None:
        print(f"✓ Quick test: only first {max_targets} target(s)")
    if start_step > 1:
        print(f"✓ Starting from step {start_step} (skipping steps 1–{start_step - 1})")

    evaluation_model = Evaluation(cfg)

    if start_step <= 4:
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
        if max_targets is not None:
            n = int(max_targets)
            keys = sorted(target_files.keys())[:n]
            target_files = {k: target_files[k] for k in keys}
            print(f"✓ Limited to first {len(target_files)} target(s) (max_targets={max_targets})")
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

        # Initialize models (evaluation_model already created above)
        p = Preprocess(cfg)
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
            gpu_list=_gpus_to_list(cfg), 
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
        
        # Make AF3 input JSON (sequence-only; AF3 refold is unconstrained, no CDR-only fixing)
        refold_model.make_af3_json_multi_process(
            backbone_dir=os.path.join(pipeline_dir, "inverse_fold", "backbones"),
            output_path=os.path.join(pipeline_dir, "refold", "af3_input.json"),
            use_backbone_as_template=False,
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
    
    # PBP evaluation (confidence, RMSD, hydrophobicity, noncovalents)
    evaluation_model.run_protein_binding_protein_evaluation(
        pipeline_dir=pipeline_dir, 
        output_csv=os.path.join(pipeline_dir, "raw_data.csv")
    )
    
    # Antigen-antibody interface analysis (paratope/epitope, BSA, H-bonds, etc.)
    if cfg.get('enable_interface_analysis', True):
        print("\n🔬 Running Interface Analysis...")
        evaluation_model.run_antibody_interface_analysis(
            pipeline_dir=pipeline_dir,
            output_csv=os.path.join(pipeline_dir, "interface_metrics.csv"),
            ab_chain_ids=cfg.get('ab_chain_ids', ['B', 'C']),
            ag_chain_ids=cfg.get('ag_chain_ids', ['A']),
        )
    
    # Developability evaluation (requires cdr_info_csv)
    if cdr_info_csv:
        print(f"\n🔬 Running Developability Evaluation...")
        print(f"   CDR info CSV: {cdr_info_csv}")
        evaluation_model.run_antibody_developability_evaluation(
            pipeline_dir=pipeline_dir,
            cdr_info_csv=cdr_info_csv,
            output_csv=os.path.join(pipeline_dir, "developability_metrics.csv"),
            num_seeds=cfg.get('developability_num_seeds', 4)
        )
    else:
        print("\n⏭ Skipping developability (no cdr_info_csv provided).")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Results saved to: {pipeline_dir}")


if __name__ == "__main__":
    main()
