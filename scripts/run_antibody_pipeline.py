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

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)

    # initialize directories
    hydra_cwd = os.getcwd()
    pipeline_dir = os.path.join(hydra_cwd, cfg.root)
    design_dir = cfg.design_dir
    os.makedirs(pipeline_dir, exist_ok=True)

    # initialize models
    p = Preprocess(cfg)
    evaluation_model = Evaluation(cfg)
    inverse_fold_model = InverseFold(cfg)
    refold_model = ReFold(cfg)

    # Check for required CDR info CSV
    cdr_info_csv = cfg.get('cdr_info_csv', None)
    if cdr_info_csv:
        cdr_info_csv = os.path.join(get_original_cwd(), cdr_info_csv) if not os.path.isabs(cdr_info_csv) else cdr_info_csv
        if not os.path.exists(cdr_info_csv):
            raise FileNotFoundError(f"CDR info CSV not found: {cdr_info_csv}")
        print(f"✓ Using CDR info CSV: {cdr_info_csv}")
        use_cdr_fix = True
    else:
        print("⚠️  Warning: No cdr_info_csv provided. Using b_factor-based fixed residues.")
        print("   For antibody design, it's recommended to provide CDR info to fix scaffold regions.")
        use_cdr_fix = False
        cdr_info_csv = None

    # run pipeline
    # p.format_output_rfantibody(
    #     input_dir=design_dir, 
    #     output_dir=os.path.join(pipeline_dir, "formatted_designs")
    # )
    inverse_fold_model.run_ligandmpnn_distributed(
        input_dir=Path(os.path.join(pipeline_dir, "formatted_designs")), 
        output_dir=os.path.join(pipeline_dir, "inverse_fold"), 
        gpu_list=str(cfg.gpus).split(','), 
        origin_cwd=get_original_cwd(),
        cdr_info_csv=cdr_info_csv,
        use_cdr_fix=use_cdr_fix
    )
    
    # make af3 input json
    refold_model.make_af3_json_multi_process(
        backbone_dir=os.path.join(pipeline_dir, "inverse_fold", "backbones"), 
        output_path=os.path.join(pipeline_dir, "refold", "af3_input.json"), 
    )
    
    # run af3
    refold_model.run_alphafold3(input_json=os.path.join(pipeline_dir, "refold", "af3_input.json"), output_dir=os.path.join(pipeline_dir, "refold", "af3_out"))
    
    # gather confidence metrics
    evaluation_model.run_protein_binding_protein_evaluation(
        pipeline_dir=pipeline_dir, 
        output_csv=os.path.join(pipeline_dir, "raw_data.csv")
    )
    
    # run developability evaluation
    # Note: cdr_info_csv should contain CDR indices for each antibody
    # Required columns: id, heavy_fv, light_fv (optional), h_cdr1_start, h_cdr1_end,
    #                   h_cdr2_start, h_cdr2_end, h_cdr3_start, h_cdr3_end,
    #                   l_cdr1_start, l_cdr1_end, l_cdr2_start, l_cdr2_end,
    #                   l_cdr3_start, l_cdr3_end
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
        print(f"\n⚠️  Skipping Developability Evaluation: cdr_info_csv not specified in config")
        print(f"   To enable, add 'cdr_info_csv: path/to/antibodies_fv.csv' to your config")

if __name__ == "__main__":
    main()