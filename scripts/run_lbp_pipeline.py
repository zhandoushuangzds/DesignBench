import os
import json
import hydra
from hydra.utils import get_original_cwd
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

@hydra.main(config_path="../configs", config_name="config_ligand_binding_protein")
def main(cfg: DictConfig):

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)

    # initialize directories
    pipeline_dir = os.path.join(os.getcwd(),cfg.root)
    design_dir = cfg.design_dir
    os.makedirs(pipeline_dir, exist_ok=True)

    # initialize models
    p = Preprocess(cfg)
    inverse_fold_model = InverseFold(cfg)
    refold_model = ReFold(cfg)
    evaluation_model = Evaluation(cfg)

    # run pipeline
    p.format_output_boltzgen(
        input_dir=design_dir, 
        output_dir=os.path.join(pipeline_dir, "formatted_designs")
    )
    inverse_fold_model.run_ligandmpnn_distributed(
        input_dir=Path(os.path.join(pipeline_dir, "formatted_designs")), 
        output_dir=os.path.join(pipeline_dir, "inverse_fold"),
        gpu_list=str(cfg.gpus).split(','),
        origin_cwd=get_original_cwd()
    )
    # make chai1 input fasta
    refold_model.make_chai1_fasta_multi_process(
        backbone_dir=os.path.join(pipeline_dir, "inverse_fold", "backbones"), 
        output_dir=os.path.join(pipeline_dir, "refold", "chai1_inputs"),
        origin_cwd=get_original_cwd()
    )
    # run chai1
    refold_model.run_chai1(
        fasta_list=list(Path(os.path.join(pipeline_dir, "refold", "chai1_inputs")).glob("*.fasta")), 
        output_dir=os.path.join(pipeline_dir, "refold", "chai1_out")
    )
    # gather confidence metrics
    evaluation_model.run_ligand_binding_protein_evaluation(
        pipeline_dir=pipeline_dir, 
        cands=os.path.join(pipeline_dir, "refold", "chai1_out", "chai1_cands.pkl"),
        output_csv=os.path.join(pipeline_dir, "raw_data.csv")
    )

if __name__ == "__main__":
    main()