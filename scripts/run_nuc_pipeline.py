import os
import json
import pandas as pd
import hydra
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from preprocess.preprocess import Preprocess
from inversefold.inversefold_api import InverseFold
from refold.refold_api import ReFold
from evaluation.evaluation_api import Evaluation

@hydra.main(config_path="../configs", config_name="config_nuc")
def main(cfg: DictConfig):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)

    pipeline_dir = os.path.join(os.getcwd(), cfg.root)
    design_dir = cfg.design_dir
    os.makedirs(pipeline_dir, exist_ok=True)

    p = Preprocess(cfg)
    inverse_fold_model = InverseFold(cfg)
    refold_model = ReFold(cfg)
    evaluation_model = Evaluation(cfg)

    p.format_output_cif(input_dir=design_dir, output_dir=os.path.join(pipeline_dir, "formatted_designs"))
    inverse_fold_model.run_odesignmpnn(
        input_root=Path(os.path.join(pipeline_dir, "formatted_designs")), 
        inverse_fold_root=Path(os.path.join(pipeline_dir, "inverse_fold"))
    )
    refold_model.make_af3_json_multi_process(
        backbone_dir=os.path.join(pipeline_dir, "inverse_fold"),
        output_path=os.path.join(pipeline_dir, "refold", "af3_input.json")
    )
    refold_model.run_alphafold3(
        input_json=os.path.join(pipeline_dir, "refold", "af3_input.json"), 
        output_dir=os.path.join(pipeline_dir, "refold", "af3_out")
    )
    evaluation_model.run_nuc_evaluation(
        pipeline_dir=pipeline_dir,
        output_csv=os.path.join(pipeline_dir, "raw_data.csv")
    )

if __name__ == "__main__":
    main()