import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(str(root_path))
from preprocess.preprocess import Preprocess
from inversefold.inversefold_api import InverseFold
from evaluation.evaluation_api import Evaluation

@hydra.main(config_path="../configs", config_name="config_protein_binding_ligand")
def main(cfg: DictConfig):

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)
    # os.environ["BABEL_LIBDIR"] = "/home/nvme01/miniforge3/envs/odesign_benchmark/lib/openbabel/3.1.0"

    pipeline_dir = os.path.join(os.getcwd(),cfg.root)
    design_dir = os.path.join(root_path, cfg.design_dir)
    os.makedirs(pipeline_dir, exist_ok=True)

    p = Preprocess(cfg)
    inverse_fold_model = InverseFold(cfg)
    m = Evaluation(cfg)

    # breakpoint()
    p.format_output_ligand(input_dir=design_dir, output_dir=os.path.join(pipeline_dir, "formatted_designs"))
    inverse_fold_model.run_odesignmpnn(input_root=Path(os.path.join(pipeline_dir, "formatted_designs")), inverse_fold_root=Path(os.path.join(pipeline_dir, "inversefold")))
    p.format_output_ligand_for_protein_binding_ligand_evaluation(input_dir=os.path.join(pipeline_dir, "inversefold"), output_dir=os.path.join(pipeline_dir, "inversefold_formatted_designs_for_evaluation"))
    m.run_protein_binding_ligand_evaluation(input_dir=os.path.join(pipeline_dir, "inversefold_formatted_designs_for_evaluation"), output_dir=os.path.join(pipeline_dir, "inversefold_formatted_designs_for_evaluation_metrics"))

if __name__ == "__main__":
    main()