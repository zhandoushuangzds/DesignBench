import os
import json
import pickle
import subprocess
import numpy as np
from pathlib import Path
import biotite.structure as struc
import concurrent.futures
from biotite.structure.io import pdb, pdbx  
from preprocess.ccd_parser import LocalCcdParser
import torch
import copy
import torch.multiprocessing as mp
from refold.chai1.chai1_distributed_inference import run_folding_on_context
from evaluation.metrics.ligand.mol_rec import *
def _chai1_mp_spawn_worker(
    local_rank: int,         # mp.spawn 会自动传入
    world_size: int,         # mp.spawn 会自动传入
    fasta_list: list,        # 以下是 'args' 中传递的自定义参数
    output_dir: str,
    config       # 传递你的 'self.config'
):
    """
    这是每个被 mp.spawn 启动的独立进程实际执行的函数。
    """
    
    # 1. 调用你修改后的 chai1 核心函数
    #    它会返回结果 (仅在 rank 0 上) 或 None (在其他 rank 上)
    results = run_folding_on_context(
        local_rank,
        world_size,
        fasta_file_list=fasta_list,
        output_dir=Path(output_dir), # 确保类型正确
        num_diffn_samples=config.refold.num_diffn_samples
        # ... 你可能需要从 config_obj 传递更多参数
    )
    
    # 2. 只有 Rank 0 进程会收到结果，并负责保存
    if local_rank == 0:
        if results:
            print(f"Rank 0: Received {len(results)} results. Saving...")
            with open(os.path.join(output_dir, 'chai1_cands.pkl'), 'wb') as f:
                pickle.dump(results, f)
            print("Rank 0: Results saved.")
        else:
            print("Rank 0: Error, no results were returned.")

from biotite.structure import get_residue_starts
from biotite.structure import BondType
from biotite.interface.rdkit import to_mol
from biotite.structure import BondList
RDKIT_TO_BIOTITE_BOND_TYPE = {
    Chem.BondType.UNSPECIFIED: BondType.ANY,
    Chem.BondType.SINGLE: BondType.SINGLE,
    Chem.BondType.DOUBLE: BondType.DOUBLE,
    Chem.BondType.TRIPLE: BondType.TRIPLE,
    Chem.BondType.QUADRUPLE: BondType.QUADRUPLE,
    Chem.BondType.DATIVE: BondType.COORDINATION,
    # [Yuanle] 以上为biotite定义的映射，额外添加AROMATIC的映射
    Chem.BondType.AROMATIC: BondType.AROMATIC,
}

NA_STD_RESIDUES_RES_NAME_TO_ONE = {
    "A": "A",
    "G": "G",
    "C": "C",
    "U": "U",
    "DA": "A",
    "DG": "G",
    "DC": "C",
    "DT": "T",    
}
def get_nucleic_acid_sequence(chain_struct):
        """get seq from structure"""
        res_starts = get_residue_starts(chain_struct, add_exclusive_stop=True)
        sequence = ""

        for res_start, res_end in zip(res_starts[:-1], res_starts[1:]):
            res = chain_struct[res_start:res_end]
            try:
                sequence += NA_STD_RESIDUES_RES_NAME_TO_ONE[res[0].res_name]
            except:
                continue
        
        return sequence if sequence else None

class ReFold:

    def __init__(self, config):

        self.config = config

    def run_alphafold3(self, input_json: str, output_dir: str):

        cmd = ["bash", f"{self.config.refold.af3_exec}", f"{self.config.refold.exp_name}", f"{input_json}", f"{output_dir}", f"{self.config.gpus}", f"{self.config.refold.run_data_pipeline}", f"{self.config.refold.cache_dir}"]
        subprocess.run(cmd, check=True)
    
    def run_chai1(self, fasta_list: list, output_dir: str):
        
        # 1. 确定要使用的 GPU 数量
        world_size = torch.cuda.device_count()
        if world_size == 0:
            print("Error: No GPUs found for distributed folding.")
            return
            
        print(f"Found {world_size} GPUs. Spawning processes...")

        # 2. (关键) 设置主进程的环境变量
        #    这必须在 mp.spawn 之前完成，以便所有子进程都能继承它们
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355" # 确保这是一个空闲端口
        
        # 3. 准备要传递给 worker 函数的参数
        #    (local_rank 和 world_size 会被自动添加在最前面)
        args_to_pass = (
            world_size,   # <--- 必须将 world_size 作为第一个参数
            fasta_list, 
            output_dir, 
            self.config
        )
        
        # 4. 启动！
        mp.spawn(
            _chai1_mp_spawn_worker, # 你在上面定义的 worker 函数
            args=args_to_pass,        # 要传递的参数
            nprocs=world_size,        # 启动的进程数
            join=True                 # 阻塞主进程，直到所有子进程完成
        )
        
        print(f"Chai1 distributed folding complete. Results saved in {output_dir}")
        # 注意：原始的 pickle.dump(...) 已被移入 worker 函数中

    def run_esmfold(self, sequences_file_json: str, output_dir: str):

        esmfold_script = os.path.join(os.path.dirname(__file__), "esmfold", "run_esmfold.sh")
        # breakpoint()
        cmd = [
            esmfold_script,
            # "-n", "batch_esmfold",
            "-s", sequences_file_json,
            "-o", output_dir,
            "-g", str(len(self.config.gpus.split(","))),
            "-p", str(self.config.refold.master_port),
            "-m", self.config.refold.esmfold_model_dir
        ]

        # breakpoint()
        
        print(f"Running ESMFold command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"ESMFold execution completed successfully")
            if result.stdout:
                print("STDOUT:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"ESMFold execution failed: {e}")
            print(f"Return code: {e.returncode}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            raise e

    # @staticmethod
    # def get_smiles_from_ligand(chain_atom_array):
    #     """
    #     Extract SMILES from ligand structure using RDKit.
    #     Attempts to generate SMILES from 3D coordinates.
        
    #     Args:
    #         chain_atom_array: Biotite AtomArray containing ligand atoms
            
    #     Returns:
    #         str: SMILES string representation of the ligand
    #     """
    #     try:
    #         from rdkit import Chem
    #         from rdkit.Chem import AllChem
            
    #         # Create RDKit molecule from atom array
    #         mol = Chem.RWMol()
    #         atom_indices = {}
            
    #         # Add atoms
    #         for i, atom in enumerate(chain_atom_array):
    #             element = atom.element
    #             rdkit_atom = Chem.Atom(element)
    #             idx = mol.AddAtom(rdkit_atom)
    #             atom_indices[i] = idx
            
    #         # Try to infer bonds from distances
    #         coords = chain_atom_array.coord
    #         for i in range(len(chain_atom_array)):
    #             for j in range(i + 1, len(chain_atom_array)):
    #                 dist = np.linalg.norm(coords[i] - coords[j])
    #                 # Simple distance-based bonding (can be improved)
    #                 if dist < 1.8:  # Typical bond length threshold
    #                     mol.AddBond(atom_indices[i], atom_indices[j], Chem.BondType.SINGLE)
            
    #         # Convert to standard molecule
    #         mol = mol.GetMol()
    #         Chem.SanitizeMol(mol)
            
    #         # Generate SMILES
    #         smiles = Chem.MolToSmiles(mol)
    #         return smiles
            
    #     except Exception as e:
    #         print(f"Warning: Failed to generate SMILES from structure: {e}")
    #         # Return a placeholder or raise error
    #         return None
    
    @staticmethod
    def get_smiles_from_ligand(gen_ligand):
 
        gen_lig_atom_index = list(range(len(gen_ligand)))
        gen_lig_positions = copy.deepcopy(gen_ligand.coord).astype(float)
        gen_lig_atom_types = [map_atom_symbol_to_atomic_number(atom_type) for atom_type in gen_ligand.element]
        rd_mol = reconstruct_mol(
            gen_lig_positions,
            gen_lig_atom_types,
            basic_mode=True,
        )
        rec_lig_bond = np.array([
            (
                gen_lig_atom_index[bond.GetBeginAtomIdx()],
                gen_lig_atom_index[bond.GetEndAtomIdx()],
                RDKIT_TO_BIOTITE_BOND_TYPE[bond.GetBondType()]
            )
            for bond in rd_mol.GetBonds()
        ])

        gen_ligand.bonds = BondList(
            gen_ligand.array_length(),
            rec_lig_bond
        )
        mol = to_mol(gen_ligand)
        smiles = ReFold.get_roundtrip_smiles(mol)
        return smiles
    
    @staticmethod
    def get_roundtrip_smiles(mol):
        try:
            smiles = Chem.MolToSmiles(mol, allBondsExplicit=True, canonical=False)
            test_mol = Chem.MolFromSmiles(smiles)
            if test_mol is not None:
                return smiles
        except:
            return None
    
    @staticmethod
    def make_af3_json_from_backbone(backbone_path: Path, run_data_pipeline: bool, unpaired_msa_cache: dict|None = None, paired_msa_cache: dict|None = None, template_cache: dict|None = None):
        single_input = {
            "name": backbone_path.stem,
            "sequences": [],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1
        }
        if backbone_path.suffix == '.cif':
            cif_file = pdbx.CIFFile.read(backbone_path)
            atom_array = pdbx.get_structure(cif_file, model=1)
        else:
            pdb_file = pdb.PDBFile.read(backbone_path)
            atom_array = pdb.get_structure(pdb_file, model=1)
        chain_ids = np.unique(atom_array.chain_id)
        for chain_id in chain_ids:
            chain_atom_array = atom_array[atom_array.chain_id == chain_id]
            if chain_atom_array.hetero.all():
                ligand_data = None
                # Get ccdCode first
                ccdCode = chain_atom_array.res_name[0].upper()
    
                # designed ligand
                if ccdCode == '-L':
                    smiles = ReFold.get_smiles_from_ligand(chain_atom_array)
                    if smiles:
                        ligand_data = {
                            "ligand": {
                                "smiles": smiles,
                                "id": [str(chain_id)]
                            }
                        }
                    else:
                        print(f"Warning: Failed to generate SMILES for designed ligand in chain {chain_id}")
                        continue
                else:
                    ligand_data = {
                        "ligand": {
                            "ccdCodes": ccdCode,
                            "id": [str(chain_id)]
                        }
                    }
                single_input["sequences"].append(ligand_data)

            elif np.isin(chain_atom_array.res_name, ['DA', 'DC', 'DG', 'DT','DN']).all():
                # sequence = str(struc.to_sequence(chain_atom_array)[0][0])
                sequence = get_nucleic_acid_sequence(sequence)
                single_input["sequences"].append({
                    "dna": {
                        "sequence": sequence,
                        "id": [str(chain_id)]
                    }
                })
            elif np.isin(chain_atom_array.res_name, ['A', 'C', 'G', 'U','N']).all():
                # sequence = str(struc.to_sequence(chain_atom_array)[0][0])
                sequence = get_nucleic_acid_sequence(chain_atom_array)
                s = {
                    "rna": {
                        "sequence": sequence,
                        "id": [str(chain_id)]
                    }
                }
                if not run_data_pipeline:
                    s["rna"]["unpairedMsa"] = ""
                single_input["sequences"].append(s)
            else:
                sequence = str(struc.to_sequence(chain_atom_array, allow_hetero=True)[0][0])
                s = {
                    "protein": {
                        "sequence": sequence,
                        "id": [str(chain_id)]
                    }
                }
                if not run_data_pipeline:
                    if unpaired_msa_cache and sequence in unpaired_msa_cache:
                        s["protein"]["unpairedMsaPath"] = unpaired_msa_cache[sequence]
                    else:
                        s["protein"]["unpairedMsa"] = ""
                    if paired_msa_cache and sequence in paired_msa_cache:
                        s["protein"]["pairedMsaPath"] = paired_msa_cache[sequence]
                    else:
                        s["protein"]["pairedMsa"] = ""
                    if template_cache and sequence in template_cache:
                        s["protein"]["templatesPath"] = template_cache[sequence]
                    else:
                        s["protein"]["templates"] = []    
                single_input["sequences"].append(s)
        return single_input
    
    @staticmethod
    def make_esmfold_json_from_backbone(backbone_path: Path,):
        single_input = {
            "name": backbone_path.stem,
            "sequence": "",
        }
        if backbone_path.suffix == '.cif':
            cif_file = pdbx.CIFFile.read(backbone_path)
            atom_array = pdbx.get_structure(cif_file, model=1)
        else:
            pdb_file = pdb.PDBFile.read(backbone_path)
            atom_array = pdb.get_structure(pdb_file, model=1)
        chain_ids = np.unique(atom_array.chain_id)
        # select the first chain only
        chain_id = chain_ids[0]
        chain_atom_array = atom_array[atom_array.chain_id == chain_id]  
        sequence = str(struc.to_sequence(chain_atom_array)[0][0])
        single_input["sequence"] = sequence
        return single_input

    def make_esmfold_json_multi_process(self, backbone_dir: str, output_dir: str):
        
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        backbone_path_list = list(Path(backbone_dir).glob("*.pdb"))
        if len(backbone_path_list) == 0:
            print(f"Warning: No backbone PDB files found in {backbone_dir}, try cif format.")
            backbone_path_list = list(Path(backbone_dir).glob("*.cif"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.refold.num_workers) as executor:
            futures = []
            for backbone_path in backbone_path_list:
                future = executor.submit(self.make_esmfold_json_from_backbone, backbone_path)
                futures.append(future)
            af3_input_list = [future.result() for future in concurrent.futures.as_completed(futures)]
        with open(output_dir, 'w') as f:
            json.dump(af3_input_list, f, indent=4)
    
    def make_af3_json_multi_process(self, backbone_dir: str, output_path: str):
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        backbone_path_list = list(Path(backbone_dir).glob("*.pdb"))
        if len(backbone_path_list) == 0:
            print(f"Warning: No backbone PDB files found in {backbone_dir}, try cif format.")
            backbone_path_list = list(Path(backbone_dir).glob("*.cif"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.refold.num_workers) as executor:
            futures = []
            for backbone_path in backbone_path_list:
                future = executor.submit(
                    self.make_af3_json_from_backbone, backbone_path, self.config.refold.run_data_pipeline,
                    unpaired_msa_cache=json.load(open(self.config.refold.unpaired_msa_cache)) if self.config.refold.unpaired_msa_cache else None,
                    paired_msa_cache=json.load(open(self.config.refold.paired_msa_cache)) if self.config.refold.paired_msa_cache else None,
                    template_cache=json.load(open(self.config.refold.template_cache)) if self.config.refold.template_cache else None
                )
                futures.append(future)
            af3_input_list = [future.result() for future in concurrent.futures.as_completed(futures)]
        with open(output_path, 'w') as f:
            json.dump(af3_input_list, f, indent=4)

    @staticmethod
    def make_chai1_fasta_from_backbone(backbone_path: Path, ccd_parser: LocalCcdParser, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Implement the logic to create a FASTA file from the backbone structure
        if backbone_path.suffix == '.cif':
            cif_file = pdbx.CIFFile.read(backbone_path)
            atom_array = pdbx.get_structure(cif_file, model=1)
        else:
            pdb_file = pdb.PDBFile.read(backbone_path)
            atom_array = pdb.get_structure(pdb_file, model=1)
        chain_ids = np.unique(atom_array.chain_id)
        fasta_strings = []
        for chain_id in chain_ids:
            chain_atom_array = atom_array[atom_array.chain_id == chain_id]
            if chain_atom_array.hetero.any():
                ccdCode = chain_atom_array.res_name[0].upper()
                smiles = ccd_parser.get_smiles(ccdCode)[0][1:-1]  # remove quotes
                fasta_strings.append(f">ligand|name={ccdCode}\n{smiles}\n")
            elif np.isin(chain_atom_array.res_name, ['DA', 'DC', 'DG', 'DT']).all():
                sequence = str(struc.to_sequence(chain_atom_array)[0][0])
                fasta_strings.append(f">dna|name=dna\n{sequence}\n")
            elif np.isin(chain_atom_array.res_name, ['A', 'C', 'G', 'U']).all():
                sequence = str(struc.to_sequence(chain_atom_array)[0][0])
                fasta_strings.append(f">rna|name=rna\n{sequence}\n")
            else:
                sequence = str(struc.to_sequence(chain_atom_array)[0][0])
                fasta_strings.append(f">protein|name=protein\n{sequence}\n")
        with open(output_path, 'w') as f:
            f.writelines(fasta_strings)
    
    def make_chai1_fasta_multi_process(self, backbone_dir: str, output_dir: str, origin_cwd: str):
        
        backbone_path_list = list(Path(backbone_dir).glob("*.pdb"))
        ccd_path = os.path.join(origin_cwd, self.config.refold.ccd_component)
        ccd_parser = LocalCcdParser(ccd_path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.refold.num_workers) as executor:
            futures = []
            for backbone_path in backbone_path_list:
                output_path = os.path.join(output_dir, f"{backbone_path.stem}.fasta")
                future = executor.submit(self.make_chai1_fasta_from_backbone, backbone_path, ccd_parser, output_path)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                future.result()