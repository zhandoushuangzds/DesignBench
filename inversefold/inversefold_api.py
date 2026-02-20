import os
import json
import pickle
import subprocess
import numpy as np
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Tuple, Callable
from biotite.structure.io import pdb, pdbx
import biotite.structure as struc
import biotite.structure.io as io
from .InverseFold.utils.tools import parse_cif_to_samples, reload_model, inference, save_sequence_replaced_cif, _AA1_TO_AA3
from .InverseFold.utils.inference_utils import extract_ligand_samples_from_cif
from .gRNAde.gRNAde import gRNAde

def _setup_distributed(local_rank, world_size):
    """Sets up the distributed environment for a worker process."""
    # Set master address and port (torchrun does this automatically)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # Make sure this port is available

    dist.init_process_group(
        backend="nccl",      # NCCL for NVIDIA GPUs
        init_method="env://",
        rank=local_rank,
        world_size=world_size
    )
    torch.cuda.set_device(local_rank)

def _cleanup_distributed():
    """Cleans up the process group."""
    dist.destroy_process_group()

def _odesign_mpnn_worker(
    local_rank: int,
    world_size: int,
    config: Any,  # Your self.config object
    inverse_fold_root: str,
    samples_all: List[Dict[str, Any]]
):
    """
    This is the function that each individual GPU process will run.
    """
    _setup_distributed(local_rank, world_size)
    device = f"cuda:{local_rank}"
    
    if local_rank == 0:
        print("Workers started. Loading models on each GPU...")

    # 1. Load the model (each process loads its own copy)
    model, dev = reload_model(
        data_name=config.inversefold.data_name, 
        model_name="ODesign", 
        device=device,
        model_dir=config.inversefold.model_dir
    )

    # 2. Get this process's unique data portion
    # We split the list based on rank: [rank::world_size]
    samples_for_this_rank = samples_all[local_rank::world_size]
    
    if local_rank == 0:
        print(f"Rank 0 processing {len(samples_for_this_rank)} samples (total {len(samples_all)}).")
    
    # 3. Run the inference loop (from your original function)
    for sample_dict in samples_for_this_rank:
        try:
            pred_seqs, scores, true_seq_masked, probs, metrics = inference(
                model=model, sample_input=sample_dict['smp'], model_name="ODesign",
                data_name=config.inversefold.data_name, topk=config.inversefold.topk,
                temp=config.inversefold.temp, use_beam=config.inversefold.use_beam,
                device=dev
            )

            # 4. Save results (from your original function)
            if config.inversefold.save_all:
                for i, seq in enumerate(pred_seqs):
                    # Use the path we saved in sample_dict
                    cif_src_path = sample_dict['cif_src_path'] 
                    # #qtfeng: Use original CIF filename (without extension) instead of smp['title'] to avoid filename conflicts
                    cif_basename = os.path.splitext(os.path.basename(cif_src_path))[0]
                    out_path = os.path.join(inverse_fold_root, f"{cif_basename}-{i+1}.cif")
                    
                    src_file = pdbx.CIFFile.read(cif_src_path)
                    src_atom_array = pdbx.get_structure(src_file, model=1)
                    
                    if sample_dict['composition'][sample_dict["cond_c_id"]][0] == 'protein':
                        seq_1_to_3 = [_AA1_TO_AA3.get(rt, 'UNK') for rt in list(seq)]
                    elif sample_dict['composition'][sample_dict["cond_c_id"]][0] == 'rna':
                        seq_1_to_3 = list(seq)
                    elif sample_dict['composition'][sample_dict["cond_c_id"]][0] == 'dna':
                        seq_1_to_3 = [f"D{rt}" for rt in list(seq)]
                    else:
                        seq_1_to_3 = seq.split(' ')

                    save_sequence_replaced_array = src_atom_array.copy()
                    r_starts = struc.get_residue_starts(save_sequence_replaced_array, add_exclusive_stop=True)
                    redesign_id = 0
                    for r_s, r_e in zip(r_starts[:-1], r_starts[1:]):
                        if save_sequence_replaced_array.chain_id[r_s] == sample_dict["cond_c_id"] and config.inversefold.data_name != "ligand":
                            save_sequence_replaced_array.res_name[r_s:r_e] = seq_1_to_3[redesign_id]
                            redesign_id += 1
                        elif save_sequence_replaced_array.chain_id[r_s] == sample_dict["cond_c_id"] and config.inversefold.data_name == "ligand":
                            save_sequence_replaced_array.element[r_s: r_e] = seq_1_to_3
                            save_sequence_replaced_array.atom_name[r_s: r_e] = np.char.add(seq_1_to_3, np.array(range(r_e - r_s), dtype=np.str_))
                        else: continue

                    io.save_structure(out_path, save_sequence_replaced_array)

            # This update now happens locally in the process and will not be
            # visible in the main process. This is OK since the function anyway
            # did not return the list. The main result is the saved files.
            sample_dict["composition"][sample_dict["cond_c_id"]] = (sample_dict["composition"][sample_dict["cond_c_id"]][0], pred_seqs)
        
        except Exception as e:
            print(f"[Rank {local_rank}] Failed to process {sample_dict['smp']['title']}: {e}")

    _cleanup_distributed()
    print(f"Rank {local_rank} is done.")

class InverseFold:

    def __init__(self, config):

        self.config = config

    def run_ligandmpnn_distributed(
        self, 
        input_dir: Path, 
        output_dir: str, 
        gpu_list: list, 
        origin_cwd: str, 
        cdr_info_csv: Optional[str] = None, 
        use_cdr_fix: bool = False,
        fixed_residues_calculator: Optional[Callable[[Path, Any], List[str]]] = None,
        cdr_df: Optional[Any] = None
    ):
        """
        Run LigandMPNN in distributed mode.
        
        Args:
            input_dir: Directory containing PDB files
            output_dir: Output directory for results
            gpu_list: List of GPU IDs
            origin_cwd: Original working directory
            cdr_info_csv: Path to CDR info CSV (required if use_cdr_fix=True and cdr_df=None)
            use_cdr_fix: If True, use CDR information to fix scaffold residues (for antibodies)
            fixed_residues_calculator: Optional function to calculate fixed residues (pdb_path, cdr_row) -> List[str]
            cdr_df: Optional pre-loaded CDR DataFrame (if None, will load from cdr_info_csv)
        """
        os.makedirs(output_dir, exist_ok=True)
        my_env = os.environ.copy()
        ligandmpnn_multi_input = {}
        
        # Load CDR info if provided
        if use_cdr_fix and cdr_df is None:
            if cdr_info_csv is None:
                raise ValueError("cdr_info_csv is required when use_cdr_fix=True and cdr_df is not provided")
            from .cdr_utils import load_cdr_info_csv, match_pdb_to_cdr_info
            cdr_df = load_cdr_info_csv(cdr_info_csv)
            print(f"Loaded CDR info for {len(cdr_df)} antibodies")
        
        # Use provided calculator or default
        if fixed_residues_calculator is None and use_cdr_fix:
            from .cdr_utils import calculate_fixed_residues_for_antibody
            fixed_residues_calculator = calculate_fixed_residues_for_antibody
        
        for pdb_path in input_dir.rglob("*.pdb"):
            if use_cdr_fix and cdr_df is not None and fixed_residues_calculator is not None:
                # Use CDR information to calculate fixed residues
                from .cdr_utils import match_pdb_to_cdr_info
                cdr_row = match_pdb_to_cdr_info(pdb_path, cdr_df)
                if cdr_row is not None:
                    try:
                        fixed_residues = fixed_residues_calculator(pdb_path, cdr_row)
                        fixed_residues_str = " ".join(fixed_residues)
                        ligandmpnn_multi_input[str(pdb_path.parent / f"{pdb_path.stem.lower()}.pdb")] = fixed_residues_str
                        print(f"Fixed {len(fixed_residues)} scaffold residues for {pdb_path.name} (CDR-based)")
                    except Exception as e:
                        print(f"Warning: Failed to calculate CDR-based fixed residues for {pdb_path.name}: {e}")
                        print(f"  Falling back to b_factor-based method")
                        # Fall back to b_factor method
                        atom_array = pdb.PDBFile.read(pdb_path).get_structure(model=1, extra_fields=['b_factor'])
                        fixed_atom_array = atom_array[atom_array.b_factor != 0.0]
                        fixed_residues = np.unique(np.char.add(fixed_atom_array.chain_id, np.array(fixed_atom_array.res_id, dtype=str)))
                        ligandmpnn_multi_input[str(pdb_path.parent / f"{pdb_path.stem.lower()}.pdb")] = " ".join(fixed_residues.tolist())
                else:
                    print(f"Warning: No CDR info found for {pdb_path.name}, using b_factor method")
                    # Fall back to b_factor method
                    atom_array = pdb.PDBFile.read(pdb_path).get_structure(model=1, extra_fields=['b_factor'])
                    fixed_atom_array = atom_array[atom_array.b_factor != 0.0]
                    fixed_residues = np.unique(np.char.add(fixed_atom_array.chain_id, np.array(fixed_atom_array.res_id, dtype=str)))
                    ligandmpnn_multi_input[str(pdb_path.parent / f"{pdb_path.stem.lower()}.pdb")] = " ".join(fixed_residues.tolist())
            else:
                # Original method: use b_factor
                atom_array = pdb.PDBFile.read(pdb_path).get_structure(model=1, extra_fields=['b_factor'])
                fixed_atom_array = atom_array[atom_array.b_factor != 0.0]
                fixed_residues = np.unique(np.char.add(fixed_atom_array.chain_id, np.array(fixed_atom_array.res_id, dtype=str)))
                ligandmpnn_multi_input[str(pdb_path.parent / f"{pdb_path.stem.lower()}.pdb")] = " ".join(fixed_residues.tolist())
        
        json.dump(ligandmpnn_multi_input, open(os.path.join(output_dir, "ligandmpnn_input.json"), 'w'), indent=4)

        num_gpus = len(gpu_list)
        all_items_list = list(ligandmpnn_multi_input.items())
        data_batches = np.array_split(all_items_list, num_gpus)
        print(f"Split {len(all_items_list)} structures (from dict) into {num_gpus} batches for parallel processing.")

        def _run_batch_on_gpu(gpu_id: str, batch_item_list: list, batch_index: int):

            if len(batch_item_list) == 0:
                print(f"GPU {gpu_id} (Batch {batch_index}) has no work. Skipping.")
                return f"GPU {gpu_id} skipped."

            worker_output_dir = output_dir
            os.makedirs(worker_output_dir, exist_ok=True)
            
            worker_json_path = os.path.join(worker_output_dir, f"ligandmpnn_input_{gpu_id}.json")
            worker_data_dict = dict(batch_item_list) # <-- Reconstruct dictionary from (key, value) tuple list
            
            with open(worker_json_path, 'w') as f:
                json.dump(worker_data_dict, f) # <-- Write this new, smaller dictionary
            
            print(f"Starting batch {batch_index} on GPU {gpu_id} ({len(worker_data_dict)} structures)...")
            
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            cmd = (
                # f'conda run -n ligandmpnn python {self.config.inversefold.exec} '
                f'python {self.config.inversefold.exec} '
                f'--model_type {self.config.inversefold.model_type} '
                f'--checkpoint_{self.config.inversefold.model_type} {self.config.inversefold.checkpoint_path} '
                f'--pdb_path_multi {worker_json_path} '  # <-- [Modified] Use worker's JSON
                f'--out_folder {worker_output_dir} '     # <-- [Modified] Use worker's output directory
                f'--batch_size {self.config.inversefold.batch_size} '
                f'--number_of_batches {self.config.inversefold.number_of_batches} '
                # f'--pack_side_chains 1 '
                f'--pack_side_chains 0 '
                f'--number_of_packs_per_design 4 ' 
                f'--pack_with_ligand_context 0 ' 
                # f'--temperature {self.config.inversefold.temperature} '
                f'--seed {self.config.inversefold.seed} '
                f'--fixed_residues_multi "{worker_json_path}" ' # <-- [Modified] Use worker's JSON
                f'--parse_atoms_with_zero_occupancy {self.config.inversefold.parse_atoms_with_zero_occupancy}'
            )
            
            # 10. Run subprocess
            try:
                subprocess.run(cmd, shell=True, check=True, env=my_env, cwd=origin_cwd)
                return f"Batch {batch_index} (GPU {gpu_id}) finished successfully."
            except subprocess.CalledProcessError as e:
                print(f"Error processing batch {batch_index} on GPU {gpu_id}: {e}")
                raise e
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i in range(num_gpus):
                gpu_id = gpu_list[i]
                batch_data_items = data_batches[i].tolist() 
                
                futures.append(
                    executor.submit(
                        _run_batch_on_gpu, 
                        gpu_id, 
                        batch_data_items, 
                        i
                    )
                )

            print("Waiting for all LigandMPNN batches to complete...")
            for future in futures:
                try:
                    result_message = future.result() 
                    print(result_message)
                except Exception as e:
                    print(f"A LigandMPNN worker failed: {e}")

        print("All LigandMPNN parallel processing is complete.")
    
    def run_proteinmpnn_distributed(self, input_dir: Path, output_dir: str, gpu_list: list, origin_cwd: str):
        os.makedirs(output_dir, exist_ok=True)
        my_env = os.environ.copy()
        ligandmpnn_multi_input = {}
        for pdb_path in input_dir.rglob("*.pdb"):
            atom_array = pdb.PDBFile.read(pdb_path).get_structure(model=1, extra_fields=['b_factor'])
            fixed_atom_array = atom_array[atom_array.b_factor != 0.0]
            fixed_residues = np.unique(np.char.add(fixed_atom_array.chain_id, np.array(fixed_atom_array.res_id, dtype=str)))
            ligandmpnn_multi_input[str(pdb_path.parent / f"{pdb_path.stem.lower()}.pdb")] = " ".join(fixed_residues.tolist())
        json.dump(ligandmpnn_multi_input, open(os.path.join(output_dir, "ligandmpnn_input.json"), 'w'), indent=4)

        num_gpus = len(gpu_list)
        all_items_list = list(ligandmpnn_multi_input.items())
        data_batches = np.array_split(all_items_list, num_gpus)
        print(f"Split {len(all_items_list)} structures (from dict) into {num_gpus} batches for parallel processing.")

        def _run_batch_on_gpu(gpu_id: str, batch_item_list: list, batch_index: int):

            if len(batch_item_list) == 0:
                print(f"GPU {gpu_id} (Batch {batch_index}) has no work. Skipping.")
                return f"GPU {gpu_id} skipped."

            worker_output_dir = output_dir
            os.makedirs(worker_output_dir, exist_ok=True)
            
            worker_json_path = os.path.join(worker_output_dir, f"ligandmpnn_input_{gpu_id}.json")
            worker_data_dict = dict(batch_item_list) # <-- Reconstruct dictionary from (key, value) tuple list
            
            with open(worker_json_path, 'w') as f:
                json.dump(worker_data_dict, f) # <-- Write this new, smaller dictionary
            
            print(f"Starting batch {batch_index} on GPU {gpu_id} ({len(worker_data_dict)} structures)...")
            
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            cmd = (
                f'python {self.config.inversefold.exec} '
                f'--model_type {self.config.inversefold.model_type} '
                f'--checkpoint_{self.config.inversefold.model_type} {self.config.inversefold.checkpoint_path} '
                f'--pdb_path_multi {worker_json_path} '  # <-- [Modified] Use worker's JSON
                f'--out_folder {worker_output_dir} '     # <-- [Modified] Use worker's output directory
                f'--batch_size {self.config.inversefold.batch_size} '
                f'--number_of_batches {self.config.inversefold.number_of_batches} '
                f'--temperature {self.config.inversefold.temperature} '
                f'--seed {self.config.inversefold.seed} '
                f'--fixed_residues_multi "{worker_json_path}"' # <-- [Modified] Use worker's JSON
            )
            
            # 10. Run subprocess
            try:
                subprocess.run(cmd, shell=True, check=True, env=my_env, cwd=origin_cwd)
                return f"Batch {batch_index} (GPU {gpu_id}) finished successfully."
            except subprocess.CalledProcessError as e:
                print(f"Error processing batch {batch_index} on GPU {gpu_id}: {e}")
                raise e
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i in range(num_gpus):
                gpu_id = gpu_list[i]
                batch_data_items = data_batches[i].tolist() 
                
                futures.append(
                    executor.submit(
                        _run_batch_on_gpu, 
                        gpu_id, 
                        batch_data_items, 
                        i
                    )
                )

            print("Waiting for all LigandMPNN batches to complete...")
            for future in futures:
                try:
                    result_message = future.result() 
                    print(result_message)
                except Exception as e:
                    print(f"A LigandMPNN worker failed: {e}")

        print("All LigandMPNN parallel processing is complete.")
    
    def run_odesignmpnn(self, input_root: Path, inverse_fold_root: str):
        """
        Starts ODesignMPNN inference on multiple GPUs.
        The 'device' parameter is ignored and is now controlled by 'local_rank'.
        """
        os.makedirs(inverse_fold_root, exist_ok=True)
        samples_all: List[Dict[str, Any]] = []

        # --- STEP 1: Data preparation (runs on main process) ---
        print("Preparing samples...")
        for cif_path in input_root.rglob("*.cif"):
            # #qtfeng: Read B_iso_or_equiv from CIF file instead of pkl file
            # #qtfeng: If B_iso_or_equiv is 0, the chain needs design (mpnn)
            # #qtfeng: If B_iso_or_equiv is 1, the chain does not need mpnn
            try:
                cif_file = pdbx.CIFFile.read(cif_path)
                atom_array = pdbx.get_structure(cif_file, model=1, extra_fields=['b_factor'])
            except Exception as e:
                print(f"Warning: Failed to read CIF file {cif_path}: {e}, skipping")
                continue
            
            # #qtfeng: Get design chains based on B_iso_or_equiv (0 means need design)
            design_chains = []
            for chain_id in np.unique(atom_array.chain_id):
                chain_atoms = atom_array[atom_array.chain_id == chain_id]
                # #qtfeng: Check if all atoms in this chain have B_iso_or_equiv == 0 (need design)
                if np.all(chain_atoms.b_factor == 0.0):
                    design_chains.append(chain_id)
            
            # #qtfeng: Skip if no chains need design (all chains have B_iso_or_equiv == 1)
            if len(design_chains) == 0:
                print(f"Info: No chains need design in {cif_path} (all B_iso_or_equiv == 1), skipping")
                continue
            
            assert len(design_chains) == 1, f"Only one design chain is allowed, but got {len(design_chains)} in {cif_path}"
            
            if self.config.inversefold.data_name != 'ligand':
                smp = parse_cif_to_samples(cif_path=cif_path, data_name=self.config.inversefold.data_name, only_chain=None)
            else:
                smp = extract_ligand_samples_from_cif(cif_path=cif_path, only_chain=None, skip_waters=True)
            
            # breakpoint()

            sample_dict = {"smp": smp[0], "composition": {}}
            
            # *** IMPORTANT: Add the source path so the worker can find the CIF file ***
            sample_dict["cif_src_path"] = str(cif_path)

            # #qtfeng: Process chains based on B_iso_or_equiv from CIF file
            # #qtfeng: Get default mol_type from config
            default_mol_type = self.config.inversefold.data_name if self.config.inversefold.data_name != 'ligand' else 'ligand'
            for c_id in np.unique(atom_array.chain_id):
                c_array = atom_array[atom_array.chain_id == c_id]
                # #qtfeng: Get mol_type for this chain
                if hasattr(c_array, 'mol_type') and len(c_array.mol_type) > 0:
                    mol_type = str(c_array.mol_type[0])
                else:
                    mol_type = default_mol_type
                
                # #qtfeng: If all atoms have B_iso_or_equiv == 1, this chain does not need design
                if np.all(c_array.b_factor == 1.0):
                    # #qtfeng: This is a fixed chain, get its sequence
                    if mol_type == 'ligand':
                        # #qtfeng: For ligand, use res_name
                        sample_dict['composition'][str(c_id)] = (mol_type, c_array.res_name[0] if len(c_array.res_name) > 0 else 'UNK')
                    else:
                        sample_dict['composition'][str(c_id)] = (mol_type, str(struc.to_sequence(c_array)[0][0]))
                else:
                    # #qtfeng: This chain needs design (B_iso_or_equiv == 0)
                    sample_dict["cond_c_id"] = str(c_id)
                    sample_dict['composition'][str(c_id)] = (mol_type, "REDESIGN_FLAG")
            samples_all.append(sample_dict)

        if not samples_all:
            print("Found no CIF files to process. Exiting.")
            return

        print(f"Found {len(samples_all)} samples. Starting multi-GPU inference...")

        # --- STEP 2: Start Multi-GPU Workers ---
        world_size = torch.cuda.device_count()
        if world_size == 0:
            print("ERROR: No GPUs found. Exiting.")
            return

        # Arguments to pass to each worker process
        # (local_rank is added automatically by mp.spawn)
        args_to_pass = (
            world_size,
            self.config,  # Pass the configuration object
            inverse_fold_root,
            samples_all,  # Pass the complete list
        )
        
        # breakpoint()

        mp.spawn(
            _odesign_mpnn_worker,  # Function to run
            args=args_to_pass,         # Arguments
            nprocs=world_size,         # Number of processes (one per GPU)
            join=True                  # Wait until all are done
        )
        
        print("Multi-GPU inference completed.")

    # def run_gRNAde(self, input_dir: Path, g_id: str):

    #     model = gRNAde(gpu_id=g_id)
    #     for pdb_path in Path(input_dir).rglob('*.pdb'):
    #         def resnames_to_seq_fallback(resnames, monomer):
    #             if monomer.upper() == 'P':
    #                 return ''.join(AA3_TO_1.get(str(r).upper(), 'X') for r in resnames)
    #             else:
    #                 return ''.join(NA3_TO_1.get(str(r).upper(), 'N') for r in resnames)
    #         sequence = []
    #         pdbfile = pdb.PDBFile.read(pdb_path)
    #         atom_array = pdb.get_structure(pdbfile, model=1)
    #         trb_path = pdb_path.parent / f"{pdb_path.stem}.trb"
    #         trb = pickle.load(open(trb_path, 'rb'))
    #         unique_chains = np.unique(atom_array.chain_id)
    #         for c_id in unique_chains:
    #             c_array = atom_array[atom_array.chain_id == c_id]
    #             c_trb = trb[trb.chain_id == c_id]
    #         sequences, samples, perplexity, recovery = model.design_from_pdb_file(pdb_path, n_samples=self.config.inversefold.n_samples, partial_seq=)
    #         sequences, samples, perplexity, recovery = model.design_from_pdb_file(pdb_path, n_samples=self.config.inversefold.n_samples, partial_seq=)
    #         for sample_id, seqrecord in enumerate(sequences):
    #             seq = [str(seqrecord.seq)] + sequence
    #             all_pred_seqs.append(':'.join(seq))
    #             all_names.append(f"{pdb_path.stem}_{sample_id+1}")

    
    # def run_odesignmpnn(self, input_root: Path, inverse_fold_root: str, device: str):
    #     """
    #     we only support one condition chain in the cif file for now
    #     """
    #     os.makedirs(inverse_fold_root, exist_ok=True)
    #     samples_all: List[Dict[str, Any]] = []

    #     for cif_path in input_root.rglob("*.cif"):
    #         trb_path = cif_path.parent / f"{cif_path.stem}.pkl"
    #         trb = pickle.load(open(trb_path, 'rb'))
    #         condition_chains = np.unique(trb.chain_id[trb.condition_token_mask])
    #         assert len(condition_chains) == 1, f"Only one condition chain is allowed, but got {condition_chains} in {cif_path}"
            
    #         if self.config.inversefold.data_name != 'ligand':
    #             smp = parse_cif_to_samples(cif_path=cif_path, data_name=self.config.inversefold.data_name, only_chain=None)
    #         else:
    #             smp = extract_ligand_samples_from_cif(cif_path=cif_path, only_chain=None, skip_waters=True)
    #         sample_dict = {"smp": smp[0], "composition": {}}
    #         for c_id in np.unique(trb.chain_id):
    #             c_array = trb[trb.chain_id == c_id]
    #             if all(c_array.condition_token_mask):
    #                 if c_array.is_ligand.all():
    #                     sample_dict['composition'][str(c_id)] = (str(c_array.mol_type[0]), c_array.res_name[0]) #(mol_type, ccdCode)
    #                 else:
    #                     sample_dict['composition'][str(c_id)] = (str(c_array.mol_type[0]), str(struc.to_sequence(c_array)[0][0]))
    #             else:
    #                 sample_dict["cond_c_id"] = str(c_id)
    #                 sample_dict['composition'][str(c_id)] = (str(c_array.mol_type[0]), "REDESIGN_FLAG")
    #         samples_all.append(sample_dict)

    #     model, dev = reload_model(data_name=self.config.inversefold.data_name, model_name="ODesign", device=device)
    #     for sample_dict in samples_all:
    #         pred_seqs, scores, true_seq_masked, probs, metrics = inference(
    #             model=model, sample_input=sample_dict['smp'], model_name="ODesign",
    #             data_name=self.config.inversefold.data_name, topk=self.config.inversefold.topk,
    #             temp=self.config.inversefold.temp, use_beam=self.config.inversefold.use_beam,
    #             device=dev
    #         )
    #         if self.config.inversefold.save_all:
    #             for i, seq in enumerate(pred_seqs):
    #                 cif_src_path=input_root / f"{sample_dict['smp']['title']}.cif"
    #                 out_path=os.path.join(inverse_fold_root, f"{sample_dict['smp']['title']}-{i+1}.cif")
    #                 src_file = pdbx.CIFFile.read(cif_src_path)
    #                 src_atom_array = pdbx.get_structure(src_file, model=1)
    #                 if sample_dict['composition'][sample_dict["cond_c_id"]][0] == 'protein':
    #                     seq_1_to_3 = [_AA1_TO_AA3.get(rt, 'UNK') for rt in list(seq)]
    #                 elif sample_dict['composition'][sample_dict["cond_c_id"]][0] == 'rna':
    #                     seq_1_to_3 = list(seq)
    #                 elif sample_dict['composition'][sample_dict["cond_c_id"]][0] == 'dna':
    #                     seq_1_to_3 = [f"D{rt}" for rt in list(seq)]
    #                 else:
    #                     seq_1_to_3 = seq.split(' ')

    #                 save_sequence_replaced_array = src_atom_array.copy()
    #                 r_starts = struc.get_residue_starts(save_sequence_replaced_array, add_exclusive_stop=True)
    #                 redesign_id = 0
    #                 for r_s, r_e in zip(r_starts[:-1], r_starts[1:]):
    #                     if save_sequence_replaced_array.chain_id[r_s] == sample_dict["cond_c_id"] and self.config.inversefold.data_name != "ligand":
    #                         save_sequence_replaced_array.res_name[r_s:r_e] = seq_1_to_3[redesign_id]
    #                         redesign_id += 1
    #                     elif save_sequence_replaced_array.chain_id[r_s] == sample_dict["cond_c_id"] and self.config.inversefold.data_name == "ligand":
    #                         save_sequence_replaced_array.element[r_s: r_e] = seq_1_to_3
    #                         save_sequence_replaced_array.atom_name[r_s: r_e] = np.char.add(seq_1_to_3, np.array(range(r_e - r_s), dtype=np.str_))
    #                     else: continue

    #                 io.save_structure(out_path, save_sequence_replaced_array)

    #         sample_dict["composition"][sample_dict["cond_c_id"]] = (sample_dict["composition"][sample_dict["cond_c_id"]][0], pred_seqs)
        
        # if self.config.refold.name == "alphafold3":
        #     af3_json = []
        #     for sample_dict in samples_all:
        #         cond_type, pred_seqs = sample_dict["composition"][sample_dict["cond_c_id"]]
        #         for i, p_seq in enumerate(pred_seqs):
        #             name = sample_dict['smp']['title'] + f"_{i+1}"
        #             sequence = []
        #             for c_id, (res_type, seq) in sample_dict['composition'].items():
        #                 if c_id == sample_dict["cond_c_id"]:
        #                     sequence.append({
        #                         res_type: {
        #                             "id": [c_id],
        #                             "sequence": p_seq,
        #                         }
        #                     })
        #                 else:
        #                     sequence.append({
        #                         res_type: {
        #                             "id": [c_id],
        #                             "sequence": seq,
        #                         }
        #                     })
        #             if not self.config.refold.run_data_pipeline:
        #                 for i, s in enumerate(sequence):
        #                     for k, v in s.items():
        #                         if k == 'protein':
        #                             sequence[i][k]["unpairedMsa"] = ""
        #                             sequence[i][k]["pairedMsa"] = ""
        #                             sequence[i][k]["templates"] = []
        #                         elif k == 'rna':
        #                             sequence[i][k]["unpairedMsa"] = ""
        #                         else:
        #                             continue
        #             af3_json.append({
        #                 "name": name,
        #                 "sequences": sequence,
        #                 "modelSeeds": [1],
        #                 "dialect": "alphafold3",
        #                 "version": 1
        #             })
        #     with open(os.path.join(refold_root, "af3_input.json"), 'w') as f:
        #         json.dump(af3_json, f, indent=4)