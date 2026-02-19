import os
import tqdm
import pickle
import operator
import subprocess
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import concurrent.futures
from evaluation.metrics.rmsd import RMSDCalculator
from evaluation.metrics.confidence import Confidence
from evaluation.metrics.usalign import USalign
from evaluation.metrics.foldseek import FoldSeek
from evaluation.metrics.analyze import calc_hydrophobicity, count_noncovalents
from evaluation.metrics.developability import DevelopabilityScorer
import json
import glob

class Evaluation():

    def __init__(self, config):

        self.config = config
    
    @staticmethod
    def _get_sequence_from_inversefold(pipeline_dir: str, sample_name: str) -> str:
        """
        Get sequence from LigandMPNN/ProteinMPNN output fasta file

        Args:
            pipeline_dir: pipeline root directory
            sample_name: sample name (without extension)
            
        Returns:
            str: protein sequence, if not found return empty string
        """
        # read sequence from LigandMPNN/ProteinMPNN output fasta file
        seqs_dir = os.path.join(pipeline_dir, "inverse_fold", "seqs")
        if os.path.exists(seqs_dir):
            # find matching fasta file
            for fasta_file in Path(seqs_dir).glob(f"{sample_name}*.fa"):
                try:
                    with open(fasta_file, 'r') as f:
                        lines = f.readlines()
                        # FASTA format: first line is header, second line is sequence
                        for i, line in enumerate(lines):
                            if line.startswith('>'):
                                if i + 1 < len(lines):
                                    seq = lines[i + 1].strip()
                                    # remove possible chain separators (e.g. ':')
                                    seq = seq.replace(':', '').replace(';', '')
                                    return seq.upper()
                except Exception as e:
                    continue
        
        # if no fasta file is found, return empty string
        return ""
    

    def run_protein_binding_ligand_evaluation(self, input_dir: str, output_dir: str, 
                          dist_cutoff: float = 10.0, 
                          exhaustiveness: int = 16,
                          num_processes: int = 8,
                          verbose: bool = True,
                          cuda_device: str = "0",
                          enable_geom: bool = True,
                          enable_chem: bool = True,
                          enable_vina: bool = True,
                          ccd_bond_length_path: str = None,
                          ccd_bond_angle_path: str = None,
                          ccd_torsion_angle_path: str = None):
        """
        Run ligand evaluation pipeline using the ligand_evaluation.py script
        
        Args:
            input_dir (str): Input directory containing CIF files
            output_dir (str): Output directory for results
            dist_cutoff (float): Distance cutoff for pocket definition (default: 10.0)
            exhaustiveness (int): Exhaustiveness for docking (default: 16)
            num_processes (int): Number of processes (default: 8)
            verbose (bool): Enable verbose output (default: True)
            cuda_device (str): CUDA device to use (default: "0")
            enable_geom (bool): Enable geometry evaluation (default: True)
            enable_chem (bool): Enable chemistry evaluation (default: True)
            enable_vina (bool): Enable Vina docking evaluation (default: True)
            ccd_bond_length_path (str): Path to CCD bond length distribution file
            ccd_bond_angle_path (str): Path to CCD bond angle distribution file
            ccd_torsion_angle_path (str): Path to CCD torsion angle distribution file
            
        Returns:
            dict: Result dictionary containing success status and output information
        """
        try:
            
            os.environ['BABEL_LIBDIR'] = self.config.metrics.babel_libdir
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

            
            # Construct the command
            script_path = os.path.join(os.path.dirname(__file__), "ligand_evaluation.py")
            
            cmd = [
                'python', script_path,
                '--input_dir', input_dir,
                '--output_dir', output_dir,
                '--dist_cutoff', str(dist_cutoff),
                '--exhaustiveness', str(exhaustiveness),
                '--num_processes', str(num_processes),
                '--ccd_bond_length_path', self.config.metrics.ccd_bond_length_path,
                '--ccd_bond_angle_path', self.config.metrics.ccd_bond_angle_path,
                '--ccd_torsion_angle_path', self.config.metrics.ccd_torsion_angle_path
            ]
            
            # Add evaluation module flags
            if enable_geom:
                cmd.append('--enable_geom')
            else:
                cmd.append('--disable_geom')
                
            if enable_chem:
                cmd.append('--enable_chem')
            else:
                cmd.append('--disable_chem')
                
            if enable_vina:
                cmd.append('--enable_vina')
            else:
                cmd.append('--disable_vina')
            
            # Add verbose flag if requested
            if verbose:
                cmd.append('--verbose')
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Run the command
            print(f"Running ligand metrics pipeline...")
            print(f"Command: {' '.join(cmd)}")
            print(f"Input directory: {input_dir}")
            print(f"Output directory: {output_dir}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Input directory exists: {os.path.exists(input_dir)}")
            if os.path.exists(input_dir):
                print(f"Files in input directory: {os.listdir(input_dir)}")
            
            result = subprocess.run(
                cmd,
                # env=env,
                capture_output=True,
                text=True,
                # cwd=current_dir
            )
            
            # Prepare result dictionary
            result_dict = {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'input_dir': input_dir,
                'output_dir': output_dir,
                'command': ' '.join(cmd)
            }
            
            if result.returncode == 0:
                print("Ligand metrics pipeline completed successfully!")
                print(f"Results saved to: {output_dir}")
            else:
                print(f"Ligand metrics pipeline failed with return code: {result.returncode}")
                print(f"Error output: {result.stderr}")
            
            return result_dict
            
        except Exception as e:
            error_dict = {
                'success': False,
                'error': str(e),
                'input_dir': input_dir,
                'output_dir': output_dir
            }
            print(f"Error running ligand metrics pipeline: {e}")
            return error_dict
    
    def run_protein_evaluation(self, pipeline_dir: str, output_csv: str, ca_rmsd_threshold: float = 2.0):
        '''
        refold model: esmfold
        '''
        
        def process_metrics_worker(refold_path: Path):
            try:
                sample_name = refold_path.name
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", "backbones", sample_name)

                ca_rmsd = RMSDCalculator.compute_protein_ca_rmsd(pred=str(refold_path), refold=inverse_fold_path)
                plddt= Confidence.gather_esmfold_confidence(str(refold_path))
                
                # Calculate hydrophobicity and noncovalents
                # Use sequence from inversefold output (designed sequence) for hydrophobicity calculation
                # noncovalents need to be calculated from refolded structure files
                sample_name_no_ext = os.path.splitext(sample_name)[0]
                sequence = Evaluation._get_sequence_from_inversefold(pipeline_dir, sample_name_no_ext)
                # If inversefold sequence not found, skip hydrophobicity calculation
                hydrophobicity = calc_hydrophobicity(sequence) if sequence else np.nan
                noncovalents = count_noncovalents(str(refold_path))
                
                result_data = {
                    'ca_rmsd': ca_rmsd,
                    'plddt': plddt,
                    'hydrophobicity': hydrophobicity,
                    'num_hydrogen_bonds': noncovalents.get('num_hydrogen_bonds', 0),
                    'num_salt_bridges': noncovalents.get('num_salt_bridges', 0),
                }
                return sample_name, result_data
                
            except Exception as e:
                print(f"Warning: get error when dealing with {refold_path.name}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)

        esmfold_result_json_path = os.path.join(pipeline_dir, "refold", "esmfold_out", "esmfold_results.json")
        with open(esmfold_result_json_path, "r") as f:
            esmfold_result_json = json.load(f)

        all_refold_paths = [Path(i['pdb_path']) for i in esmfold_result_json]
        print(f"{len(all_refold_paths)} files were found for evaluation.")
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for refold_path in all_refold_paths:
                future = executor.submit(process_metrics_worker, refold_path)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_refold_paths), desc="computing metrics"):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient='index')
        
        # Extract design names from sample names (e.g., oqo-1-4.pdb -> oqo-1)
        def extract_design_name(sample_name: str) -> str:
            # Remove .pdb extension and extract design name (remove last number)
            name_without_ext = sample_name.replace('.pdb', '')
            # Split by '-' and remove the last part (refold number)
            parts = name_without_ext.rsplit('-', 1)
            if len(parts) == 2:
                return parts[0]
            return name_without_ext
        
        # Add design_name column
        df['design_name'] = df.index.map(extract_design_name)
        
        # Mark structures as designable if ca_rmsd < threshold
        df['is_designable'] = df['ca_rmsd'] < ca_rmsd_threshold
        
        # Group by design_name and check if all refold structures are designable
        design_groups = df.groupby('design_name')
        designable_designs = set()
        for design_name, group in design_groups:
            # A design is designable if all its refold structures have ca_rmsd < threshold
            if group['is_designable'].all():
                designable_designs.add(design_name)
        
        # Calculate designability
        total_designs = len(design_groups)
        num_designable = len(designable_designs)
        designability = num_designable / total_designs if total_designs > 0 else 0.0
        
        print(f"Designability: {num_designable}/{total_designs} = {designability:.4f}")
        
        # Filter to only designable structures for diversity and novelty computation
        designable_df = df[df['design_name'].isin(designable_designs)]
        
        # Initialize summary data
        summary_data = {
            'designability': {
                'value': designability,
                'num_designable': num_designable,
                'total_designs': total_designs
            },
            'diversity': {
                'value': 0.0,
                'clusters': 0,
                'num_designable': num_designable
            },
            'novelty': {
                'value': 0.0,
                'max_tmscore_avg': 0.0,
                'num_designable': num_designable
            }
        }
        
        if len(designable_df) == 0:
            print("Warning: No designable structures found. Setting diversity and novelty to 0.0")
            df['diversity'] = 0.0
            df['novelty'] = 0.0
            df['designability'] = designability
        else:
            # Create a temporary directory with only designable structures
            # Use formatted_designs instead of refold results to avoid duplicates
            # (refold generates multiple structures per design based on MPNN count)
            formatted_designs_dir = os.path.join(pipeline_dir, "formatted_designs")
            metrics_output_dir = os.path.join(pipeline_dir, "metrics")
            designable_structure_dir = os.path.join(metrics_output_dir, "designable_structures")
            os.makedirs(designable_structure_dir, exist_ok=True)
            
            # Copy designable structure files from formatted_designs to temporary directory
            # Use design names (e.g., oqo-1) instead of refold sample names (e.g., oqo-1-1.pdb)
            for design_name in designable_designs:
                src_path = os.path.join(formatted_designs_dir, f"{design_name}.pdb")
                dst_path = os.path.join(designable_structure_dir, f"{design_name}.pdb")
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"Warning: Design structure not found: {src_path}")
            
            print(f"Computing diversity and novelty for {len(designable_designs)} designable structures...")
            
            # Initialize FoldSeek with config
            foldseek_config = {}
            if hasattr(self.config.metrics, 'foldseek_bin'):
                foldseek_config['foldseek_bin'] = self.config.metrics.foldseek_bin
            if hasattr(self.config.metrics, 'foldseek_database'):
                foldseek_config['foldseek_database'] = self.config.metrics.foldseek_database
            foldseek_config['verbose'] = getattr(self.config.metrics, 'verbose', False)
            
            # Fallback: try to load from foldseek.yaml if not in metrics config
            if 'foldseek_database' not in foldseek_config or not foldseek_config['foldseek_database']:
                import yaml
                foldseek_yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'foldseek.yaml')
                if os.path.exists(foldseek_yaml_path):
                    try:
                        with open(foldseek_yaml_path, 'r') as f:
                            foldseek_yaml = yaml.safe_load(f)
                            if 'foldseek_bin' in foldseek_yaml and 'foldseek_bin' not in foldseek_config:
                                foldseek_config['foldseek_bin'] = foldseek_yaml['foldseek_bin']
                            if 'foldseek_database' in foldseek_yaml:
                                foldseek_config['foldseek_database'] = foldseek_yaml['foldseek_database']
                            print(f"Loaded FoldSeek config from {foldseek_yaml_path}")
                    except Exception as e:
                        print(f"Warning: Failed to load foldseek.yaml: {e}")
            
            # Debug: print configuration
            print(f"FoldSeek config - bin: {foldseek_config.get('foldseek_bin', 'NOT SET')}")
            print(f"FoldSeek config - database: {foldseek_config.get('foldseek_database', 'NOT SET')}")
            
            foldseek = FoldSeek(foldseek_config)
            
            # Compute diversity for designable structures
            diversity_result = foldseek.compute_diversity(
                structure_dir=designable_structure_dir,
                output_dir=metrics_output_dir,
                dump=True
            )
            
            # Compute novelty for designable structures
            novelty_result = foldseek.compute_novelty(
                structure_dir=designable_structure_dir,
                output_dir=metrics_output_dir,
                dump=True
            )
            
            # Update summary data with diversity results
            # Diversity should be based on designable_designs: num_clusters / num_designable
            num_clusters = diversity_result.get('num_clusters', 0)
            diversity_value = num_clusters / num_designable if num_designable > 0 else 0.0
            summary_data['diversity'] = {
                'value': diversity_value,
                'clusters': num_clusters,
                'num_designable': num_designable
            }
            
            # Update summary data with novelty results
            # Novelty value is already computed
            novelty_value = novelty_result.get('novelty', 0.0)
            summary_data['novelty'] = {
                'value': novelty_value,
                'max_tmscore_avg': novelty_result.get('max_tmscore_avg', 0.0),
                'num_designable': num_designable
            }
            
            # Add diversity and novelty to all rows (same value for all samples)
            df['diversity'] = diversity_result.get('diversity', 0.0)
            df['novelty'] = novelty_result.get('novelty', 0.0)
            df['designability'] = designability
        
        # Save summary JSON
        output_dir = os.path.dirname(output_csv)
        summary_output_path = os.path.join(output_dir, 'raw_summary.json')
        with open(summary_output_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Summary saved to {summary_output_path}")
        
        df.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")

    # maybe need to seperate into different functions for easy use and development
    def run_nuc_binding_ligand_evaluation(self, pipeline_dir: str, output_csv: str):
        
        def process_metrics_worker(refold_path: Path):
            try:
                sample_name = refold_path.parent.name
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", f"{sample_name.upper()}.cif")
                trb_path = os.path.join(pipeline_dir, "formatted_designs", f"{sample_name.rsplit('-',1)[0]}-1.pkl")
                summary_confidence_path = os.path.join(refold_path.parent, f"{refold_path.parent.name}_summary_confidences.json")
                confidence_path = os.path.join(refold_path.parent, f"{refold_path.parent.name}_confidences.json")

                # breakpoint()
                
                
                plddt, ipae, min_ipae, iptm, ptm_binder = Confidence.gather_af3_confidence(confidence_path, summary_confidence_path, trb_path)
                # breakpoint()
                ligand_rmsd = RMSDCalculator.compute_nuc_align_ligand_rmsd(pred=str(refold_path), refold=inverse_fold_path)
                
                # Calculate noncovalents (all tasks can use this)
                # Note: This is nucleic acid + ligand, so hydrophobicity may not be applicable
                noncovalents = count_noncovalents(str(refold_path))
                
                result_data = {
                    'ligand_rmsd': ligand_rmsd,
                    'plddt': plddt,
                    'ipae': ipae,
                    'min_ipae': min_ipae,
                    'iptm': iptm,
                    'ptm_binder': ptm_binder,
                    'num_hydrogen_bonds': noncovalents.get('num_hydrogen_bonds', 0),
                    'num_salt_bridges': noncovalents.get('num_salt_bridges', 0),
                }
                return sample_name, result_data
                
            except Exception as e:
                print(f"Warning: Error processing {refold_path.name}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)
        all_refold_paths = list(Path(os.path.join(pipeline_dir, "refold", "af3_out")).rglob("*_model.cif"))
        print(f"{len(all_refold_paths)} files were found for evaluation.")
        futures = []

        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for refold_path in all_refold_paths:
                future = executor.submit(process_metrics_worker, refold_path)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_refold_paths), desc="computing metrics"):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient='index')
        df.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")

    # maybe need to seperate into different functions for easy use and development
    def run_protein_binding_protein_evaluation(self, pipeline_dir: str, output_csv: str):
        
        def process_metrics_worker(refold_path: Path):
            try:
                sample_name = refold_path.parent.name
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", "backbones", f"{sample_name}.pdb")
                # trb_path = os.path.join(pipeline_dir, "formatted_designs", f"{sample_name.rsplit('-',1)[0]}.pkl")
                summary_confidence_path = os.path.join(refold_path.parent, f"{refold_path.parent.name}_summary_confidences.json")
                confidence_path = os.path.join(refold_path.parent, f"{refold_path.parent.name}_confidences.json")
                
                try:
                    ca_rmsd = RMSDCalculator.compute_protein_ca_rmsd(pred=str(refold_path), refold=inverse_fold_path)
                except:
                    ca_rmsd = np.inf
                    print(f"{refold_path} fail for calculate rmsd, set to inf, please check the case")
                plddt, ipae, min_ipae, iptm, ptm_binder = Confidence.gather_af3_confidence(confidence_path, summary_confidence_path, inverse_fold_path)
                
                # Calculate hydrophobicity and noncovalents
                # Use sequence from inversefold output (designed sequence) for hydrophobicity calculation
                # noncovalents need to be calculated from refolded structure files
                sample_name_no_ext = os.path.splitext(sample_name)[0]
                sequence = Evaluation._get_sequence_from_inversefold(pipeline_dir, sample_name_no_ext)
                # If inversefold sequence not found, skip hydrophobicity calculation
                hydrophobicity = calc_hydrophobicity(sequence) if sequence else np.nan
                noncovalents = count_noncovalents(str(refold_path))
                
                result_data = {
                    'ca_rmsd': ca_rmsd,
                    'plddt': plddt,
                    'ipae': ipae,
                    'min_ipae': min_ipae,
                    'iptm': iptm,
                    'ptm_binder': ptm_binder,
                    'hydrophobicity': hydrophobicity,
                    'num_hydrogen_bonds': noncovalents.get('num_hydrogen_bonds', 0),
                    'num_salt_bridges': noncovalents.get('num_salt_bridges', 0),
                }
                return sample_name, result_data
                
            except Exception as e:
                print(f"Warning: Error processing {refold_path.name}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)
        all_refold_paths = list(Path(os.path.join(pipeline_dir, "refold", "af3_out")).rglob("*_model.cif"))
        print(f"{len(all_refold_paths)} files were found for evaluation.")
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for refold_path in all_refold_paths:
                future = executor.submit(process_metrics_worker, refold_path)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_refold_paths), desc="computing metrics"):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient='index')
        df.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")

    # maybe need to seperate into different functions for easy use and development
    def run_abag_evaluation(self, pipeline_dir: str, output_csv: str):
        
        def process_metrics_worker(refold_path: Path):
            try:
                sample_name = refold_path.parent.name
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", "backbones", f"{sample_name}.pdb")
                # trb_path = os.path.join(pipeline_dir, "formatted_designs", f"{sample_name.rsplit('-',1)[0]}.pkl")
                summary_confidence_path = os.path.join(refold_path.parent, f"{refold_path.parent.name}_summary_confidences.json")
                confidence_path = os.path.join(refold_path.parent, f"{refold_path.parent.name}_confidences.json")
                
                try:
                    ca_rmsd = RMSDCalculator.compute_protein_ca_rmsd(pred=str(refold_path), refold=inverse_fold_path)
                except:
                    ca_rmsd = np.inf
                    print(f"{refold_path} fail for calculate rmsd, set to inf, please check the case")
                plddt, ipae, min_ipae, iptm, ptm_binder = Confidence.gather_af3_confidence(confidence_path, summary_confidence_path, inverse_fold_path)
                
                # Calculate hydrophobicity and noncovalents
                # Use sequence from inversefold output (designed sequence) for hydrophobicity calculation
                # noncovalents need to be calculated from refolded structure files
                sample_name_no_ext = os.path.splitext(sample_name)[0]
                sequence = Evaluation._get_sequence_from_inversefold(pipeline_dir, sample_name_no_ext)
                # If inversefold sequence not found, skip hydrophobicity calculation
                hydrophobicity = calc_hydrophobicity(sequence) if sequence else np.nan
                noncovalents = count_noncovalents(str(refold_path))
                
                result_data = {
                    'ca_rmsd': ca_rmsd,
                    'plddt': plddt,
                    'ipae': ipae,
                    'min_ipae': min_ipae,
                    'iptm': iptm,
                    'ptm_binder': ptm_binder,
                    'hydrophobicity': hydrophobicity,
                    'num_hydrogen_bonds': noncovalents.get('num_hydrogen_bonds', 0),
                    'num_salt_bridges': noncovalents.get('num_salt_bridges', 0),
                }
                return sample_name, result_data
                
            except Exception as e:
                print(f"Warning: Error processing {refold_path.name}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)
        all_refold_paths = list(Path(os.path.join(pipeline_dir, "refold", "af3_out")).rglob("*_model.cif"))
        print(f"{len(all_refold_paths)} files were found for evaluation.")
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for refold_path in all_refold_paths:
                future = executor.submit(process_metrics_worker, refold_path)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_refold_paths), desc="computing metrics"):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient='index')
        df.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")
    
    def run_ligand_binding_protein_evaluation(self, pipeline_dir: str, cands: str, output_csv: str):
        
        def process_metrics_worker(cand: Path):
            try:
                refold_paths = cand.cif_paths
                result_data_all = defaultdict(dict)
                refold_path = refold_paths[0]
                sample_name = refold_path.parent.name
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", "backbones", f"{sample_name}.pdb")
                # trb_path = os.path.join(pipeline_dir, "formatted_designs", f"{sample_name.rsplit('-',1)[0]}.pkl")
                plddt, ipae, min_ipae, iptm, ptm_binder = Confidence.gather_chai1_confidence(cand, inverse_fold_path)
                
                # Calculate hydrophobicity and noncovalents
                # Use sequence from inversefold output (designed sequence) for hydrophobicity calculation
                # noncovalents need to be calculated from refolded structure files
                sample_name_no_ext = os.path.splitext(sample_name)[0]
                sequence = Evaluation._get_sequence_from_inversefold(pipeline_dir, sample_name_no_ext)
                # If inversefold sequence not found, skip hydrophobicity calculation
                hydrophobicity = calc_hydrophobicity(sequence) if sequence else np.nan
                noncovalents = count_noncovalents(str(refold_path))
                
                result_data_all[f"{sample_name}"] = {
                    'plddt': plddt,
                    'ipae': ipae,
                    'min_ipae': min_ipae,
                    'iptm': iptm,
                    'ptm_binder': ptm_binder,
                    'hydrophobicity': hydrophobicity,
                    'num_hydrogen_bonds': noncovalents.get('num_hydrogen_bonds', 0),
                    'num_salt_bridges': noncovalents.get('num_salt_bridges', 0),
                }
                return result_data_all
                
            except Exception as e:
                print(f"Warning: Error processing {refold_path}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)
        cands = pickle.load(open(cands, 'rb'))
        print(f"{len(cands)} files were found for evaluation.")
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for cand in cands:
                future = executor.submit(process_metrics_worker, cand)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(cands), desc="computing metrics"):
                result_data = future.result()
                if result_data is not None:
                    raw_data.update(result_data)

        df = pd.DataFrame.from_dict(raw_data, orient='index')
        df.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")
    
    def run_protein_binding_nuc_evaluation(self, pipeline_dir: str, output_csv: str):
        
        def process_metrics_worker(refold_path: Path):
            try:
                sample_name = refold_path.parent.name
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", f"{sample_name}.cif")
                trb_path = os.path.join(pipeline_dir, "formatted_designs", f"{sample_name.rsplit('-',1)[0]}.pkl")
                rmsd = RMSDCalculator.compute_protein_align_nuc_rmsd(pred=str(refold_path), refold=inverse_fold_path, trb=trb_path)
                
                # Calculate hydrophobicity and noncovalents
                # Use sequence from inversefold output (designed sequence) for hydrophobicity calculation
                # noncovalents need to be calculated from refolded structure files
                sample_name_no_ext = os.path.splitext(sample_name)[0]
                sequence = Evaluation._get_sequence_from_inversefold(pipeline_dir, sample_name_no_ext)
                # If inversefold sequence not found, skip hydrophobicity calculation
                hydrophobicity = calc_hydrophobicity(sequence) if sequence else np.nan
                noncovalents = count_noncovalents(str(refold_path))
                
                result_data = {
                    'rmsd': rmsd,
                    'hydrophobicity': hydrophobicity,
                    'num_hydrogen_bonds': noncovalents.get('num_hydrogen_bonds', 0),
                    'num_salt_bridges': noncovalents.get('num_salt_bridges', 0),
                }
                return sample_name, result_data
                
            except Exception as e:
                print(f"Warning: Error processing {refold_path.name}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)
        all_refold_paths = list(Path(os.path.join(pipeline_dir, "refold", "af3_out")).rglob("*_model.cif"))
        print(f"{len(all_refold_paths)} files were found for evaluation.")
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for refold_path in all_refold_paths:
                future = executor.submit(process_metrics_worker, refold_path)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_refold_paths), desc="computing metrics"):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient='index')
        df.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")
    
    def run_nuc_evaluation(self, pipeline_dir: str, output_csv: str):
        
        def process_metrics_worker(refold_path: Path):
            try:
                sample_name = refold_path.parent.name
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", f"{sample_name}.cif")
                rmsd = RMSDCalculator.compute_C4_rmsd(pred=str(refold_path), refold=inverse_fold_path)
                tmscore = USalign.compute_tmscore(pred=inverse_fold_path, refold=str(refold_path))
                
                # Calculate noncovalents (all tasks can use this)
                # Note: This is nucleic acid evaluation, so hydrophobicity may not be applicable
                noncovalents = count_noncovalents(str(refold_path))
                
                result_data = {
                    'rmsd': rmsd,
                    'tmscore': tmscore,
                    'num_hydrogen_bonds': noncovalents.get('num_hydrogen_bonds', 0),
                    'num_salt_bridges': noncovalents.get('num_salt_bridges', 0),
                }
                return sample_name, result_data
                
            except Exception as e:
                print(f"Warning: Error processing {refold_path.name}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)
        all_refold_paths = list(Path(os.path.join(pipeline_dir, "refold", "af3_out")).rglob("*_model.cif"))
        print(f"{len(all_refold_paths)} files were found for evaluation.")
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for refold_path in all_refold_paths:
                future = executor.submit(process_metrics_worker, refold_path)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_refold_paths), desc="computing metrics"):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient='index')

        op_map = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq
        }
        all_condition_series = []
        for metric_name, threshold in self.config.metrics.threshold.items():
            if metric_name not in df.columns:
                print(f"Warning: Metric '{metric_name}' in config does not exist in DataFrame, skipping.")
                continue

        try:
            # 2.1 Parse condition string, e.g., ">/0.45"
            op_str, val_str = threshold.split('/')
            threshold_value = float(val_str)
            
            # 2.2 Get corresponding operator function, e.g., operator.gt
            op_func = op_map[op_str]
            
            # 2.3 Apply operator to entire column (vectorized operation)
            # Example: op_func(df_metrics['metric_1'], 0.45)
            # This returns a boolean Series indicating which rows satisfy the condition
            condition_series = op_func(df[metric_name], threshold_value)
            all_condition_series.append(condition_series)

        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: Cannot parse condition '{threshold}' (metric: {metric_name}). Please check format. Error: {e}")
            # If a condition parsing fails, we can either make all samples fail
            # or skip this condition. Here we create an all-False Series
            all_condition_series.append(pd.Series(False, index=df.index))
        
        if not all_condition_series:
            print("No valid thresholds applied.")
            # If config is empty, can define all as success or all as failure
            success_series = pd.Series(True, index=df.index, name="is_successful")
        else:
            # 3.1 Combine all boolean Series into a new DataFrame
            combined_mask = pd.concat(all_condition_series, axis=1)
            
            # 3.2 Check each row (axis=1) if *all* conditions are True
            success_series = combined_mask.all(axis=1)
        
        df['if_success'] = success_series
        df.to_csv(output_csv, index=True)

        success_backbone_list = defaultdict(list)
        for sample_name in df.index:
            sample_case = sample_name.rsplit('-',1)[0]
            case = sample_name.split('-')[0]
            if sample_case in success_backbone_list:
                continue
            if_success = df.loc[sample_name, 'if_success']
            if if_success:
                success_backbone_list[case].append(f"{sample_case}.cif")
        os.makedirs(os.path.join(pipeline_dir, "cluster"), exist_ok=True)
        os.makedirs(os.path.join(pipeline_dir, "cluster", "success"), exist_ok=True)
        for case, backbones in success_backbone_list.items():
            with open(os.path.join(pipeline_dir, "cluster", "success", f"{case}_success_backbones.list"), 'w') as f:
                f.write('\n'.join(backbones))
        
        USalign.compute_qTMclust_metrics(
            success_dir=os.path.join(pipeline_dir, "cluster", "success"),
            gen_dir=os.path.join(pipeline_dir, "formatted_designs"),
            tm_thresh=self.config.metrics.tm_thres,
        )
    
    def run_motif_scaffolding_evaluation(
        self,
        input_dir: str,
        output_dir: str,
        metadata_file: str = None,
        motif_name: str = None
    ):
        """
        Run motif scaffolding evaluation pipeline.
        
        Assumes standardized input:
        - input_dir: Directory containing PDB files (real residues, not Poly-Ala)
        - metadata_file: Path to scaffold_info.csv with motif_placements
        - motif_name: Name of the motif (for MotifBench reference)
        
        Args:
            input_dir: Directory containing input PDB files
            output_dir: Directory to save evaluation results
            metadata_file: Path to scaffold_info.csv (optional)
            motif_name: Name of the motif (optional, for MotifBench)
            
        Returns:
            dict: Evaluation results dictionary
        """
        from evaluation import get_evaluator
        
        # Get evaluator for motif scaffolding task
        evaluator = get_evaluator("motif_scaffolding", self.config)
        
        # If motif_name provided, pass it to evaluator
        if motif_name and hasattr(evaluator, 'motif_name'):
            evaluator.motif_name = motif_name
        
        # Run evaluation
        results = evaluator.run_evaluation(
            input_dir=input_dir,
            output_dir=output_dir,
            metadata_file=metadata_file
        )
        
        return results
    
    def run_antibody_developability_evaluation(
        self,
        pipeline_dir: str,
        cdr_info_csv: str,
        output_csv: str,
        num_seeds: int = 4
    ):
        """
        Run antibody developability evaluation using DevelopabilityScorer.
        
        Args:
            pipeline_dir: Root directory of the pipeline (contains refold/af3_out)
            cdr_info_csv: Path to CSV file with CDR indices (must contain columns:
                         id, heavy_fv, light_fv (optional), h_cdr1_start, h_cdr1_end,
                         h_cdr2_start, h_cdr2_end, h_cdr3_start, h_cdr3_end,
                         l_cdr1_start, l_cdr1_end, l_cdr2_start, l_cdr2_end,
                         l_cdr3_start, l_cdr3_end)
            output_csv: Output CSV file path for developability metrics
            num_seeds: Number of seeds to process per antibody (default: 4)
        """
        import pandas as pd
        import numpy as np
        from collections import defaultdict
        import glob
        
        print("🔬 Starting Antibody Developability Evaluation...")
        
        # Load CDR info CSV
        if not os.path.exists(cdr_info_csv):
            raise FileNotFoundError(f"CDR info CSV not found: {cdr_info_csv}")
        
        df = pd.read_csv(cdr_info_csv)
        print(f"   Loaded {len(df)} antibodies from {cdr_info_csv}")
        
        # Initialize scorer
        scorer = DevelopabilityScorer()
        results = []
        
        # Find AF3 output directory
        af3_out_dir = os.path.join(pipeline_dir, "refold", "af3_out")
        if not os.path.exists(af3_out_dir):
            raise FileNotFoundError(f"AF3 output directory not found: {af3_out_dir}")
        
        print(f"   Searching for CIF files in {af3_out_dir}")
        
        for idx, row in df.iterrows():
            ab_id = row.get('id', f"antibody_{idx}")
            print(f"\n[{idx+1}/{len(df)}] Processing {ab_id}...")
            
            # Detect mode
            mode = scorer.detect_mode(row)
            print(f"   Mode: {mode}")
            
            # Find CIF files for this antibody
            # Search pattern: **/antibody_id/**/seed-*/model.cif or similar
            search_patterns = [
                os.path.join(af3_out_dir, f"**/{ab_id}/**/seed-*/model.cif"),
                os.path.join(af3_out_dir, f"**/{ab_id}/**/*.cif"),
                os.path.join(af3_out_dir, f"**/*{ab_id}*/**/seed-*/model.cif"),
                os.path.join(af3_out_dir, f"**/*{ab_id}*/**/*.cif"),
            ]
            
            files = []
            for pattern in search_patterns:
                found = glob.glob(pattern, recursive=True)
                if found:
                    files = sorted(found)
                    break
            
            if not files:
                print(f"   ⚠️  No CIF files found for {ab_id}")
                # Create default result
                default_result = {
                    'id': ab_id,
                    'Mode': mode,
                    'PSH': np.nan, 'PPC': np.nan, 'PNC': np.nan,
                    'Total_CDR_Length': np.nan
                }
                if mode == 'TAP':
                    default_result['SFvCSP'] = np.nan
                else:
                    default_result['CDR3_Length'] = np.nan
                    default_result['CDR3_Compactness'] = np.nan
                results.append(default_result)
                continue
            
            # Filter to seed files if available
            seed_files = [f for f in files if 'seed-' in f]
            if seed_files:
                files = seed_files[:num_seeds]
            else:
                files = files[:num_seeds]
            
            print(f"   Found {len(files)} CIF file(s)")
            
            # Process seeds
            ab_res = defaultdict(list)
            for cif_file in files:
                try:
                    m = scorer.calculate_metrics(cif_file, row)
                    if m:
                        for k, v in m.items():
                            if k != 'Mode':  # Mode is the same for all seeds
                                ab_res[k].append(v)
                except Exception as e:
                    print(f"   ⚠️  Error processing {os.path.basename(cif_file)}: {e}")
                    continue
            
            # Average across seeds
            entry = {'id': ab_id}
            if ab_res:
                entry['Mode'] = mode
                for k in ab_res.keys():
                    vals = ab_res[k]
                    if k == 'CDR3_Length':
                        # CDR3_Length should be the same across seeds, use first value
                        entry[k] = vals[0] if vals else np.nan
                    elif k.endswith('_Category'):
                        # Categories: use most common (mode)
                        if vals:
                            from collections import Counter
                            most_common = Counter(vals).most_common(1)[0][0]
                            entry[k] = most_common
                        else:
                            entry[k] = 'Unknown'
                    else:
                        entry[k] = np.nanmean(vals) if vals else np.nan
            else:
                # No valid results
                entry['Mode'] = mode
                entry.update({k: np.nan for k in ['PSH', 'PPC', 'PNC', 'Total_CDR_Length']})
                if mode == 'TAP':
                    entry['SFvCSP'] = np.nan
                else:
                    entry['CDR3_Length'] = np.nan
                    entry['CDR3_Compactness'] = np.nan
            
            results.append(entry)
            
            # Print summary
            psh_str = f"{entry.get('PSH', np.nan):.2f}" if not np.isnan(entry.get('PSH', np.nan)) else "NaN"
            ppc_str = f"{entry.get('PPC', np.nan):.0f}" if not np.isnan(entry.get('PPC', np.nan)) else "NaN"
            pnc_str = f"{entry.get('PNC', np.nan):.0f}" if not np.isnan(entry.get('PNC', np.nan)) else "NaN"
            
            if mode == 'TAP':
                sfvcsp_str = f"{entry.get('SFvCSP', np.nan):.2f}" if not np.isnan(entry.get('SFvCSP', np.nan)) else "NaN"
                print(f"   PSH: {psh_str}, PPC: {ppc_str}, PNC: {pnc_str}, SFvCSP: {sfvcsp_str}")
            else:
                cdr3_len_str = f"{entry.get('CDR3_Length', np.nan):.0f}" if not np.isnan(entry.get('CDR3_Length', np.nan)) else "NaN"
                compact_str = f"{entry.get('CDR3_Compactness', np.nan):.3f}" if not np.isnan(entry.get('CDR3_Compactness', np.nan)) else "NaN"
                print(f"   PSH: {psh_str}, PPC: {ppc_str}, PNC: {pnc_str}, CDR3_Length: {cdr3_len_str}, CDR3_Compactness: {compact_str}")
        
        # Save results
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_csv, index=False)
        print(f"\n✅ Developability metrics saved to {output_csv}")
        print(f"   Processed {len(results)} antibodies")
        
        # Print summary
        if len(result_df) > 0:
            print(f"\n📊 Summary:")
            tap_count = len(result_df[result_df['Mode'] == 'TAP'])
            tnp_count = len(result_df[result_df['Mode'] == 'TNP'])
            print(f"   TAP (mAb): {tap_count}")
            print(f"   TNP (Nanobody): {tnp_count}")
        
        return result_df