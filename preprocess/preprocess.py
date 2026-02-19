import os
import shutil
import numpy as np
from pathlib import Path
from collections import Counter
import biotite.structure.io as io
from biotite.structure.io import pdbx, pdb
from biotite.structure import get_residue_starts

class Preprocess():

    def __init__(self, config):
        
        pass

    def format_output_rfantibody(self, input_dir: str, output_dir: str):

        os.makedirs(output_dir, exist_ok=True)
        count = Counter()
        for case_dir in Path(input_dir).iterdir():
            case_name = case_dir.name
            for pdbpath in case_dir.glob("*.pdb"):
                count[case_name] += 1
                output_path = os.path.join(output_dir, f"{case_name}-{count[case_name]}.pdb")
                atom_array = read_structure(str(pdbpath))
                atom_array.b_factor[atom_array.chain_id != 'H'] = 1.0
                io.save_structure(output_path, atom_array)
                print(f"Saved processed structure to {output_path}")
    
    def format_output_boltzgen(self, input_dir: str, output_dir: str):
        
        os.makedirs(output_dir, exist_ok=True)
        count = Counter()
        for case_dir in Path(input_dir).iterdir():
            case_name = case_dir.name
            for cif_path in case_dir.rglob("intermediate_designs/*.cif"):
                npz = str(cif_path).replace('cif', 'npz')
                mol_type = np.load(npz)['mol_type']
                ligand_mask = mol_type == 3
                count[case_name] += 1
                output_path = os.path.join(output_dir, f"{case_name}-{count[case_name]}.pdb")
                atom_array = read_structure(str(cif_path))
                condition_mask = atom_array.b_factor == 0
                atom_array.b_factor[condition_mask] = 1.0
                atom_array.b_factor[~condition_mask] = 0.0
                r_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
                for idx, (s ,e) in enumerate(zip(r_starts[:-1], r_starts[1:])):
                    atom_array.hetero[s:e] = ligand_mask[idx]
                io.save_structure(output_path, atom_array)
                print(f"Saved processed structure to {output_path}")

    def format_output_cif(self, input_dir: str, output_dir: str):
        
        os.makedirs(output_dir, exist_ok=True)
        count = Counter()
        for case_dir in Path(input_dir).iterdir():
            case_name = case_dir.name
            for cif_path in case_dir.rglob("*.cif"):
                count[case_name] += 1
                output_path = os.path.join(output_dir, f"{case_name}-{count[case_name]}.cif")
                shutil.copy(cif_path, output_path)
                print(f"Copied CIF file to {output_path}")

    def format_output_pdb(self, input_dir: str, output_dir: str):
        """Collect PDB/CIF from design_dir: one subdir per target, each with .pdb or .cif design outputs."""
        os.makedirs(output_dir, exist_ok=True)
        count = Counter()
        for case_dir in Path(input_dir).iterdir():
            if not case_dir.is_dir():
                continue
            case_name = case_dir.name
            for cif_path in case_dir.rglob("*.cif"):
                count[case_name] += 1
                output_path = os.path.join(output_dir, f"{case_name.lower()}-{count[case_name]}.pdb")
                atom_array = read_structure(str(cif_path))
                io.save_structure(output_path, atom_array)
                print(f"Converted CIF to PDB and saved to {output_path}")
            for pdb_path in case_dir.rglob("*.pdb"):
                count[case_name] += 1
                output_path = os.path.join(output_dir, f"{case_name.lower()}-{count[case_name]}.pdb")
                atom_array = read_structure(str(pdb_path))
                io.save_structure(output_path, atom_array)
                print(f"Copied PDB to {output_path}")
                
    # need pdb
    def format_output_for_foldseek(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for cif_path in Path(input_dir).rglob("*.cif"):
            convert_cif_to_pdb_dir(cif_path=str(cif_path), struct_output_dir=output_dir)

    def format_output_ligand(self, input_dir: str, output_dir: str):

        os.makedirs(output_dir, exist_ok=True)
        count = Counter()
        for case_dir in Path(input_dir).iterdir():
            case_name = case_dir.name
            for cif_path in case_dir.rglob("*.cif"):
                count[case_name] += 1
                output_path = os.path.join(output_dir, f"{case_name}-{count[case_name]}.cif")
                # trb_path = os.path.join(os.path.dirname(cif_path), "traceback.pkl")
                # output_trb_path = os.path.join(output_dir, f"{case_name}-{count[case_name]}.pkl")
                cif_file = pdbx.CIFFile.read(cif_path)
                atom_array = pdbx.get_structure(cif_file, model=1, extra_fields=['b_factor', 'occupancy'])
                atom_array.atom_name[atom_array.hetero] = np.char.add(['C']*sum(atom_array.hetero), np.array(range(sum(atom_array.hetero)), dtype=np.str_))
                io.save_structure(output_path, atom_array)
                # shutil.copy(trb_path, output_trb_path)
    
    def format_output_ligand_for_protein_binding_ligand_evaluation(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for cif_path in Path(input_dir).rglob("*.cif"):
            output_path = os.path.join(output_dir, cif_path.name)  
            cif_file = pdbx.CIFFile.read(cif_path)
            block = cif_file.block
            atom_site = block.get("atom_site")
            atom_site["occupancy"] = pdbx.CIFColumn(pdbx.CIFData(["1.0" for _ in range(len(atom_site['group_PDB']))]))
            atom_site['B_iso_or_equiv'] = pdbx.CIFColumn(pdbx.CIFData(["1.0" for _ in range(len(atom_site['group_PDB']))]))
            cif_file.write(output_path)
    
    def rna_preprocess(self, input_dir: str, output_dir: str):

        count = Counter()
        for case_dir in Path(input_dir).iterdir():
            case = case_dir.name
            for cif_path in case_dir.rglob('*.cif'):
                count[case] += 1
                cif_file = pdbx.CIFFile.read(cif_path)
                arr = pdbx.get_structure(cif_file, model=1)
                arr.atom_name[(arr.atom_name == 'N') & (arr.res_name == 'C')] = "N9" # concerned gRNAde
                pdb_file = pdb.PDBFile()
                pdb_file.set_structure(arr)
                pdb_file.write(os.path.join(output_dir, f'{case}_{count[case]}.pdb'))
                shutil.copy(os.path.join(os.path.dirname(cif_path), "traceback.pkl"), os.path.join(output_dir, f"{case}_{count[case]}.pkl"))

    def make_ligandmpnn_input(self):

        pass
    
    def make_af3_input(self):

        pass
    
    def make_chai1_input(self):

        pass


import gemmi
import biotite.structure.io.pdbx as pdbx
def convert_cif_to_pdb(cif_path: str, struct_output_path: str) -> str:
    """Convert mmCIF file to PDB format with B_iso_or_equiv support"""
    pdb_path = struct_output_path
    
    if not os.path.exists(pdb_path):
        try:
            # First, check if B_iso_or_equiv exists in the CIF file
            cif_file = pdbx.CIFFile.read(cif_path)
            block = cif_file.block
            atom_site = block.get("atom_site")
            
            # Check if B_iso_or_equiv attribute exists, if not add it
            need_temp_file = False
            if "B_iso_or_equiv" not in atom_site:
                print(f"Adding B_iso_or_equiv to {cif_path}")
                atom_site["B_iso_or_equiv"] = pdbx.CIFColumn(pdbx.CIFData(["1.0" for _ in range(len(atom_site['group_PDB']))]))
                need_temp_file = True
            else:
                print(f"B_iso_or_equiv already exists in {cif_path}")
            
            # Create temporary CIF file with B_iso_or_equiv if needed
            temp_cif_path = cif_path
            if need_temp_file:
                temp_cif_path = os.path.join(os.path.dirname(pdb_path), f"temp_{os.path.basename(cif_path)}")
                cif_file.write(temp_cif_path)
            
            # Use gemmi Python API to convert mmCIF to PDB
            structure = gemmi.read_structure(temp_cif_path)
            structure.write_pdb(pdb_path)
            
            # Clean up temporary file if created
            if temp_cif_path != cif_path and os.path.exists(temp_cif_path):
                os.remove(temp_cif_path)
                
            print(f"Successfully converted {cif_path} to {pdb_path}")
            
        except Exception as e:
            print(f"Error converting {cif_path} to PDB using gemmi API: {e}")
            # Fallback to command line if API fails
            print(f"Falling back to command line conversion: gemmi convert {cif_path} {pdb_path}")
            os.system(f"gemmi convert {cif_path} {pdb_path}")
    else:
        print(f"PDB file already exists: {pdb_path}")
    
    return pdb_path

def convert_cif_to_pdb_dir(cif_path: str, struct_output_dir: str) -> str:
    """Convert mmCIF file to PDB format with B_iso_or_equiv support"""
    filename = os.path.basename(os.path.splitext(cif_path)[0]) + '.pdb'
    pdb_path = os.path.join(struct_output_dir, filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(struct_output_dir, exist_ok=True)
    
    if not os.path.exists(pdb_path):
        try:
            # First, check if B_iso_or_equiv exists in the CIF file
            cif_file = pdbx.CIFFile.read(cif_path)
            block = cif_file.block
            atom_site = block.get("atom_site")
            
            # Check if B_iso_or_equiv attribute exists, if not add it
            need_temp_file = False
            if "B_iso_or_equiv" not in atom_site:
                print(f"Adding B_iso_or_equiv to {cif_path}")
                atom_site["B_iso_or_equiv"] = pdbx.CIFColumn(pdbx.CIFData(["1.0" for _ in range(len(atom_site['group_PDB']))]))
                need_temp_file = True
            else:
                print(f"B_iso_or_equiv already exists in {cif_path}")
            
            # Create temporary CIF file with B_iso_or_equiv if needed
            temp_cif_path = cif_path
            if need_temp_file:
                temp_cif_path = os.path.join(struct_output_dir, f"temp_{os.path.basename(cif_path)}")
                cif_file.write(temp_cif_path)
            
            # Use gemmi Python API to convert mmCIF to PDB
            structure = gemmi.read_structure(temp_cif_path)
            structure.write_pdb(pdb_path)
            
            # Clean up temporary file if created
            if temp_cif_path != cif_path and os.path.exists(temp_cif_path):
                os.remove(temp_cif_path)
                
            print(f"Successfully converted {cif_path} to {pdb_path}")
            
        except Exception as e:
            print(f"Error converting {cif_path} to PDB using gemmi API: {e}")
            # Fallback to command line if API fails
            print(f"Falling back to command line conversion: gemmi convert {cif_path} {pdb_path}")
            os.system(f"gemmi convert {cif_path} {pdb_path}")
    else:
        print(f"PDB file already exists: {pdb_path}")
    
    return pdb_path

def read_structure(fpath: str):
    """Read structure from a file (PDB or mmCIF)"""
    ext = os.path.splitext(fpath)[1].lower()
    if ext == '.pdb':
        return pdb.PDBFile.read(fpath).get_structure(model=1, extra_fields=["b_factor"])
    elif ext in ['.cif', '.mmcif']:
        cif_file = pdbx.CIFFile.read(fpath)
        return pdbx.get_structure(cif_file, model=1, extra_fields=["b_factor"])
    else:
        raise ValueError(f"Unsupported file extension: {ext}")