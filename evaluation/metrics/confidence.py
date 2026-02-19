import json
import pickle
from biopandas.pdb import PandasPdb
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import pdb, pdbx

def calculate_ipae_info(pae_matrix: np.ndarray, chain_indices: np.ndarray) -> dict:
    """
    Calculate iPAE (interface PAE) related information from complete PAE matrix.

    Args:
        pae_matrix (np.ndarray): N x N PAE matrix, where N is total number of residues.
        chain_indices (np.ndarray): Array of length N, indicating chain index for each residue (0, 1, 2...).

    Returns:
        dict: Dictionary containing iPAE statistics:
              - 'mean_ipae': Mean of all inter-chain PAE values.
              - 'min_ipae': Minimum of all inter-chain PAE values.
              - 'ipae_values': 1D array of all inter-chain PAE values.
              - 'ipae_mask': N x N boolean mask, True indicates inter-chain positions.
              - 'ipae_blocks': Dictionary storing iPAE submatrices for each chain pair.
    """
    num_residues = pae_matrix.shape[0]
    if num_residues != len(chain_indices):
        raise ValueError("PAE matrix dimensions do not match chain indices length.")

    # --- Core calculation: Use broadcasting to create inter-chain mask ---
    # Convert chain indices to column and row vectors
    chain_col = chain_indices[:, np.newaxis]
    chain_row = chain_indices[np.newaxis, :]
    
    # Mask is True when two residues have different chain indices
    ipae_mask = (chain_col != chain_row)
    
    # Extract all iPAE values using mask
    ipae_values = pae_matrix[ipae_mask]
    
    if ipae_values.size == 0:
        # If only one chain, no iPAE values
        return {
            'mean_ipae': np.nan, 'min_ipae': np.nan,
            'ipae_values': np.array([]), 'ipae_mask': ipae_mask,
            'ipae_blocks': {}
        }

    # Calculate key statistics
    mean_ipae = ipae_values.mean()
    min_ipae = ipae_values.min()

    # Extract iPAE submatrices for each chain pair
    ipae_blocks = {}
    unique_chains = np.unique(chain_indices)
    for i in range(len(unique_chains)):
        for j in range(i + 1, len(unique_chains)):
            chain_id_1 = unique_chains[i]
            chain_id_2 = unique_chains[j]
            
            mask_chain_1 = (chain_indices == chain_id_1)
            mask_chain_2 = (chain_indices == chain_id_2)
            
            # Extract A-B and B-A iPAE blocks
            block_ab = pae_matrix[mask_chain_1, :][:, mask_chain_2]
            block_ba = pae_matrix[mask_chain_2, :][:, mask_chain_1]

            ipae_blocks[f'chain_{chain_id_1}-chain_{chain_id_2}'] = block_ab
            ipae_blocks[f'chain_{chain_id_2}-chain_{chain_id_1}'] = block_ba


    return {
        'mean_ipae': mean_ipae,
        'min_ipae': min_ipae,
        'ipae_values': ipae_values,
        'ipae_mask': ipae_mask,
        'ipae_blocks': ipae_blocks
    }

def letter_to_number(char: str):
    """
    Convert a single English letter to its corresponding number in the alphabet (A=1, B=2, ...).

    Args:
        char: A single character string.

    Returns:
        If input is an English letter, returns integer between 1-26.
        Otherwise returns None.
    """
    # Check if input is a single character string
    if not isinstance(char, str) or len(char) != 1:
        return None
    
    # Convert to uppercase for case-insensitive comparison
    char_upper = char.upper()
    
    # Check if it's an English letter
    if 'A' <= char_upper <= 'Z':
        # Calculate using ASCII code
        return ord(char_upper) - ord('A') + 1
    else:
        return None

class Confidence:

    def __init__(self):
        pass
    
    @staticmethod
    def gather_af3_confidence(confidence_path: str, summary_confidence_path: str, pdbpath: str):
        with open(summary_confidence_path, "r") as f:
            summary_confidence = json.load(f)
        with open(confidence_path, "r") as f:
            confidence = json.load(f)
        
        iptm = summary_confidence['iptm']
        ipae_info = calculate_ipae_info(np.array(confidence['pae']), np.array(confidence['token_chain_ids']))
        ipae, min_ipae = ipae_info['mean_ipae'], ipae_info['min_ipae']
        # atom_array = pdb.PDBFile.read(pdbpath).get_structure(model=1, extra_fields=['b_factor'])
        chains_to_design ="B"
        chains_to_design = str(atom_array.chain_id[atom_array.b_factor != 0][0])
        ptm_binder = summary_confidence['chain_ptm'][np.unique(atom_array.chain_id).tolist().index(chains_to_design)]
        plddt = sum(confidence['atom_plddts']) / len(confidence['atom_plddts'])
        return plddt, ipae, min_ipae, iptm, ptm_binder
    
    @staticmethod
    def gather_chai1_confidence(cand: str, inverse_fold_path: str):
        token_asym_id = cand.token_asym_id.numpy()
        token_asym_id = token_asym_id[token_asym_id != 0]
        plddt = np.mean(cand.plddt.squeeze(0).numpy())
        pae = cand.pae.squeeze(0).numpy()
        ipae_info = calculate_ipae_info(pae, token_asym_id)
        ipae, min_ipae = ipae_info['mean_ipae'], ipae_info['min_ipae']
        iptm = cand.ranking_data[0].ptm_scores.interface_ptm.numpy()
        # trb = pickle.load(open(trb, 'rb'))
        atom_array = pdb.get_structure(pdb.PDBFile.read(inverse_fold_path), model=1, extra_fields=['b_factor'])
        chains_to_design = str(atom_array.chain_id[atom_array.b_factor == 0][0])
        binder_id = letter_to_number(chains_to_design)
        ptm_binder = cand.ranking_data[0].ptm_scores.per_chain_ptm[0, binder_id - 1].numpy()
        return plddt, ipae, min_ipae, iptm, ptm_binder
    
    @staticmethod
    def gather_esmfold_confidence(pdb_path, chain_id=None):
        """
        Extract pLDDT values from B-factor column of a PDB file using PandasPdb
        
        Parameters:
        -----------
        pdb_path : str
            Path to the PDB file
        chain_id : str, optional
            Chain ID to filter for. If None, returns data for all chains
            
        Returns:
        --------
        pandas.DataFrame or pandas.Series
            B-factor values (pLDDT) for the specified chain or all chains
        """
        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_path)
        
        # Get only CA atoms to avoid duplicate values per residue
        ca_atoms = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']
        
        if chain_id:
            # breakpoint()
            # Filter for specific chain
            chain_data = ca_atoms[ca_atoms['chain_id'] == chain_id]
            return chain_data['b_factor'].mean()
        else:
            # Return data for all chains
            return ca_atoms['b_factor'].mean()