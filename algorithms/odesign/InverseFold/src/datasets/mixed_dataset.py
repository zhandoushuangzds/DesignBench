# -*- coding: utf-8 -*-
# src/datasets/mixed_dataset.py
import os.path as osp
from typing import Optional
from torch.utils.data import Dataset, ConcatDataset

from .cath_dataset import CATHDataset
from .nucleotide_dataset import NucleotideDataset

class MixedDataset(Dataset):
    """
    把 Protein(CATH) + RNA + DNA 三个数据集合并为一个数据集。
    不改变样本结构（单条样本仍是“原始样本”，等 featurizer 处理）。
    """
    def __init__(self,
                 protein_path: str,
                 rna_path: str,
                 dna_path: str,
                 split: str = "train",
                 max_length: int = 500,
                 cath_version: str = "4.3",
                 rna_jsonl_name: Optional[str] = None,
                 dna_jsonl_name: Optional[str] = None):
        super().__init__()

        ds_protein = CATHDataset(
            path=protein_path,
            split=split,
            max_length=max_length,
            version=cath_version,
        )
        ds_rna = NucleotideDataset(
            path=rna_path,
            split=split,
            max_length=max_length,
            jsonl_name=rna_jsonl_name,
            kind='rna',
        )
        ds_dna = NucleotideDataset(
            path=dna_path,
            split=split,
            max_length=max_length,
            jsonl_name=dna_jsonl_name,
            kind='dna',
        )

        self._concat = ConcatDataset([ds_protein, ds_rna, ds_dna])

    def __len__(self):
        return len(self._concat)

    def __getitem__(self, idx):
        return self._concat[idx]
