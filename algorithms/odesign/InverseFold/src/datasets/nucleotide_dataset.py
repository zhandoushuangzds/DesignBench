# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch.utils.data as data


class NucleotideDataset(data.Dataset):
    """
    从 .jsonl 读取核酸数据，支持 RNA / DNA。
    每行一个样本（dict），输出与 featurize_UniIF().featurize 兼容的“原始样本”：

      {
        'title': str,
        'type': 'rna' | 'dna',
        'seq': [str],                    # 单链 -> list包一层（与多链接口统一）
        'P':   [np.ndarray(L,3)],
        'O5':  [np.ndarray(L,3)],
        'C5':  [np.ndarray(L,3)],
        'C4':  [np.ndarray(L,3)],
        'C3':  [np.ndarray(L,3)],
        'O3':  [np.ndarray(L,3)],
        'chain_mask': [np.ones(L)],
        'chain_encoding': [np.ndarray(L)],  # 来自 entry['chain_idxs'] + 1；若无则全 1
      }

    文件名选择（按顺序优先）：
      1) __init__(jsonl_name=...) 指定的文件
      2) <split>_data.jsonl
      3) <split>.jsonl

    split 支持 'valid'→'val'
    """

    def __init__(self,
                 path: str,
                 split: str = 'train',
                 max_length: int = 500,
                 jsonl_name: Optional[str] = None,
                 kind: str = 'rna'):
        super().__init__()
        split = 'val' if split in ('valid', 'validation') else split
        assert split in ('train', 'val', 'test'), f"split must in train/val/test: {split}"

        kind = str(kind).lower()
        assert kind in ('rna', 'dna'), f"kind must be 'rna' or 'dna', got {kind}"

        self.path = path
        self.split = split
        self.max_length = max_length
        self.kind = kind
        self.atom_lst = ['P', 'O5', 'C5', 'C4', 'C3', 'O3','N']

        # 只读 jsonl
        cand = []
        if jsonl_name:
            cand.append(os.path.join(path, jsonl_name))
        cand += [
            # os.path.join(path, "test_data_new.jsonl"),
            os.path.join(path, f"{split}_data_new.jsonl"),
            os.path.join(path, f"{split}_data.jsonl"),
            os.path.join(path, f"{split}.jsonl"),
        ]
        for p in cand:
            if os.path.exists(p):
                self.jsonl_file = p
                break
        else:
            raise FileNotFoundError(f"jsonl 未找到，已尝试：{cand}")

        self.data: List[Dict[str, Any]] = self._load_jsonl(self.jsonl_file)


    def _load_jsonl(self, fn: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        num_base = 0

        is_rna = (self.kind == 'rna')
        allowed = set('AUGC') if is_rna else set('ATGC')

        with open(fn, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {self.kind.upper()}(JSONL): {os.path.basename(fn)}"):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue

                seq_raw = entry.get("seq") or ""
                if not isinstance(seq_raw, str):
                    # 兼容：若是 list 且第一个元素为字符串
                    if isinstance(seq_raw, list) and seq_raw and isinstance(seq_raw[0], str):
                        seq_raw = seq_raw[0]
                    else:
                        continue

                seq_main = seq_raw.upper()
                # RNA: T->U；DNA: U->T
                seq_main = seq_main.replace('T', 'U') if is_rna else seq_main.replace('U', 'T')
                if not seq_main or not set(seq_main).issubset(allowed):
                    continue
                L = len(seq_main)

                chain_idxs = entry.get("chain_idxs", None)
                if isinstance(chain_idxs, (list, np.ndarray)) and len(chain_idxs) == L:
                    chain_enc = np.asarray(chain_idxs, dtype=np.int32) + 1
                else:
                    chain_enc = np.ones(L, dtype=np.int32)

                coords = entry.get("coords", {}) or {}
                title = self._make_title(entry, default_index=len(items))

                item = self._coords_to_item(
                    coords=coords,
                    title=title,
                    seq=seq_main,
                    chain_enc=chain_enc
                )
                if item is not None:
                    items.append(item)
                    num_base += 1

        self.num_base = num_base
        return items

    def _make_title(self, entry: Dict[str, Any], default_index: int) -> str:
        t = entry.get("name", "") or entry.get("title", "")
        if t:
            return t

        pdb_id = entry.get("pdb_id", "")
        asm = entry.get("assembly_id", None)
        row = entry.get("row_index", None)

        chain = None
        resolved = entry.get("resolved_ids", {})
        if isinstance(resolved, dict):
            chain = resolved.get("auth") or resolved.get("label")
        if not chain:
            chain = entry.get("requested_chain_id", None)

        parts = []
        if pdb_id: parts.append(str(pdb_id))
        if asm is not None: parts.append(f"A{asm}")
        if row is not None: parts.append(f"row{row}")
        if chain: parts.append(f"chain{chain}")

        if parts:
            return "_".join(parts)
        else:
            return f"{self.kind.upper()}_{self.split}_{default_index}"

    # ------------------------------
    # coords -> 样本（单链）
    # ------------------------------
    def _coords_to_item(self, coords: Dict[str, Any], title: str, seq: str, chain_enc: np.ndarray) -> Optional[Dict[str, Any]]:
        if not seq:
            return None
        s = seq.upper()
        L_ = len(s)

        def pick(atom_no_prime: str) -> np.ndarray:
            """
            从 coords 中取出对应原子列：
            P -> 'P'；其它加撇号：'O5' -> "O5'"
            空/长度不符 -> 用 NaN(L,3)
            """
            key = atom_no_prime if atom_no_prime == 'P' or atom_no_prime == 'N' else (atom_no_prime + "'")
            arr = (coords or {}).get(key)
            if arr is None:
                return np.full((L_, 3), np.nan, dtype=np.float32)
            out = np.array(
                [([np.nan, np.nan, np.nan] if (a is None or a is False) else a) for a in arr],
                dtype=np.float32
            )
            if out.shape != (L_, 3):
                return np.full((L_, 3), np.nan, dtype=np.float32)
            return out

        # 取出所有原子的坐标
        coords_lst = {atom: pick(atom) for atom in self.atom_lst}

        if len(chain_enc) != L_:
            return None
        enc = chain_enc.astype(np.float32)

        return {
            "title": title,
            "type": self.kind,  # 'rna' or 'dna'
            "seq": [s],
            "chain_mask":     [np.ones(L_, dtype=np.float32)],
            "chain_encoding": [enc],
            **coords_lst,
        }



    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        # print(f"Item: {item}")
        seq = item["seq"][0]
        msk = item["chain_mask"][0]
        enc = item["chain_encoding"][0]

        L = len(seq)
        coords_slice = {}
        if L > self.max_length:
            import random
            t0 = random.randint(0, L - self.max_length)
            t1 = t0 + self.max_length
            seq = seq[t0:t1]
            msk, enc = msk[t0:t1], enc[t0:t1]
            for atom in self.atom_lst:
                coords_slice[atom] = item[atom][t0:t1]
        else:
            for atom in self.atom_lst:
                coords_slice[atom] = item[atom]

        return {
            "title": item["title"],
            "type":  self.kind,
            "seq": [seq],
            "chain_mask":     [msk],
            "chain_encoding": [enc],
            **coords_slice,
        }



# # -*- coding: utf-8 -*-
# import os
# import json
# import numpy as np
# from typing import List, Dict, Any, Optional
# from tqdm import tqdm
# import torch.utils.data as data


# class NucleotideDataset(data.Dataset):
#     """
#     从 .jsonl 读取核酸数据，支持 RNA / DNA。
#     每行一个样本（dict），输出与 featurize_UniIF().featurize 兼容的“原始样本”：

#       {
#         'title': str,
#         'type': 'rna' | 'dna',
#         'seq': [str],                    # 单链 -> list包一层（与多链接口统一）
#         'P':   [np.ndarray(L,3)],
#         'O5':  [np.ndarray(L,3)],
#         'C5':  [np.ndarray(L,3)],
#         'C4':  [np.ndarray(L,3)],
#         'C3':  [np.ndarray(L,3)],
#         'O3':  [np.ndarray(L,3)],
#         'chain_mask': [np.ones(L)],
#         'chain_encoding': [np.ndarray(L)],  # 来自 entry['chain_idxs'] + 1；若无则全 1
#       }

#     文件名选择（按顺序优先）：
#       1) __init__(jsonl_name=...) 指定的文件
#       2) <split>_data.jsonl
#       3) <split>.jsonl

#     split 支持 'valid'→'val'
#     """

#     def __init__(self,
#                  path: str,
#                  split: str = 'train',
#                  max_length: int = 500,
#                  jsonl_name: Optional[str] = None,
#                  kind: str = 'rna',
#                  min_seq_len: Optional[int] = 75,   # NEW: 最小长度（含）
#                  max_seq_len: Optional[int] = 150):  # NEW: 最大长度（含）
#         super().__init__()
#         split = 'val' if split in ('valid', 'validation') else split
#         assert split in ('train', 'val', 'test'), f"split must in train/val/test: {split}"

#         kind = str(kind).lower()
#         assert kind in ('rna', 'dna'), f"kind must be 'rna' or 'dna', got {kind}"

#         self.path = path
#         self.split = split
#         self.max_length = max_length
#         self.kind = kind
#         self.atom_lst = ['P', 'O5', 'C5', 'C4', 'C3', 'O3','N']

#         # NEW: 保存长度筛选配置
#         self.min_seq_len = min_seq_len
#         self.max_seq_len = max_seq_len

#         # 只读 jsonl
#         cand = []
#         if jsonl_name:
#             cand.append(os.path.join(path, jsonl_name))
#         cand += [
#             os.path.join(path, f"{split}_data_new.jsonl"),
#             os.path.join(path, f"{split}_data.jsonl"),
#             os.path.join(path, f"{split}.jsonl"),
#         ]
#         for p in cand:
#             if os.path.exists(p):
#                 self.jsonl_file = p
#                 break
#         else:
#             raise FileNotFoundError(f"jsonl 未找到，已尝试：{cand}")

#         self.data: List[Dict[str, Any]] = self._load_jsonl(self.jsonl_file)


#     def _load_jsonl(self, fn: str) -> List[Dict[str, Any]]:
#         items: List[Dict[str, Any]] = []
#         num_base = 0

#         is_rna = (self.kind == 'rna')
#         allowed = set('AUGC') if is_rna else set('ATGC')

#         # NEW: 统计筛选情况（可选）
#         dropped_by_length = 0

#         with open(fn, "r", encoding="utf-8") as f:
#             for line in tqdm(f, desc=f"Loading {self.kind.upper()}(JSONL): {os.path.basename(fn)}"):
#                 line = line.strip()
#                 if not line:
#                     continue
#                 try:
#                     entry = json.loads(line)
#                 except Exception:
#                     continue

#                 seq_raw = entry.get("seq") or ""
#                 if not isinstance(seq_raw, str):
#                     # 兼容：若是 list 且第一个元素为字符串
#                     if isinstance(seq_raw, list) and seq_raw and isinstance(seq_raw[0], str):
#                         seq_raw = seq_raw[0]
#                     else:
#                         continue

#                 seq_main = seq_raw.upper()
#                 # RNA: T->U；DNA: U->T
#                 seq_main = seq_main.replace('T', 'U') if is_rna else seq_main.replace('U', 'T')
#                 if not seq_main or not set(seq_main).issubset(allowed):
#                     continue
#                 L = len(seq_main)

#                 # NEW: 长度筛选（闭区间）
#                 if (self.min_seq_len is not None and L < self.min_seq_len) or \
#                    (self.max_seq_len is not None and L > self.max_seq_len):
#                     dropped_by_length += 1
#                     continue

#                 chain_idxs = entry.get("chain_idxs", None)
#                 if isinstance(chain_idxs, (list, np.ndarray)) and len(chain_idxs) == L:
#                     chain_enc = np.asarray(chain_idxs, dtype=np.int32) + 1
#                 else:
#                     chain_enc = np.ones(L, dtype=np.int32)

#                 coords = entry.get("coords", {}) or {}
#                 title = self._make_title(entry, default_index=len(items))

#                 item = self._coords_to_item(
#                     coords=coords,
#                     title=title,
#                     seq=seq_main,
#                     chain_enc=chain_enc
#                 )
#                 if item is not None:
#                     items.append(item)
#                     num_base += 1

#         self.num_base = num_base
#         # 可选：打印一下被长度筛掉的数量
#         if self.min_seq_len is not None or self.max_seq_len is not None:
#             print(f"[INFO] Length filter kept {len(items)} items, dropped_by_length={dropped_by_length}. "
#                   f"Range=[{self.min_seq_len},{self.max_seq_len}]")
#         return items

#     def _make_title(self, entry: Dict[str, Any], default_index: int) -> str:
#         t = entry.get("name", "") or entry.get("title", "")
#         if t:
#             return t

#         pdb_id = entry.get("pdb_id", "")
#         asm = entry.get("assembly_id", None)
#         row = entry.get("row_index", None)

#         chain = None
#         resolved = entry.get("resolved_ids", {})
#         if isinstance(resolved, dict):
#             chain = resolved.get("auth") or resolved.get("label")
#         if not chain:
#             chain = entry.get("requested_chain_id", None)

#         parts = []
#         if pdb_id: parts.append(str(pdb_id))
#         if asm is not None: parts.append(f"A{asm}")
#         if row is not None: parts.append(f"row{row}")
#         if chain: parts.append(f"chain{chain}")

#         if parts:
#             return "_".join(parts)
#         else:
#             return f"{self.kind.upper()}_{self.split}_{default_index}"

#     # ------------------------------
#     # coords -> 样本（单链）
#     # ------------------------------
#     def _coords_to_item(self, coords: Dict[str, Any], title: str, seq: str, chain_enc: np.ndarray) -> Optional[Dict[str, Any]]:
#         if not seq:
#             return None
#         s = seq.upper()
#         L_ = len(s)

#         def pick(atom_no_prime: str) -> np.ndarray:
#             """
#             从 coords 中取出对应原子列：
#             P -> 'P'；其它加撇号：'O5' -> "O5'"
#             空/长度不符 -> 用 NaN(L,3)
#             """
#             key = atom_no_prime if atom_no_prime == 'P' or atom_no_prime == 'N' else (atom_no_prime + "'")
#             arr = (coords or {}).get(key)
#             if arr is None:
#                 return np.full((L_, 3), np.nan, dtype=np.float32)
#             out = np.array(
#                 [([np.nan, np.nan, np.nan] if (a is None or a is False) else a) for a in arr],
#                 dtype=np.float32
#             )
#             if out.shape != (L_, 3):
#                 return np.full((L_, 3), np.nan, dtype=np.float32)
#             return out

#         # 取出所有原子的坐标
#         coords_lst = {atom: pick(atom) for atom in self.atom_lst}

#         if len(chain_enc) != L_:
#             return None
#         enc = chain_enc.astype(np.float32)

#         return {
#             "title": title,
#             "type": self.kind,  # 'rna' or 'dna'
#             "seq": [s],
#             "chain_mask":     [np.ones(L_, dtype=np.float32)],
#             "chain_encoding": [enc],
#             **coords_lst,
#         }

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         item = self.data[idx]
#         seq = item["seq"][0]
#         msk = item["chain_mask"][0]
#         enc = item["chain_encoding"][0]

#         L = len(seq)
#         coords_slice = {}
#         if L > self.max_length:
#             # 注意：如果你希望严格不截断，把 max_length 设成 >= max_seq_len
#             import random
#             t0 = random.randint(0, L - self.max_length)
#             t1 = t0 + self.max_length
#             seq = seq[t0:t1]
#             msk, enc = msk[t0:t1], enc[t0:t1]
#             for atom in self.atom_lst:
#                 coords_slice[atom] = item[atom][t0:t1]
#         else:
#             for atom in self.atom_lst:
#                 coords_slice[atom] = item[atom]

#         return {
#             "title": item["title"],
#             "type":  self.kind,
#             "seq": [seq],
#             "chain_mask":     [msk],
#             "chain_encoding": [enc],
#             **coords_slice,
#         }
