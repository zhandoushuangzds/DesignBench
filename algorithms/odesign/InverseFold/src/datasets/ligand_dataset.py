# # -*- coding: utf-8 -*-
# import os
# import json
# from typing import List, Dict, Any, Optional
# from tqdm import tqdm
# import numpy as np
# import torch.utils.data as data
# import re

# # ---- 全元素周期表（1–118，IUPAC 正式符号）----
# _ALL_ELEMS = {
#     # 1–10
#     "H","He","Li","Be","B","C","N","O","F","Ne",
#     # 11–18
#     "Na","Mg","Al","Si","P","S","Cl","Ar",
#     # 19–36
#     "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
#     # 37–54
#     "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
#     # 55–86
#     "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
#     "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
#     # 87–118
#     "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
#     "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
# }

# # 常见别名/写法纠正（全大写/拼写误差/元素缩写）
# _ALIAS_MAP = {
#     # 卤素 & 常见两字符
#     "CL": "Cl", "BR": "Br", "SI": "Si", "AL": "Al", "SE": "Se",
#     "ZN": "Zn", "FE": "Fe", "MG": "Mg", "NA": "Na", "CA": "Ca",
#     "MN": "Mn", "CO": "Co", "NI": "Ni", "CU": "Cu", "AG": "Ag",
#     "CD": "Cd", "SN": "Sn", "SB": "Sb", "TE": "Te", "XE": "Xe",
#     "CS": "Cs", "BA": "Ba", "LA": "La", "CE": "Ce", "PT": "Pt",
#     "AU": "Au", "HG": "Hg", "PB": "Pb", "BI": "Bi", "IR": "Ir",
#     "OS": "Os", "PD": "Pd", "RH": "Rh", "RU": "Ru", "MO": "Mo",
#     "TI": "Ti", "ZR": "Zr", "NB": "Nb", "TA": "Ta", "RE": "Re",
#     "GA": "Ga", "GE": "Ge", "AS": "As", "KR": "Kr", "RB": "Rb",
#     "SR": "Sr", "YB": "Yb", "LU": "Lu", "HF": "Hf", "W": "W",   # W 保持
#     "I": "I", "B": "B", "C": "C", "N": "N", "O": "O", "F": "F",
#     # 近期命名元素
#     "OG": "Og", "TS": "Ts", "LV": "Lv", "MC": "Mc", "FL": "Fl", "NH": "Nh", "CN": "Cn",
#     # 常见“写错大小写”的单字符
#     "K": "K", "P": "P", "S": "S", "V": "V"
# }

# # 去除尾随数字用，例如 "C1" -> "C"；也处理如 "CL1" -> "CL"
# _TAIL_NUM = re.compile(r"^([A-Za-z]{1,3})(\d+)$")

# def _normalize_element(sym: str) -> str:
#     """
#     规范化元素符号：
#       - 去掉尾随数字（如 C1 -> C）
#       - D/T -> H（氘/氚）
#       - 大小写修正、常见别名映射
#       - 不识别时返回首字母大写（并交给后续UNK处理）
#     """
#     if sym is None:
#         return "X"

#     s = str(sym).strip()
#     if not s:
#         return "X"

#     # 去尾随数字: "CL1" -> "CL"
#     m = _TAIL_NUM.match(s)
#     if m:
#         s = m.group(1)

#     # 先统一成大写，便于别名表查找
#     su = s.upper()

#     # 同位素：氘/氚 -> H
#     if su in ("D", "T"):
#         return "H"

#     # 直接别名表命中
#     if su in _ALIAS_MAP:
#         fixed = _ALIAS_MAP[su]
#         if fixed in _ALL_ELEMS:
#             return fixed

#     # 常规大小写规范：首字母大写，其余小写
#     sn = su[0] + su[1:].lower() if len(su) > 1 else su
#     if sn in _ALL_ELEMS:
#         return sn

#     # 有些来源给出全小写，例如 'br'
#     sl = s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()
#     if sl in _ALL_ELEMS:
#         return sl

#     # 实在不认识，返回首字母大写版本；后续可映射到 UNK
#     return (s[0].upper() + s[1:].lower()) if len(s) > 1 else s.upper()


# class LigandDataset(data.Dataset):
#     """
#     JSONL -> 配体样本（坐标 -> 原子类型）。
#     只做：
#       - 元素规范化（覆盖 1–118，全别名修正）
#       - 严格长度校验：len(elements) == len(coords) 且 coords.shape == (N,3)
#       - 可选超长裁剪（默认关闭）
#     不做：
#       - token/ID 映射（放到 featurizer）
#       - 默认不拼 seq 字符串
#     """

#     def __init__(self,
#                  path: str,
#                  split: str = "train",
#                  jsonl_name: Optional[str] = None,
#                  max_atoms: int = 0,
#                  drop_invalid: bool = True,
#                  keep_meta: bool = True,
#                  make_seq_string: bool = False):
#         super().__init__()
#         split = "val" if split in ("valid", "validation") else split
#         assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"

#         self.path = path
#         self.split = split
#         self.max_atoms = int(max_atoms)
#         self.drop_invalid = bool(drop_invalid)
#         self.keep_meta = bool(keep_meta)
#         self.make_seq_string = bool(make_seq_string)

#         cand = []
#         if jsonl_name:
#             cand.append(os.path.join(path, jsonl_name))
#         cand += [
#             os.path.join(path, "ligand_samples.jsonl"),
#             os.path.join(path, f"{split}_data.jsonl"),
#             os.path.join(path, f"{split}.jsonl"),
#             os.path.join(path, "ligand_samples.jsonl"),
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
#         with open(fn, "r", encoding="utf-8") as f:
#             for line in tqdm(f, desc=f"Loading Ligands(JSONL): {os.path.basename(fn)}"):
#                 line = line.strip()
#                 if not line:
#                     continue
#                 try:
#                     entry = json.loads(line)
#                 except Exception:
#                     if self.drop_invalid:
#                         continue
#                     entry = {}

#                 it = self._entry_to_item(entry)
#                 if it is not None:
#                     items.append(it)
#         return items

#     def _entry_to_item(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#         title = str(entry.get("title") or entry.get("name") or
#                     f"LIG_{self.split}_{len(getattr(self, 'data', []))}")

#         elements = entry.get("elements", None)
#         coords = entry.get("coords", None)

#         if not isinstance(elements, list) or not isinstance(coords, list):
#             return None if self.drop_invalid else None

#         arr = np.asarray(coords, dtype=np.float32)
#         if arr.ndim != 2 or arr.shape[1] != 3:
#             return None if self.drop_invalid else None

#         N = arr.shape[0]
#         elems_norm = [_normalize_element(e) for e in elements]
#         if len(elems_norm) != N or N < 1:
#             return None if self.drop_invalid else None

#         # 可选裁剪（一般不裁剪小分子）
#         if self.max_atoms > 0 and N > self.max_atoms:
#             idx = np.arange(self.max_atoms, dtype=np.int64)
#             arr = arr[idx]
#             elems_norm = [elems_norm[i] for i in idx]
#             N = self.max_atoms

#         atom_names = entry.get("atoms", None)
#         if isinstance(atom_names, list) and len(atom_names) == N:
#             atom_names_out = [str(a) for a in atom_names]
#         else:
#             atom_names_out = [f"{elems_norm[i]}{i+1}" for i in range(N)]

#         chain_mask = entry.get("chain_mask", None)
#         if isinstance(chain_mask, list) and len(chain_mask) == N:
#             mask_np = np.array(chain_mask, dtype=np.float32)
#         else:
#             mask_np = np.ones((N,), dtype=np.float32)

#         chain_encoding = entry.get("chain_encoding", None)
#         if isinstance(chain_encoding, list) and len(chain_encoding) == N:
#             enc_np = np.array(chain_encoding, dtype=np.float32)
#         else:
#             enc_np = np.ones((N,), dtype=np.float32)

#         out: Dict[str, Any] = {
#             "title": title,
#             "type": "ligand",
#             "elements": elems_norm,              # 监督标签（后续在 featurizer 里映射到ID/UNK）
#             "coords": arr.astype(np.float32),    # (N,3)
#             "atom_names": atom_names_out,
#             "chain_mask": [mask_np],
#             "chain_encoding": [enc_np],
#             "meta": {}
#         }

#         if self.keep_meta:
#             for k in ("smiles", "comp_id", "pdb_id", "chain_id", "res_id", "type"):
#                 if k in entry:
#                     out["meta"][k] = entry[k]
#             out["meta"]["N"] = int(N)

#         if self.make_seq_string:
#             out["seq"] = [" ".join(elems_norm)]  # 仅占位，默认关闭

#         return out

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         return self.data[idx]

# -*- coding: utf-8 -*-
# src/datasets/ligand_dataset.py  （你原来这段 Dataset 放哪就替换哪）

import os
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np
import torch.utils.data as data
import re

# ---- 全元素周期表（1–118，IUPAC 正式符号）----
_ALL_ELEMS = {
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
}
_ALIAS_MAP = {
    "CL": "Cl", "BR": "Br", "SI": "Si", "AL": "Al", "SE": "Se",
    "ZN": "Zn", "FE": "Fe", "MG": "Mg", "NA": "Na", "CA": "Ca",
    "MN": "Mn", "CO": "Co", "NI": "Ni", "CU": "Cu", "AG": "Ag",
    "CD": "Cd", "SN": "Sn", "SB": "Sb", "TE": "Te", "XE": "Xe",
    "CS": "Cs", "BA": "Ba", "LA": "La", "CE": "Ce", "PT": "Pt",
    "AU": "Au", "HG": "Hg", "PB": "Pb", "BI": "Bi", "IR": "Ir",
    "OS": "Os", "PD": "Pd", "RH": "Rh", "RU": "Ru", "MO": "Mo",
    "TI": "Ti", "ZR": "Zr", "NB": "Nb", "TA": "Ta", "RE": "Re",
    "GA": "Ga", "GE": "Ge", "AS": "As", "KR": "Kr", "RB": "Rb",
    "SR": "Sr", "YB": "Yb", "LU": "Lu", "HF": "Hf", "W": "W",
    "I": "I", "B": "B", "C": "C", "N": "N", "O": "O", "F": "F",
    "OG": "Og", "TS": "Ts", "LV": "Lv", "MC": "Mc", "FL": "Fl", "NH": "Nh", "CN": "Cn",
    "K": "K", "P": "P", "S": "S", "V": "V"
}
_TAIL_NUM = re.compile(r"^([A-Za-z]{1,3})(\d+)$")

def _normalize_element(sym: str) -> str:
    if sym is None:
        return "X"
    s = str(sym).strip()
    if not s:
        return "X"
    m = _TAIL_NUM.match(s)
    if m:
        s = m.group(1)
    su = s.upper()
    if su in ("D", "T"):
        return "H"
    if su in _ALIAS_MAP:
        fixed = _ALIAS_MAP[su]
        if fixed in _ALL_ELEMS:
            return fixed
    sn = su[0] + su[1:].lower() if len(su) > 1 else su
    if sn in _ALL_ELEMS:
        return sn
    sl = s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()
    if sl in _ALL_ELEMS:
        return sl
    return (s[0].upper() + s[1:].lower()) if len(s) > 1 else s.upper()


def _take_to_len(arr_like, N, dtype=np.float32, fill=0.0) -> np.ndarray:
    if arr_like is None:
        return np.full((N,), fill_value=fill, dtype=dtype)
    try:
        arr = np.asarray(arr_like, dtype=dtype).reshape(-1)
    except Exception:
        return np.full((N,), fill_value=fill, dtype=dtype)
    if arr.shape[0] >= N:
        return arr[:N]
    out = np.full((N,), fill_value=fill, dtype=dtype)
    out[:arr.shape[0]] = arr
    return out


class LigandDataset(data.Dataset):
    """
    JSONL -> 配体样本（坐标 -> 原子类型）。
    现在会把你已有的节点级特征（charges/hybridization/is_aromatic）
    和分子级计数（hbd_count/hba_count）一并读出来，交给 Featurizer 使用。
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 jsonl_name: Optional[str] = None,
                 max_atoms: int = 0,
                 drop_invalid: bool = True,
                 keep_meta: bool = True,
                 make_seq_string: bool = False):
        super().__init__()
        split = "val" if split in ("valid", "validation") else split
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"

        self.path = path
        self.split = split
        self.max_atoms = int(max_atoms)
        self.drop_invalid = bool(drop_invalid)
        self.keep_meta = bool(keep_meta)
        self.make_seq_string = bool(make_seq_string)

        cand = []
        if jsonl_name:
            cand.append(os.path.join(path, jsonl_name))
        cand += [
            # os.path.join(path, "ligand_samples.jsonl"),
            os.path.join(path, f"{split}_new.jsonl"),
            os.path.join(path, f"{split}.jsonl"),
            os.path.join(path, "ligand_samples.jsonl"),
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
        with open(fn, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading Ligands(JSONL): {os.path.basename(fn)}"):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    if self.drop_invalid:
                        continue
                    entry = {}
                it = self._entry_to_item(entry)
                if it is not None:
                    items.append(it)
        return items

    def _entry_to_item(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        title = str(entry.get("title") or entry.get("name") or
                    f"LIG_{self.split}_{len(getattr(self, 'data', []))}")

        elements = entry.get("elements", None)
        coords = entry.get("coords", None)

        if not isinstance(elements, list) or not isinstance(coords, list):
            return None if self.drop_invalid else None

        arr = np.asarray(coords, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None if self.drop_invalid else None

        N = arr.shape[0]
        elems_norm = [_normalize_element(e) for e in elements]
        if len(elems_norm) != N or N < 1:
            return None if self.drop_invalid else None

        # 可选裁剪（一般不裁剪小分子）
        if self.max_atoms > 0 and N > self.max_atoms:
            idx = np.arange(self.max_atoms, dtype=np.int64)
            arr = arr[idx]
            elems_norm = [elems_norm[i] for i in idx]
            N = self.max_atoms

        atom_names = entry.get("atoms", None)
        if isinstance(atom_names, list) and len(atom_names) == N:
            atom_names_out = [str(a) for a in atom_names]
        else:
            atom_names_out = [f"{elems_norm[i]}{i+1}" for i in range(N)]

        chain_mask = entry.get("chain_mask", None)
        if isinstance(chain_mask, list) and len(chain_mask) == N:
            mask_np = np.array(chain_mask, dtype=np.float32)
        else:
            mask_np = np.ones((N,), dtype=np.float32)

        chain_encoding = entry.get("chain_encoding", None)
        if isinstance(chain_encoding, list) and len(chain_encoding) == N:
            enc_np = np.array(chain_encoding, dtype=np.float32)
        else:
            enc_np = np.ones((N,), dtype=np.float32)

        # ========= 新增：节点级与分子级属性的安全读取 =========
        charges_vec = _take_to_len(entry.get("charges", None), N, dtype=np.float32, fill=0.0)
        charges_vec = np.round(charges_vec.astype(np.float32), 3)
        hyb_vec     = _take_to_len(entry.get("hybridization", None), N, dtype=np.float32, fill=0.0)
        arom_vec    = _take_to_len(entry.get("is_aromatic", None), N, dtype=np.float32, fill=0.0)

        # 分子级计数（缺失则置 0）
        hbd_count = int(entry.get("hbd_count", 0) or 0)
        hba_count = int(entry.get("hba_count", 0) or 0)
        # ====================================================

        out: Dict[str, Any] = {
            "title": title,
            "type": "ligand",
            "elements": elems_norm,              # 监督标签（featurizer 里映射ID）
            "coords": arr.astype(np.float32),    # (N,3)
            "atom_names": [str(a) for a in atom_names_out],
            "chain_mask": [mask_np],
            "chain_encoding": [enc_np],

            # 交给 Featurizer 的新增字段
            "charges": charges_vec.tolist(),
            "hybridization": hyb_vec.tolist(),
            "is_aromatic": arom_vec.tolist(),
            "hbd_count": hbd_count,
            "hba_count": hba_count,

            "meta": {}
        }

        if self.keep_meta:
            for k in ("smiles", "comp_id", "pdb_id", "chain_id", "res_id", "type"):
                if k in entry:
                    out["meta"][k] = entry[k]
            out["meta"]["N"] = int(N)

        if self.make_seq_string:
            out["seq"] = [" ".join(elems_norm)]

        return out

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
