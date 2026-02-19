# -*- coding: utf-8 -*-
import os
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import re

# ----------------------------
# 与 LigandDataset 一致的元素规范化
# ----------------------------
_ALL_ELEMS = {
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru",
    "Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce",
    "Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
    "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
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

# ----------------------------
# 受体侧坐标键（标准化为不带撇的键名）
# ----------------------------
PROT_KEYS = ["N", "CA", "C", "O"]
NA_KEYS   = ["P", "O5", "C5", "C4", "C3", "O3", "N"]  # N 为锚点

# 输入里的核酸原子名兼容（带撇/星号）
PRIME_IN_MAP = {
    "P":  ("P",),
    "O5": ("O5", "O5'", "O5*"),
    "C5": ("C5", "C5'", "C5*"),
    "C4": ("C4", "C4'", "C4*"),
    "C3": ("C3", "C3'", "C3*"),
    "O3": ("O3", "O3'", "O3*"),
    # N 锚点：预处理里键为 "N"；极端情况下也可能给到 "N1"/"N9"
}

def _to_array_L3_from_list(arr_list: Iterable[Any], L: int) -> np.ndarray:
    out = np.full((L, 3), np.nan, dtype=np.float32)
    if arr_list is None:
        return out
    i = 0
    for a in arr_list:
        if i >= L:
            break
        if isinstance(a, (list, tuple)) and len(a) == 3:
            try:
                out[i, 0] = np.float32(a[0]) if a[0] is not None else np.nan
                out[i, 1] = np.float32(a[1]) if a[1] is not None else np.nan
                out[i, 2] = np.float32(a[2]) if a[2] is not None else np.nan
            except Exception:
                pass
        i += 1
    return out

def _pick_na_atom(coords_in: Dict[str, Any], key_no_prime: str, L: int) -> np.ndarray:
    if not isinstance(coords_in, dict):
        return np.full((L, 3), np.nan, dtype=np.float32)
    # 尝试兼容带撇/星号的键名
    for cand in (PRIME_IN_MAP.get(key_no_prime, (key_no_prime,)) + (key_no_prime,)):
        raw = coords_in.get(cand)
        if isinstance(raw, (list, tuple)):
            return _to_array_L3_from_list(raw, L)
    return np.full((L, 3), np.nan, dtype=np.float32)

def _pick_na_anchor_N(coords_in: Dict[str, Any], L: int) -> np.ndarray:
    if isinstance(coords_in, dict):
        for k in ("N", "N1", "N9"):
            raw = coords_in.get(k)
            if isinstance(raw, (list, tuple)):
                return _to_array_L3_from_list(raw, L)
    return np.full((L, 3), np.nan, dtype=np.float32)

def _pick_prot_atom(coords_in: Dict[str, Any], atom: str, L: int) -> np.ndarray:
    if not isinstance(coords_in, dict):
        return np.full((L, 3), np.nan, dtype=np.float32)
    raw = coords_in.get(atom)
    return _to_array_L3_from_list(raw, L)

# ----------------------------
# 数据集（单配体-单受体）
# ----------------------------
class LigandComplexDataset(data.Dataset):
    """
    读取 preprocess 生成的 ligand–receptor JSONL（每行一个配体-受体对）：
      {
        "pdb_id": "XXXX",
        "eval_type": "ligand_prot" | "ligand_nuc",
        "ligand":  {"comp_id","chain_id","atoms","elements","coords"},
        "receptor": {
            "type": "protein" | "rna" | "dna",
            "chain_id": "...",
            "seq": "....",
            "coords": { protein: N/CA/C/O; nucleic: P/O5'/C5'/C4'/C3'/O3'/N ... }
        },
        ...
      }

    输出样本（受体为单链）：
      {
        "title": str,
        "type": "protein" | "rna" | "dna",
        "seq": [str],
        "coords": { ... }           # ← 坐标统一放在 coords 字典中（与 ligand 对齐）
        "chain_mask": [np.ones(L)],
        "chain_encoding": [np.ones(L)],
        "chain_names": [str],
        "pdb_id": str,
        "eval_type": str,
        "chain_spans": np.ndarray([(0,L)], int64),

        "ligand": {
          "elements": [str]*N,
          "coords":   np.ndarray(N,3),
          "atom_names": [str]*N,
          "meta": {"comp_id":..., "chain_id":..., "pdb_id":..., "N": int}
        }
      }
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 *,
                 jsonl_name: Optional[str] = None,
                 max_length: int = 500,
                 max_ligand_atoms: int = 0,
                 preload: bool = True,
                 return_title: bool = True,
                 verbose: bool = True):
        super().__init__()
        split = "val" if split in ("valid", "validation") else split
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"

        self.path = path
        self.split = split
        self.max_length = int(max_length)
        self.max_ligand_atoms = int(max_ligand_atoms)
        self.preload = bool(preload)
        self.return_title = bool(return_title)
        self.verbose = bool(verbose)

        # 选择文件
        cand = []
        if jsonl_name:
            cand.append(os.path.join(path, jsonl_name))
        cand += [
            # os.path.join(path, "test.jsonl"),
            os.path.join(path, f"{split}.jsonl"),
            os.path.join(path, f"{split}_data.jsonl"),
            os.path.join(path, f"{split}_data_new.jsonl"),
        ]
        for p in cand:
            if os.path.exists(p):
                self.jsonl_file = p
                break
        else:
            raise FileNotFoundError(f"jsonl 未找到，已尝试：{cand}")

        self._items: Optional[List[Dict[str, Any]]] = None
        self._offsets: Optional[List[int]] = None
        self._fh = None
        self._warned_extra_once = False

        if self.preload:
            self._items = self._load_all_json(self.jsonl_file, verbose=self.verbose)
        else:
            self._offsets = self._build_offsets(self.jsonl_file, verbose=self.verbose)

    # --------- 多进程安全 ---------
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_fh"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._fh = None

    def _ensure_open(self):
        if self._fh is None:
            self._fh = open(self.jsonl_file, "rb", buffering=4 * 1024 * 1024)

    # --------- I/O 帮助 ---------
    @staticmethod
    def _build_offsets(fn: str, verbose: bool = True) -> List[int]:
        offs: List[int] = []
        with open(fn, "rb") as f:
            it = iter(int, 1)
            pbar = tqdm(it, desc=f"Indexing {os.path.basename(fn)}", disable=not verbose)
            for _ in pbar:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offs.append(pos)
        return offs

    @staticmethod
    def _load_all_json(fn: str, verbose: bool = True) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        with open(fn, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(fn)}", disable=not verbose):
                s = (line or "").strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    try:
                        dec = json.JSONDecoder()
                        obj, _end = dec.raw_decode(s.lstrip())
                    except Exception:
                        continue
                items.append(obj)
        return items

    def __len__(self) -> int:
        return len(self._items) if self.preload else len(self._offsets)

    def _json_loads_one(self, s: str) -> Dict[str, Any]:
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            try:
                dec = json.JSONDecoder()
                obj, end = dec.raw_decode(s.lstrip())
                rest = s.lstrip()[end:].strip()
                if rest and not self._warned_extra_once:
                    self._warned_extra_once = True
                    print("[WARN] JSONL line contained multiple JSON objects; "
                          "used the first and ignored the rest. (suppressing further warnings)")
                return obj
            except Exception:
                snippet = s[:120].replace("\n", "\\n")
                raise ValueError(f"Failed to parse JSONL line. Head(120)='{snippet}'") from e

    def _read_row(self, idx: int) -> Dict[str, Any]:
        if self.preload:
            return self._items[idx]
        else:
            self._ensure_open()
            off = self._offsets[idx]
            self._fh.seek(off)
            line = self._fh.readline()
            s = line.decode("utf-8", errors="ignore").strip()
            return self._json_loads_one(s)

    # --------- 主入口 ---------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self._read_row(idx)

        pdb_id    = str(obj.get("pdb_id") or obj.get("name") or f"LC_{idx}")
        eval_type = str(obj.get("eval_type") or "").lower().strip()

        lig = obj.get("ligand", {}) or {}
        rec = obj.get("receptor", {}) or {}

        # ---------- 受体 ----------
        r_type  = str(rec.get("type") or "").lower().strip()
        r_chain = str(rec.get("chain_id") or "A")
        r_seq   = str(rec.get("seq") or "").upper()
        r_co_in = rec.get("coords", {}) or {}
        L_full  = len(r_seq)
        if L_full == 0:
            raise RuntimeError(f"Empty receptor sequence in row {idx} (pdb_id={pdb_id}).")

        # 把输入 coords 标准化为统一键名（不带撇），并得到 ndarray(L,3)
        def std_prot_coords(coords_in: Dict[str, Any], L: int) -> Dict[str, np.ndarray]:
            return {
                "N":  _pick_prot_atom(coords_in, "N",  L),
                "CA": _pick_prot_atom(coords_in, "CA", L),
                "C":  _pick_prot_atom(coords_in, "C",  L),
                "O":  _pick_prot_atom(coords_in, "O",  L),
            }
        def std_na_coords(coords_in: Dict[str, Any], L: int) -> Dict[str, np.ndarray]:
            return {
                "P":  _pick_na_atom(coords_in, "P",  L),
                "O5": _pick_na_atom(coords_in, "O5", L),
                "C5": _pick_na_atom(coords_in, "C5", L),
                "C4": _pick_na_atom(coords_in, "C4", L),
                "C3": _pick_na_atom(coords_in, "C3", L),
                "O3": _pick_na_atom(coords_in, "O3", L),
                "N":  _pick_na_anchor_N(coords_in,  L),  # 锚点
            }

        if r_type == "protein":
            r_coords_full = std_prot_coords(r_co_in, L_full)
            valid_keys = PROT_KEYS
        else:
            # 缺省把非 protein 都当 nucleic（已由 preprocess 区分 dna/rna）
            r_coords_full = std_na_coords(r_co_in, L_full)
            valid_keys = NA_KEYS

        # 训练集随机裁剪（只裁剪受体）
        if self.split == "train" and self.max_length and L_full > self.max_length:
            t0 = random.randint(0, L_full - self.max_length)
            t1 = t0 + self.max_length
            r_seq = r_seq[t0:t1]
            L = self.max_length
            def slicer(a: np.ndarray) -> np.ndarray:
                return a[t0:t1] if isinstance(a, np.ndarray) and a.shape[0] >= t1 \
                                else np.full((L,3), np.nan, dtype=np.float32)
            r_coords = {k: slicer(v) for k, v in r_coords_full.items() if k in valid_keys}
        else:
            L = L_full
            r_coords = {k: (v if isinstance(v, np.ndarray) and v.shape == (L,3)
                            else np.full((L,3), np.nan, dtype=np.float32))
                        for k, v in r_coords_full.items() if k in valid_keys}

        # ---------- 配体（严格按 preprocess 结构） ----------
        lig_elems_raw = lig.get("elements") or []
        lig_coords_raw = lig.get("coords") or []
        n_lig = len(lig_coords_raw)

        lig_elems = [_normalize_element(e) for e in lig_elems_raw]
        lig_coords = np.array(lig_coords_raw, dtype=np.float32).reshape(n_lig, 3) if n_lig else np.zeros((0,3), dtype=np.float32)

        # 可选：裁剪配体原子数（通常不裁剪）
        if self.max_ligand_atoms > 0 and n_lig > self.max_ligand_atoms:
            idx_sel = np.arange(self.max_ligand_atoms, dtype=np.int64)
            lig_coords = lig_coords[idx_sel]
            lig_elems  = [lig_elems[i] for i in idx_sel]
            n_lig = lig_coords.shape[0]

        lig_atom_names = lig.get("atoms") or [f"{lig_elems[i]}{i+1}" for i in range(n_lig)]
        if len(lig_atom_names) != n_lig:
            lig_atom_names = [f"{lig_elems[i]}{i+1}" for i in range(n_lig)]

        # ---------- 组装输出（coords 统一打包） ----------
        item: Dict[str, Any] = {
            "type":           [r_type] if r_type in ("protein","rna","dna") else ("rna" if set(r_seq)<=set("AUGC") else "protein"),
            "seq":            [r_seq],
            "coords":         r_coords,  # ← 受体坐标放到 coords 里
            "chain_mask":     [np.ones(L, dtype=np.float32)],
            "chain_encoding": [np.ones(L, dtype=np.float32)],  # 单链 → 全 1
            "chain_names":    [r_chain],
            "pdb_id":          pdb_id,
            "eval_type":       eval_type,
            "chain_spans":     np.array([(0, L)], dtype=np.int64),

            "ligand": {
                "elements":   lig_elems,
                "coords":     lig_coords.astype(np.float32),     # (N_lig, 3)
                "atom_names": [str(a) for a in lig_atom_names],
                "meta": {
                    "comp_id": lig.get("comp_id"),
                    "chain_id": lig.get("chain_id"),
                    "pdb_id":   pdb_id,
                    "N":        int(n_lig),
                }
            }
        }

        if self.return_title:
            item["title"] = f"{pdb_id}_{r_chain}|{lig.get('chain_id','L')}_{lig.get('comp_id','LIG')}"

        return item
