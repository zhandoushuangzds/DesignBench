# -*- coding: utf-8 -*-
# src/datasets/pdb_complex.py

import os
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
import torch.utils.data as data

# 统一坐标键（与 Featurize_Complex / ComplexDataset 对齐）
PROT_KEYS = ["N", "CA", "C", "O"]
NA_KEYS   = ["P", "O5", "C5", "C4", "C3", "O3"]

# 输入里的核酸原子名兼容（带撇/星号）
PRIME_IN_MAP = {
    "P":  ("P",),
    "O5": ("O5", "O5'", "O5*"),
    "C5": ("C5", "C5'", "C5*"),
    "C4": ("C4", "C4'", "C4*"),
    "C3": ("C3", "C3'", "C3*"),
    "O3": ("O3", "O3'", "O3*"),
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

def _infer_kind_from_seq(seq: str) -> str:
    s = "".join([c for c in (seq or "").upper() if c.isalpha()])
    if not s:
        return "protein"
    letters = set(s)
    if letters.issubset(set("AUGC")):
        return "rna"
    if letters.issubset(set("ATGCN")):
        return "dna"
    return "protein"

def _canon_kind(section_key: str) -> str:
    k = (section_key or "").strip().lower()
    if k in ("protein", "proteins"):
        return "protein"
    if k in ("rna", "rnas"):
        return "rna"
    if k in ("dna", "dnas"):
        return "dna"
    return k

class PDB_Complex(data.Dataset):
    """
    按 PDB 聚合的 JSONL（每行包含 PROTEIN/RNA/DNA -> 多条链 {seq, coords}），
    转成 Featurize_Complex 需要的复合物样本。

    文件名选择顺序：
      1) 显式 jsonl_name
      2) new_<split>.jsonl
      3) output_all_validation.jsonl
      4) debug_test.jsonl

    split 归一为：train / valid / test

    训练专属裁剪：
      - 若 split=='train' 且链数 > max_train_chains（默认 2），
        优先保留 RNA/DNA，再从蛋白中随机补齐到上限。

    single_chain_only:
      - True：若裁剪后仍超过 1 条链，则自动取第一条链
      - False：保留多链
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 *,
                 jsonl_name: Optional[str] = None,
                 include_types: Optional[Iterable[str]] = None,
                 include_chain_ids: Optional[Dict[str, Iterable[str]]] = None,
                 target_type: Optional[str] = None,
                 target_chains: Optional[Dict[str, Iterable[str]]] = None,
                 max_length: int = 500,
                 max_train_chains: int = 2,
                 return_title: bool = True,
                 verbose: bool = True,
                 preload: bool = False,
                 single_chain_only: bool = False):
        super().__init__()

        split = "valid" if split in ("valid", "val", "validation") else split
        assert split in ("train", "valid", "test")
        self.path = path
        self.split = split
        self.max_length = int(max_length)
        self.max_train_chains = int(max_train_chains) if max_train_chains is not None else 0
        self.return_title = bool(return_title)
        self.preload = bool(preload)
        self.verbose = bool(verbose)
        self.single_chain_only = bool(single_chain_only)

        # 文件选择
        cand = []
        if jsonl_name:
            cand.append(os.path.join(path, jsonl_name))
        cand += [
            # os.path.join(path, "debug_test.jsonl"),
            os.path.join(path, f"new_{split}.jsonl"),
            os.path.join(path, "output_all_validation.jsonl"),
            os.path.join(path, "debug_test.jsonl"),
        ]
        for p in cand:
            if os.path.exists(p):
                self.jsonl_file = p
                break
        else:
            raise FileNotFoundError(f"jsonl 未找到，已尝试：{cand}")

        # 过滤/目标设定
        self.include_types = { _canon_kind(t) for t in include_types } if include_types else None
        self.include_chain_ids = ({ _canon_kind(k): {str(x) for x in v}
                                    for k, v in include_chain_ids.items() } if include_chain_ids else None)
        self.target_type = _canon_kind(target_type) if target_type else None
        self.target_chains = ({ _canon_kind(k): {str(x) for x in v}
                                 for k, v in target_chains.items() } if target_chains else None)

        # 数据源
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

    # --------- 索引/加载 ---------
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

    # --------- 结构工具 ---------
    @staticmethod
    def _iter_section(obj: Dict[str, Any], section: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
        block = obj.get(section) or obj.get(section.upper()) or obj.get(section.capitalize())
        if not isinstance(block, dict):
            return []
        return block.items()

    @staticmethod
    def _pick_na_atom(coords: Dict[str, Any], key_no_prime: str, L: int) -> np.ndarray:
        if not isinstance(coords, dict):
            return np.full((L, 3), np.nan, dtype=np.float32)
        for cand in PRIME_IN_MAP[key_no_prime]:
            raw = coords.get(cand)
            if isinstance(raw, (list, tuple)):
                return _to_array_L3_from_list(raw, L)
        return np.full((L, 3), np.nan, dtype=np.float32)

    @staticmethod
    def _pick_prot_atom(coords: Dict[str, Any], atom: str, L: int) -> np.ndarray:
        if not isinstance(coords, dict):
            return np.full((L, 3), np.nan, dtype=np.float32)
        raw = coords.get(atom)
        return _to_array_L3_from_list(raw, L)

    @staticmethod
    def _resolve_is_target(types: List[str],
                           chain_ids: List[Tuple[str, str]],
                           target_type: Optional[str],
                           target_chains: Optional[Dict[str, set]]) -> List[bool]:
        """
        规则：
        - 若给了 target_chains 且命中 → 仅命中的为 True；
        - 否则若给了 target_type → 该类型为 True；
        - 否则 → 全 False。
        """
        M = len(types)
        is_tgt = [False] * M

        if target_chains:
            hit = False
            for i, (t, cid) in enumerate(chain_ids):
                tset = target_chains.get(t, None)
                if tset and cid in tset:
                    is_tgt[i] = True
                    hit = True
            if hit:
                return is_tgt

        if target_type:
            for i, t in enumerate(types):
                if t == target_type:
                    is_tgt[i] = True
            return is_tgt

        return is_tgt

    def _subset_for_training(self,
                             kinds: List[str],
                             chains: List[str],
                             seqs: List[str],
                             coords_list: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str], List[Dict[str, Any]]]:
        """
        训练集链裁剪：优先保留核酸（rna/dna），不足再从蛋白随机补齐到 max_train_chains。
        返回保持原有相对顺序的子集。
        """
        M = len(kinds)
        limit = int(self.max_train_chains) if self.max_train_chains else 0
        if self.split != "train" or limit <= 0 or M <= limit:
            return kinds, chains, seqs, coords_list

        na_idx   = [i for i, k in enumerate(kinds) if k in ("rna", "dna")]
        prot_idx = [i for i, k in enumerate(kinds) if k == "protein"]

        if len(na_idx) >= limit:
            chosen = random.sample(na_idx, k=limit)
        else:
            need = limit - len(na_idx)
            chosen = na_idx + random.sample(prot_idx, k=need)

        chosen = sorted(chosen)  # 保持原有顺序
        kinds_s   = [kinds[i] for i in chosen]
        chains_s  = [chains[i] for i in chosen]
        seqs_s    = [seqs[i] for i in chosen]
        coords_s  = [coords_list[i] for i in chosen]
        return kinds_s, chains_s, seqs_s, coords_s

    # --------- 返回：一个 PDB 内所有链组成的“复合物样本” ---------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self._read_row(idx)
        raw_title = obj.get("name") or obj.get("title") or f"pdb_{idx}"
        pdb_id = os.path.splitext(str(raw_title))[0]

        kinds: List[str] = []
        chains: List[str] = []
        seqs:  List[str] = []
        coords_list: List[Dict[str, Any]] = []

        # 三类统一收集
        for sec in ("PROTEIN", "RNA", "DNA"):
            kind = _canon_kind(sec)
            if self.include_types and (kind not in self.include_types):
                continue

            for chain_id, rec in self._iter_section(obj, sec):
                chain_id = str(chain_id)
                if self.include_chain_ids:
                    allow = self.include_chain_ids.get(kind, None)
                    if allow is not None and chain_id not in allow:
                        continue

                seq = (rec or {}).get("seq") or ""
                coords = (rec or {}).get("coords") or {}

                _ = _infer_kind_from_seq(seq)  # 仅兜底参考，不覆盖块类型

                kinds.append(kind)
                chains.append(chain_id)
                seqs.append(seq)
                coords_list.append(coords)

        if not kinds:
            raise RuntimeError(f"No chains parsed in row {idx} (title={raw_title}).")

        # --- 训练集链裁剪：优先保留核酸 ---
        kinds, chains, seqs, coords_list = self._subset_for_training(kinds, chains, seqs, coords_list)

        # ★ single_chain_only：若仍多链，直接取第一条
        if self.single_chain_only and len(kinds) > 1:
            kinds       = kinds[:1]
            chains      = chains[:1]
            seqs        = seqs[:1]
            coords_list = coords_list[:1]

        # 目标解析
        tgt = self._resolve_is_target(
            kinds,
            list(zip(kinds, chains)),
            self.target_type,
            ({k: set(v) for k, v in self.target_chains.items()} if self.target_chains else None)
        )

        # ---- 组装输出（每条链一个条目）----
        out: Dict[str, Any] = {
            "type": [],
            "seq": [],
            "is_target": [],
            "chain_mask": [],
            "chain_encoding": [],
            "N":  [], "CA": [], "C":  [], "O":  [],
            "P":  [], "O5": [], "C5": [], "C4": [], "C3": [], "O3": [],
            # ★ 新增
            "chain_names": [],          # e.g., ["I","A",...]
            "pdb_id": pdb_id,          # e.g., "9MM6"
        }

        # 记录裁剪后用于扁平化的区间（可选）
        chain_spans: List[Tuple[int, int]] = []
        running = 0

        # 每条链裁剪、取坐标并填充
        for i, (k, cid, seq, co) in enumerate(zip(kinds, chains, seqs, coords_list)):
            s = (seq or "").upper()
            L_full = len(s)
            if L_full == 0:
                continue

            # 随机裁剪每条链
            if self.max_length and self.max_length > 0 and L_full > self.max_length:
                t0 = random.randint(0, L_full - self.max_length)
                t1 = t0 + self.max_length
                s = s[t0:t1]
                L = self.max_length
                def slice_na(arr: np.ndarray) -> np.ndarray:
                    return arr[t0:t1] if isinstance(arr, np.ndarray) and arr.shape[0] >= t1 else np.full((L,3), np.nan, dtype=np.float32)
                if k == "protein":
                    N  = self._pick_prot_atom(co, "N",  L_full)
                    CA = self._pick_prot_atom(co, "CA", L_full)
                    C  = self._pick_prot_atom(co, "C",  L_full)
                    O  = self._pick_prot_atom(co, "O",  L_full)
                    N, CA, C, O = map(slice_na, (N, CA, C, O))
                else:
                    P  = self._pick_na_atom(co, "P",  L_full)
                    O5 = self._pick_na_atom(co, "O5", L_full)
                    C5 = self._pick_na_atom(co, "C5", L_full)
                    C4 = self._pick_na_atom(co, "C4", L_full)
                    C3 = self._pick_na_atom(co, "C3", L_full)
                    O3 = self._pick_na_atom(co, "O3", L_full)
                    P, O5, C5, C4, C3, O3 = map(slice_na, (P, O5, C5, C4, C3, O3))
            else:
                L = L_full
                if k == "protein":
                    N  = self._pick_prot_atom(co, "N",  L)
                    CA = self._pick_prot_atom(co, "CA", L)
                    C  = self._pick_prot_atom(co, "C",  L)
                    O  = self._pick_prot_atom(co, "O",  L)
                else:
                    P  = self._pick_na_atom(co, "P",  L)
                    O5 = self._pick_na_atom(co, "O5", L)
                    C5 = self._pick_na_atom(co, "C5", L)
                    C4 = self._pick_na_atom(co, "C4", L)
                    C3 = self._pick_na_atom(co, "C3", L)
                    O3 = self._pick_na_atom(co, "O3", L)

            out["type"].append(k)
            out["seq"].append(s)
            out["is_target"].append(bool(tgt[i]))
            out["chain_mask"].append(np.ones(L, dtype=np.float32))
            out["chain_encoding"].append(np.full(L, fill_value=(i + 1), dtype=np.float32))
            out["chain_names"].append(cid)

            if k == "protein":
                out["N"].append(N); out["CA"].append(CA); out["C"].append(C); out["O"].append(O)
                for kk in NA_KEYS:
                    out[kk].append(np.full((L, 3), np.nan, dtype=np.float32))
            else:
                out["P"].append(P); out["O5"].append(O5); out["C5"].append(C5)
                out["C4"].append(C4); out["C3"].append(C3); out["O3"].append(O3)
                for kk in PROT_KEYS:
                    out[kk].append(np.full((L, 3), np.nan, dtype=np.float32))

            # ★ 累加扁平区间
            chain_spans.append((running, running + L))
            running += L

        # ★ 提供 chain_spans 便于下游（如需要按链切 FASTA）
        out["chain_spans"] = np.array(chain_spans, dtype=np.int64)

        # ★ title：改为每链一个，形如 "<pdbid>_<chain>"
        if self.return_title:
            out["title"] = [f"{pdb_id}_{cid}" for cid in out["chain_names"]]

        return out




# # -*- coding: utf-8 -*-
# # src/datasets/pdb_complex.py

# import os
# import json
# import random
# from typing import Any, Dict, Iterable, List, Optional, Tuple

# import numpy as np
# from tqdm import tqdm
# import torch.utils.data as data

# # 统一坐标键（与 Featurize_Complex / ComplexDataset 对齐）
# PROT_KEYS = ["N", "CA", "C", "O"]
# NA_KEYS   = ["P", "O5", "C5", "C4", "C3", "O3"]

# # 输入里的核酸原子名兼容（带撇/星号）
# PRIME_IN_MAP = {
#     "P":  ("P",),
#     "O5": ("O5", "O5'", "O5*"),
#     "C5": ("C5", "C5'", "C5*"),
#     "C4": ("C4", "C4'", "C4*"),
#     "C3": ("C3", "C3'", "C3*"),
#     "O3": ("O3", "O3'", "O3*"),
# }

# def _to_array_L3_from_list(arr_list: Iterable[Any], L: int) -> np.ndarray:
#     out = np.full((L, 3), np.nan, dtype=np.float32)
#     if arr_list is None:
#         return out
#     i = 0
#     for a in arr_list:
#         if i >= L:
#             break
#         if isinstance(a, (list, tuple)) and len(a) == 3:
#             try:
#                 out[i, 0] = np.float32(a[0]) if a[0] is not None else np.nan
#                 out[i, 1] = np.float32(a[1]) if a[1] is not None else np.nan
#                 out[i, 2] = np.float32(a[2]) if a[2] is not None else np.nan
#             except Exception:
#                 pass
#         i += 1
#     return out

# def _infer_kind_from_seq(seq: str) -> str:
#     s = "".join([c for c in (seq or "").upper() if c.isalpha()])
#     if not s:
#         return "protein"
#     letters = set(s)
#     if letters.issubset(set("AUGC")):
#         return "rna"
#     if letters.issubset(set("ATGCN")):
#         return "dna"
#     return "protein"

# def _canon_kind(section_key: str) -> str:
#     k = (section_key or "").strip().lower()
#     if k in ("protein", "proteins"):
#         return "protein"
#     if k in ("rna", "rnas"):
#         return "rna"
#     if k in ("dna", "dnas"):
#         return "dna"
#     return k

# class PDB_Complex(data.Dataset):
#     """
#     按 PDB 聚合的 JSONL（每行包含 PROTEIN/RNA/DNA -> 多条链 {seq, coords}），
#     转成 Featurize_Complex 需要的复合物样本。

#     文件名选择顺序：
#       1) 显式 jsonl_name
#       2) <split>.jsonl
#       3) output_all_validation.jsonl
#       4) debug_test.jsonl

#     split 归一为：train / validation / test

#     训练专属裁剪：
#       - 若 split=='train' 且链数 > max_train_chains（默认 2），
#         优先保留 RNA/DNA，再从蛋白中随机补齐到上限。

#     single_chain_only:
#       - 默认 True：若裁剪后仍超过 1 条链，则**自动取第一条链**（不再报错）
#       - 设为 False：保留原有多链复合物行为
#     """

#     def __init__(self,
#                  path: str,
#                  split: str = "train",
#                  *,
#                  jsonl_name: Optional[str] = None,
#                  include_types: Optional[Iterable[str]] = None,
#                  include_chain_ids: Optional[Dict[str, Iterable[str]]] = None,
#                  target_type: Optional[str] = None,
#                  target_chains: Optional[Dict[str, Iterable[str]]] = None,
#                  max_length: int = 500,
#                  max_train_chains: int = 2,
#                  return_title: bool = True,
#                  verbose: bool = True,
#                  preload: bool = False,
#                  single_chain_only: bool = False  # ★ 默认仅单链；多链时自动取第一条
#                  ):
#         super().__init__()

#         split = "valid" if split in ("valid", "val", "validation") else split
#         self.path = path
#         self.split = split
#         self.max_length = int(max_length)
#         self.max_train_chains = int(max_train_chains) if max_train_chains is not None else 0
#         self.return_title = bool(return_title)
#         self.preload = bool(preload)
#         self.verbose = bool(verbose)
#         self.single_chain_only = bool(single_chain_only)

#         # 文件选择
#         cand = []
#         if jsonl_name:
#             cand.append(os.path.join(path, jsonl_name))
#         cand += [
#             os.path.join(path, f"new_{split}.jsonl"),
#             os.path.join(path, "output_all_validation.jsonl"),
#             os.path.join(path, "debug_test.jsonl"),
#         ]
#         for p in cand:
#             if os.path.exists(p):
#                 self.jsonl_file = p
#                 break
#         else:
#             raise FileNotFoundError(f"jsonl 未找到，已尝试：{cand}")

#         # 过滤/目标设定
#         self.include_types = { _canon_kind(t) for t in include_types } if include_types else None
#         self.include_chain_ids = ({ _canon_kind(k): {str(x) for x in v}
#                                     for k, v in include_chain_ids.items() } if include_chain_ids else None)
#         self.target_type = _canon_kind(target_type) if target_type else None
#         self.target_chains = ({ _canon_kind(k): {str(x) for x in v}
#                                  for k, v in target_chains.items() } if target_chains else None)

#         # 数据源
#         self._items: Optional[List[Dict[str, Any]]] = None
#         self._offsets: Optional[List[int]] = None
#         self._fh = None
#         self._warned_extra_once = False

#         if self.preload:
#             self._items = self._load_all_json(self.jsonl_file, verbose=self.verbose)
#         else:
#             self._offsets = self._build_offsets(self.jsonl_file, verbose=self.verbose)

#     # --------- 多进程安全 ---------
#     def __getstate__(self):
#         d = self.__dict__.copy()
#         d["_fh"] = None
#         return d

#     def __setstate__(self, d):
#         self.__dict__.update(d)
#         self._fh = None

#     def _ensure_open(self):
#         if self._fh is None:
#             self._fh = open(self.jsonl_file, "rb", buffering=4 * 1024 * 1024)

#     # --------- 索引/加载 ---------
#     @staticmethod
#     def _build_offsets(fn: str, verbose: bool = True) -> List[int]:
#         offs: List[int] = []
#         with open(fn, "rb") as f:
#             it = iter(int, 1)
#             pbar = tqdm(it, desc=f"Indexing {os.path.basename(fn)}", disable=not verbose)
#             for _ in pbar:
#                 pos = f.tell()
#                 line = f.readline()
#                 if not line:
#                     break
#                 offs.append(pos)
#         return offs

#     @staticmethod
#     def _load_all_json(fn: str, verbose: bool = True) -> List[Dict[str, Any]]:
#         items: List[Dict[str, Any]] = []
#         with open(fn, "r", encoding="utf-8") as f:
#             for line in tqdm(f, desc=f"Loading {os.path.basename(fn)}", disable=not verbose):
#                 s = line.strip()
#                 if not s:
#                     continue
#                 try:
#                     obj = json.loads(s)
#                 except json.JSONDecodeError:
#                     try:
#                         dec = json.JSONDecoder()
#                         obj, _end = dec.raw_decode(s.lstrip())
#                     except Exception:
#                         continue
#                 items.append(obj)
#         return items

#     def __len__(self) -> int:
#         return len(self._items) if self.preload else len(self._offsets)

#     def _json_loads_one(self, s: str) -> Dict[str, Any]:
#         try:
#             return json.loads(s)
#         except json.JSONDecodeError as e:
#             try:
#                 dec = json.JSONDecoder()
#                 obj, end = dec.raw_decode(s.lstrip())
#                 rest = s.lstrip()[end:].strip()
#                 if rest and not self._warned_extra_once:
#                     self._warned_extra_once = True
#                     print("[WARN] JSONL line contained multiple JSON objects; "
#                           "used the first and ignored the rest. (suppressing further warnings)")
#                 return obj
#             except Exception:
#                 snippet = s[:120].replace("\n", "\\n")
#                 raise ValueError(f"Failed to parse JSONL line. Head(120)='{snippet}'") from e

#     def _read_row(self, idx: int) -> Dict[str, Any]:
#         if self.preload:
#             return self._items[idx]
#         else:
#             self._ensure_open()
#             off = self._offsets[idx]
#             self._fh.seek(off)
#             line = self._fh.readline()
#             s = line.decode("utf-8", errors="ignore").strip()
#             return self._json_loads_one(s)

#     # --------- 结构工具 ---------
#     @staticmethod
#     def _iter_section(obj: Dict[str, Any], section: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
#         block = obj.get(section) or obj.get(section.upper()) or obj.get(section.capitalize())
#         if not isinstance(block, dict):
#             return []
#         return block.items()

#     @staticmethod
#     def _pick_na_atom(coords: Dict[str, Any], key_no_prime: str, L: int) -> np.ndarray:
#         if not isinstance(coords, dict):
#             return np.full((L, 3), np.nan, dtype=np.float32)
#         for cand in PRIME_IN_MAP[key_no_prime]:
#             raw = coords.get(cand)
#             if isinstance(raw, (list, tuple)):
#                 return _to_array_L3_from_list(raw, L)
#         return np.full((L, 3), np.nan, dtype=np.float32)

#     @staticmethod
#     def _pick_prot_atom(coords: Dict[str, Any], atom: str, L: int) -> np.ndarray:
#         if not isinstance(coords, dict):
#             return np.full((L, 3), np.nan, dtype=np.float32)
#         raw = coords.get(atom)
#         return _to_array_L3_from_list(raw, L)

#     @staticmethod
#     def _resolve_is_target(types: List[str],
#                         chain_ids: List[Tuple[str, str]],
#                         target_type: Optional[str],
#                         target_chains: Optional[Dict[str, set]]) -> List[bool]:
#         """
#         规则（修改后）：
#         - 若给了 target_chains 且命中 → 仅命中的为 True；
#         - 否则若给了 target_type → 该类型为 True；
#         - 否则（target_type=None 且未给/未命中 target_chains）→ 全 False（不选择 target）。
#         """
#         M = len(types)
#         is_tgt = [False] * M

#         # 1) 链 ID 精确指定优先
#         if target_chains:
#             hit = False
#             for i, (t, cid) in enumerate(chain_ids):
#                 tset = target_chains.get(t, None)
#                 if tset and cid in tset:
#                     is_tgt[i] = True
#                     hit = True
#             if hit:
#                 return is_tgt
#             # 若提供了 target_chains 但未命中，不再回退为默认，继续看 target_type

#         # 2) 类型指定
#         if target_type:
#             for i, t in enumerate(types):
#                 if t == target_type:
#                     is_tgt[i] = True
#             return is_tgt  # 哪怕没有匹配也直接返回（保持全 False）

#         # 3) 不做任何默认选择（与原实现不同）
#         return is_tgt

#     def _subset_for_training(self,
#                              kinds: List[str],
#                              chains: List[str],
#                              seqs: List[str],
#                              coords_list: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str], List[Dict[str, Any]]]:
#         """
#         训练集链裁剪：优先保留核酸（rna/dna），不足再从蛋白随机补齐到 max_train_chains。
#         返回保持原有相对顺序的子集。
#         """
#         M = len(kinds)
#         limit = int(self.max_train_chains) if self.max_train_chains else 0
#         if self.split != "train" or limit <= 0 or M <= limit:
#             return kinds, chains, seqs, coords_list

#         na_idx   = [i for i, k in enumerate(kinds) if k in ("rna", "dna")]
#         prot_idx = [i for i, k in enumerate(kinds) if k == "protein"]

#         if len(na_idx) >= limit:
#             chosen = random.sample(na_idx, k=limit)
#         else:
#             need = limit - len(na_idx)
#             chosen = na_idx + random.sample(prot_idx, k=need)

#         chosen = sorted(chosen)  # 保持原有顺序
#         kinds_s   = [kinds[i] for i in chosen]
#         chains_s  = [chains[i] for i in chosen]
#         seqs_s    = [seqs[i] for i in chosen]
#         coords_s  = [coords_list[i] for i in chosen]
#         return kinds_s, chains_s, seqs_s, coords_s

#     # --------- 返回：一个 PDB 内所有链拼成的“复合物原始样本” ---------
#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         obj = self._read_row(idx)
#         title = obj.get("name") or obj.get("title") or f"pdb_{idx}"

#         kinds: List[str] = []
#         chains: List[str] = []
#         seqs:  List[str] = []
#         coords_list: List[Dict[str, Any]] = []

#         # 三类统一收集
#         for sec in ("PROTEIN", "RNA", "DNA"):
#             kind = _canon_kind(sec)
#             if self.include_types and (kind not in self.include_types):
#                 continue

#             for chain_id, rec in self._iter_section(obj, sec):
#                 chain_id = str(chain_id)
#                 if self.include_chain_ids:
#                     allow = self.include_chain_ids.get(kind, None)
#                     if allow is not None and chain_id not in allow:
#                         continue

#                 seq = (rec or {}).get("seq") or ""
#                 coords = (rec or {}).get("coords") or {}

#                 _ = _infer_kind_from_seq(seq)  # 仅兜底参考，不覆盖块类型

#                 kinds.append(kind)
#                 chains.append(chain_id)
#                 seqs.append(seq)
#                 coords_list.append(coords)

#         if not kinds:
#             raise RuntimeError(f"No chains parsed in row {idx} (title={title}).")

#         # --- 训练集链裁剪：优先保留核酸 ---
#         kinds, chains, seqs, coords_list = self._subset_for_training(kinds, chains, seqs, coords_list)

#         # ★ single_chain_only：若仍多链，直接取第一条（不报错）
#         if self.single_chain_only and len(kinds) > 1:
#             kinds       = kinds[:1]
#             chains      = chains[:1]
#             seqs        = seqs[:1]
#             coords_list = coords_list[:1]

#         # 目标解析（训练里你“全随机 mask”不依赖它，但保持字段以兼容评估/下游）
#         tgt = self._resolve_is_target(
#             kinds,
#             list(zip(kinds, chains)),
#             self.target_type,
#             ({k: set(v) for k, v in self.target_chains.items()} if self.target_chains else None)
#         )

#         # 组装输出
#         out: Dict[str, Any] = {
#             "type": [],
#             "seq": [],
#             "is_target": [],
#             "chain_mask": [],
#             "chain_encoding": [],
#             "N":  [], "CA": [], "C":  [], "O":  [],
#             "P":  [], "O5": [], "C5": [], "C4": [], "C3": [], "O3": [],
#         }
#         if self.return_title:
#             out["title"] = title

#         for i, (k, cid, seq, co) in enumerate(zip(kinds, chains, seqs, coords_list)):
#             s = (seq or "").upper()
#             L = len(s)

#             # 随机裁剪每条链
#             if self.max_length and self.max_length > 0 and L > self.max_length:
#                 t0 = random.randint(0, L - self.max_length)
#                 t1 = t0 + self.max_length
#                 s, L = s[t0:t1], self.max_length

#                 def slice_na(arr: np.ndarray) -> np.ndarray:
#                     return arr[t0:t1] if isinstance(arr, np.ndarray) and arr.shape[0] >= t1 else arr

#                 if k == "protein":
#                     N  = self._pick_prot_atom(co, "N",  len(seq))
#                     CA = self._pick_prot_atom(co, "CA", len(seq))
#                     C  = self._pick_prot_atom(co, "C",  len(seq))
#                     O  = self._pick_prot_atom(co, "O",  len(seq))
#                     N, CA, C, O = map(slice_na, (N, CA, C, O))
#                 else:
#                     P  = self._pick_na_atom(co, "P",  len(seq))
#                     O5 = self._pick_na_atom(co, "O5", len(seq))
#                     C5 = self._pick_na_atom(co, "C5", len(seq))
#                     C4 = self._pick_na_atom(co, "C4", len(seq))
#                     C3 = self._pick_na_atom(co, "C3", len(seq))
#                     O3 = self._pick_na_atom(co, "O3", len(seq))
#                     P, O5, C5, C4, C3, O3 = map(slice_na, (P, O5, C5, C4, C3, O3))
#             else:
#                 if k == "protein":
#                     N  = self._pick_prot_atom(co, "N",  L)
#                     CA = self._pick_prot_atom(co, "CA", L)
#                     C  = self._pick_prot_atom(co, "C",  L)
#                     O  = self._pick_prot_atom(co, "O",  L)
#                 else:
#                     P  = self._pick_na_atom(co, "P",  L)
#                     O5 = self._pick_na_atom(co, "O5", L)
#                     C5 = self._pick_na_atom(co, "C5", L)
#                     C4 = self._pick_na_atom(co, "C4", L)
#                     C3 = self._pick_na_atom(co, "C3", L)
#                     O3 = self._pick_na_atom(co, "O3", L)

#             out["type"].append(k)
#             out["seq"].append(s)
#             out["is_target"].append(bool(tgt[i]))  # 训练时可忽略，保留字段做兼容
#             out["chain_mask"].append(np.ones(L, dtype=np.float32))
#             out["chain_encoding"].append(np.full(L, fill_value=(i + 1), dtype=np.float32))

#             if k == "protein":
#                 out["N"].append(N); out["CA"].append(CA); out["C"].append(C); out["O"].append(O)
#                 for kk in NA_KEYS:
#                     out[kk].append(np.full((L, 3), np.nan, dtype=np.float32))
#             else:
#                 out["P"].append(P); out["O5"].append(O5); out["C5"].append(C5)
#                 out["C4"].append(C4); out["C3"].append(C3); out["O3"].append(O3)
#                 for kk in PROT_KEYS:
#                     out[kk].append(np.full((L, 3), np.nan, dtype=np.float32))

#         return out
