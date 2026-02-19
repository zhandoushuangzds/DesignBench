# -*- coding: utf-8 -*-
# src/datasets/complex_dataset.py

import os
import json
import mmap
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm
import torch.utils.data as data


# ---- 坐标键配置（与你现有代码一致：蛋白无撇号；核酸统一到无撇号键） ----
PROT_KEYS = ["N", "CA", "C", "O"]
NA_KEYS   = ["P", "O5", "C5", "C4", "C3", "O3"]

# 从 JSONL 的 coords（带撇号） -> 统一键（不带撇号）
PRIME_MAP = {
    "O5": "O5'",
    "C5": "C5'",
    "C4": "C4'",
    "C3": "C3'",
    "O3": "O3'",
    "O4": "O4'",   # 可能存在；默认不输出 O4，仅保留获取映射
    "C1": "C1'",
    "C2": "C2'",
}


class ChainStore:
    """
    轻量链存储：从 chains.jsonl + chains.idx 按需随机读取链记录，并用 LRU 缓存。
    - chains.idx: 每行 "key<TAB>byte_offset"
    - chains.jsonl: 每行一个 JSON（包含 "key","seq","coords"...）
    加强版：
    - 使用 mmap 定位 '\n' 切整行，避免读到“半行/多行 JSON”
    - 兼容 UTF-8 BOM 行首
    - 解析后核对 rec['key']，与 idx 保持一致
    """
    def __init__(self, dir_path: str, filename: str = "chains.jsonl", indexname: str = "chains.idx", max_cache: int = 256):
        self.dir = dir_path
        self.fn = os.path.join(dir_path, filename)
        self.idx = os.path.join(dir_path, indexname)
        self.max_cache = int(max_cache)

        if not (os.path.exists(self.fn) and os.path.exists(self.idx)):
            raise FileNotFoundError(f"Missing {self.fn} or {self.idx}")

        # 加载偏移索引（很小）
        self.offsets: Dict[str, int] = {}
        with open(self.idx, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                k, off = line.split("\t")
                self.offsets[k] = int(off)

        # 二进制随机读 + mmap
        self._fh = open(self.fn, "rb")
        # 只读 mmap，避免频繁系统调用；文件可能很大，mmap 仍然高效
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

        # LRU
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def _touch(self, k: str):
        self._cache.move_to_end(k)
        if len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)

    def get(self, key: str) -> Dict[str, Any]:
        if key in self._cache:
            self._touch(key)
            return self._cache[key]

        off = self.offsets.get(key)
        if off is None:
            raise KeyError(f"Chain key not found: {key}")

        mm = self._mm
        # 从 off 定位到下一行 '\n'（或文件尾），切出“一整行”
        end = mm.find(b"\n", off)
        if end == -1:
            end = len(mm)
        line = mm[off:end]

        # 兼容 UTF-8 BOM（极少见）
        if line.startswith(b"\xef\xbb\xbf"):
            line = line[3:]

        # 解析 & 校验
        rec = json.loads(line.decode("utf-8"))
        if rec.get("key") != key:
            raise RuntimeError(f"chains.idx offset mismatch for {key}: got {rec.get('key')} at {off}")

        self._cache[key] = rec
        self._touch(key)
        return rec

    def close(self):
        try:
            if hasattr(self, "_mm") and self._mm:
                self._mm.close()
        finally:
            try:
                if hasattr(self, "_fh") and self._fh:
                    self._fh.close()
            except Exception:
                pass


def _resolve_jsonl_path(path: str, split: str) -> str:
    """查找 JSONL：<path>/<split>.jsonl -> <path>/<split>_data.jsonl -> <path>/pairs.jsonl -> <path>/samples.jsonl"""
    split = "val" if split in ("val","valid", "validation") else split
    cands = [
        os.path.join(path, f"{split}.jsonl"),
        os.path.join(path, f"{split}_data.jsonl"),
        os.path.join(path, "test.jsonl"),     # 新：两段式样本
        os.path.join(path, "samples.jsonl"),   # 旧：内嵌 coords 的样本
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"JSONL not found. Tried: {cands}")


def _infer_kind_from_seq(seq: str) -> str:
    """根据序列推断 kind：protein / rna / dna。"""
    s = "".join([c for c in (seq or "").upper() if c.isalpha()])
    if not s:
        return "protein"
    letters = set(s)
    if letters.issubset(set("AUGC")):
        return "rna"
    if letters.issubset(set("ATGCN")):
        return "dna"
    return "protein"


def _to_array_L3(arr_list: List[Any], L: int) -> np.ndarray:
    """JSON 坐标列 -> (L,3) float32；None/False/错形状 -> NaN。"""
    if arr_list is None:
        return np.full((L, 3), np.nan, dtype=np.float32)
    out = []
    for a in arr_list:
        if a is None or a is False or not isinstance(a, (list, tuple)) or len(a) != 3:
            out.append([np.nan, np.nan, np.nan])
        else:
            x, y, z = a
            out.append([
                np.nan if x is None else float(x),
                np.nan if y is None else float(y),
                np.nan if z is None else float(z),
            ])
    out = np.asarray(out, dtype=np.float32)
    if out.shape != (L, 3):
        return np.full((L, 3), np.nan, dtype=np.float32)
    return out


def _pick_na_atom(coords: Dict[str, Any], atom_no_prime: str, L: int) -> np.ndarray:
    """从 coords（带撇号的键）里拿核酸原子，输出 (L,3)。"""
    if atom_no_prime == "P":
        raw = coords.get("P")
    else:
        raw = coords.get(PRIME_MAP.get(atom_no_prime, atom_no_prime + "'"))
    return _to_array_L3(raw, L)


def _pick_prot_atom(coords: Dict[str, Any], atom_key: str, L: int) -> np.ndarray:
    raw = coords.get(atom_key)
    return _to_array_L3(raw, L)


def _to_bool(x: Any, default: bool = False) -> bool:
    """把多种形式（bool/int/str）稳妥转为布尔值。"""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return True
        if s in {"0", "false", "f", "no", "n"}:
            return False
    return default


class ComplexDataset(data.Dataset):
    """
    读取 samples/pairs.jsonl（每行一条链），按 pair_uid 归组为“复合物样本”。

    输出（坐标键与单链版一致，只是每个键变成 list，与链对齐）：
      {
        "title": pair_uid,
        "type":  [ "protein"|"rna"|"dna", ... ],
        "seq":   [ seq_i, ... ],
        "is_target": [ True/False, ... ],
        "chain_mask":     [ np.ones(L_i), ... ],
        "chain_encoding": [ np.full(L_i, chain_idx+1), ... ],
        "N"/"CA"/"C"/"O" / "P"/"O5"/"C5"/"C4"/"C3"/"O3": [np(L,3) or None, ...],
        "meta": {
          "pair_uid": "...",
          "pair_side": ["chain1","chain2",...],
          "chain_ids": [{"auth":..., "label":...}, ...]
        }
      }
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 max_length: int = 500,
                 min_chains: int = 1,
                 require_bimolecular: bool = False,
                 fill_missing_with_nan: bool = True,
                 sort_by_pair_side: bool = True,
                 chain_cache_size: int = 256):
        """
        - path: 目录路径（内部会按 split 查找 jsonl；两段式写在同目录下）
        - split: train/val/test（valid/validation 会归一到 val）
        - max_length: 每条链长度上限（>则随机截断）
        - min_chains: 至少多少条链才算一个样本（默认 1；设为 2 可只保留成对样本）
        - require_bimolecular: True 时要求复合物中至少包含两类分子（如 protein+RNA）
        - fill_missing_with_nan: 对“不适用该原子键的链”是否填 (L,3) NaN；False 则放 None
        - sort_by_pair_side: 优先按 pair_side=chain1/chain2 排序，增强稳定性
        - chain_cache_size: 链 LRU 缓存大小（仅两段式引用模式生效）
        """
        super().__init__()
        self.path = path
        self.split = "val" if split in ("valid", "validation") else split
        assert self.split in ("train", "val", "test"), f"split must be train/val/test, got {self.split}"

        self.max_length = int(max_length)
        self.min_chains = int(min_chains)
        self.require_bimolecular = bool(require_bimolecular)
        self.fill_missing_with_nan = bool(fill_missing_with_nan)
        self.sort_by_pair_side = bool(sort_by_pair_side)

        # 1) pairs/samples
        self.jsonl_file = _resolve_jsonl_path(self.path, self.split)

        # 2) 尝试加载链存储（两段式）——若不存在则兼容旧数据（内嵌 coords/seq）
        try:
            self.chain_store = ChainStore(self.path, max_cache=int(chain_cache_size))
        except FileNotFoundError:
            self.chain_store = None

        self._complexes = self._load_and_group(self.jsonl_file)

        if len(self._complexes) == 0:
            raise RuntimeError(f"No complexes loaded from {self.jsonl_file} (check filters or file content).")

    def __del__(self):
        if getattr(self, "chain_store", None) is not None:
            self.chain_store.close()

    # -------- 加载并按 pair_uid 分组 --------
    def _load_and_group(self, fn: str) -> List[List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        with open(fn, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading Complex JSONL: {os.path.basename(fn)}"):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue

                uid = entry.get("pair_uid")
                if not uid:
                    continue

                # 至少具备：内嵌 seq/coords 或 引用 coords_ref/seq_ref
                has_seq = isinstance(entry.get("seq"), str)
                has_coords = isinstance(entry.get("coords"), dict)
                has_ref = isinstance(entry.get("coords_ref"), str) or isinstance(entry.get("seq_ref"), str)
                if not (has_seq or has_coords or has_ref):
                    continue

                groups[str(uid)].append(entry)

        complexes: List[List[Dict[str, Any]]] = []
        for uid, chains in groups.items():
            if self.sort_by_pair_side:
                def side_rank(s: Optional[str]) -> int:
                    s = (s or "").lower()
                    return 0 if s == "chain1" else (1 if s == "chain2" else 2)
                chains = sorted(
                    chains,
                    key=lambda e: (
                        side_rank(e.get("pair_side")),
                        (e.get("resolved_ids") or {}).get("auth", ""),
                        (e.get("resolved_ids") or {}).get("label", ""),
                    )
                )
            if len(chains) < self.min_chains:
                continue

            if self.require_bimolecular:
                kinds = []
                for e in chains:
                    rt = (e.get("resolved_type") or "").strip().lower()
                    if rt == "prot":
                        kinds.append("protein")
                    elif rt == "nuc":
                        kinds.append("nuc")
                if len(set(kinds)) < 2:
                    continue

            complexes.append(chains)
        return complexes

    # ---- 引用 （拿到 seq/coords），输出条目结构保持不变 ----
    def _materialize_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(entry.get("coords"), dict) and isinstance(entry.get("seq"), str):
            return entry  # 老格式，直接用

        if self.chain_store is None:
            raise RuntimeError("Entry uses refs but chains.jsonl / chains.idx not found beside dataset path.")

        key = entry.get("coords_ref") or entry.get("seq_ref") or entry.get("chain_ref")
        if not key:
            raise RuntimeError("Entry has no coords/seq nor refs.")
        rec = self.chain_store.get(key)

        out = dict(entry)
        out["seq"] = rec["seq"]
        out["coords"] = rec["coords"]
        return out

    def __len__(self) -> int:
        return len(self._complexes)

    # -------- 单个复合物打包为样本 --------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_chains = self._complexes[idx]
        chains = [self._materialize_entry(e) for e in raw_chains]

        out: Dict[str, Any] = {
            "title": chains[0].get("pair_uid", f"complex_{idx}"),
            "type": [],
            "seq": [],
            "is_target": [],
            "chain_mask": [],
            "chain_encoding": [],
            "N":  [], "CA": [], "C":  [], "O":  [],
            "P":  [], "O5": [], "C5": [], "C4": [], "C3": [], "O3": [],
            "meta": {
                "pair_uid": chains[0].get("pair_uid"),
                "pair_side": [],
                "chain_ids": [],
            }
        }

        def missing(L_: int) -> Optional[np.ndarray]:
            if not self.fill_missing_with_nan:
                return None
            return np.full((L_, 3), np.nan, dtype=np.float32)

        for chain_idx, entry in enumerate(chains):
            seq: str = entry.get("seq", "")
            rt = (entry.get("resolved_type") or "").strip().lower()
            if rt == "prot":
                kind = "protein"
            elif rt == "nuc":
                kind = _infer_kind_from_seq(seq)  # rna/dna
            else:
                kind = _infer_kind_from_seq(seq)

            s = seq or ""
            L = len(s)
            if self.max_length and L > self.max_length:
                import random
                t0 = random.randint(0, L - self.max_length)
                t1 = t0 + self.max_length
                s = s[t0:t1]
                L = len(s)
                slice_indices = (t0, t1)
            else:
                slice_indices = None

            out["type"].append(kind)
            out["seq"].append(s)
            out["is_target"].append(_to_bool(entry.get("is_target", False)))
            out["chain_mask"].append(np.ones(L, dtype=np.float32))
            out["chain_encoding"].append(np.full(L, fill_value=(chain_idx + 1), dtype=np.float32))

            coords: Dict[str, Any] = entry.get("coords", {}) or {}

            def maybe_slice(arr: np.ndarray) -> np.ndarray:
                if slice_indices is None:
                    return arr
                t0, t1 = slice_indices
                return arr[t0:t1]

            if kind == "protein":
                for k in PROT_KEYS:
                    arr = _pick_prot_atom(coords, k, len(entry.get("seq", "")))
                    out[k].append(maybe_slice(arr))
                for k in NA_KEYS:
                    out[k].append(missing(L))
            else:
                for k in NA_KEYS:
                    arr = _pick_na_atom(coords, k, len(entry.get("seq", "")))
                    out[k].append(maybe_slice(arr))
                for k in PROT_KEYS:
                    out[k].append(missing(L))

            out["meta"]["pair_side"].append(entry.get("pair_side"))
            out["meta"]["chain_ids"].append(entry.get("resolved_ids"))

        return out
