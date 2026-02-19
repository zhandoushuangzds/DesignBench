# -*- coding: utf-8 -*-
# src/datasets/complex_dataset.py

import os
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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


def _resolve_jsonl_path(path: str, split: str) -> str:
    """查找 JSONL：<path>/<split>.jsonl -> <path>/<split>_data.jsonl -> <path>/samples.jsonl"""
    split = "valid" if split in ("val", "validation") else split
    cands = [
        os.path.join(path, f"{split}.jsonl"),
        os.path.join(path, f"{split}_data.jsonl"),
        os.path.join(path, "samples.jsonl"),
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
    读取你生成的 samples.jsonl（每行一条链），按 pair_uid 归组为“复合物样本”。

    输出（坐标键与单链版一致，只是每个键变成 list，与链对齐）：
      {
        "title": pair_uid,
        "type":  [ "protein"|"rna"|"dna", ... ],
        "seq":   [ seq_i, ... ],
        "is_target": [ True/False, ... ],        # ← 新增：与链对齐
        "chain_mask":     [ np.ones(L_i), ... ],
        "chain_encoding": [ np.full(L_i, chain_idx+1), ... ],

        # 坐标键（每个键都是 list，长度=链数；对不适用的链填 NaN 或 None）
        "N":  [np(L,3) or None, ...], "CA": [...], "C": [...], "O": [...],
        "P":  [np(L,3) or None, ...], "O5": [...], "C5": [...], "C4": [...], "C3": [...], "O3": [...],

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
                 sort_by_pair_side: bool = True):
        """
        - path: 目录路径（内部会按 split 查找 jsonl）
        - split: train/val/test（valid/validation 会归一到 val）
        - max_length: 每条链长度上限（>则随机截断）
        - min_chains: 至少多少条链才算一个样本（默认 1；设为 2 可只保留成对样本）
        - require_bimolecular: True 时要求复合物中至少包含两类分子（如 protein+RNA）
        - fill_missing_with_nan: 对“不适用该原子键的链”是否填 (L,3) NaN；False 则放 None
        - sort_by_pair_side: 优先按 pair_side=chain1/chain2 排序，增强稳定性
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

        self.jsonl_file = _resolve_jsonl_path(self.path, self.split)
        self._complexes = self._load_and_group(self.jsonl_file)

        if len(self._complexes) == 0:
            raise RuntimeError(f"No complexes loaded from {self.jsonl_file} (check filters or file content).")

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
                seq = entry.get("seq")
                coords = entry.get("coords")
                if not uid or not isinstance(seq, str) or not isinstance(coords, dict):
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

    def __len__(self) -> int:
        return len(self._complexes)

    # -------- 单个复合物打包为样本 --------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chains = self._complexes[idx]

        out: Dict[str, Any] = {
            "title": chains[0].get("pair_uid", f"complex_{idx}"),
            "type": [],
            "seq": [],
            "is_target": [],                    # ← 新增字段：与链对齐
            "chain_mask": [],
            "chain_encoding": [],

            # 坐标键（值为 list，与链对齐）
            "N":  [], "CA": [], "C":  [], "O":  [],
            "P":  [], "O5": [], "C5": [], "C4": [], "C3": [], "O3": [],

            "meta": {
                "pair_uid": chains[0].get("pair_uid"),
                "pair_side": [],
                "chain_ids": [],
            }
        }

        for chain_idx, entry in enumerate(chains):
            seq: str = entry.get("seq", "")
            # kind：先看 resolved_type（prot/nuc），核酸再由序列细分 rna/dna；都缺则仅序列推断
            rt = (entry.get("resolved_type") or "").strip().lower()
            if rt == "prot":
                kind = "protein"
            elif rt == "nuc":
                kind = _infer_kind_from_seq(seq)  # rna/dna
            else:
                kind = _infer_kind_from_seq(seq)

            # 随机截断到 max_length（与其他 Dataset 保持一致）
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
            out["is_target"].append(_to_bool(entry.get("is_target", False)))  # ← 新增：每链的 is_target
            out["chain_mask"].append(np.ones(L, dtype=np.float32))
            out["chain_encoding"].append(np.full(L, fill_value=(chain_idx + 1), dtype=np.float32))

            coords: Dict[str, Any] = entry.get("coords", {}) or {}

            # 不适用键的占位
            def missing(L_: int) -> Optional[np.ndarray]:
                if not self.fill_missing_with_nan:
                    return None
                return np.full((L_, 3), np.nan, dtype=np.float32)

            # 取并切片
            def maybe_slice(arr: np.ndarray) -> np.ndarray:
                if slice_indices is None:
                    return arr
                t0, t1 = slice_indices
                return arr[t0:t1]

            if kind == "protein":
                # 写蛋白键
                for k in PROT_KEYS:
                    arr = _pick_prot_atom(coords, k, len(entry.get("seq", "")))
                    out[k].append(maybe_slice(arr))
                # 核酸键占位
                for k in NA_KEYS:
                    out[k].append(missing(L))
            else:
                # 写核酸键（RNA/DNA 统一）
                for k in NA_KEYS:
                    arr = _pick_na_atom(coords, k, len(entry.get("seq", "")))
                    out[k].append(maybe_slice(arr))
                # 蛋白键占位
                for k in PROT_KEYS:
                    out[k].append(missing(L))

            # meta
            out["meta"]["pair_side"].append(entry.get("pair_side"))
            out["meta"]["chain_ids"].append(entry.get("resolved_ids"))

        return out
