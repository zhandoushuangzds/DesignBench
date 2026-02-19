# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Set
from tqdm import tqdm
import torch.utils.data as data


class CATHDataset(data.Dataset):
    """
    从 CATH 数据集读取蛋白单链样本（默认 4.3）。
    目录期望包含：
      - chain_set.jsonl
      - chain_set_splits.json
      - (可选) test_split_L100.json
      - (可选) test_split_sc.json

    输出与 featurize_UniIF().featurize 兼容（单链）：
      {
        "title": str,
        "type": "protein",
        "seq": [str],                              # 注意：list 包一层
        "N":  np.ndarray(L,3),
        "CA": np.ndarray(L,3),
        "C":  np.ndarray(L,3),
        "O":  np.ndarray(L,3),
        "chain_mask":     [np.ones(L, dtype=np.float32)],
        "chain_encoding": [np.ones(L, dtype=np.float32)],
      }
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 max_length: int = 500,
                 version: str = "4.3",
                 test_name: Optional[str] = None,
                 jsonl_name: str = "chain_set.jsonl",
                 splits_name: str = "chain_set_splits.json"):
        super().__init__()
        # 规范 split
        split_norm = "val" if split in ("valid", "validation") else split
        assert split_norm in ("train", "val", "test"), f"split must be train/val/test, got {split}"

        self.version = str(version)          # 保留接口，默认 4.3
        self.path = path
        self.split = split_norm
        self.max_length = max_length
        self.test_name = (test_name or "").lower()
        self.atom_lst = ["N", "CA", "C", "O"]

        jsonl_path = os.path.join(path, jsonl_name)
        splits_path = os.path.join(path, splits_name)
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Missing {jsonl_path}")
        if not os.path.exists(splits_path):
            raise FileNotFoundError(f"Missing {splits_path}")

        # 用“返回 items 的函数”来构建数据
        self.data: List[Dict[str, Any]] = self._load_jsonl(jsonl_path, splits_path)
        if len(self.data) == 0:
            raise RuntimeError(f"No samples loaded for split={self.split} (after filtering).")


    def _load_jsonl(self, jsonl_path: str, splits_path: str) -> List[Dict[str, Any]]:
        # 读取基础划分
        with open(splits_path, "r", encoding="utf-8") as f:
            dataset_splits = json.load(f)

        # 测试集可选替换
        if self.split == "test" and self.test_name in ("l100", "sc"):
            repl_file = "test_split_L100.json" if self.test_name == "l100" else "test_split_sc.json"
            repl_path = os.path.join(self.path, repl_file)
            if os.path.exists(repl_path):
                with open(repl_path, "r", encoding="utf-8") as f:
                    test_splits = json.load(f)
                if "test" in test_splits:
                    dataset_splits["test"] = test_splits["test"]

        # 目标 name 集合
        if self.split == "train":
            wanted_names: Set[str] = set(dataset_splits.get("train", []))
        elif self.split == "val":
            wanted_names = set(dataset_splits.get("validation", []))
        else:
            wanted_names = set(dataset_splits.get("test", []))

        if not wanted_names:
            raise RuntimeError(f"No names found for split={self.split} in {splits_path}")

        alphabet_set = set("ACDEFGHIKLMNPQRSTVWY")
        items: List[Dict[str, Any]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading CATH({self.version}) {self.split}: {os.path.basename(jsonl_path)}"):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue

                name = entry.get("name") or entry.get("title")
                if not name or name not in wanted_names:
                    continue

                seq_raw = entry.get("seq", "")
                if not isinstance(seq_raw, str) or len(seq_raw) == 0:
                    continue
                seq = seq_raw.upper()
                # 仅保留标准 20 AA
                if set(seq).difference(alphabet_set):
                    continue
                L = len(seq)

                coords = entry.get("coords", {}) or {}

                def pick(atom_key: str) -> np.ndarray:
                    """coords[atom_key] -> (L,3)；空或错形状则填 NaN"""
                    arr = coords.get(atom_key)
                    if arr is None:
                        return np.full((L, 3), np.nan, dtype=np.float32)
                    out = np.array(
                        [([np.nan, np.nan, np.nan] if (a is None or a is False) else a) for a in arr],
                        dtype=np.float32
                    )
                    if out.shape != (L, 3):
                        return np.full((L, 3), np.nan, dtype=np.float32)
                    return out

                item = {
                    "title": name,
                    "type": "protein",
                    "seq": [seq],
                    "N":  pick("N"),
                    "CA": pick("CA"),
                    "C":  pick("C"),
                    "O":  pick("O"),
                    "chain_mask":     [np.ones(L, dtype=np.float32)],
                    "chain_encoding": [np.ones(L, dtype=np.float32)],
                }
                items.append(item)

        return items


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        seq = item["seq"][0]
        L = len(seq)

        # 长序列随机截断到 max_length
        if L > self.max_length:
            import random
            t0 = random.randint(0, L - self.max_length)
            t1 = t0 + self.max_length
            return {
                "title": item["title"],
                "type":  "protein",
                "seq": [seq[t0:t1]],
                "N":  item["N"][t0:t1],
                "CA": item["CA"][t0:t1],
                "C":  item["C"][t0:t1],
                "O":  item["O"][t0:t1],
                "chain_mask":     [item["chain_mask"][0][t0:t1]],
                "chain_encoding": [item["chain_encoding"][0][t0:t1]],
            }

        # 不截断
        return {
            "title": item["title"],
            "type":  "protein",
            "seq": [seq],
            "N":  item["N"],
            "CA": item["CA"],
            "C":  item["C"],
            "O":  item["O"],
            "chain_mask":     item["chain_mask"],
            "chain_encoding": item["chain_encoding"],
        }
