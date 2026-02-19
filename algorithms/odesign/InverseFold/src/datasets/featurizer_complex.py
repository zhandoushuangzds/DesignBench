# -*- coding: utf-8 -*-
# src/datasets/featurizer_complex.py

import torch
import numpy as np
from typing import List, Optional, Sequence, Union, Dict, Any,Tuple
import torch.nn.functional as F

from torch_geometric.nn.pool import knn_graph
from torch_scatter import scatter_sum

from ..tools.affine_utils import Rigid, Rotation, get_interact_feats
from ..modules.if_module import *  # rbf()

# ---------------- Tokenizer (统一版) ----------------
class Unitokenizer_Complex:
    """
    Unified tokenizer with a dedicated mask token for protein / DNA / RNA.

    Protein: 20 aa -> [0..19], plus X -> 26
    Nucleotides: A,C,G,T,U -> [20..24], plus N -> 27
    MASK: '*' -> 25
    vocab_size = 28
    """
    def __init__(self, map_N_mode: str = "keep", mask_char: str = "*", unknown_to_mask: bool = False):
        # protein
        self.alphabet_protein = 'ACDEFGHIKLMNPQRSTVWY'         # 0..19
        self.protein_to_id = {ch: i for i, ch in enumerate(self.alphabet_protein)}
        self.protein_X_id = 26                                  # X -> 26

        # nucleotides
        self.nuc5 = ['A', 'C', 'G', 'T', 'U']                  # 20..24
        self.nuc_to_id = {b: 20 + i for i, b in enumerate(self.nuc5)}
        self.nuc_N_id = 27                                      # N -> 27
        self.nuc_to_id['N'] = self.nuc_N_id

        # mask
        self.mask_char = mask_char
        self.mask_id = 25

        # vocab size (20 aa + 5 nuc + MASK + X + N) = 28
        self.vocab_size = 28

        self.map_N_mode = map_N_mode    # 'keep' | 'A' | 'random'
        self.unknown_to_mask = unknown_to_mask

        # convenient id sets
        self.protein_ids = list(range(20)) + [self.protein_X_id]
        self.dna_ids = [self.nuc_to_id[x] for x in ['A','C','G','T']] + [self.nuc_N_id]
        self.rna_ids = [self.nuc_to_id[x] for x in ['A','C','G','U']] + [self.nuc_N_id]

    def _map_Ns(self, s: str) -> str:
        if 'N' not in s: return s
        if self.map_N_mode == 'keep':
            return s
        if self.map_N_mode == 'A':
            return s.replace('N', 'A')
        elif self.map_N_mode == 'random':
            import random
            return ''.join(ch if ch!='N' else random.choice('ACGT') for ch in s)
        else:
            return s

    def _auto_kind(self, s: str) -> str:
        S = set(s)
        S.discard(self.mask_char)
        # allow N in both dna/rna auto-detect
        if S.issubset(set('AUGCN')):   return 'rna'
        if S.issubset(set('ATGCN')):   return 'dna'
        return 'protein'

    def encode(self, seq: str, kind: Optional[str] = None) -> List[int]:
        s = seq.upper()
        if kind is None:
            kind = self._auto_kind(s)

        if kind == 'protein':
            out = []
            for ch in s:
                if ch == self.mask_char:
                    out.append(self.mask_id); continue
                if ch in self.protein_to_id:
                    out.append(self.protein_to_id[ch]); continue
                if ch == 'X':
                    out.append(self.protein_X_id); continue
                # unknown AA
                out.append(self.mask_id if self.unknown_to_mask else self.protein_X_id)
            return out

        if kind == 'dna':
            s = s.replace('U','T')
            s = self._map_Ns(s)
        elif kind == 'rna':
            s = s.replace('T','U')
            s = self._map_Ns(s)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        out = []
        for ch in s:
            if ch == self.mask_char:
                out.append(self.mask_id); continue
            if ch in self.nuc_to_id:
                out.append(self.nuc_to_id[ch])
            else:
                # unknown base -> mask or 'A'
                out.append(self.mask_id if self.unknown_to_mask else self.nuc_to_id['A'])
        return out

    def decode(self, ids: Sequence[int], kind: Optional[str] = None) -> str:
        out = []
        for t in ids:
            if t == self.mask_id:
                out.append(self.mask_char)
            elif 0 <= t < 20:
                out.append(self.alphabet_protein[t])
            elif t == self.protein_X_id:
                out.append('X')
            elif 20 <= t <= 24 or t == self.nuc_N_id:
                if t == self.nuc_N_id:
                    out.append('N'); continue
                base = self.nuc5[t-20]
                if kind == 'dna' and base == 'U': base = 'T'
                if kind == 'rna' and base == 'T': base = 'U'
                out.append(base)
            else:
                out.append(self.mask_char)
        return ''.join(out)

# ---------------- Featurizer (统一版) ----------------

class Featurize_Complex:
    """
    把单体或复合物样本统一拼成一张图（单体自动视作“只有一条链的复合物”）。
    使用 6 槽位坐标 X[L_total, 6, 3]：
        - protein: [N, CA, C, O, NaN, NaN]；anchors=(0,1,2)
        - RNA:     [P, O5, C5, C4, C3, O3]；anchors=(0,1,2)
        - DNA:     [O5, C5, C4, P, C3, O3]；anchors=(0,1,2)
    仅产出模型所需字段；不再在 featurizer 内做 mask/target 决策。
    """
    def __init__(self, knn_k: int = 48, virtual_frame_num: int = 3) -> None:
        self.tokenizer = Unitokenizer_Complex()
        self.virtual_frame_num = virtual_frame_num
        self.knn_k = knn_k

        self.A_TOTAL = 6
        self.MIXED_NODE_IN = 114
        self.MIXED_EDGE_IN = 272

    # ---------- 输入标准化：把单体转成“只有一条链的复合物” ----------
    @staticmethod
    def _normalize_input(item: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(item)  # 浅拷贝
        t = out.get('type', None)

        # 已是复合物格式
        if isinstance(t, (list, tuple)):
            return out

        # 单体格式 -> 包装为复合物（保持 title/chain_names/pdb_id 等元信息不变）
        if isinstance(t, str):
            out['type'] = [t]
            seq = out.get('seq', '')
            if isinstance(seq, str):
                out['seq'] = [seq]
            elif isinstance(seq, (list, tuple)):
                out['seq'] = [seq[0]]
            else:
                raise ValueError("Single-chain input requires 'seq' as str or [str].")

            for key, val in list(out.items()):
                if key in ('N','CA','C','O','P','O5','C5','C4','C3','O3'):
                    if isinstance(val, np.ndarray):
                        out[key] = [val]
                    elif isinstance(val, (list, tuple)):
                        if len(val) != 1 or not isinstance(val[0], np.ndarray):
                            raise ValueError(f"Expect single-chain coords for key '{key}' as np.ndarray or [np.ndarray].")
                    else:
                        pass

            if 'chain_mask' in out and isinstance(out['chain_mask'], np.ndarray):
                out['chain_mask'] = [out['chain_mask']]
            if 'chain_encoding' in out and isinstance(out['chain_encoding'], np.ndarray):
                out['chain_encoding'] = [out['chain_encoding']]

            # 注意：不过度改动 title/chain_names/pdb_id，保持原样
            return out

        raise ValueError("Input 'type' must be either str (single chain) or list (complex).")

    # ---------- 将 ComplexDataset 的 item 转成拼接后的 (S, X, mask 等) ----------
    def _flatten_complex(self, item):
        item = self._normalize_input(item)

        kinds: List[str] = [str(x).lower() for x in item["type"]]  # ['protein','rna',...]
        seqs:  List[str] = item["seq"]
        type2id = {"protein": 0, "rna": 1, "dna": 2}

        X_list = []
        type_vec_full = []
        S_flat = []
        mask_full = []
        enc_full = []
        slot_expect = []

        # —— 新增：为计算过滤后的 chain_spans 做准备 ——
        chain_idx_full: List[int] = []   # 长度为 L_total（过滤前），记录每个残基属于哪条链
        lens_per_chain: List[int] = []   # 每条链长度（过滤前）

        # 可能存在的元信息（来自上游 Dataset）
        titles_in = item.get("title", None)          # 可能是 list[str] 或 list-of-lists
        chain_names_in = item.get("chain_names", None)  # list[str] 或 list-of-lists
        pdb_id_in = item.get("pdb_id", None)

        def get_arr(key: str, idx: int, L: int) -> np.ndarray:
            arr = item.get(key, None)
            if arr is None or idx >= len(arr) or arr[idx] is None:
                return np.full((L, 3), np.nan, dtype=np.float32)
            a = arr[idx]
            if not isinstance(a, np.ndarray) or a.shape != (L, 3):
                return np.full((L, 3), np.nan, dtype=np.float32)
            return a.astype(np.float32, copy=False)

        for i, kind in enumerate(kinds):
            seq = (seqs[i] or "").upper()
            L = len(seq)
            lens_per_chain.append(L)
            chain_idx_full.extend([i] * L)

            if kind == "protein":
                N  = get_arr("N",  i, L)
                CA = get_arr("CA", i, L)
                C  = get_arr("C",  i, L)
                O  = get_arr("O",  i, L)
                S6 = np.stack([N, CA, C, O,
                               np.full((L,3), np.nan, np.float32),
                               np.full((L,3), np.nan, np.float32)], axis=1)
                expect = np.zeros((L, 6), dtype=bool); expect[:, :4] = True

            elif kind == "rna":
                P  = get_arr("P",  i, L)
                O5 = get_arr("O5", i, L)
                C5 = get_arr("C5", i, L)
                C4 = get_arr("C4", i, L)
                C3 = get_arr("C3", i, L)
                O3 = get_arr("O3", i, L)
                S6 = np.stack([P, O5, C5, C4, C3, O3], axis=1)
                expect = np.ones((L, 6), dtype=bool)

            elif kind == "dna":
                O5 = get_arr("O5", i, L)
                C5 = get_arr("C5", i, L)
                C4 = get_arr("C4", i, L)
                P  = get_arr("P",  i, L)
                C3 = get_arr("C3", i, L)
                O3 = get_arr("O3", i, L)
                S6 = np.stack([O5, C5, C4, P, C3, O3], axis=1)
                expect = np.ones((L, 6), dtype=bool)

            else:
                raise ValueError(f"Unknown kind: {kind}")

            X_list.append(torch.from_numpy(S6))              # (L,6,3)
            slot_expect.append(torch.from_numpy(expect))     # (L,6)

            # token ids & type ids
            S_flat.extend(self.tokenizer.encode(seq, kind))
            type_id = type2id.get(kind, 0)
            type_vec_full.extend([type_id] * L)

            # masks / encodings
            mask_full.append(torch.from_numpy(item["chain_mask"][i].astype(np.float32)))
            enc_full.append(torch.from_numpy(item["chain_encoding"][i].astype(np.float32)))

        X6 = torch.cat(X_list, dim=0)                          # [L_total, 6, 3]
        slot_expect_mask = torch.cat(slot_expect, dim=0)       # [L_total, 6]
        S_flat = torch.tensor(S_flat, dtype=torch.long)        # [L_total]
        type_vec_full = torch.tensor(type_vec_full, dtype=torch.long)
        chain_mask_full = torch.cat(mask_full, dim=0).float()  # [L_total]
        chain_encoding_full = torch.cat(enc_full, dim=0).float()

        # ---------- 有效位点过滤 ----------
        L_total = X6.shape[0]
        valid_per_slot = torch.isfinite(X6).all(dim=2)
        eff_ok = (~slot_expect_mask) | valid_per_slot
        mask_bool = eff_ok.all(dim=1)  # [L_total]
        if mask_bool.sum() < 2:
            return None

        # ---------- 压紧到 N ----------
        X = X6[mask_bool]                                  # [N, 6, 3]
        S = S_flat[mask_bool]                              # [N]
        type_vec = type_vec_full[mask_bool]                # [N]
        chain_mask = chain_mask_full[mask_bool]            # [N]
        chain_encoding = chain_encoding_full[mask_bool]    # [N]
        X[torch.isnan(X)] = 0.0

        # ---------- 计算过滤后的 chain_spans ----------
        # chain_idx_full: [L_total]，每个位置所属链的索引
        chain_idx_full_t = torch.tensor(chain_idx_full, dtype=torch.long, device=mask_bool.device)
        kept_chain_idx = chain_idx_full_t[mask_bool]  # [N]
        C = len(lens_per_chain)
        counts = [(kept_chain_idx == i).sum().item() for i in range(C)]
        spans = []
        cur = 0
        for cnt in counts:
            spans.append([cur, cur + cnt])
            cur += cnt
        chain_spans = torch.tensor(spans, dtype=torch.long)  # [C,2]（注意：某条链可能在过滤后长度为 0）

        # ---------- 单 batch 其它几何/图结构（原样） ----------
        N = X.shape[0]
        device = X.device
        batch_id = torch.zeros(N, dtype=torch.long, device=device)

        center = X[:, 1, :]
        edge_idx = knn_graph(center, k=self.knn_k, batch=batch_id, loop=False, flow='target_to_source')
        key = (edge_idx[1] * (edge_idx[0].max() + 1) + edge_idx[0]).long()
        order = torch.argsort(key)
        edge_idx = edge_idx[:, order]

        T = Rigid.make_transform_from_reference(X[:, 0].float(), X[:, 1].float(), X[:, 2].float())
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        T_ts = T[dst_idx, None].invert().compose(T[src_idx, None])

        num_global = self.virtual_frame_num
        X_c = T._trans
        X_m = X_c.mean(dim=0, keepdim=True)
        X_c = X_c - X_m
        U, Ssvd, V = torch.svd(X_c.T @ X_c)
        d = (torch.det(U) * torch.det(V)) < 0.0
        D = torch.zeros_like(V); D[[0, 1], [0, 1]] = 1; D[2, 2] = -1 * d + 1 * (~d)
        V = D @ V
        R = torch.matmul(U, V.permute(0, 1))

        rot_g = [R] * num_global
        trans_g = [X_m] * num_global

        feat = get_interact_feats(T, T_ts, X.float(), edge_idx, batch_id)
        _V, _E = feat["_V"], feat["_E"]

        T_g = Rigid(Rotation(torch.stack(rot_g)), torch.cat(trans_g, dim=0))
        num_nodes = scatter_sum(torch.ones_like(batch_id), batch_id)
        global_src = torch.cat([batch_id + k * num_nodes.shape[0] for k in range(num_global)]) + num_nodes
        global_dst = torch.arange(batch_id.shape[0], device=device).repeat(num_global)
        edge_idx_g = torch.stack([global_dst, global_src])
        edge_idx_g_inv = torch.stack([global_src, global_dst])
        edge_idx_g = torch.cat([edge_idx_g, edge_idx_g_inv], dim=1)

        batch_id_g = torch.zeros(num_global, dtype=batch_id.dtype, device=device)
        T_all = Rigid.cat([T, T_g], dim=0)
        idx_min, _ = edge_idx_g.min(dim=0)
        T_gs = T_all[idx_min, None].invert().compose(T_all[idx_min, None])

        rbf_ts = rbf(T_ts._trans.norm(dim=-1), 0, 50, 16)[:, 0].view(_E.shape[0], -1)
        rbf_gs = rbf(T_gs._trans.norm(dim=-1), 0, 50, 16)[:, 0].view(edge_idx_g.shape[1], -1)

        _V_g = torch.arange(num_global, device=device)
        _E_g = torch.zeros([edge_idx_g.shape[1], 128], device=device)

        # ---------- 输出（仅新增元信息字段，其它不变） ----------
        batch_out = {
            "type_vec": type_vec,            # [N] 0/1/2
            "T": T, "T_g": T_g, "T_ts": T_ts, "T_gs": T_gs,
            "rbf_ts": rbf_ts, "rbf_gs": rbf_gs,

            "X": X,                           # [N,6,3]
            "_V": _V, "_E": _E,
            "_V_g": _V_g, "_E_g": _E_g,

            "S": S,                            # 真值
            "edge_idx": edge_idx, "edge_idx_g": edge_idx_g,
            "batch_id": batch_id, "batch_id_g": batch_id_g,
            "num_nodes": num_nodes,            # [1]

            "mask": chain_mask,                # [N]
            "chain_mask": chain_mask,          # [N]
            "chain_encoding": chain_encoding,  # [N]

            # 展开刚体（保持原接口）
            "T_rot":  T._rots._rot_mats,   "T_trans":  T._trans,
            "T_g_rot":T_g._rots._rot_mats, "T_g_trans":T_g._trans,
            "T_ts_rot":T_ts._rots._rot_mats,"T_ts_trans":T_ts._trans,
            "T_gs_rot":T_gs._rots._rot_mats,"T_gs_trans":T_gs._trans,

            "K_g": self.virtual_frame_num,

            # ★★★ 新增：供 MInterface 按链切 FASTA 的元信息 ★★★
            "chain_spans": chain_spans,                 # [C,2]（过滤后索引空间）
            "chain_names": chain_names_in,              # 原样透传（可能是 list[str]/list-of-lists/None）
            "title": titles_in,                         # 原样透传
            "pdb_id": pdb_id_in,                        # 原样透传
        }
        return batch_out

    # ---------- collate ----------
    def featurize(self, batch: List[dict]):
        samples = []
        for one in batch:
            feat = self._get_features_persample(one)
            if feat is not None:
                samples.append(feat)
        if not samples:
            return {"_skip_batch": True}
        return self.custom_collate_fn(samples)

    def custom_collate_fn(self, batch: List[dict]):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        K_g = int(batch[0]["K_g"])
        num_nodes_list = [int(b["num_nodes"][0]) for b in batch]
        B = len(batch)
        total_real = sum(num_nodes_list)

        prefix_real = torch.tensor([0] + list(torch.cumsum(torch.tensor(num_nodes_list[:-1]), dim=0).tolist()),
                                   dtype=torch.long)
        base_virtual = total_real

        def remap_indices(local_idx: torch.Tensor, N_i: int, base_real_i: int, base_virt_i: int) -> torch.Tensor:
            is_virtual = (local_idx >= N_i)
            out = local_idx.clone()
            out[~is_virtual] += base_real_i
            out[is_virtual] = (local_idx[is_virtual] - N_i) + (base_virtual + base_virt_i)
            return out

        ret = {}
        # 必须键
        cat_keys = [
            "X", "_V", "_E", "_V_g", "_E_g",
            "S",
            "type_vec",
            "mask", "chain_mask", "chain_encoding",
        ]
        for k in cat_keys:
            ret[k] = torch.cat([b[k] for b in batch], dim=0)

        # 刚体
        for k in ["T", "T_g", "T_ts", "T_gs"]:
            T_cat = Rigid.cat([b[k] for b in batch], dim=0)
            ret[k + "_rot"]   = T_cat._rots._rot_mats
            ret[k + "_trans"] = T_cat._trans

        ret["num_nodes"] = torch.tensor(num_nodes_list, dtype=torch.long)
        ret["batch_id"]  = torch.cat([torch.full((num_nodes_list[i],), i, dtype=torch.long) for i in range(B)], dim=0)
        ret["batch_id_g"] = torch.cat([torch.full((K_g,), i, dtype=torch.long) for i in range(B)], dim=0)

        # 边重映射
        edge_parts = []
        for i, b in enumerate(batch):
            shift = prefix_real[i]
            edge_parts.append(b["edge_idx"] + shift)
        ret["edge_idx"] = torch.cat(edge_parts, dim=1)

        edge_g_parts = []
        for i, b in enumerate(batch):
            N_i = num_nodes_list[i]
            base_real_i = int(prefix_real[i].item())
            base_virt_i = i * K_g
            src_local = b["edge_idx_g"][0]
            dst_local = b["edge_idx_g"][1]
            src_global = remap_indices(src_local, N_i, base_real_i, base_virt_i)
            dst_global = remap_indices(dst_local, N_i, base_real_i, base_virt_i)
            edge_g_parts.append(torch.stack([src_global, dst_global], dim=0))
        ret["edge_idx_g"] = torch.cat(edge_g_parts, dim=1)

        # 其他
        ret["rbf_ts"] = torch.cat([b["rbf_ts"] for b in batch], dim=0)
        ret["rbf_gs"] = torch.cat([b["rbf_gs"] for b in batch], dim=0)
        ret["_V_g"]   = torch.cat([b["_V_g"] for b in batch], dim=0)
        ret["_E_g"]   = torch.cat([b["_E_g"] for b in batch], dim=0)
        ret["K_g"]    = K_g

        # ★★★ 新增：把元信息以“逐样本列表”的形式带出（MInterface 已支持 list-of-…） ★★★
        ret["chain_spans"]  = [b.get("chain_spans", None) for b in batch]   # list of [C,2] tensors / None
        ret["chain_names"]  = [b.get("chain_names", None) for b in batch]   # list of list[str] / None
        ret["title"]        = [b.get("title", None) for b in batch]         # list of list[str]/str/None
        ret["pdb_id"]       = [b.get("pdb_id", None) for b in batch]        # list of str / None

        return ret

    def _get_features_persample(self, item: dict):
        # 兼容你现有接口：名字不变
        return self._flatten_complex(item)


    
    
# class Featurize_Complex:
#     """
#     把单体或复合物样本统一拼成一张图（单体自动视作“只有一条链的复合物”）。
#     使用 6 槽位坐标 X[L_total, 6, 3]：
#         - protein: [N, CA, C, O, NaN, NaN]；anchors=(0,1,2)
#         - RNA:     [P, O5, C5, C4, C3, O3]；anchors=(0,1,2)
#         - DNA:     [O5, C5, C4, P, C3, O3]；anchors=(0,1,2)
#     仅产出模型所需字段；不再在 featurizer 内做 mask/target 决策。
#     """
#     def __init__(self, knn_k: int = 48, virtual_frame_num: int = 3) -> None:
#         self.tokenizer = Unitokenizer_Complex()
#         self.virtual_frame_num = virtual_frame_num
#         self.knn_k = knn_k

#         self.A_TOTAL = 6
#         self.MIXED_NODE_IN = 114
#         self.MIXED_EDGE_IN = 272

#     # ---------- 输入标准化：把单体转成“只有一条链的复合物” ----------
#     @staticmethod
#     def _normalize_input(item: Dict[str, Any]) -> Dict[str, Any]:
#         out = dict(item)  # 浅拷贝
#         t = out.get('type', None)

#         # 已是复合物格式
#         if isinstance(t, (list, tuple)):
#             return out

#         # 单体格式 -> 包装为复合物
#         if isinstance(t, str):
#             out['type'] = [t]
#             seq = out.get('seq', '')
#             if isinstance(seq, str):
#                 out['seq'] = [seq]
#             elif isinstance(seq, (list, tuple)):
#                 out['seq'] = [seq[0]]
#             else:
#                 raise ValueError("Single-chain input requires 'seq' as str or [str].")

#             for key, val in list(out.items()):
#                 if key in ('N','CA','C','O','P','O5','C5','C4','C3','O3'):
#                     if isinstance(val, np.ndarray):
#                         out[key] = [val]
#                     elif isinstance(val, (list, tuple)):
#                         if len(val) != 1 or not isinstance(val[0], np.ndarray):
#                             raise ValueError(f"Expect single-chain coords for key '{key}' as np.ndarray or [np.ndarray].")
#                     else:
#                         pass

#             if 'chain_mask' in out and isinstance(out['chain_mask'], np.ndarray):
#                 out['chain_mask'] = [out['chain_mask']]
#             if 'chain_encoding' in out and isinstance(out['chain_encoding'], np.ndarray):
#                 out['chain_encoding'] = [out['chain_encoding']]

#             return out

#         raise ValueError("Input 'type' must be either str (single chain) or list (complex).")

#     # ---------- 将 ComplexDataset 的 item 转成拼接后的 (S, X, mask 等) ----------
#     def _flatten_complex(self, item):
#         item = self._normalize_input(item)

#         kinds: List[str] = [str(x).lower() for x in item["type"]]  # ['protein','rna',...]
#         seqs:  List[str] = item["seq"]
#         type2id = {"protein": 0, "rna": 1, "dna": 2}

#         X_list = []
#         type_vec_full = []
#         S_flat = []
#         mask_full = []
#         enc_full = []
#         slot_expect = []

#         def get_arr(key: str, idx: int, L: int) -> np.ndarray:
#             arr = item.get(key, None)
#             if arr is None or idx >= len(arr) or arr[idx] is None:
#                 return np.full((L, 3), np.nan, dtype=np.float32)
#             a = arr[idx]
#             if not isinstance(a, np.ndarray) or a.shape != (L, 3):
#                 return np.full((L, 3), np.nan, dtype=np.float32)
#             return a.astype(np.float32, copy=False)

#         for i, kind in enumerate(kinds):
#             seq = (seqs[i] or "").upper()
#             L = len(seq)

#             if kind == "protein":
#                 N  = get_arr("N",  i, L)
#                 CA = get_arr("CA", i, L)
#                 C  = get_arr("C",  i, L)
#                 O  = get_arr("O",  i, L)
#                 S6 = np.stack([N, CA, C, O,
#                                np.full((L,3), np.nan, np.float32),
#                                np.full((L,3), np.nan, np.float32)], axis=1)
#                 expect = np.zeros((L, 6), dtype=bool); expect[:, :4] = True

#             elif kind == "rna":
#                 P  = get_arr("P",  i, L)
#                 O5 = get_arr("O5", i, L)
#                 C5 = get_arr("C5", i, L)
#                 C4 = get_arr("C4", i, L)
#                 C3 = get_arr("C3", i, L)
#                 O3 = get_arr("O3", i, L)
#                 S6 = np.stack([P, O5, C5, C4, C3, O3], axis=1)
#                 expect = np.ones((L, 6), dtype=bool)

#             elif kind == "dna":
#                 O5 = get_arr("O5", i, L)
#                 C5 = get_arr("C5", i, L)
#                 C4 = get_arr("C4", i, L)
#                 P  = get_arr("P",  i, L)
#                 C3 = get_arr("C3", i, L)
#                 O3 = get_arr("O3", i, L)
#                 S6 = np.stack([O5, C5, C4, P, C3, O3], axis=1)
#                 expect = np.ones((L, 6), dtype=bool)

#             else:
#                 raise ValueError(f"Unknown kind: {kind}")

#             X_list.append(torch.from_numpy(S6))              # (L,6,3)
#             slot_expect.append(torch.from_numpy(expect))     # (L,6)

#             # token ids & type ids
#             S_flat.extend(self.tokenizer.encode(seq, kind))
#             type_id = type2id.get(kind, 0)
#             type_vec_full.extend([type_id] * L)

#             # masks / encodings（必须提供；若上游缺失，需要保证给出与 L 对齐的数组）
#             mask_full.append(torch.from_numpy(item["chain_mask"][i].astype(np.float32)))
#             enc_full.append(torch.from_numpy(item["chain_encoding"][i].astype(np.float32)))

#         X6 = torch.cat(X_list, dim=0)                          # [L_total, 6, 3]
#         slot_expect_mask = torch.cat(slot_expect, dim=0)       # [L_total, 6]
#         S_flat = torch.tensor(S_flat, dtype=torch.long)        # [L_total]
#         type_vec_full = torch.tensor(type_vec_full, dtype=torch.long)
#         chain_mask_full = torch.cat(mask_full, dim=0).float()  # [L_total]
#         chain_encoding_full = torch.cat(enc_full, dim=0).float()

#         return S_flat, type_vec_full, chain_mask_full, chain_encoding_full, X6, slot_expect_mask

#     def _get_features_persample(self, item: dict):
#         # 1) 展平
#         S_flat, type_vec_full, chain_mask_full, chain_encoding_full, X6, slot_expect = self._flatten_complex(item)

#         L_total = X6.shape[0]
#         if L_total < 2:
#             return None

#         # 2) 有效节点：仅要求“应当存在的槽”有效
#         valid_per_slot = torch.isfinite(X6).all(dim=2)
#         eff_ok = (~slot_expect) | valid_per_slot
#         mask_bool = eff_ok.all(dim=1)  # [L_total]
#         if mask_bool.sum() < 2:
#             return None

#         # 3) 压紧到 N
#         X = X6[mask_bool]                                  # [N, 6, 3]
#         S = S_flat[mask_bool]                              # [N]
#         type_vec = type_vec_full[mask_bool]                # [N]
#         chain_mask = chain_mask_full[mask_bool]            # [N]
#         chain_encoding = chain_encoding_full[mask_bool]    # [N]

#         # 数值稳定
#         X[torch.isnan(X)] = 0.0

#         # 4) 单 batch
#         N = X.shape[0]
#         device = X.device
#         batch_id = torch.zeros(N, dtype=torch.long, device=device)

#         # 中心点：slot=1（CA/O5/C5）
#         center = X[:, 1, :]

#         # 5) KNN 图
#         edge_idx = knn_graph(center, k=self.knn_k, batch=batch_id, loop=False, flow='target_to_source')
#         key = (edge_idx[1] * (edge_idx[0].max() + 1) + edge_idx[0]).long()
#         order = torch.argsort(key)
#         edge_idx = edge_idx[:, order]

#         # 6) 刚体与相对刚体（anchors = slots 0/1/2）
#         T = Rigid.make_transform_from_reference(X[:, 0].float(), X[:, 1].float(), X[:, 2].float())
#         src_idx, dst_idx = edge_idx[0], edge_idx[1]
#         T_ts = T[dst_idx, None].invert().compose(T[src_idx, None])

#         # 7) 全局虚拟帧
#         num_global = self.virtual_frame_num
#         X_c = T._trans
#         X_m = X_c.mean(dim=0, keepdim=True)
#         X_c = X_c - X_m
#         U, Ssvd, V = torch.svd(X_c.T @ X_c)
#         d = (torch.det(U) * torch.det(V)) < 0.0
#         D = torch.zeros_like(V); D[[0, 1], [0, 1]] = 1; D[2, 2] = -1 * d + 1 * (~d)
#         V = D @ V
#         R = torch.matmul(U, V.permute(0, 1))

#         rot_g = [R] * num_global
#         trans_g = [X_m] * num_global

#         # 8) 节点/边几何特征
#         feat = get_interact_feats(T, T_ts, X.float(), edge_idx, batch_id)
#         _V, _E = feat["_V"], feat["_E"]  # Mixed 维度由下游 embedding 统一处理

#         # 9) 虚拟全局边
#         T_g = Rigid(Rotation(torch.stack(rot_g)), torch.cat(trans_g, dim=0))
#         num_nodes = scatter_sum(torch.ones_like(batch_id), batch_id)   # [1]
#         global_src = torch.cat([batch_id + k * num_nodes.shape[0] for k in range(num_global)]) + num_nodes
#         global_dst = torch.arange(batch_id.shape[0], device=device).repeat(num_global)
#         edge_idx_g = torch.stack([global_dst, global_src])
#         edge_idx_g_inv = torch.stack([global_src, global_dst])
#         edge_idx_g = torch.cat([edge_idx_g, edge_idx_g_inv], dim=1)

#         batch_id_g = torch.zeros(num_global, dtype=batch_id.dtype, device=device)
#         T_all = Rigid.cat([T, T_g], dim=0)
#         idx_min, _ = edge_idx_g.min(dim=0)
#         T_gs = T_all[idx_min, None].invert().compose(T_all[idx_min, None])

#         rbf_ts = rbf(T_ts._trans.norm(dim=-1), 0, 50, 16)[:, 0].view(_E.shape[0], -1)
#         rbf_gs = rbf(T_gs._trans.norm(dim=-1), 0, 50, 16)[:, 0].view(edge_idx_g.shape[1], -1)

#         _V_g = torch.arange(num_global, device=device)
#         _E_g = torch.zeros([edge_idx_g.shape[1], 128], device=device)

#         # 10) 输出（不含 S_in/loss_mask；由 MInterface 决策后写入）
#         batch_out = {
#             "type_vec": type_vec,            # [N] 0/1/2
#             "T": T, "T_g": T_g, "T_ts": T_ts, "T_gs": T_gs,
#             "rbf_ts": rbf_ts, "rbf_gs": rbf_gs,

#             "X": X,                           # [N,6,3]
#             "_V": _V, "_E": _E,
#             "_V_g": _V_g, "_E_g": _E_g,

#             "S": S,                            # 真值
#             "edge_idx": edge_idx, "edge_idx_g": edge_idx_g,
#             "batch_id": batch_id, "batch_id_g": batch_id_g,
#             "num_nodes": num_nodes,            # [1]

#             # 可见位与链编码（MInterface 用它们来构造 loss_mask/target）
#             "mask": chain_mask,                # [N]
#             "chain_mask": chain_mask,          # [N]
#             "chain_encoding": chain_encoding,  # [N]

#             # 展开后的刚体张量（模型 forward 里会重建 Rigid）
#             "T_rot":  T._rots._rot_mats,   "T_trans":  T._trans,
#             "T_g_rot":T_g._rots._rot_mats, "T_g_trans":T_g._trans,
#             "T_ts_rot":T_ts._rots._rot_mats,"T_ts_trans":T_ts._trans,
#             "T_gs_rot":T_gs._rots._rot_mats,"T_gs_trans":T_gs._trans,

#             "K_g": self.virtual_frame_num
#         }
#         return batch_out

#     # ---------- collate ----------
#     def featurize(self, batch: List[dict]):
#         samples = []
#         for one in batch:
#             feat = self._get_features_persample(one)
#             if feat is not None:
#                 samples.append(feat)
#         if not samples:
#             # 不返回 None；返回一个哑批标记，交给 forward 提前退出
#             return {"_skip_batch": True}
#         return self.custom_collate_fn(samples)

#     def custom_collate_fn(self, batch: List[dict]):
#         batch = [b for b in batch if b is not None]
#         if not batch:
#             return None

#         K_g = int(batch[0]["K_g"])
#         num_nodes_list = [int(b["num_nodes"][0]) for b in batch]
#         B = len(batch)
#         total_real = sum(num_nodes_list)

#         prefix_real = torch.tensor([0] + list(torch.cumsum(torch.tensor(num_nodes_list[:-1]), dim=0).tolist()),
#                                    dtype=torch.long)
#         base_virtual = total_real

#         def remap_indices(local_idx: torch.Tensor, N_i: int, base_real_i: int, base_virt_i: int) -> torch.Tensor:
#             is_virtual = (local_idx >= N_i)
#             out = local_idx.clone()
#             out[~is_virtual] += base_real_i
#             out[is_virtual] = (local_idx[is_virtual] - N_i) + (base_virtual + base_virt_i)
#             return out

#         ret = {}
#         # 仅拼接必须键；S_in/loss_mask 由 MInterface 在 batch 上另行写入
#         cat_keys = [
#             "X", "_V", "_E", "_V_g", "_E_g",
#             "S",
#             "type_vec",
#             "mask", "chain_mask", "chain_encoding",
#         ]
#         for k in cat_keys:
#             ret[k] = torch.cat([b[k] for b in batch], dim=0)

#         # 刚体
#         for k in ["T", "T_g", "T_ts", "T_gs"]:
#             T_cat = Rigid.cat([b[k] for b in batch], dim=0)
#             ret[k + "_rot"]   = T_cat._rots._rot_mats
#             ret[k + "_trans"] = T_cat._trans

#         ret["num_nodes"] = torch.tensor(num_nodes_list, dtype=torch.long)
#         ret["batch_id"]  = torch.cat([torch.full((num_nodes_list[i],), i, dtype=torch.long) for i in range(B)], dim=0)
#         ret["batch_id_g"] = torch.cat([torch.full((K_g,), i, dtype=torch.long) for i in range(B)], dim=0)

#         # 边重映射
#         edge_parts = []
#         for i, b in enumerate(batch):
#             shift = prefix_real[i]
#             edge_parts.append(b["edge_idx"] + shift)
#         ret["edge_idx"] = torch.cat(edge_parts, dim=1)

#         edge_g_parts = []
#         for i, b in enumerate(batch):
#             N_i = num_nodes_list[i]
#             base_real_i = int(prefix_real[i].item())
#             base_virt_i = i * K_g
#             src_local = b["edge_idx_g"][0]
#             dst_local = b["edge_idx_g"][1]
#             src_global = remap_indices(src_local, N_i, base_real_i, base_virt_i)
#             dst_global = remap_indices(dst_local, N_i, base_real_i, base_virt_i)
#             edge_g_parts.append(torch.stack([src_global, dst_global], dim=0))
#         ret["edge_idx_g"] = torch.cat(edge_g_parts, dim=1)

#         # 其他
#         ret["rbf_ts"] = torch.cat([b["rbf_ts"] for b in batch], dim=0)
#         ret["rbf_gs"] = torch.cat([b["rbf_gs"] for b in batch], dim=0)
#         ret["_V_g"]   = torch.cat([b["_V_g"] for b in batch], dim=0)
#         ret["_E_g"]   = torch.cat([b["_E_g"] for b in batch], dim=0)
#         ret["K_g"]    = K_g
#         return ret
    
    
