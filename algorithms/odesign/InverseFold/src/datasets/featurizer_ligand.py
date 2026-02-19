# -*- coding: utf-8 -*-
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Sequence, Union

from torch_geometric.nn.pool import knn_graph
from torch_scatter import scatter_sum

# 你工程里已有的工具
from ..tools.affine_utils import Rigid, Rotation, get_interact_feats
from ..modules.if_module import *  # rbf()


from .featurizer_complex import Unitokenizer_Complex  # 若路径不同，请改 import

class LigandTokenizer:
    BASE_ELEMS = [
        "C","O","N","P","S","Fe","Os","F","Mg","Cl","Br","W","B","Co","I","Mo",
        "As","Be","Mn","Cu","Ta","V","Al","Ir","Hg","Se","Ni","Ru","D","Pt",
        "Ca","Re","Zn","Si"
    ]
    def __init__(self) -> None:
        self.elem_to_id = {e: i for i, e in enumerate(self.BASE_ELEMS)}
        self.RARE_TOKEN = "RARE"; self.UNK_TOKEN = "UNK"; self.MASK_TOKEN = "*"
        self.rare_id = len(self.elem_to_id)       # 34
        self.unk_id  = self.rare_id + 1           # 35
        self.mask_id = self.unk_id + 1            # 36
        self.id_to_token = list(self.BASE_ELEMS) + [self.RARE_TOKEN, self.UNK_TOKEN, self.MASK_TOKEN]
        self.vocab_size = len(self.id_to_token)

    @staticmethod
    def _canon(sym: str):
        if sym is None: return None
        s = str(sym).strip()
        if not s: return None
        return s.upper() if len(s) == 1 else (s[0].upper() + s[1:].lower())

    def encode(self, elements: Sequence[str]) -> List[int]:
        out = []
        for e in elements:
            c = self._canon(e)
            if c is None: out.append(self.unk_id)
            elif c in self.elem_to_id: out.append(self.elem_to_id[c])
            else: out.append(self.rare_id)
        return out

    def decode(self, ids: Union[Sequence[int], np.ndarray, torch.Tensor], join: bool=False) -> Union[List[str], str]:
        """
        ids -> 元素符号列表（或拼接字符串）
        - 越界 id 映射为 '*'
        - 可以根据需要把 'RARE'/'UNK' 转成 'X'（下面注释处）
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()

        out: List[str] = []
        for t in ids:
            try:
                i = int(t)
            except Exception:
                i = self.mask_id
            if 0 <= i < self.vocab_size:
                tok = self.id_to_token[i]
            else:
                tok = self.MASK_TOKEN

            # 如果不想看到 'RARE'/'UNK'，可以统一成 'X'
            # if tok in (self.RARE_TOKEN, self.UNK_TOKEN):
            #     tok = 'X'

            out.append(tok)
        return " ".join(out) if join else out
    


# --------- 小工具 ---------
_PRIME_IN_MAP = {
    "O5'": "O5", 'O5"': "O5", 'O5*': "O5",
    "C5'": "C5", 'C5"': "C5", 'C5*': "C5",
    "C4'": "C4", 'C4"': "C4", 'C4*': "C4",
    "C3'": "C3", 'C3"': "C3", 'C3*': "C3",
    "O3'": "O3", 'O3"': "O3", 'O3*': "O3",
}

def _is_finite_row(x: np.ndarray) -> np.ndarray:
    """x: (L,3) -> (L,), True 表示该行坐标都是有限数"""
    if x.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.isfinite(x).all(axis=1)

def _get_from_coords(coords: Dict[str, Any], key: str, L: int) -> np.ndarray:
    """从 coords 字典里取对应原子坐标；不存在则返回 NaN 填充阵列 (L,3)"""
    arr = None
    if key in coords:
        arr = coords[key]
    else:
        # 尝试 prime 映射（O5' 等）
        for k, v in _PRIME_IN_MAP.items():
            if v == key and k in coords:
                arr = coords[k]
                break
    if arr is None:
        return np.full((L, 3), np.nan, dtype=np.float32)
    a = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
    if a.shape[0] != L:
        # 长度不一致也回退 NaN
        return np.full((L, 3), np.nan, dtype=np.float32)
    return a

def _stack_X6_protein(N: np.ndarray, CA: np.ndarray, C: np.ndarray, O: np.ndarray) -> np.ndarray:
    L = CA.shape[0]
    Z = np.full((L, 3), np.nan, dtype=np.float32)
    return np.stack([N, CA, C, O, Z, Z], axis=1)

def _stack_X6_rna(P: np.ndarray, O5: np.ndarray, C5: np.ndarray, C4: np.ndarray, C3: np.ndarray, O3: np.ndarray) -> np.ndarray:
    return np.stack([P, O5, C5, C4, C3, O3], axis=1)

def _stack_X6_dna(O5: np.ndarray, C5: np.ndarray, C4: np.ndarray, P: np.ndarray, C3: np.ndarray, O3: np.ndarray) -> np.ndarray:
    return np.stack([O5, C5, C4, P, C3, O3], axis=1)

def _build_ligand_X6(lig_xyz: np.ndarray, rec_centers: Optional[np.ndarray] = None) -> np.ndarray:
    """
    lig_xyz: (N,3)
    槽位定义（与之前一致）：
      0: 最近邻（优先在 ligand 内部找；如不足，再用受体中心兜底）
      1: 自身
      2: 第二近邻（同上）
      3..5: NaN
    """
    lig_xyz = np.asarray(lig_xyz, dtype=np.float32).reshape(-1, 3)
    N = lig_xyz.shape[0]
    X6 = np.full((N, 6, 3), np.nan, dtype=np.float32)
    if N == 0:
        return X6
    X6[:, 1, :] = lig_xyz  # 槽 1 = 自身

    if N >= 2:
        # 两个最近邻（ligand 内部）
        import torch
        X = torch.from_numpy(lig_xyz)
        D = torch.cdist(X, X, p=2)  # [N,N]
        D.fill_diagonal_(float("inf"))
        k2 = 2 if N >= 3 else 1
        knn2 = torch.topk(D, k=k2, largest=False).indices  # [N,k2]
        j1 = knn2[:, 0].numpy()
        j2 = (knn2[:, 1].numpy() if k2 > 1 else j1)
        X6[:, 0, :] = lig_xyz[j1]
        X6[:, 2, :] = lig_xyz[j2]
    else:
        # N==1：需要兜底最近邻
        if rec_centers is not None and rec_centers.size > 0:
            import torch
            R = torch.from_numpy(np.asarray(rec_centers, dtype=np.float32))
            x = torch.from_numpy(lig_xyz[0:1])
            d = torch.cdist(x, R, p=2)[0]  # [Nr]
            j = int(torch.argmin(d).item())
            X6[0, 0, :] = R[j].numpy()
            X6[0, 2, :] = R[j].numpy()
        else:
            # 实在没有就拷贝自身（退化）
            X6[0, 0, :] = lig_xyz[0]
            X6[0, 2, :] = lig_xyz[0]
    return X6


# -------------------- Featurizer（支持只有 Ligand 的输入） --------------------
class Featurize_Ligand:
    """
    单图版：受体（Protein/RNA/DNA，可为空） + 配体（原子图）合并建图。
    - 只对受体做口袋裁剪（配体不裁），裁不出则整链兜底（保留受体有效位点）；无受体则跳过。
    - 输出的 S 只包含配体元素 token；受体位置一律填 lig_tokenizer.mask_id。
    - chain_mask: 受体=0, 配体=1（训练/评估只作用在配体上）。
    - 兼容 ligand-only：没有受体时也能正常建图与前向。
    """

    def __init__(self,
                 knn_k: int = 32,
                 virtual_frame_num: int = 3,
                 ensure_cross: bool = True,
                 K_cross: int = 12,
                 # 口袋裁剪
                 R_pocket: float = 20.0,
                 min_near: int = 64,
                 R_far: float = 30.0,
                 expand_W: int = 2,
                 use_receptor_for_global: bool = True):
        self.knn_k = int(knn_k)
        self.virtual_frame_num = int(virtual_frame_num)
        self.ensure_cross = bool(ensure_cross)
        self.K_cross = int(K_cross)

        self.R_pocket = float(R_pocket)
        self.min_near = int(min_near)
        self.R_far = float(R_far)
        self.expand_W = int(expand_W)
        self.use_receptor_for_global = bool(use_receptor_for_global)

        # 与混合几何特征维度对齐（接口占位）
        self.A_TOTAL = 6
        self.MIXED_NODE_IN = 114
        self.MIXED_EDGE_IN = 272

        self.lig_tok = LigandTokenizer()

    # ---------- 入口（batch） ----------
    def featurize(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        samples = []
        for it in batch:
            feat = self._per_sample(it)
            if feat is not None:
                samples.append(feat)
        if not samples:
            return None
        return self._collate(samples)

    # ---------- 单样本 ----------
    def _per_sample(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # ---- 受体（允许为空）----
        kind_list = (item.get("type") or [])
        seq_list = (item.get("seq") or [])
        has_receptor = (
            isinstance(kind_list, (list, tuple)) and len(kind_list) > 0
            and str(kind_list[0]).lower() in ("protein", "rna", "dna")
            and isinstance((seq_list[0] if seq_list else ""), str)
            and len(seq_list[0]) > 0
        )

        if not has_receptor:
            # 空受体：所有与受体相关的量都置空
            kind = None
            L = 0
            X6_rec = np.zeros((0, 6, 3), dtype=np.float32)
            rec_valid = np.zeros((0,), dtype=bool)
            rep = np.zeros((0, 3), dtype=np.float32)
            span_rec = np.array([[0, 0]], dtype=np.int64)
        else:
            kind = str(kind_list[0]).lower()
            seq = seq_list[0]
            L = len(seq)
            coords_dict = item.get("coords") or {}

            if kind == "protein":
                N = _get_from_coords(coords_dict, "N", L)
                CA = _get_from_coords(coords_dict, "CA", L)
                C = _get_from_coords(coords_dict, "C", L)
                O = _get_from_coords(coords_dict, "O", L)
                X6_rec = _stack_X6_protein(N, CA, C, O)
                rep = CA
                expect_mask = np.zeros((L, 6), dtype=bool); expect_mask[:, :4] = True
            elif kind == "rna":
                P = _get_from_coords(coords_dict, "P", L)
                O5 = _get_from_coords(coords_dict, "O5", L)
                C5 = _get_from_coords(coords_dict, "C5", L)
                C4 = _get_from_coords(coords_dict, "C4", L)
                C3 = _get_from_coords(coords_dict, "C3", L)
                O3 = _get_from_coords(coords_dict, "O3", L)
                X6_rec = _stack_X6_rna(P, O5, C5, C4, C3, O3)
                rep = O5
                expect_mask = np.ones((L, 6), dtype=bool)
            else:  # dna
                O5 = _get_from_coords(coords_dict, "O5", L)
                C5 = _get_from_coords(coords_dict, "C5", L)
                C4 = _get_from_coords(coords_dict, "C4", L)
                P = _get_from_coords(coords_dict, "P", L)
                C3 = _get_from_coords(coords_dict, "C3", L)
                O3 = _get_from_coords(coords_dict, "O3", L)
                X6_rec = _stack_X6_dna(O5, C5, C4, P, C3, O3)
                rep = O5
                expect_mask = np.ones((L, 6), dtype=bool)

            valid_per_slot = np.isfinite(X6_rec).all(axis=2)  # (L,6)
            rec_valid = ((~expect_mask) | valid_per_slot).all(axis=1)  # (L,)

            # 只对受体裁剪
            keep_idx = self._pocket_or_full_fast(rep, self._get_lig_xyz(item), rec_valid)
            X6_rec = X6_rec[keep_idx]
            rep = rep[keep_idx] if rep.size > 0 else rep
            span_rec = np.array([[0, X6_rec.shape[0]]], dtype=np.int64)

        # ---- 配体（必须存在）----
        lig = item.get("ligand") or {}
        lig_coords = lig.get("coords", None)
        lig_elems = lig.get("elements", None)
        if lig_coords is None or lig_elems is None:
            return None
        lig_coords = np.asarray(lig_coords, dtype=np.float32).reshape(-1, 3)
        N_lig = lig_coords.shape[0]
        if N_lig < 1:
            return None

        # 配体 X6；必要时用裁剪后的受体中心兜底第二邻居
        rec_center_after = X6_rec[:, 1, :] if X6_rec.shape[0] > 0 else None
        X6_lig = _build_ligand_X6(lig_coords, rec_centers=rec_center_after)

        # ---- 合并节点：受体在前、配体在后 ----
        X6_full = np.concatenate([X6_rec, X6_lig], axis=0)  # (N_all,6,3)
        X6_full_np = X6_full.copy()
        N_rec = X6_rec.shape[0]
        N_all = N_rec + N_lig

        # type_vec：受体 0/1/2，配体 3；若无受体则只有 3
        if has_receptor:
            t_id = {"protein": 0, "rna": 1, "dna": 2}[kind]
            type_vec = np.concatenate([np.full(N_rec, t_id, np.int64),
                                       np.full(N_lig, 3, np.int64)], axis=0)
        else:
            type_vec = np.full(N_lig, 3, np.int64)

        # 监督只在配体：受体=0，配体=1；没有受体时全为 1
        if has_receptor:
            chain_mask = np.concatenate([np.zeros(N_rec, np.float32),
                                         np.ones(N_lig, np.float32)], axis=0)
            chain_encoding = np.concatenate([np.zeros(N_rec, np.float32),
                                             np.ones(N_lig, np.float32)], axis=0)
        else:
            chain_mask = np.ones(N_lig, np.float32)
            chain_encoding = np.ones(N_lig, np.float32)

        # NaN -> 0
        X6_full_np[~np.isfinite(X6_full_np)] = 0.0
        X_full = torch.from_numpy(X6_full_np).float()  # [N_all,6,3]

        # ---- S（只含配体 token；受体位置全 mask_id）----
        S_np = np.full(N_all, self.lig_tok.mask_id, dtype=np.int64)
        lig_ids = self.lig_tok.encode(lig_elems)
        S_np[N_rec:N_all] = np.array(lig_ids, dtype=np.int64)

        # ---- 建图：KNN + （可选）配体-受体 cross 保证 ----
        X_center = torch.from_numpy(X6_full_np[:, 1, :]).float()
        batch_id = torch.zeros(N_all, dtype=torch.long)

        if N_all <= 1:
            edge_idx = torch.zeros((2, 0), dtype=torch.long)
        else:
            k_eff = min(max(self.knn_k, 1), N_all - 1)
            edge_idx_knn = knn_graph(X_center, k=k_eff, batch=batch_id, loop=False, flow='target_to_source')

            if self.ensure_cross and (N_rec > 0) and (N_lig > 0):
                E_cross = self._build_cross_edges(X_center, N_rec, self.K_cross)
                edge_idx = torch.cat([edge_idx_knn, E_cross], dim=1)
                # 排序 + 去重（不使用 return_index，兼容较老的 PyTorch）
                key = edge_idx[1] * N_all + edge_idx[0]
                order = torch.argsort(key)
                edge_idx = edge_idx[:, order]
                key_sorted = key[order]
                keep = torch.ones(key_sorted.numel(), dtype=torch.bool, device=key_sorted.device)
                keep[1:] = key_sorted[1:] != key_sorted[:-1]
                edge_idx = edge_idx[:, keep]
            else:
                edge_idx = edge_idx_knn

            # 最后再做一次稳定排序
            key2 = (edge_idx[1] * (edge_idx[0].max() + 1) + edge_idx[0]).long()
            edge_idx = edge_idx[:, torch.argsort(key2)]

        E = edge_idx.shape[1]

        # ---- 刚体与相对刚体 ----
        T = Rigid.make_transform_from_reference(X_full[:, 0].float(),
                                                X_full[:, 1].float(),
                                                X_full[:, 2].float())
        if E > 0:
            src_idx, dst_idx = edge_idx[0], edge_idx[1]
            T_ts = T[dst_idx, None].invert().compose(T[src_idx, None])
        else:
            # 空边时构造空 Rigid（形状 [0,1]），保证下游 rbf/拼接不崩
            T_ts = T[:0, None]
        feat = get_interact_feats(T, T_ts, X_full.float(), edge_idx, batch_id)
        _V, _E = feat["_V"], feat["_E"]

        # ---- 全局虚拟帧/边 ----
        # 没受体时退化为“用配体中心估计全局方向”
        X_for_global = (X_full[:N_rec, 1, :] if (self.use_receptor_for_global and N_rec > 0)
                        else X_full[:, 1, :])
        X_m = X_for_global.mean(dim=0, keepdim=True) if X_for_global.numel() > 0 else torch.zeros(1, 3)
        X_c = X_for_global - X_m
        cov = X_c.T @ X_c if X_c.numel() > 0 else torch.zeros(3, 3)
        U, Ssvd, Vh = torch.linalg.svd(cov, full_matrices=False)
        V = Vh.mT
        D = torch.eye(3, device=cov.device)
        if torch.det(U @ V.T) < 0:
            D[2, 2] = -1.0
        Rm = U @ D @ V.T

        K_g = self.virtual_frame_num
        T_g = Rigid(Rotation(torch.stack([Rm] * K_g)), torch.cat([X_m] * K_g, dim=0))

        # 全局边（双向）
        global_nodes = torch.arange(N_all, N_all + K_g)
        if N_all > 0:
            global_src = global_nodes.repeat_interleave(N_all)
            global_dst = torch.arange(N_all).repeat(K_g)
            edge_idx_g = torch.stack([global_dst, global_src], dim=0)
            edge_idx_g = torch.cat([edge_idx_g, edge_idx_g.flip(0)], dim=1)
        else:
            edge_idx_g = torch.zeros((2, 0), dtype=torch.long)

        T_all = Rigid.cat([T, T_g], dim=0)
        if edge_idx_g.shape[1] > 0:
            T_gs = T_all[edge_idx_g[1], None].invert().compose(T_all[edge_idx_g[0], None])
        else:
            T_gs = T[:0, None]  # 空

        # RBF
        dist_ts = T_ts._trans.norm(dim=-1) if E > 0 else torch.zeros((0, 1))
        dist_gs = T_gs._trans.norm(dim=-1) if edge_idx_g.shape[1] > 0 else torch.zeros((0, 1))
        rbf_ts = rbf(dist_ts, 0, 50, 16)[:, 0].reshape(E, -1)
        rbf_gs = rbf(dist_gs, 0, 50, 16)[:, 0].reshape(edge_idx_g.shape[1], -1)

        # ---- 输出（单样本）----
        out = {
            # 图/几何
            "X": X_full,                      # [N_all,6,3]
            "edge_idx": edge_idx,             # [2,E]
            "rbf_ts": rbf_ts, "_V": _V, "_E": _E,
            "T": T, "T_ts": T_ts,

            "edge_idx_g": edge_idx_g,
            "rbf_gs": rbf_gs,
            "T_g": T_g, "T_gs": T_gs,
            "_V_g": torch.arange(K_g, dtype=torch.long),  # 虚拟节点索引（给 embedding）

            # 监督/掩码（只在配体）
            "S": torch.from_numpy(S_np).long(),           # [N_all]
            "mask": torch.from_numpy(chain_mask).float(),
            "chain_mask": torch.from_numpy(chain_mask).float(),
            "chain_encoding": torch.from_numpy(chain_encoding).float(),

            # 元信息/统计
            "type_vec": torch.from_numpy(type_vec).long(),     # [N_all] 0/1/2/3
            "num_nodes": torch.tensor([N_all], dtype=torch.long),
            "batch_id": torch.zeros(N_all, dtype=torch.long),
            "batch_id_g": torch.zeros(K_g, dtype=torch.long),
            "K_g": K_g,

            # 仅受体链 span；ligand-only 时为 [[0,0]]
            "chain_spans": torch.from_numpy(span_rec if has_receptor else np.array([[0, 0]], dtype=np.int64)),
            "chain_names": item.get("chain_names", None),
            "title": item.get("title", None),
            "pdb_id": item.get("pdb_id", None),
            "eval_type": item.get("eval_type", None),
        }

        # 展开刚体（与旧接口一致）
        out.update({
            "T_rot": T._rots._rot_mats, "T_trans": T._trans,
            "T_g_rot": T_g._rots._rot_mats, "T_g_trans": T_g._trans,
            "T_ts_rot": T_ts._rots._rot_mats, "T_ts_trans": T_ts._trans,
            "T_gs_rot": T_gs._rots._rot_mats, "T_gs_trans": T_gs._trans,
        })
        return out

    def _get_lig_xyz(self, item: Dict[str, Any]) -> np.ndarray:
        lig = item.get("ligand") or {}
        lig_coords = lig.get("coords", None)
        if lig_coords is None:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(lig_coords, dtype=np.float32).reshape(-1, 3)

    # ---------- 口袋 or 整链（torch 加速版；无受体则直接返回空） ----------
    def _pocket_or_full_fast(self, rep_np: np.ndarray, lig_np: np.ndarray, rec_valid_np: np.ndarray) -> np.ndarray:
        """
        用 torch.cdist 加速最近距离计算。
        return: (L,) 的布尔向量，True -> 保留
        """
        L = rep_np.shape[0]
        if L == 0:
            return np.zeros((0,), dtype=bool)

        rec_valid = torch.from_numpy(rec_valid_np.astype(np.bool_))
        rep = torch.from_numpy(rep_np).float()     # [L,3]
        lig = torch.from_numpy(lig_np).float()     # [N,3]

        is_finite = torch.isfinite(rep).all(dim=1)
        idx_cand = torch.nonzero(rec_valid & is_finite, as_tuple=False).view(-1)
        if idx_cand.numel() == 0:
            # 没有任何可用候选 -> 仅保留“有效位点”
            keep = (rec_valid & is_finite).cpu().numpy()
            if keep.sum() < 2:
                keep = rec_valid_np
            return keep

        # AABB 预筛
        lo = lig.min(dim=0).values - self.R_pocket
        hi = lig.max(dim=0).values + self.R_pocket
        in_box_mask = ((rep[idx_cand] >= lo) & (rep[idx_cand] <= hi)).all(dim=1)
        if not in_box_mask.any():
            # 盒内没有候选 -> 整链兜底（仅在有效位点）
            keep = (rec_valid & is_finite).cpu().numpy()
            if keep.sum() < 2:
                keep = rec_valid_np
            return keep

        idx_use = idx_cand[in_box_mask]                         # [m]
        # 最近距离（一次 cdist 就够）
        D = torch.cdist(rep[idx_use], lig, p=2)                 # [m,N]
        dmin_i = D.min(dim=1).values                            # [m]
        near = (dmin_i <= self.R_pocket)
        near_count = int(near.sum().item())
        dmin_star = float(dmin_i.min().item()) if dmin_i.numel() > 0 else float("inf")

        use_full = (near_count < self.min_near) or (dmin_star > self.R_far)
        if use_full:
            keep = (rec_valid & is_finite).cpu().numpy()
            if keep.sum() < 2:
                keep = rec_valid_np
            return keep

        keep = torch.zeros(L, dtype=torch.bool)
        sel_idx = idx_use[near]
        keep[sel_idx] = True

        # 序列扩张 ±W
        if self.expand_W > 0 and keep.any():
            pos = torch.nonzero(keep, as_tuple=False).view(-1)
            for k in range(1, self.expand_W + 1):
                keep[(pos - k).clamp_min(0)] = True
                keep[(pos + k).clamp_max(L - 1)] = True

        keep &= rec_valid & is_finite
        out = keep.cpu().numpy()
        if out.sum() < 2:
            out = (rec_valid & is_finite).cpu().numpy()
        return out

    # ---------- cross 边（配体→受体，双向） ----------
    def _build_cross_edges(self, X_center: torch.Tensor, N_rec: int, K_cross: int) -> torch.Tensor:
        """
        X_center: [N_all,3]（slot1 中心）
        受体：0..N_rec-1，配体：N_rec..N_all-1
        返回 [2, E_cross]
        """
        N_all = X_center.shape[0]
        if N_rec <= 0 or N_rec >= N_all:
            return torch.zeros((2, 0), dtype=torch.long)
        X_rec = X_center[:N_rec]            # [Nr,3]
        X_lig = X_center[N_rec:]            # [Nl,3]
        D = torch.cdist(X_lig, X_rec, p=2)  # [Nl,Nr]
        k_eff = min(max(int(K_cross), 1), max(N_rec, 1))
        idx_knn = torch.topk(D, k=k_eff, largest=False).indices     # [Nl,K]
        src_lig = (torch.arange(X_lig.shape[0]).unsqueeze(1).repeat(1, k_eff) + N_rec).reshape(-1)
        dst_rec = idx_knn.reshape(-1)
        e1 = torch.stack([src_lig, dst_rec], dim=0)  # lig -> rec
        e2 = torch.stack([dst_rec, src_lig], dim=0)  # rec -> lig
        return torch.cat([e1, e2], dim=1)

    # ---------- collate ----------
    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        K_g = int(batch[0]["K_g"])
        num_nodes_list = [int(b["num_nodes"][0]) for b in batch]
        total_real = sum(num_nodes_list)
        prefix_real = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(num_nodes_list[:-1]), dim=0).tolist()),
            dtype=torch.long
        )
        base_virtual = total_real

        def remap_indices(local_idx: torch.Tensor, N_i: int, base_real_i: int, base_virt_i: int) -> torch.Tensor:
            is_virtual = (local_idx >= N_i)
            out = local_idx.clone()
            out[~is_virtual] += base_real_i
            out[is_virtual] = (local_idx[is_virtual] - N_i) + (base_virtual + base_virt_i)
            return out

        ret: Dict[str, torch.Tensor] = {}
        cat_keys = [
            "X", "_V", "_E", "rbf_ts", "rbf_gs",
            "type_vec", "mask", "chain_mask", "chain_encoding",
            "S", "_V_g",
        ]
        for k in cat_keys:
            ret[k] = torch.cat([b[k] for b in batch], dim=0)

        # 刚体拼接（导出 rot/trans）
        for k in ["T", "T_g", "T_ts", "T_gs"]:
            T_cat = Rigid.cat([b[k] for b in batch], dim=0)
            ret[k + "_rot"] = T_cat._rots._rot_mats
            ret[k + "_trans"] = T_cat._trans

        ret["num_nodes"] = torch.tensor(num_nodes_list, dtype=torch.long)
        ret["batch_id"] = torch.cat([torch.full((num_nodes_list[i],), i, dtype=torch.long) for i in range(B)], dim=0)
        ret["batch_id_g"] = torch.cat([torch.full((K_g,), i, dtype=torch.long) for i in range(B)], dim=0)

        # 实边重映射
        edge_parts = []
        for i, b in enumerate(batch):
            shift = prefix_real[i]
            edge_parts.append(b["edge_idx"] + shift)
        ret["edge_idx"] = torch.cat(edge_parts, dim=1)

        # 全局边重映射
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

        # 其它元信息以 list 形式透传
        ret["chain_spans"] = [b.get("chain_spans", None) for b in batch]
        ret["chain_names"] = [b.get("chain_names", None) for b in batch]
        ret["title"] = [b.get("title", None) for b in batch]
        ret["pdb_id"] = [b.get("pdb_id", None) for b in batch]
        ret["eval_type"] = [b.get("eval_type", None) for b in batch]
        ret["K_g"] = K_g

        return ret


# class Featurize_Ligand:
#     """
#     单图版：受体（Protein/RNA/DNA） + 配体（原子图）合并建图。
#     输出中仅提供配体序列通道：
#       - S: [N_all]，受体位置全部为 tokenizer.mask_id，配体位置为元素 token；
#       - chain_mask: 受体=0，配体=1（不改 MInterface 也能只在配体上监督/遮蔽）。
#     """

#     def __init__(self,
#                  knn_k: int = 32,
#                  virtual_frame_num: int = 3,
#                  ensure_cross: bool = True,
#                  K_cross: int = 12,
#                  # 口袋裁剪
#                  R_pocket: float = 20.0,
#                  min_near: int = 64,
#                  R_far: float = 30.0,
#                  expand_W: int = 2,
#                  use_receptor_for_global: bool = True):
#         self.knn_k = int(knn_k)
#         self.virtual_frame_num = int(virtual_frame_num)
#         self.ensure_cross = bool(ensure_cross)
#         self.K_cross = int(K_cross)

#         self.R_pocket = float(R_pocket)
#         self.min_near = int(min_near)
#         self.R_far = float(R_far)
#         self.expand_W = int(expand_W)
#         self.use_receptor_for_global = bool(use_receptor_for_global)

#         # 与现有实现对齐（Mixed 维度）
#         self.A_TOTAL = 6
#         self.MIXED_NODE_IN = 114
#         self.MIXED_EDGE_IN = 272

#         # 只保留配体词表
#         self.lig_tok = LigandTokenizer()

#     # ---------- 入口（batch） ----------
#     def featurize(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
#         samples = []
#         for it in batch:
#             feat = self._per_sample(it)
#             if feat is not None:
#                 samples.append(feat)
#         if not samples:
#             return None
#         return self._collate(samples)

#     # ---------- 单样本 ----------
#     def _per_sample(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#         """
#         预期 item（LigandComplexDataset 单链包装）：
#           受体：{"type":[...], "seq":[...], "coords":{...}, "chain_mask":[...], "chain_encoding":[...]}
#           配体：{"elements":[...], "coords":(N,3), ...}
#         """
#         # ---- 受体 ----
#         kind = (item.get("type") or [""])[0]
#         kind = str(kind).lower()
#         assert kind in ("protein", "rna", "dna"), f"kind must be protein/rna/dna, got {kind}"
#         seq  = (item.get("seq") or [""])[0]
#         assert isinstance(seq, str) and len(seq) > 0, "seq must be non-empty string"
#         L = len(seq)

#         coords_dict = item.get("coords") or {}
#         chain_mask_rec_in  = (item.get("chain_mask") or [np.ones(L, np.float32)])[0].astype(np.float32)
#         chain_enc_rec_in   = (item.get("chain_encoding") or [np.ones(L, np.float32)])[0].astype(np.float32)

#         if kind == "protein":
#             N  = _get_from_coords(coords_dict, "N",  L)
#             CA = _get_from_coords(coords_dict, "CA", L)
#             C  = _get_from_coords(coords_dict, "C",  L)
#             O  = _get_from_coords(coords_dict, "O",  L)
#             X6_rec = _stack_X6_protein(N, CA, C, O)
#             rep = CA
#             expect_mask = np.zeros((L, 6), dtype=bool); expect_mask[:, :4] = True
#         elif kind == "rna":
#             P  = _get_from_coords(coords_dict, "P",  L)
#             O5 = _get_from_coords(coords_dict, "O5", L)
#             C5 = _get_from_coords(coords_dict, "C5", L)
#             C4 = _get_from_coords(coords_dict, "C4", L)
#             C3 = _get_from_coords(coords_dict, "C3", L)
#             O3 = _get_from_coords(coords_dict, "O3", L)
#             X6_rec = _stack_X6_rna(P, O5, C5, C4, C3, O3)
#             rep = O5
#             expect_mask = np.ones((L, 6), dtype=bool)
#         else:
#             O5 = _get_from_coords(coords_dict, "O5", L)
#             C5 = _get_from_coords(coords_dict, "C5", L)
#             C4 = _get_from_coords(coords_dict, "C4", L)
#             P  = _get_from_coords(coords_dict, "P",  L)
#             C3 = _get_from_coords(coords_dict, "C3", L)
#             O3 = _get_from_coords(coords_dict, "O3", L)
#             X6_rec = _stack_X6_dna(O5, C5, C4, P, C3, O3)
#             rep = O5
#             expect_mask = np.ones((L, 6), dtype=bool)

#         valid_per_slot = np.isfinite(X6_rec).all(axis=2)  # (L,6)
#         rec_valid = ((~expect_mask) | valid_per_slot).all(axis=1)  # (L,)

#         # ---- 配体 ----
#         lig = item.get("ligand") or {}
#         lig_coords = lig.get("coords", None)
#         lig_elems  = lig.get("elements", None)
#         if lig_coords is None or lig_elems is None:
#             return None
#         lig_coords = np.asarray(lig_coords, dtype=np.float32).reshape(-1, 3)
#         N_lig = lig_coords.shape[0]
#         if N_lig < 1:
#             return None

#         # ---- 口袋裁剪（只裁受体）----
#         keep_idx = self._pocket_or_full(rep, lig_coords, rec_valid)
#         # 压紧受体
#         X6_rec = X6_rec[keep_idx]
#         # 受体代表点（中心）用于 ligand 槽位兜底
#         rec_center_after = X6_rec[:, 1, :]  # slot1

#         # ---- 配体 X6（用受体中心兜底构 anchor）----
#         X6_lig = _build_ligand_X6(lig_coords, rec_centers=rec_center_after)

#         # 过滤后链 span（仅受体；配体不计入）
#         span_rec = np.array([[0, X6_rec.shape[0]]], dtype=np.int64)

#         # ---- 合并节点：受体在前、配体在后 ----
#         X6_full = np.concatenate([X6_rec, X6_lig], axis=0)           # (N_all,6,3)
#         X6_full_np = X6_full.copy()
#         N_rec = X6_rec.shape[0]
#         N_all = N_rec + N_lig

#         # type_vec: 0/1/2/3
#         t_id = {"protein": 0, "rna": 1, "dna": 2}[kind]
#         type_vec = np.concatenate([np.full(N_rec, t_id, np.int64),
#                                    np.full(N_lig, 3,    np.int64)], axis=0)

#         # mask / encoding：受体=0，配体=1（只让配体参与训练/评估）
#         chain_mask = np.concatenate([np.zeros(N_rec, np.float32),
#                                      np.ones(N_lig,  np.float32)], axis=0)
#         chain_encoding = np.concatenate([np.zeros(N_rec, np.float32),
#                                          np.ones(N_lig,  np.float32)], axis=0)

#         # NaN -> 0
#         X6_full_np[~np.isfinite(X6_full_np)] = 0.0
#         X_full = torch.from_numpy(X6_full_np).float()                 # [N_all,6,3]

#         # ---- S（只含配体 token；受体位置全 mask_id）----
#         S_np = np.full(N_all, self.lig_tok.mask_id, dtype=np.int64)
#         lig_ids = self.lig_tok.encode(lig_elems)
#         S_np[N_rec:N_all] = np.array(lig_ids, dtype=np.int64)

#         # ---- 建图：KNN + 交互 cross ----
#         X_center = torch.from_numpy(X6_full_np[:, 1, :]).float()
#         batch_id = torch.zeros(N_all, dtype=torch.long)
#         k_eff = min(max(self.knn_k, 1), max(N_all - 1, 1))
#         edge_idx_knn = knn_graph(X_center, k=k_eff, batch=batch_id, loop=False, flow='target_to_source')

#         if self.ensure_cross and (N_rec > 0) and (N_lig > 0):
#             E_cross = self._build_cross_edges(X_center, N_rec, self.K_cross)
#             edge_idx = torch.cat([edge_idx_knn, E_cross], dim=1)
#             # 去重 + 排序
#             # 先按 key 排序
#             key = edge_idx[1] * N_all + edge_idx[0]          # [E]
#             order = torch.argsort(key)                        # 稳定排序
#             edge_idx = edge_idx[:, order]
#             key_sorted = key[order]

#             # 相邻去重（保留每组重复中的第一个）
#             keep = torch.ones(key_sorted.numel(), dtype=torch.bool, device=key_sorted.device)
#             keep[1:] = key_sorted[1:] != key_sorted[:-1]
#             edge_idx = edge_idx[:, keep]

#         else:
#             edge_idx = edge_idx_knn

#         order_key = (edge_idx[1] * (edge_idx[0].max() + 1) + edge_idx[0]).long()
#         edge_idx = edge_idx[:, torch.argsort(order_key)]
#         E = edge_idx.shape[1]

#         # ---- 刚体与相对刚体 ----
#         T = Rigid.make_transform_from_reference(X_full[:, 0].float(),
#                                                 X_full[:, 1].float(),
#                                                 X_full[:, 2].float())
#         src_idx, dst_idx = edge_idx[0], edge_idx[1]
#         T_ts = T[dst_idx, None].invert().compose(T[src_idx, None])
#         feat = get_interact_feats(T, T_ts, X_full.float(), edge_idx, batch_id)
#         _V, _E = feat["_V"], feat["_E"]

#         # ---- 全局虚拟边 ----
#         X_for_global = (X_full[:N_rec, 1, :] if (self.use_receptor_for_global and N_rec > 0)
#                         else X_full[:, 1, :])
#         X_m = X_for_global.mean(dim=0, keepdim=True)
#         X_c = X_for_global - X_m
#         cov = X_c.T @ X_c
#         U, Ssvd, V = torch.svd(cov)
#         D = torch.eye(3)
#         if torch.det(U @ V.T) < 0:
#             D[2, 2] = -1.0
#         Rm = U @ D @ V.T

#         K_g = self.virtual_frame_num
#         T_g = Rigid(Rotation(torch.stack([Rm]*K_g)), torch.cat([X_m]*K_g, dim=0))

#         # 全局边（双向）
#         global_nodes = torch.arange(N_all, N_all + K_g)
#         global_src = global_nodes.repeat_interleave(N_all)
#         global_dst = torch.arange(N_all).repeat(K_g)
#         edge_idx_g = torch.stack([global_dst, global_src], dim=0)
#         edge_idx_g = torch.cat([edge_idx_g, edge_idx_g.flip(0)], dim=1)
#         T_all = Rigid.cat([T, T_g], dim=0)
#         T_gs = T_all[edge_idx_g[1], None].invert().compose(T_all[edge_idx_g[0], None])

#         # RBF
#         dist_ts = T_ts._trans.norm(dim=-1)
#         dist_gs = T_gs._trans.norm(dim=-1)
#         rbf_ts = rbf(dist_ts, 0, 50, 16)[:, 0].reshape(E, -1)
#         rbf_gs = rbf(dist_gs, 0, 50, 16)[:, 0].reshape(edge_idx_g.shape[1], -1)

#         # ---- 输出（单样本）----
#         out = {
#             "X": X_full,                                        # [N_all,6,3]
#             "type_vec": torch.from_numpy(
#                 np.concatenate([np.full(N_rec, {"protein":0,"rna":1,"dna":2}[kind], np.int64),
#                                 np.full(N_lig, 3, np.int64)], axis=0)
#             ).long(),
#             "mask": torch.from_numpy(chain_mask).float(),       # [N_all] 受体=0, 配体=1
#             "chain_mask": torch.from_numpy(chain_mask).float(),
#             "chain_encoding": torch.from_numpy(chain_encoding).float(),

#             "edge_idx": edge_idx,                               # [2,E]
#             "rbf_ts": rbf_ts, "_V": _V, "_E": _E,
#             "T": T, "T_ts": T_ts,

#             "edge_idx_g": edge_idx_g, "rbf_gs": rbf_gs,
#             "T_g": T_g, "T_gs": T_gs,
#             "_V_g": torch.arange(K_g, dtype=torch.long),    
#             "num_nodes": torch.tensor([N_all], dtype=torch.long),
#             "batch_id": torch.zeros(N_all, dtype=torch.long),
#             "batch_id_g": torch.zeros(K_g, dtype=torch.long),
#             "K_g": K_g,

#             # ★★★ 核心：S 仅含配体标签；受体位置全部为 mask_id ★★★
#             "S": torch.from_numpy(S_np).long(),                 # [N_all]

#             # 可选元信息
#             "chain_spans": torch.from_numpy(span_rec),          # 仅受体链 span
#             "chain_names": item.get("chain_names", None),
#             "title": item.get("title", None),
#             "pdb_id": item.get("pdb_id", None),
#             "eval_type": item.get("eval_type", None),
#         }

#         # 展开刚体（与旧接口一致）
#         out.update({
#             "T_rot":   T._rots._rot_mats,   "T_trans":   T._trans,
#             "T_g_rot": T_g._rots._rot_mats, "T_g_trans": T_g._trans,
#             "T_ts_rot":T_ts._rots._rot_mats,"T_ts_trans":T_ts._trans,
#             "T_gs_rot":T_gs._rots._rot_mats,"T_gs_trans":T_gs._trans,
#         })
#         return out

#     # ---------- 口袋 or 整链 ----------
#     def _pocket_or_full(self, rep: np.ndarray, lig_xyz: np.ndarray, rec_valid: np.ndarray) -> np.ndarray:
#         """
#         rep: (L,3) 受体代表点（Protein=CA；NA=O5）
#         lig_xyz: (N,3) 配体原子
#         rec_valid: (L,) 受体有效位点
#         return: (L,) 的布尔向量，True -> 保留
#         """
#         L = rep.shape[0]
#         idx_cand = np.where(rec_valid & _is_finite_row(rep))[0]
#         if idx_cand.size == 0:
#             return rec_valid

#         lo = lig_xyz.min(0) - self.R_pocket
#         hi = lig_xyz.max(0) + self.R_pocket
#         in_box = np.where(((rep[idx_cand] >= lo) & (rep[idx_cand] <= hi)).all(axis=1))[0]
#         if in_box.size == 0:
#             keep = rec_valid.copy()
#             return keep

#         idx_use = idx_cand[in_box]
#         D = _cdist(rep[idx_use].astype(np.float32), lig_xyz.astype(np.float32))  # (m,N)
#         dmin_i = D.min(axis=1)
#         near = (dmin_i <= self.R_pocket)
#         near_count = int(near.sum())
#         dmin_star = float(dmin_i.min()) if dmin_i.size > 0 else np.inf

#         use_full = (near_count < self.min_near) or (dmin_star > self.R_far)
#         if use_full:
#             return rec_valid.copy()

#         keep = np.zeros(L, dtype=bool)
#         sel_idx = idx_use[near]
#         keep[sel_idx] = True

#         if self.expand_W > 0:
#             sel = np.where(keep)[0]
#             for k in range(1, self.expand_W + 1):
#                 keep[np.clip(sel - k, 0, L - 1)] = True
#                 keep[np.clip(sel + k, 0, L - 1)] = True

#         keep &= rec_valid
#         if keep.sum() < 2:
#             keep = rec_valid.copy()
#         return keep

#     # ---------- cross 边（配体→受体，双向） ----------
#     def _build_cross_edges(self, X_center: torch.Tensor, N_rec: int, K_cross: int) -> torch.Tensor:
#         """
#         X_center: [N_all,3]（slot1 中心）
#         受体：0..N_rec-1，配体：N_rec..N_all-1
#         返回 [2, E_cross]
#         """
#         N_all = X_center.shape[0]
#         if N_rec <= 0 or N_rec >= N_all:
#             return torch.zeros((2, 0), dtype=torch.long)
#         X_rec = X_center[:N_rec]               # [Nr,3]
#         X_lig = X_center[N_rec:]               # [Nl,3]
#         D = torch.cdist(X_lig, X_rec, p=2)     # [Nl,Nr]
#         k_eff = min(max(int(K_cross), 1), max(N_rec, 1))
#         idx_knn = torch.topk(D, k=k_eff, largest=False).indices     # [Nl,K]
#         src_lig = (torch.arange(X_lig.shape[0]).unsqueeze(1).repeat(1, k_eff) + N_rec).reshape(-1)
#         dst_rec = idx_knn.reshape(-1)
#         e1 = torch.stack([src_lig, dst_rec], dim=0)  # lig -> rec
#         e2 = torch.stack([dst_rec, src_lig], dim=0)  # rec -> lig
#         return torch.cat([e1, e2], dim=1)

#     # ---------- collate ----------
#     def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         B = len(batch)
#         K_g = int(batch[0]["K_g"])
#         num_nodes_list = [int(b["num_nodes"][0]) for b in batch]
#         total_real = sum(num_nodes_list)
#         prefix_real = torch.tensor(
#             [0] + list(torch.cumsum(torch.tensor(num_nodes_list[:-1]), dim=0).tolist()),
#             dtype=torch.long
#         )
#         base_virtual = total_real

#         def remap_indices(local_idx: torch.Tensor, N_i: int, base_real_i: int, base_virt_i: int) -> torch.Tensor:
#             is_virtual = (local_idx >= N_i)
#             out = local_idx.clone()
#             out[~is_virtual] += base_real_i
#             out[is_virtual] = (local_idx[is_virtual] - N_i) + (base_virtual + base_virt_i)
#             return out

#         ret: Dict[str, torch.Tensor] = {}
#         cat_keys = [
#             "X", "_V", "_E", "rbf_ts", "rbf_gs",
#             "type_vec", "mask", "chain_mask", "chain_encoding",
#             "S","_V_g",
#         ]
#         for k in cat_keys:
#             ret[k] = torch.cat([b[k] for b in batch], dim=0)

#         # 刚体拼接（导出 rot/trans）
#         for k in ["T", "T_g", "T_ts", "T_gs"]:
#             T_cat = Rigid.cat([b[k] for b in batch], dim=0)
#             ret[k + "_rot"]   = T_cat._rots._rot_mats
#             ret[k + "_trans"] = T_cat._trans

#         ret["num_nodes"] = torch.tensor(num_nodes_list, dtype=torch.long)
#         ret["batch_id"]  = torch.cat([torch.full((num_nodes_list[i],), i, dtype=torch.long) for i in range(B)], dim=0)
#         ret["batch_id_g"] = torch.cat([torch.full((K_g,), i, dtype=torch.long) for i in range(B)], dim=0)

#         # 实边重映射
#         edge_parts = []
#         for i, b in enumerate(batch):
#             shift = prefix_real[i]
#             edge_parts.append(b["edge_idx"] + shift)
#         ret["edge_idx"] = torch.cat(edge_parts, dim=1)

#         # 全局边重映射
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

#         # 其它元信息以 list 形式透传
#         ret["chain_spans"] = [b.get("chain_spans", None) for b in batch]
#         ret["chain_names"] = [b.get("chain_names", None) for b in batch]
#         ret["title"]       = [b.get("title", None) for b in batch]
#         ret["pdb_id"]      = [b.get("pdb_id", None) for b in batch]
#         ret["eval_type"]   = [b.get("eval_type", None) for b in batch]
#         ret["K_g"]         = K_g
#         return ret






# # -*- coding: utf-8 -*-
# # src/datasets/featurizer_ligand.py

# import torch
# import numpy as np
# from typing import List, Sequence, Dict, Any

# from torch_geometric.nn.pool import knn_graph
# from torch_scatter import scatter_sum

# from ..tools.affine_utils import Rigid, Rotation, get_interact_feats
# from ..modules.if_module import *  # rbf()


# # ---------------- Tokenizer ----------------
# class LigandTokenizer:
#     """
#     只覆盖常见元素；其余 -> RARE；并包含 UNK、MASK
#     vocab 顺序：BASE_ELEMS + ['RARE','UNK','*']
#     """
#     BASE_ELEMS = [
#         "C","O","N","P","S","Fe","Os","F","Mg","Cl","Br","W","B","Co","I","Mo",
#         "As","Be","Mn","Cu","Ta","V","Al","Ir","Hg","Se","Ni","Ru","D","Pt",
#         "Ca","Re","Zn","Si"
#     ]
#     def __init__(self) -> None:
#         self.elem_to_id = {e: i for i, e in enumerate(self.BASE_ELEMS)}
#         self.RARE_TOKEN = "RARE"; self.UNK_TOKEN = "UNK"; self.MASK_TOKEN = "*"
#         self.rare_id = len(self.elem_to_id)       # 34
#         self.unk_id  = self.rare_id + 1           # 35
#         self.mask_id = self.unk_id + 1            # 36
#         self.id_to_token = list(self.BASE_ELEMS) + [self.RARE_TOKEN, self.UNK_TOKEN, self.MASK_TOKEN]
#         self.vocab_size = len(self.id_to_token)

#     @staticmethod
#     def _canon(sym: str):
#         if sym is None: return None
#         s = str(sym).strip()
#         if not s: return None
#         return s.upper() if len(s) == 1 else (s[0].upper() + s[1:].lower())

#     def encode(self, elements: Sequence[str]) -> List[int]:
#         out = []
#         for e in elements:
#             c = self._canon(e)
#             if c is None: out.append(self.unk_id)
#             elif c in self.elem_to_id: out.append(self.elem_to_id[c])
#             else: out.append(self.rare_id)
#         return out


# # ---------------- Featurizer ----------------
# class Featurize_Ligand:
#     """
#     维持“核酸版”的老接口 & 风格：
#       - anchors = (slot0, slot1, slot2)；这里设为(最近邻, 当前, 第二近邻)
#       - KNN 实边：edge_idx, T_ts, rbf_ts（二维 [E,16]，通过 [:,0].reshape(E,-1) 实现）
#       - 全局虚拟边：edge_idx_g, T_gs, rbf_gs（同上）
#       - _V_g 为 Long；N<3 丢弃
#     """
#     def __init__(self, knn_k: int = 48, virtual_frame_num: int = 3, min_atoms: int = 3):
#         self.tokenizer = LigandTokenizer()
#         self.knn_k = int(knn_k)
#         self.virtual_frame_num = int(virtual_frame_num)
#         self.min_atoms = int(min_atoms)

#         # 与下游对齐的占位（保持和核酸版完全一致）
#         self.A_TOTAL = 6
#         self.MIXED_NODE_IN = 114
#         self.MIXED_EDGE_IN = 272

#     @staticmethod
#     def _build_local_slots(coords: torch.Tensor) -> torch.Tensor:
#         """
#         coords: [N,3] -> X6: [N,6,3]
#         槽位定义：
#           0: 最近邻
#           1: 当前原子（中心）
#           2: 第二近邻
#           3..5: NaN 填充（保持 6 槽位一致）
#         """
#         N = coords.shape[0]
#         device = coords.device
#         X6 = torch.full((N, 6, 3), float("nan"), dtype=coords.dtype, device=device)
#         if N == 0:
#             return X6

#         # 最近邻 / 次近邻
#         dmat = torch.cdist(coords, coords, p=2)  # [N,N]
#         dmat.fill_diagonal_(float("inf"))
#         # 取两个最近的（N>=3 正常；N==2 时 fallback）
#         k2 = 2 if N >= 3 else 1
#         knn2 = torch.topk(dmat, k=k2, largest=False).indices  # [N,k2]
#         j1 = knn2[:, 0]
#         j2 = knn2[:, 1] if knn2.shape[1] > 1 else j1  # N==2 时次近邻退化为最近邻

#         X6[:, 0] = coords[j1]
#         X6[:, 1] = coords
#         X6[:, 2] = coords[j2]
#         return X6

#     def _get_features_persample(self, item: Dict[str, Any]):
#         # ---- 基本检查 ----
#         coords_np = item.get("coords", None)
#         if coords_np is None:
#             return None
#         coords = torch.as_tensor(coords_np, dtype=torch.float32)  # [N,3]
#         if coords.ndim != 2 or coords.shape[1] != 3:
#             return None
#         N = int(coords.shape[0])
#         if N < self.min_atoms:   # N<3 时无法稳定构造局部坐标系/两邻居，直接丢弃
#             return None

#         elements = item.get("elements", None)
#         if not isinstance(elements, list) or len(elements) != N:
#             return None

#         # ---- tokens / masks / encodings ----
#         S_ids = torch.tensor(self.tokenizer.encode(elements), dtype=torch.long)  # [N]
#         # mask
#         cm_in = item.get("chain_mask", None)
#         if isinstance(cm_in, (list, tuple)) and len(cm_in) >= 1:
#             chain_mask = torch.from_numpy(np.asarray(cm_in[0], dtype=np.float32))
#         else:
#             chain_mask = torch.ones(N, dtype=torch.float32)
#         # encoding
#         ce_in = item.get("chain_encoding", None)
#         if isinstance(ce_in, (list, tuple)) and len(ce_in) >= 1:
#             chain_encoding = torch.from_numpy(np.asarray(ce_in[0], dtype=np.float32))
#         else:
#             chain_encoding = torch.ones(N, dtype=torch.float32)

#         # ---- 6 槽位 & anchors ----
#         X6 = self._build_local_slots(coords)              # [N,6,3]
#         X6_filled = X6.clone()
#         X6_filled[torch.isnan(X6_filled)] = 0.0

#         type_vec = torch.zeros(N, dtype=torch.long)       # 单类型占位
#         batch_id = torch.zeros(N, dtype=torch.long)

#         # ---- KNN 实边 ----
#         center = X6_filled[:, 1, :]                       # 中心 = 槽 1
#         k_eff = min(max(self.knn_k, 1), max(N - 1, 1))
#         edge_idx = knn_graph(center, k=k_eff, batch=batch_id, loop=False, flow='target_to_source')
#         # 排序确保确定性
#         key = (edge_idx[1] * (edge_idx[0].max() + 1) + edge_idx[0]).long()
#         order = torch.argsort(key)
#         edge_idx = edge_idx[:, order]                     # [2,E]
#         E = edge_idx.shape[1]

#         # ---- 刚体与相对刚体（anchors = 0/1/2）----
#         T = Rigid.make_transform_from_reference(
#             X6_filled[:, 0].float(), X6_filled[:, 1].float(), X6_filled[:, 2].float()
#         )
#         src_idx, dst_idx = edge_idx[0], edge_idx[1]
#         # 维持老风格：显式加 None，让形状变成 [E,1]
#         T_ts = T[dst_idx, None].invert().compose(T[src_idx, None])  # [E,1]

#         # ---- 全局虚拟帧（K_g）----
#         K_g = self.virtual_frame_num
#         X_c = T._trans                                    # [N,3]
#         X_m = X_c.mean(dim=0, keepdim=True)               # [1,3]
#         X_c = X_c - X_m
#         cov = X_c.T @ X_c                                 # [3,3]
#         U, Ssvd, V = torch.svd(cov)
#         # right-handed 修正
#         D = torch.eye(3, device=coords.device)
#         if torch.det(U @ V.T) < 0:
#             D[2, 2] = -1.0
#         Rm = U @ D @ V.T                                  # [3,3]
#         rot_g = [Rm] * K_g
#         trans_g = [X_m] * K_g
#         T_g = Rigid(Rotation(torch.stack(rot_g)), torch.cat(trans_g, dim=0))  # [K_g]

#         # ---- 几何特征（实边）----
#         feat = get_interact_feats(T, T_ts, X6_filled.float(), edge_idx, batch_id)
#         _V, _E = feat["_V"], feat["_E"]

#         # ---- 虚拟全局边（双向，长度严格一致）----
#         # 实节点: 0..N-1，虚拟节点: N..N+K_g-1
#         global_nodes = torch.arange(N, N + K_g, device=coords.device)       # [K_g]
#         global_src = global_nodes.repeat_interleave(N)                       # [K_g*N] (node <- global)
#         global_dst = torch.arange(N, device=coords.device).repeat(K_g)       # [K_g*N]
#         edge_idx_g = torch.stack([global_dst, global_src], dim=0)            # [2, K_g*N]
#         edge_idx_g = torch.cat([edge_idx_g, edge_idx_g.flip(0)], dim=1)      # [2, 2*K_g*N]
#         E_g = edge_idx_g.shape[1]

#         # 与 edge_idx_g 对齐的相对刚体，维持 [E_g,1] 形状风格
#         T_all = Rigid.cat([T, T_g], dim=0)            # [N + K_g]
#         T_src = T_all[edge_idx_g[0], None]            # [E_g,1]
#         T_dst = T_all[edge_idx_g[1], None]            # [E_g,1]
#         T_gs  = T_dst.invert().compose(T_src)         # [E_g,1]

#         # ---- RBF：老风格（[:,0].reshape）稳定为二维 [*,16] ----
#         dist_ts = T_ts._trans.norm(dim=-1)                # [E,1]
#         dist_gs = T_gs._trans.norm(dim=-1)                # [E_g,1]
#         rbf_ts = rbf(dist_ts, 0, 50, 16)[:, 0].reshape(E,   -1)  # [E,16]
#         rbf_gs = rbf(dist_gs, 0, 50, 16)[:, 0].reshape(E_g, -1)  # [E_g,16]

#         # 也挂到 Rigid 上，兼容上游（若需要构造）
#         setattr(T_ts, "rbf", rbf_ts)
#         setattr(T_gs, "rbf", rbf_gs)

#         # ---- 其余信号 ----
#         chain_features = torch.ones(E, dtype=torch.int32)  # ligand 单链：全同链
#         S = S_ids.clone(); S_tgt = S_ids.clone()
#         # 单体：输入全 MASK，整条链监督
#         S_in = torch.full_like(S_ids, fill_value=self.tokenizer.mask_id)
#         loss_mask = chain_mask

#         # 统计占位（与核酸版接口一致）
#         mat9 = torch.zeros(3, 3, dtype=torch.long)
#         edge_stats_detail = {
#             "same_chain": int(E),
#             "cross_same_type": 0,
#             "cross_diff_type": 0,
#             "total_edges": int(E),
#             "same_frac": 1.0 if E > 0 else 0.0,
#             "cross_same_type_frac": 0.0,
#             "cross_diff_type_frac": 0.0,
#             "type_pair_counts_3x3": mat9.cpu().tolist(),
#             "type_legend": ["protein","rna","dna"],
#         }

#         # _V_g 必须是 Long 索引，供 nn.Embedding 使用
#         _V_g = torch.arange(self.virtual_frame_num, dtype=torch.long)
#         _E_g = torch.zeros((E_g, 128), dtype=torch.float32)

#         out = {
#             "type_vec": type_vec,  # [N] 0

#             # --- 刚体 + RBF（样本级，后面在 collate 里转成 rot/trans 张量） ---
#             "T": T, "T_g": T_g, "T_ts": T_ts, "T_gs": T_gs,
#             "rbf_ts": rbf_ts, "rbf_gs": rbf_gs,

#             # --- 几何特征/图 ---
#             "X": X6_filled, "_V": _V, "_E": _E,
#             "_V_g": _V_g, "_E_g": _E_g,
#             "edge_idx": edge_idx, "edge_idx_g": edge_idx_g,

#             # --- 序列/监督 ---
#             "S": S, "S_tgt": S_tgt, "S_in": S_in,
#             "loss_mask": loss_mask,

#             # --- 其他 ---
#             "batch_id": batch_id,
#             "batch_id_g": torch.zeros(self.virtual_frame_num, dtype=torch.long),
#             "num_nodes": torch.tensor([N], dtype=torch.long),
#             "mask": chain_mask, "chain_mask": chain_mask, "chain_encoding": chain_encoding,
#             "K_g": self.virtual_frame_num,

#             # 统计（可选）
#             "chain_features": chain_features,
#             "edge_stats_detail": edge_stats_detail,
#         }
#         return out

#     # ---------- 批处理 ----------
#     def featurize(self, batch: List[dict]):
#         samples = []
#         for one in batch:
#             feat = self._get_features_persample(one)
#             if feat is not None:
#                 samples.append(feat)
#         if not samples:
#             return None
#         return self.custom_collate_fn(samples)

#     def custom_collate_fn(self, batch: List[dict]):
#         batch = [b for b in batch if b is not None]
#         if not batch:
#             return None

#         K_g = int(batch[0]["K_g"])
#         num_nodes_list = [int(b["num_nodes"][0]) for b in batch]
#         B = len(batch)
#         total_real = sum(num_nodes_list)

#         prefix_real = torch.tensor(
#             [0] + list(torch.cumsum(torch.tensor(num_nodes_list[:-1]), dim=0).tolist()),
#             dtype=torch.long
#         )
#         base_virtual = total_real

#         def remap_indices(local_idx: torch.Tensor, N_i: int, base_real_i: int, base_virt_i: int) -> torch.Tensor:
#             is_virtual = (local_idx >= N_i)
#             out = local_idx.clone()
#             out[~is_virtual] += base_real_i
#             out[is_virtual] = (local_idx[is_virtual] - N_i) + (base_virtual + base_virt_i)
#             return out

#         ret: Dict[str, torch.Tensor] = {}

#         # 纯张量拼接
#         cat_keys = [
#             "X", "_V", "_E",
#             "S", "S_tgt", "S_in",
#             "type_vec",
#             "mask", "loss_mask", "chain_mask", "chain_encoding",
#             "rbf_ts", "rbf_gs",
#         ]
#         for k in cat_keys:
#             ret[k] = torch.cat([b[k] for b in batch], dim=0)

#         # 注意：_V_g 必须是 Long 索引
#         ret["_V_g"] = torch.cat([b["_V_g"] for b in batch], dim=0).long()
#         ret["_E_g"] = torch.cat([b["_E_g"] for b in batch], dim=0)

#         # 把 Rigid 拼接后只导出 rot/trans 张量，不把 Rigid 本体放进 batch
#         for k in ["T", "T_g", "T_ts", "T_gs"]:
#             T_cat = Rigid.cat([b[k] for b in batch], dim=0)
#             ret[k + "_rot"]   = T_cat._rots._rot_mats    # 张量
#             ret[k + "_trans"] = T_cat._trans             # 张量
#             # 不要把 Rigid 对象塞进 ret，避免 Lightning .to(device) 报错

#         ret["num_nodes"] = torch.tensor(num_nodes_list, dtype=torch.long)
#         ret["batch_id"]  = torch.cat([torch.full((num_nodes_list[i],), i, dtype=torch.long) for i in range(B)], dim=0)
#         ret["batch_id_g"] = torch.cat([torch.full((K_g,), i, dtype=torch.long) for i in range(B)], dim=0)

#         # 实边全局偏移
#         edge_parts = []
#         for i, b in enumerate(batch):
#             shift = prefix_real[i]
#             edge_parts.append(b["edge_idx"] + shift)
#         ret["edge_idx"] = torch.cat(edge_parts, dim=1)

#         # 全局边（实 + 虚）偏移
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

#         # —— 自检（训练时可注释）——
#         # assert ret["rbf_ts"].shape == (ret["edge_idx"].shape[1], 16)
#         # assert ret["rbf_gs"].shape == (ret["edge_idx_g"].shape[1], 16)

#         ret["K_g"] = K_g
#         return ret



# # # -*- coding: utf-8 -*-
# # # src/datasets/featurizer_ligand.py

# # # -*- coding: utf-8 -*-
# # # src/datasets/featurizer_ligand.py

# # import torch
# # import numpy as np
# # from typing import List, Sequence, Dict, Any, Optional

# # from torch_geometric.nn.pool import knn_graph

# # from ..tools.affine_utils import Rigid, Rotation, get_interact_feats
# # from ..modules.if_module import *  # rbf()


# # # ---------------- Tokenizer ----------------
# # class LigandTokenizer:
# #     BASE_ELEMS = [
# #         "C","O","N","P","S","Fe","Os","F","Mg","Cl","Br","W","B","Co","I","Mo",
# #         "As","Be","Mn","Cu","Ta","V","Al","Ir","Hg","Se","Ni","Ru","D","Pt",
# #         "Ca","Re","Zn","Si"
# #     ]
# #     def __init__(self) -> None:
# #         self.elem_to_id = {e: i for i, e in enumerate(self.BASE_ELEMS)}
# #         self.RARE_TOKEN = "RARE"; self.UNK_TOKEN = "UNK"; self.MASK_TOKEN = "*"
# #         self.rare_id = len(self.elem_to_id)       # 34
# #         self.unk_id  = self.rare_id + 1           # 35
# #         self.mask_id = self.unk_id + 1            # 36
# #         self.id_to_token = list(self.BASE_ELEMS) + [self.RARE_TOKEN, self.UNK_TOKEN, self.MASK_TOKEN]
# #         self.vocab_size = len(self.id_to_token)

# #     @staticmethod
# #     def _canon(sym: str):
# #         if sym is None: return None
# #         s = str(sym).strip()
# #         if not s: return None
# #         return s.upper() if len(s) == 1 else (s[0].upper() + s[1:].lower())

# #     def encode(self, elements: Sequence[str]) -> List[int]:
# #         out = []
# #         for e in elements:
# #             c = self._canon(e)
# #             if c is None: out.append(self.unk_id)
# #             elif c in self.elem_to_id: out.append(self.elem_to_id[c])
# #             else: out.append(self.rare_id)
# #         return out


# # # --------- 工具函数 ----------
# # def _take_to_len(arr_like, N, dtype=np.float32, fill=0.0) -> np.ndarray:
# #     """将输入安全拉到长度 N 的 1D 数组。"""
# #     if arr_like is None:
# #         return np.full((N,), fill_value=fill, dtype=dtype)
# #     try:
# #         arr = np.asarray(arr_like, dtype=dtype).reshape(-1)
# #     except Exception:
# #         return np.full((N,), fill_value=fill, dtype=dtype)
# #     if arr.shape[0] >= N:
# #         return arr[:N]
# #     out = np.full((N,), fill_value=fill, dtype=dtype)
# #     out[:arr.shape[0]] = arr
# #     return out

# # def _hyb_to_onehot(hyb_vec: torch.Tensor) -> torch.Tensor:
# #     """
# #     杂化稳健分桶：≤2 -> sp；==3 -> sp2；≥4 -> sp3+；非法/全零兜底到 sp3+。
# #     输入/输出： [N] -> [N,3]
# #     """
# #     hyb = hyb_vec.to(torch.float32)
# #     is_sp   = (hyb <= 2.0)
# #     is_sp2  = (hyb == 3.0)
# #     is_sp3p = (hyb >= 4.0)
# #     oh = torch.stack([is_sp, is_sp2, is_sp3p], dim=-1).to(torch.float32)
# #     bad = (oh.sum(dim=-1, keepdim=True) == 0)
# #     if bad.any():
# #         oh[bad.squeeze(-1)] = torch.tensor([0.0, 0.0, 1.0], dtype=oh.dtype, device=oh.device)
# #     return oh

# # def _zscore_clip(x: np.ndarray, clip: float = 3.0, eps: float = 1e-6) -> np.ndarray:
# #     """分子内 z-score，再裁剪（避免常数列/极值）。"""
# #     x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
# #     mu = float(x.mean()) if x.size > 0 else 0.0
# #     sd = float(x.std())  if x.size > 0 else 0.0
# #     x = (x - mu) / (sd + eps)
# #     return np.clip(x, -clip, clip).astype(np.float32)


# # # ---------------- Featurizer ----------------
# # class Featurize_Ligand:
# #     """
# #     保持原有接口，新增“节点侧拼接特征”：
# #       额外节点特征 = charges_RBF(8) + hybridization_3桶(3) + is_aromatic(1)
# #                    + HBD_count_broadcast(1) + HBA_count_broadcast(1)  → 14 维
# #       最终 _V: [N, 114+14] = [N,128]
# #     """
# #     def __init__(self,
# #                  knn_k: int = 48,
# #                  virtual_frame_num: int = 3,
# #                  min_atoms: int = 3,
# #                  debug_checks: bool = True):
# #         self.tokenizer = LigandTokenizer()
# #         self.knn_k = int(knn_k)
# #         self.virtual_frame_num = int(virtual_frame_num)
# #         self.min_atoms = int(min_atoms)
# #         self.debug_checks = bool(debug_checks)

# #         # 与下游对齐
# #         self.A_TOTAL = 6
# #         self.MIXED_NODE_IN = 114 + 14
# #         self.MIXED_EDGE_IN = 272

# #         # 额外特征超参
# #         self.CHARGE_RBF_BINS = 8
# #         self.CHARGE_RBF_MIN = -3.0  # 与 z-score clip 对齐
# #         self.CHARGE_RBF_MAX = +3.0
# #         self.HCOUNT_CAP = 10.0

# #     @staticmethod
# #     def _build_local_slots(coords: torch.Tensor) -> torch.Tensor:
# #         """
# #         coords: [N,3] -> X6: [N,6,3]
# #         槽 0: 最近邻；1: 当前；2: 次近邻；3..5: NaN
# #         """
# #         N = coords.shape[0]
# #         device = coords.device
# #         X6 = torch.full((N, 6, 3), float("nan"), dtype=coords.dtype, device=device)
# #         if N == 0:
# #             return X6
# #         dmat = torch.cdist(coords, coords, p=2)
# #         dmat.fill_diagonal_(float("inf"))
# #         k2 = 2 if N >= 3 else 1
# #         knn2 = torch.topk(dmat, k=k2, largest=False).indices
# #         j1 = knn2[:, 0]
# #         j2 = knn2[:, 1] if knn2.shape[1] > 1 else j1
# #         X6[:, 0] = coords[j1]
# #         X6[:, 1] = coords
# #         X6[:, 2] = coords[j2]
# #         return X6

# #     def _get_features_persample(self, item: Dict[str, Any]):
# #         # ---- 基本检查 ----
# #         coords_np = item.get("coords", None)
# #         if coords_np is None:
# #             return None
# #         coords = torch.as_tensor(coords_np, dtype=torch.float32)
# #         if coords.ndim != 2 or coords.shape[1] != 3:
# #             return None
# #         N = int(coords.shape[0])
# #         if N < self.min_atoms:
# #             return None

# #         elements = item.get("elements", None)
# #         if not isinstance(elements, list) or len(elements) != N:
# #             return None

# #         # ---- tokens / masks / encodings ----
# #         S_ids = torch.tensor(self.tokenizer.encode(elements), dtype=torch.long)
# #         cm_in = item.get("chain_mask", None)
# #         chain_mask = torch.from_numpy(np.asarray(cm_in[0], dtype=np.float32)) if isinstance(cm_in, (list, tuple)) and len(cm_in) >= 1 else torch.ones(N, dtype=torch.float32)
# #         ce_in = item.get("chain_encoding", None)
# #         chain_encoding = torch.from_numpy(np.asarray(ce_in[0], dtype=np.float32)) if isinstance(ce_in, (list, tuple)) and len(ce_in) >= 1 else torch.ones(N, dtype=torch.float32)

# #         # ---- 6 槽位 & anchors ----
# #         X6 = self._build_local_slots(coords)     # [N,6,3]
# #         X6_filled = X6.clone()
# #         X6_filled[torch.isnan(X6_filled)] = 0.0

# #         type_vec = torch.zeros(N, dtype=torch.long)
# #         batch_id = torch.zeros(N, dtype=torch.long)

# #         # ---- KNN 实边 ----
# #         center = X6_filled[:, 1, :]
# #         k_eff = min(max(self.knn_k, 1), max(N - 1, 1))
# #         edge_idx = knn_graph(center, k=k_eff, batch=batch_id, loop=False, flow='target_to_source')
# #         key = (edge_idx[1] * (edge_idx[0].max() + 1) + edge_idx[0]).long()
# #         order = torch.argsort(key)
# #         edge_idx = edge_idx[:, order]
# #         E = edge_idx.shape[1]

# #         # ---- 刚体与相对刚体（anchors = 0/1/2）----
# #         T = Rigid.make_transform_from_reference(
# #             X6_filled[:, 0].float(), X6_filled[:, 1].float(), X6_filled[:, 2].float()
# #         )
# #         src_idx, dst_idx = edge_idx[0], edge_idx[1]
# #         T_ts = T[dst_idx, None].invert().compose(T[src_idx, None])  # [E,1]

# #         # ---- 全局虚拟帧（K_g） — 稳化 SVD ----
# #         K_g = self.virtual_frame_num
# #         X_c = T._trans
# #         X_m = X_c.mean(dim=0, keepdim=True)
# #         X_c = X_c - X_m
# #         cov = X_c.T @ X_c
# #         cov = cov + 1e-6 * torch.eye(3, device=cov.device, dtype=cov.dtype)  # jitter
# #         # torch.linalg.svd 更稳
# #         U, Ssvd, Vh = torch.linalg.svd(cov, full_matrices=False)
# #         V = Vh.transpose(-1, -2)
# #         D = torch.eye(3, device=coords.device)
# #         if torch.det(U @ V.T) < 0:
# #             D[2, 2] = -1.0
# #         Rm = U @ D @ V.T
# #         rot_g = [Rm] * K_g
# #         trans_g = [X_m] * K_g
# #         T_g = Rigid(Rotation(torch.stack(rot_g)), torch.cat(trans_g, dim=0))  # [K_g]

# #         # ---- 几何特征（实边）----
# #         feat = get_interact_feats(T, T_ts, X6_filled.float(), edge_idx, batch_id)
# #         _V, _E = feat["_V"], feat["_E"]   # _V: [N,114]

# #         # ================== 新增：节点侧拼接特征 ==================
# #         # charges: 清洗 → 分子内 z-score → 裁剪 → RBF
# #         charges_np = _take_to_len(item.get("charges", None), N, dtype=np.float32, fill=0.0)
# #         charges_np = _zscore_clip(charges_np, clip=3.0, eps=1e-6)
# #         charges_t = torch.from_numpy(charges_np).to(dtype=_V.dtype)
# #         ch_rbf = rbf(
# #             charges_t.clamp(self.CHARGE_RBF_MIN, self.CHARGE_RBF_MAX),
# #             self.CHARGE_RBF_MIN, self.CHARGE_RBF_MAX, n_bins=self.CHARGE_RBF_BINS
# #         )  # [N,8]

# #         # hybridization -> 3 桶（含兜底）
# #         hyb_np = _take_to_len(item.get("hybridization", None), N, dtype=np.float32, fill=0.0)
# #         hyb_np = np.nan_to_num(hyb_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
# #         hyb_t  = torch.from_numpy(hyb_np).to(dtype=_V.dtype)
# #         hyb_oh = _hyb_to_onehot(hyb_t)  # [N,3]

# #         # is_aromatic -> [N,1]（clip 到 [0,1]）
# #         arom_np = _take_to_len(item.get("is_aromatic", None), N, dtype=np.float32, fill=0.0)
# #         arom_np = np.nan_to_num(arom_np, nan=0.0, posinf=0.0, neginf=0.0)
# #         arom_np = np.clip(arom_np, 0.0, 1.0).astype(np.float32)
# #         arom_col = torch.from_numpy(arom_np).to(dtype=_V.dtype).unsqueeze(-1)  # [N,1]

# #         # 分子级计数（截断+归一，广播）
# #         hbd = float(item.get("hbd_count", 0) or 0.0)
# #         hba = float(item.get("hba_count", 0) or 0.0)
# #         hbd_norm = min(max(hbd, 0.0), self.HCOUNT_CAP) / self.HCOUNT_CAP
# #         hba_norm = min(max(hba, 0.0), self.HCOUNT_CAP) / self.HCOUNT_CAP
# #         hbd_col = torch.full((N, 1), fill_value=hbd_norm, dtype=_V.dtype)
# #         hba_col = torch.full((N, 1), fill_value=hba_norm, dtype=_V.dtype)

# #         extra_V = torch.cat([ch_rbf, hyb_oh, arom_col, hbd_col, hba_col], dim=-1)  # [N,14]
# #         if self.debug_checks:
# #             assert torch.isfinite(extra_V).all(), "extra_V has NaN/Inf"
# #             assert extra_V.shape[1] == 14

# #         _V = torch.cat([_V, extra_V], dim=-1)  # [N,128]
# #         if self.debug_checks:
# #             assert _V.shape[1] == 128, f"_V dim mismatch: {_V.shape}"

# #         # ---- 虚拟全局边（双向）----
# #         global_nodes = torch.arange(N, N + K_g, device=coords.device)
# #         global_src = global_nodes.repeat_interleave(N)
# #         global_dst = torch.arange(N, device=coords.device).repeat(K_g)
# #         edge_idx_g = torch.stack([global_dst, global_src], dim=0)
# #         edge_idx_g = torch.cat([edge_idx_g, edge_idx_g.flip(0)], dim=1)
# #         E_g = edge_idx_g.shape[1]

# #         T_all = Rigid.cat([T, T_g], dim=0)
# #         T_src = T_all[edge_idx_g[0], None]
# #         T_dst = T_all[edge_idx_g[1], None]
# #         T_gs  = T_dst.invert().compose(T_src)

# #         dist_ts = T_ts._trans.norm(dim=-1)
# #         dist_gs = T_gs._trans.norm(dim=-1)
# #         rbf_ts = rbf(dist_ts, 0, 50, 16)[:, 0].reshape(E,   -1)
# #         rbf_gs = rbf(dist_gs, 0, 50, 16)[:, 0].reshape(E_g, -1)

# #         setattr(T_ts, "rbf", rbf_ts)
# #         setattr(T_gs, "rbf", rbf_gs)

# #         if self.debug_checks:
# #             for k, t in [("_V", _V), ("_E", _E), ("rbf_ts", rbf_ts), ("rbf_gs", rbf_gs)]:
# #                 assert torch.isfinite(t).all(), f"{k} has NaN/Inf"

# #         chain_features = torch.ones(E, dtype=torch.int32)
# #         S = S_ids.clone(); S_tgt = S_ids.clone()
# #         S_in = torch.full_like(S_ids, fill_value=self.tokenizer.mask_id)
# #         loss_mask = chain_mask

# #         mat9 = torch.zeros(3, 3, dtype=torch.long)
# #         edge_stats_detail = {
# #             "same_chain": int(E),
# #             "cross_same_type": 0,
# #             "cross_diff_type": 0,
# #             "total_edges": int(E),
# #             "same_frac": 1.0 if E > 0 else 0.0,
# #             "cross_same_type_frac": 0.0,
# #             "cross_diff_type_frac": 0.0,
# #             "type_pair_counts_3x3": mat9.cpu().tolist(),
# #             "type_legend": ["protein","rna","dna"],
# #         }

# #         _V_g = torch.arange(self.virtual_frame_num, dtype=torch.long)
# #         _E_g = torch.zeros((E_g, 128), dtype=torch.float32)

# #         out = {
# #             "type_vec": type_vec,

# #             "T": T, "T_g": T_g, "T_ts": T_ts, "T_gs": T_gs,
# #             "rbf_ts": rbf_ts, "rbf_gs": rbf_gs,

# #             "X": X6_filled, "_V": _V, "_E": _E,
# #             "_V_g": _V_g, "_E_g": _E_g,
# #             "edge_idx": edge_idx, "edge_idx_g": edge_idx_g,

# #             "S": S, "S_tgt": S_tgt, "S_in": S_in,
# #             "loss_mask": loss_mask,

# #             "batch_id": batch_id,
# #             "batch_id_g": torch.zeros(self.virtual_frame_num, dtype=torch.long),
# #             "num_nodes": torch.tensor([N], dtype=torch.long),
# #             "mask": chain_mask, "chain_mask": chain_mask, "chain_encoding": chain_encoding,
# #             "K_g": self.virtual_frame_num,

# #             "chain_features": chain_features,
# #             "edge_stats_detail": edge_stats_detail,
# #         }
# #         return out

# #     # ---------- 批处理 ----------
# #     def featurize(self, batch: List[dict]):
# #         samples = []
# #         for one in batch:
# #             feat = self._get_features_persample(one)
# #             if feat is not None:
# #                 samples.append(feat)
# #         if not samples:
# #             return None
# #         return self.custom_collate_fn(samples)

# #     def custom_collate_fn(self, batch: List[dict]):
# #         batch = [b for b in batch if b is not None]
# #         if not batch:
# #             return None

# #         K_g = int(batch[0]["K_g"])
# #         num_nodes_list = [int(b["num_nodes"][0]) for b in batch]
# #         B = len(batch)
# #         total_real = sum(num_nodes_list)

# #         prefix_real = torch.tensor(
# #             [0] + list(torch.cumsum(torch.tensor(num_nodes_list[:-1]), dim=0).tolist()),
# #             dtype=torch.long
# #         )
# #         base_virtual = total_real

# #         def remap_indices(local_idx: torch.Tensor, N_i: int, base_real_i: int, base_virt_i: int) -> torch.Tensor:
# #             is_virtual = (local_idx >= N_i)
# #             out = local_idx.clone()
# #             out[~is_virtual] += base_real_i
# #             out[is_virtual] = (local_idx[is_virtual] - N_i) + (base_virtual + base_virt_i)
# #             return out

# #         ret: Dict[str, torch.Tensor] = {}

# #         cat_keys = [
# #             "X", "_V", "_E",
# #             "S", "S_tgt", "S_in",
# #             "type_vec",
# #             "mask", "loss_mask", "chain_mask", "chain_encoding",
# #             "rbf_ts", "rbf_gs",
# #         ]
# #         for k in cat_keys:
# #             ret[k] = torch.cat([b[k] for b in batch], dim=0)

# #         ret["_V_g"] = torch.cat([b["_V_g"] for b in batch], dim=0).long()
# #         ret["_E_g"] = torch.cat([b["_E_g"] for b in batch], dim=0)

# #         for k in ["T", "T_g", "T_ts", "T_gs"]:
# #             T_cat = Rigid.cat([b[k] for b in batch], dim=0)
# #             ret[k + "_rot"]   = T_cat._rots._rot_mats
# #             ret[k + "_trans"] = T_cat._trans

# #         ret["num_nodes"] = torch.tensor(num_nodes_list, dtype=torch.long)
# #         ret["batch_id"]  = torch.cat([torch.full((num_nodes_list[i],), i, dtype=torch.long) for i in range(B)], dim=0)
# #         ret["batch_id_g"] = torch.cat([torch.full((K_g,), i, dtype=torch.long) for i in range(B)], dim=0)

# #         edge_parts = []
# #         for i, b in enumerate(batch):
# #             shift = prefix_real[i]
# #             edge_parts.append(b["edge_idx"] + shift)
# #         ret["edge_idx"] = torch.cat(edge_parts, dim=1)

# #         edge_g_parts = []
# #         for i, b in enumerate(batch):
# #             N_i = num_nodes_list[i]
# #             base_real_i = int(prefix_real[i].item())
# #             base_virt_i = i * K_g
# #             src_local = b["edge_idx_g"][0]
# #             dst_local = b["edge_idx_g"][1]
# #             src_global = remap_indices(src_local, N_i, base_real_i, base_virt_i)
# #             dst_global = remap_indices(dst_local, N_i, base_real_i, base_virt_i)
# #             edge_g_parts.append(torch.stack([src_global, dst_global], dim=0))
# #         ret["edge_idx_g"] = torch.cat(edge_g_parts, dim=1)

# #         ret["K_g"] = K_g
# #         return ret

