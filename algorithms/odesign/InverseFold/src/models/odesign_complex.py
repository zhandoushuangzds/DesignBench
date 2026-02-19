# # -*- coding: utf-8 -*-
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from ..modules.if_module import build_MLP, StructureEncoder, MLPDecoder
# from ..datasets.featurizer_complex import Unitokenizer_Complex
# from ..tools.affine_utils import Rigid, Rotation


# class ODesign_Complex_Model(nn.Module):
#     """
#     Node-side Group-FiLM（对齐新版 interface）：
#     - 优先使用 batch['S_in']；若无则回退到 batch['S']；若仍不可用则跳过 FiLM
#     - 兼容 S_in 形状 [N] / [B,N]，自动对齐到节点数；必要时尝试使用 batch['mask']
#     - 不改边特征/结构编码器；不区分单链/多链
#     """
#     def __init__(self, args, **kwargs):
#         super().__init__()
#         self.__dict__.update(locals())

#         geo_layer, attn_layer = args.geo_layer, args.attn_layer
#         node_layer, edge_layer = args.node_layer, args.edge_layer
#         encoder_layer, hidden_dim = args.encoder_layer, args.hidden_dim
#         dropout, mask_rate = args.dropout, args.mask_rate

#         # ---- Tokenizer / vocab ----
#         self.tokenizer = Unitokenizer_Complex()
#         self.mask_id = self.tokenizer.mask_id
#         self.vocab_size = self.tokenizer.vocab_size
#         V = self.vocab_size

#         # ---- Node/Edge embedding（保持 114 / 272）----
#         NODE_IN = 114
#         EDGE_IN = 272
#         self.node_embedding = build_MLP(2, NODE_IN, hidden_dim, hidden_dim)
#         self.edge_embedding = build_MLP(2, EDGE_IN, hidden_dim, hidden_dim)

#         # ---- 虚拟节点（保持接口）----
#         self.virtual_embedding = nn.Embedding(30, hidden_dim)

#         # ===================== Group-FiLM（仅 node 侧） =====================
#         self.seq_dim = getattr(args, "seq_dim", 32)            # 小词嵌入维
#         self.seq_embedding = nn.Embedding(V, self.seq_dim)

#         self.num_groups = getattr(args, "film_groups", 8)      # 组数
#         assert hidden_dim % self.num_groups == 0, \
#             f"hidden_dim={hidden_dim} must be divisible by film_groups={self.num_groups}"

#         self.seq_to_cond = nn.Sequential(
#             nn.Linear(self.seq_dim, 32),
#             nn.GELU(),
#             nn.Linear(32, 32),
#             nn.GELU()
#         )
#         self.to_gamma = nn.Linear(32, self.num_groups)
#         self.to_beta  = nn.Linear(32, self.num_groups)

#         self.pre_ln = nn.LayerNorm(hidden_dim)                 # FiLM 前先归一化
#         self.cond_dropout = nn.Dropout(getattr(args, "film_dropout", 0.1))

#         # ---- Backbone ----
#         self.encoder = StructureEncoder(
#             geo_layer, attn_layer, node_layer, edge_layer,
#             encoder_layer, hidden_dim, dropout, mask_rate
#         )
#         self.decoder = MLPDecoder(hidden_dim, vocab=V)

#         self._init_params()

#     def _init_params(self):
#         for n, p in self.named_parameters():
#             if p.dim() > 1 and "virtual_embedding" not in n:
#                 nn.init.xavier_uniform_(p)
#         # 让 FiLM 初始几乎无效（渐进启用条件）
#         nn.init.zeros_(self.to_gamma.weight); nn.init.zeros_(self.to_gamma.bias)
#         nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)

#     @staticmethod
#     def _broadcast_group_params(param_g: torch.Tensor, hidden_dim: int, num_groups: int):
#         """
#         [N, g] -> [N, H] 组级广播
#         """
#         N, g = param_g.shape
#         per = hidden_dim // num_groups
#         return param_g.unsqueeze(-1).expand(N, g, per).reshape(N, hidden_dim)

#     def _get_seq_tokens_from_batch(self, batch, N_nodes: int):
#         """
#         取得用于 FiLM 的 token 向量，并与节点数对齐：
#         - 优先 S_in；无则回退 S；若都无或不匹配则返回 None
#         - 兼容 [N] / [B,N]；必要时用 batch['mask'] 对齐（mask==1 的位置）
#         """
#         tok = batch.get('S_in', None)
#         if tok is None:
#             tok = batch.get('S', None)
#         if tok is None:
#             return None

#         # 若已经是一维且长度匹配
#         if tok.dim() == 1 and tok.numel() == N_nodes:
#             return tok.long()

#         # 若是二维，尝试展平
#         if tok.dim() == 2:
#             if tok.numel() == N_nodes:
#                 return tok.reshape(-1).long()

#             # 若提供了 mask，可用 mask==1 选择元素
#             m = batch.get('mask', None)
#             if m is not None:
#                 # 接受 m=[N] 或 [B,N] 的两类
#                 if m.dtype != torch.bool:
#                     m_bool = (m > 0)
#                 else:
#                     m_bool = m
#                 try:
#                     sel = torch.masked_select(tok, m_bool)  # 广播选择
#                     if sel.numel() == N_nodes:
#                         return sel.long()
#                 except Exception:
#                     pass

#         # 回退：不施加 FiLM
#         return None

#     def forward(self, batch, num_global=3):
#         # -------- 刚体与图信息 --------
#         X, h_V0, h_E0 = batch['X'], batch['_V'], batch['_E']
#         edge_idx, batch_id = batch['edge_idx'], batch['batch_id']
#         edge_idx_g, batch_id_g = batch['edge_idx_g'], batch['batch_id_g']

#         T    = Rigid(Rotation(batch['T_rot']),    batch['T_trans'])
#         T_g  = Rigid(Rotation(batch['T_g_rot']),  batch['T_g_trans'])
#         T_ts = Rigid(Rotation(batch['T_ts_rot']), batch['T_ts_trans'])
#         T_gs = Rigid(Rotation(batch['T_gs_rot']), batch['T_gs_trans'])
#         T_ts.rbf = batch['rbf_ts']
#         T_gs.rbf = batch['rbf_gs']

#         # -------- 几何 -> 隐空间 --------
#         h_V = self.node_embedding(h_V0)                 # [N,H]
#         h_E = self.edge_embedding(h_E0)                 # [E,H]
#         h_V_g = self.virtual_embedding(batch['_V_g'])   # [K_g,H]
#         h_E_g = torch.zeros((edge_idx_g.shape[1], h_E.shape[1]),
#                             device=h_V.device, dtype=h_V.dtype)

#         # ===================== Group-FiLM 注入（与新版 interface 对齐） =====================
#         N_nodes, H = h_V.shape[0], h_V.shape[1]
#         toks = self._get_seq_tokens_from_batch(batch, N_nodes)  # [N] 或 None

#         if toks is not None:
#             # 小嵌入 → 条件
#             e = self.seq_embedding(toks.to(h_V.device))         # [N, d_s]
#             u = self.seq_to_cond(e)                             # [N, 32]
#             gamma_g = self.cond_dropout(self.to_gamma(u))       # [N, g]
#             beta_g  = self.cond_dropout(self.to_beta(u))        # [N, g]

#             gamma = self._broadcast_group_params(gamma_g, H, self.num_groups).to(h_V.dtype)
#             beta  = self._broadcast_group_params(beta_g,  H, self.num_groups).to(h_V.dtype)

#             # 先 LN 再 FiLM
#             h_V = self.pre_ln(h_V)
#             h_V = h_V * (1.0 + gamma) + beta
#         else:
#             # 无法可靠对齐 tokens：走纯几何 + pre-LN
#             h_V = self.pre_ln(h_V)

#         # ===================== 进入结构编码器 =====================
#         h_E_0_cat = torch.cat(
#             [h_E0, torch.zeros((edge_idx_g.shape[1], h_E0.shape[1]),
#                                device=h_V.device, dtype=h_V.dtype)],
#             dim=0
#         )

#         h_V = self.encoder(
#             h_S=None,
#             T=T, T_g=T_g,
#             h_V=h_V, h_V_g=h_V_g,
#             h_E=h_E, h_E_g=h_E_g,
#             T_ts=T_ts, T_gs=T_gs,
#             edge_idx=edge_idx, edge_idx_g=edge_idx_g,
#             batch_id=batch_id, batch_id_g=batch_id_g,
#             h_E_0=h_E_0_cat
#         )

#         # ===================== 解码 =====================
#         logits = self.decoder.readout(h_V)          # [N,V]（与你的 loss 计算对齐）
#         log_probs = F.log_softmax(logits, dim=-1)

#         return {'log_probs': log_probs, 'logits': logits}

#     # 兼容接口：外侧会先调 _get_features，再调 forward
#     def _get_features(self, batch):
#         return batch

       

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.if_module import build_MLP, StructureEncoder, MLPDecoder
from ..datasets.featurizer_complex import Unitokenizer_Complex
from ..tools.affine_utils import Rigid, Rotation


# -----------------------
# 小型 Token-Routed Experts（MoE）模块
# -----------------------
class TokenMoE(nn.Module):
    """
    Token-Routed Experts（node-only）
    - 路由条件：token_emb (ds) + 几何摘要 geom (dg)
    - Top-k 路由，专家为轻量 MLP（两层），残差注入
    - 可返回负载均衡辅助损失（Switch Transformer 风格）
    """
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_experts: int = 4,
        top_k: int = 1,
        ds: int = 32,         # token embedding dim
        dg: int = 32,         # geom summary dim
        fuse_mid: int = 64,   # router 条件融合中间维
        ff_mult: float = 2.0, # 每个专家的 FF 宽度：ff_mult * H
        act: str = "gelu",
        dropout: float = 0.1,
        router_temp: float = 1.0,
        router_noisy_std: float = 0.0,  # 训练期可加噪
        residual_scale: float = None,   # 默认为 1/sqrt(num_experts)
        mask_logit_bias: float = 0.0,   # 对 mask 位路由 logits 的加性偏置（增强被遮位）
    ):
        super().__init__()
        assert top_k >= 1 and top_k <= num_experts
        self.H = hidden_dim
        self.E = num_experts
        self.K = top_k
        self.router_temp = router_temp
        self.router_noisy_std = router_noisy_std
        self.mask_logit_bias = mask_logit_bias

        if residual_scale is None:
            residual_scale = 1.0 / math.sqrt(num_experts)
        self.residual_scale = residual_scale

        # 条件融合：token(ds) + geom(dg) -> fuse_mid
        self.fuse = nn.Sequential(
            nn.Linear(ds + dg, fuse_mid),
            nn.GELU(),
        )
        # router：fuse_mid -> num_experts
        self.router = nn.Linear(fuse_mid, num_experts)

        # 几何摘要：对 LN(h) 线性压缩 -> dg
        self.pre_ln = nn.LayerNorm(hidden_dim)
        self.geom_proj = nn.Sequential(
            nn.Linear(hidden_dim, dg),
            nn.GELU(),
        )

        # 专家们：每个是 H -> ffH -> H 的小 MLP
        ff_hidden = int(ff_mult * hidden_dim)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ff_hidden),
                nn.GELU() if act.lower() == "gelu" else nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_hidden, hidden_dim),
            ) for _ in range(num_experts)
        ])
        self.expert_drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, tok_emb: torch.Tensor, is_mask: torch.Tensor, *, training: bool = False):
        """
        h:       [N,H]
        tok_emb: [N,ds]  已经过外部 nn.Embedding
        is_mask: [N]     Bool/0-1，指示被遮位（可影响路由偏置）
        返回：
            h_out: [N,H]  注入后的节点特征
            aux:   dict   可能包含 'load_balance_loss'
        """
        N, H = h.shape
        assert H == self.H

        # 几何摘要（来自 LN(h)）
        h_ln = self.pre_ln(h)                  # [N,H]
        g = self.geom_proj(h_ln)               # [N,dg]

        # 路由 logits
        cond = torch.cat([tok_emb, g], dim=-1) # [N, ds+dg]
        u = self.fuse(cond)                    # [N, fuse_mid]
        logits = self.router(u)                # [N, E]

        # mask 位路由偏置（可选）
        if self.mask_logit_bias != 0.0:
            logits = logits + (is_mask.float().unsqueeze(-1) * self.mask_logit_bias)

        # 训练期可选加噪
        if training and self.router_noisy_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.router_noisy_std

        # softmax 概率（温度）
        probs = F.softmax(logits / max(self.router_temp, 1e-6), dim=-1)  # [N,E]

        # Top-k 选择
# ---- Top-1 快速路径（分桶 + bincount；无大矩阵）----
        if self.K == 1:
            top_prob, top_idx = probs.max(dim=-1)  # [N], [N]
            y = torch.zeros_like(h)                # [N,H]

            # 1) 按 expert 排序，做一次性连续前向
            order = torch.argsort(top_idx)         # [N]
            h_sorted = h[order]
            idx_sorted = top_idx[order]
            # 找每个 expert 的边界
            # e_starts[e]: 此 expert 在 h_sorted 的起始下标；若无样本则相等
            e_counts = torch.bincount(idx_sorted, minlength=self.E)       # [E]
            e_starts = torch.cumsum(torch.cat([e_counts.new_zeros(1), e_counts[:-1]]), 0)  # [E]
            # 逐专家一次性前向（无布尔掩码切片）
            for e in range(self.E):
                cnt = e_counts[e].item()
                if cnt == 0: 
                    continue
                s = e_starts[e].item()
                t = s + cnt
                out_e = self.expert_drop(self.experts[e](h_sorted[s:t]))  # [cnt,H]
                y[order[s:t]] = out_e

            # 2) 残差注入（按 token 概率缩放）
            h_out = h + self.residual_scale * (top_prob.unsqueeze(-1) * y)

            # 3) 负载均衡损失（无 [N,E] 矩阵）
            importance = probs.sum(dim=0)                                  # [E]
            load = torch.bincount(top_idx, minlength=self.E).float()       # [E]
            lb_loss = (self.E * (importance * load).sum()) / (N * N + 1e-8)

            return h_out, {"load_balance_loss": lb_loss}


# -----------------------
# 主模型：在 encoder 前加入 Token-Routed Experts
# -----------------------
class ODesign_Complex_Model(nn.Module):
    """
    Node-side 小 MoE（前置门控）
    - 仅改 node 输入侧，edge/encoder/decoder 不变
    - 优先使用 batch['S_in']，回退 batch['S']；兼容 [N]/[B,N]；必要时用 batch['mask'] 对齐
    - 可选地返回一个 'aux_losses'，包含负载均衡 loss（外层可选择性加权使用）
    """
    def __init__(self, args, **kwargs):
        super().__init__()
        self.__dict__.update(locals())

        # ---- 基本配置 ----
        geo_layer, attn_layer = args.geo_layer, args.attn_layer
        node_layer, edge_layer = args.node_layer, args.edge_layer
        encoder_layer, hidden_dim = args.encoder_layer, args.hidden_dim
        dropout, mask_rate = args.dropout, args.mask_rate
        H = hidden_dim

        # ---- Tokenizer / vocab ----
        self.tokenizer = Unitokenizer_Complex()
        self.mask_id = self.tokenizer.mask_id
        self.vocab_size = self.tokenizer.vocab_size
        V = self.vocab_size

        # ---- 基础几何嵌入 ----
        NODE_IN = 114
        EDGE_IN = 272
        self.node_embedding = build_MLP(2, NODE_IN, H, H)
        self.edge_embedding = build_MLP(2, EDGE_IN, H, H)

        # ---- 虚拟节点 ----
        self.virtual_embedding = nn.Embedding(30, H)

        # ---- token 小嵌入（供 MoE 路由使用）----
        self.seq_dim = getattr(args, "seq_dim", 32)
        self.seq_embedding = nn.Embedding(V, self.seq_dim)

        # ---- 小 MoE（前置）----
        self.num_experts = getattr(args, "moe_num_experts", 4)
        self.top_k = getattr(args, "moe_top_k",1)
        self.moe = TokenMoE(
            hidden_dim=H,
            num_experts=self.num_experts,
            top_k=self.top_k,
            ds=self.seq_dim,
            dg=getattr(args, "moe_geom_dim", 16),
            fuse_mid=getattr(args, "moe_fuse_mid", 32),
            ff_mult=float(getattr(args, "moe_ff_mult", 2.0)),
            act=str(getattr(args, "moe_act", "gelu")),
            dropout=float(getattr(args, "moe_dropout", 0.1)),
            router_temp=float(getattr(args, "moe_router_temp", 1.0)),
            router_noisy_std=float(getattr(args, "moe_router_noisy_std", 0.1)),
            residual_scale=float(getattr(args, "moe_residual_scale", 1.0 / math.sqrt(max(1, getattr(args, "moe_num_experts", 4))))),
            mask_logit_bias=float(getattr(args, "moe_mask_logit_bias", 0.0)),
        )

        # ---- Backbone ----
        self.encoder = StructureEncoder(
            geo_layer, attn_layer, node_layer, edge_layer,
            encoder_layer, H, dropout, mask_rate
        )
        self.decoder = MLPDecoder(H, vocab=V)

        self._init_params()

    def _init_params(self):
        for n, p in self.named_parameters():
            if p.dim() > 1 and "virtual_embedding" not in n:
                nn.init.xavier_uniform_(p)

    # ---------- 取得 tokens 并与节点对齐 ----------
    def _get_seq_tokens_from_batch(self, batch, N_nodes: int):
        tok = batch.get('S_in', None)
        if tok is None:
            tok = batch.get('S', None)
        if tok is None:
            return None

        if tok.dim() == 1 and tok.numel() == N_nodes:
            return tok.long()

        if tok.dim() == 2:
            if tok.numel() == N_nodes:
                return tok.reshape(-1).long()
            m = batch.get('mask', None)
            if m is not None:
                m_bool = (m > 0) if m.dtype != torch.bool else m
                try:
                    sel = torch.masked_select(tok, m_bool)
                    if sel.numel() == N_nodes:
                        return sel.long()
                except Exception:
                    pass
        return None

    # ---------- 前向 ----------
    def forward(self, batch, num_global=3):
        # 图/刚体
        X, h_V0, h_E0 = batch['X'], batch['_V'], batch['_E']
        edge_idx, batch_id = batch['edge_idx'], batch['batch_id']
        edge_idx_g, batch_id_g = batch['edge_idx_g'], batch['batch_id_g']

        T    = Rigid(Rotation(batch['T_rot']),    batch['T_trans'])
        T_g  = Rigid(Rotation(batch['T_g_rot']),  batch['T_g_trans'])
        T_ts = Rigid(Rotation(batch['T_ts_rot']), batch['T_ts_trans'])
        T_gs = Rigid(Rotation(batch['T_gs_rot']), batch['T_gs_trans'])
        T_ts.rbf = batch['rbf_ts']
        T_gs.rbf = batch['rbf_gs']

        # 基础嵌入
        h_V = self.node_embedding(h_V0)                 # [N,H]
        h_E = self.edge_embedding(h_E0)                 # [E,H]
        h_V_g = self.virtual_embedding(batch['_V_g'])   # [K_g,H]
        h_E_g = torch.zeros((edge_idx_g.shape[1], h_E.shape[1]),
                            device=h_V.device, dtype=h_V.dtype)

        # tokens（对齐到节点）
        N, H = h_V.shape
        toks = self._get_seq_tokens_from_batch(batch, N)   # [N] or None

        aux_losses = {}

        # ========== 小 MoE（前置） ==========
        if toks is not None:
            tok_emb = self.seq_embedding(toks.to(h_V.device))     # [N, ds]
            is_mask = (toks == self.mask_id)                      # [N]
            h_V, aux = self.moe(h_V, tok_emb, is_mask, training=self.training)
            aux_losses.update(aux)
        # 若无 tokens，跳过 MoE（退化为纯几何）

        # ========== 结构编码器 ==========
        h_E_0_cat = torch.cat(
            [h_E0, torch.zeros((edge_idx_g.shape[1], h_E0.shape[1]),
                               device=h_V.device, dtype=h_V.dtype)],
            dim=0
        )

        h_V = self.encoder(
            h_S=None,
            T=T, T_g=T_g,
            h_V=h_V, h_V_g=h_V_g,
            h_E=h_E, h_E_g=h_E_g,
            T_ts=T_ts, T_gs=T_gs,
            edge_idx=edge_idx, edge_idx_g=edge_idx_g,
            batch_id=batch_id, batch_id_g=batch_id_g,
            h_E_0=h_E_0_cat
        )

        # ========== 解码 ==========
        logits = self.decoder.readout(h_V)              # [N,V]
        log_probs = F.log_softmax(logits, dim=-1)

        out = {'log_probs': log_probs, 'logits': logits}
        # if aux_losses:
        #     out['aux_losses'] = aux_losses   # 外层可选择性加权使用
        return out

    # 兼容接口
    def _get_features(self, batch):
        return batch
