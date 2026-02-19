# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.if_module import *  # StructureEncoder, MLPDecoder, build_MLP, rbf()
from ..datasets.featurizer_ligand import LigandTokenizer
from ..tools.affine_utils import Rigid, Rotation


class ODesign_Ligand_Model(nn.Module):
    """
    仅用几何/图特征进行配体元素预测（不注入任何序列 embedding）。
    - 词表：LigandTokenizer（元素表）
    - 输入：与 LigandComplex featurizer 对齐的整图 batch（受体+配体）
    - 输出：logits / log_probs（整图每个节点），训练时请用外部 loss_mask=ligand_mask 仅在配体位点监督
    """

    def __init__(self, args, **kwargs):
        super().__init__()
        self.__dict__.update(locals())

        geo_layer, attn_layer = args.geo_layer, args.attn_layer
        node_layer, edge_layer = args.node_layer, args.edge_layer
        encoder_layer, hidden_dim = args.encoder_layer, args.hidden_dim
        dropout, mask_rate = args.dropout, args.mask_rate

        # —— 配体元素词表（预测目标）——
        self.tokenizer = LigandTokenizer()
        self.mask_id = self.tokenizer.mask_id
        V = self.tokenizer.vocab_size

        # —— Mixed 输入维度（与 featurizer 输出一致）——
        NODE_IN = 114
        EDGE_IN = 272

        # 几何→隐空间投影
        self.node_embedding = build_MLP(2, NODE_IN, hidden_dim, hidden_dim)
        self.edge_embedding = build_MLP(2, EDGE_IN, hidden_dim, hidden_dim)

        # 虚拟全局节点 embedding（固定 30 够用；由 batch['_V_g'] 索引）
        self.virtual_embedding = nn.Embedding(30, hidden_dim)

        # 结构编码器与解码器
        self.encoder = StructureEncoder(
            geo_layer, attn_layer, node_layer, edge_layer,
            encoder_layer, hidden_dim, dropout, mask_rate
        )
        self.decoder = MLPDecoder(hidden_dim, vocab=V)

        self._init_params()

    def _init_params(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch, num_global=3):
        """
        期望 batch 键（来自 LigandComplex featurizer）：
          X, _V, _E, edge_idx, edge_idx_g
          T_rot/T_trans, T_g_rot/T_g_trans, T_ts_rot/T_ts_trans, T_gs_rot/T_gs_trans
          rbf_ts, rbf_gs, _V_g, batch_id, batch_id_g
        训练时 loss_mask 应在外部传入（推荐使用 ligand_mask）。
        """
        # 原始几何/图特征
        X, h_V0, h_E0 = batch['X'], batch['_V'], batch['_E']
        edge_idx, batch_id = batch['edge_idx'], batch['batch_id']
        edge_idx_g, batch_id_g = batch['edge_idx_g'], batch['batch_id_g']

        # 刚体构造
        T    = Rigid(Rotation(batch['T_rot']),    batch['T_trans'])
        T_g  = Rigid(Rotation(batch['T_g_rot']),  batch['T_g_trans'])
        T_ts = Rigid(Rotation(batch['T_ts_rot']), batch['T_ts_trans'])
        T_gs = Rigid(Rotation(batch['T_gs_rot']), batch['T_gs_trans'])
        T_ts.rbf = batch['rbf_ts']
        T_gs.rbf = batch['rbf_gs']

        # 几何投影到隐空间
        h_V  = self.node_embedding(h_V0)                           # [N, hidden]
        h_E  = self.edge_embedding(h_E0)                           # [E, hidden]
        h_V_g = self.virtual_embedding(batch['_V_g'])              # [K_g, hidden]
        h_E_g = torch.zeros((edge_idx_g.shape[1], h_E.shape[1]),
                            device=h_V.device, dtype=h_V.dtype)    # 全局边初始 0

        # 进入结构编码器（保持你原来的接口形状）
        # 注意：部分实现里会需要原始边特征拼接为 h_E_0，这里与之前版本保持一致
        h_E_0_cat = torch.cat(
            [h_E0,
             torch.zeros((edge_idx_g.shape[1], h_E0.shape[1]),
                         device=h_V.device, dtype=h_V0.dtype)]
        )
        h_V = self.encoder(
            h_S=None,                    # 不注入任何序列信息
            T=T, T_g=T_g,
            h_V=h_V, h_V_g=h_V_g,
            h_E=h_E, h_E_g=h_E_g,
            T_ts=T_ts, T_gs=T_gs,
            edge_idx=edge_idx, edge_idx_g=edge_idx_g,
            batch_id=batch_id, batch_id_g=batch_id_g,
            h_E_0=h_E_0_cat
        )

        # 解码到配体词表
        logits = self.decoder.readout(h_V)          # [N, V]
        log_probs = F.log_softmax(logits, dim=-1)   # [N, V]

        return {'log_probs': log_probs, 'logits': logits}

    # 兼容接口：上层会先通过 featurizer 得到 batch，这里直接返回即可
    def _get_features(self, batch):
        return batch
