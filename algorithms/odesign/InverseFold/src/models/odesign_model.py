import torch
import torch.nn as nn
from ..modules.if_module import *
from ..datasets.featurizer import UniTokenizer
# from ..datasets.featurizer_complex import Unitokenizer_Complex
from ..tools.affine_utils import Rigid, Rotation
from torch.nn.utils.rnn import pad_sequence

class ODesign_Model(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(ODesign_Model, self).__init__()
        self.__dict__.update(locals())
        geo_layer, attn_layer, node_layer, edge_layer, encoder_layer, hidden_dim, dropout, mask_rate = args.geo_layer, args.attn_layer, args.node_layer, args.edge_layer, args.encoder_layer, args.hidden_dim, args.dropout, args.mask_rate
        self.use_ar =  getattr(args, 'use_ar', args.get('use_ar', False) if isinstance(args, dict) else False)
        
        self.tokenizer = UniTokenizer()
        self.vocab_size = vocab_size = self.tokenizer.vocab_size

        if args['dataset'] in ('DNA', 'RNA','Mixed'):
            self.node_embedding = build_MLP(2, 114+19, hidden_dim, hidden_dim)
            self.edge_embedding = build_MLP(2, 272+38, hidden_dim, hidden_dim)
        if args['dataset']=='Protein':
            self.node_embedding = build_MLP(2, 76, hidden_dim, hidden_dim)
            self.edge_embedding = build_MLP(2, 196, hidden_dim, hidden_dim)
        self.virtual_embedding = nn.Embedding(30, hidden_dim) 
        self.encoder = StructureEncoder(geo_layer, attn_layer, node_layer, edge_layer, encoder_layer, hidden_dim, dropout, mask_rate)
        self.decoder_mlp = MLPDecoder(hidden_dim,vocab=vocab_size)
        
        self.decoder = SimpleARDecoder(hidden_dim, vocab=vocab_size,
                                        n_layers=4, n_heads=8,
                                        dropout=dropout, use_struct=True, tie_weights=True)
        # self.decoder = CrossARDecoder(hidden_dim, vocab=vocab_size,
        #                             n_layers=4, n_heads=8,cross_from='half',
        #                             dropout=dropout, use_struct=True, tie_weights=True)

        self.chain_embeddings = nn.Embedding(2, 16)

        self._init_params()

    def _init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    @staticmethod
    def _segment_by_chain(x, batch_id, chain_encoding):
        """
        把一维长序列按 (batch_id, chain_encoding) 的连续段切分成若干 [Li, ...] 段。
        x: [N, d] 或 [N]；返回 list[tensor]
        """
        device = batch_id.device
        N = batch_id.size(0)
        # 链起点：batch_id 或 chain_encoding 发生变化的位置
        ce = chain_encoding.long()
        start = torch.zeros(N, dtype=torch.bool, device=device)
        start[0] = True
        start[1:] = (batch_id[1:] != batch_id[:-1]) | (ce[1:] != ce[:-1])

        starts = torch.nonzero(start, as_tuple=False).flatten()
        ends = torch.cat([starts[1:], torch.tensor([N], device=device)])
        if x.dim() == 1:
            segs = [x[s:e] for s, e in zip(starts.tolist(), ends.tolist())]
        else:
            segs = [x[s:e, ...] for s, e in zip(starts.tolist(), ends.tolist())]
        return segs
    
    
    def forward(self, batch, num_global = 3):
        X, h_V, h_E, edge_idx, batch_id, chain_features = batch['X'], batch['_V'], batch['_E'], batch['edge_idx'], batch['batch_id'], batch['chain_features']
        edge_idx_g, batch_id_g = batch['edge_idx_g'], batch['batch_id_g']
        T = Rigid(Rotation(batch['T_rot']), batch['T_trans'])
        T_g = Rigid(Rotation(batch['T_g_rot']), batch['T_g_trans'])
        T_ts = Rigid(Rotation(batch['T_ts_rot']), batch['T_ts_trans'])
        T_gs = Rigid(Rotation(batch['T_gs_rot']), batch['T_gs_trans'])
        rbf_ts, rbf_gs = batch['rbf_ts'], batch['rbf_gs']
        T_gs.rbf = rbf_gs
        T_ts.rbf = rbf_ts

        h_E_0 = h_E
        h_V = self.node_embedding(h_V)
        h_E = self.edge_embedding(h_E)
        h_V_g = self.virtual_embedding(batch['_V_g'])
        h_E_g = torch.zeros((edge_idx_g.shape[1], h_E.shape[1]), device=h_V.device, dtype=h_V.dtype)
        h_S = None
        h_E_0 = torch.cat([h_E_0, torch.zeros((edge_idx_g.shape[1], h_E_0.shape[1]), device=h_V.device, dtype=h_V.dtype)])

        def chk(name, x):
            ok = torch.isfinite(x).all()
            if not ok: print(f"[NaN] {name} has NaN/Inf")
            return ok

        assert chk("h_V_in", h_V) and chk("h_E_in", h_E)
        assert chk("T_ts_rbf", T_ts.rbf) and chk("T_rot", T._rots._rot_mats) and chk("T_trans", T._trans)
        assert chk("T_ts_rot", T_ts._rots._rot_mats)
        assert chk("T_ts_trans", T_ts._trans)
        h_V = self.encoder(h_S, T, T_g, 
                            h_V, h_V_g,
                            h_E, h_E_g,
                            T_ts, T_gs, 
                            edge_idx, edge_idx_g,
                            batch_id, batch_id_g, h_E_0)
        
        if self.use_ar:
            # === AR 解码（teacher forcing）===
            H_enc = h_V   # [N, d]
            device = H_enc.device
            chain_enc = batch['chain_encoding'].to(device)    # [N]
            S_all     = batch['S'].long().to(device)          # [N]

            seg_H = self._segment_by_chain(H_enc, batch_id, chain_enc)   # list of [Li, d]
            seg_S = self._segment_by_chain(S_all,  batch_id, chain_enc)  # list of [Li]

            B = len(seg_H)
            L_max = max(s.shape[0] for s in seg_S)

            H_pad = pad_sequence(seg_H, batch_first=True, padding_value=0.0)       # [B, L, d]
            S_pad = pad_sequence(seg_S, batch_first=True, padding_value=0)         # [B, L]
            # valid_mask: 前 Li 为 True，后面 PAD 为 False
            lengths = torch.tensor([t.size(0) for t in seg_S], device=device)
            arangeL = torch.arange(L_max, device=device).unsqueeze(0).expand(B, -1)
            valid_mask = arangeL < lengths.unsqueeze(1)                             # [B, L] bool

            # ==== ⬇️ 自回归解码（并行训练）====
            log_probs_pad, logits_pad = self.decoder(H_pad, S_pad, valid_mask)     # [B, L, V]

            # ==== ⬇️ 为了兼容原返回形状 [N, V]：只取有效位置并压平 ====
            V = logits_pad.shape[-1]
            sel = valid_mask.view(-1)
            logits = logits_pad.reshape(-1, V)[sel]        # [N, V]
            log_probs = log_probs_pad.reshape(-1, V)[sel]  # [N, V]
                
        else:
            log_probs, logits = self.decoder_mlp(h_V)
            print('---------------------------------------------------')
        
        return {'log_probs': log_probs, 'logits':logits}


    # === 追加到 ODesign_Model 类内 ===
    def encode_and_pad(self, batch):
        """
        只做 encoder 与分链+PAD，供 AR 推断使用。
        返回:
        H_pad: [B, L, d]
        valid_mask: [B, L]  (True=有效，False=PAD)
        lengths: [B]
        aux: dict  # 保存一些还原用的信息
        """
        # === 复制 forward 里的 encoder 前半部分 ===
        X, h_V, h_E = batch['X'], batch['_V'], batch['_E']
        edge_idx, batch_id = batch['edge_idx'], batch['batch_id']
        edge_idx_g, batch_id_g = batch['edge_idx_g'], batch['batch_id_g']

        T  = Rigid(Rotation(batch['T_rot']),  batch['T_trans'])
        T_g= Rigid(Rotation(batch['T_g_rot']),batch['T_g_trans'])
        T_ts=Rigid(Rotation(batch['T_ts_rot']),batch['T_ts_trans'])
        T_gs=Rigid(Rotation(batch['T_gs_rot']),batch['T_gs_trans'])
        T_gs.rbf = batch['rbf_gs']
        T_ts.rbf = batch['rbf_ts']

        h_E_0 = h_E
        h_V = self.node_embedding(h_V)
        h_E = self.edge_embedding(h_E)
        h_V_g = self.virtual_embedding(batch['_V_g'])
        h_E_g = torch.zeros((edge_idx_g.shape[1], h_E.shape[1]),
                            device=h_V.device, dtype=h_V.dtype)
        h_E_0 = torch.cat([
            h_E_0,
            torch.zeros((edge_idx_g.shape[1], h_E_0.shape[1]),
                        device=h_V.device, dtype=h_V.dtype)
        ])

        H_enc = self.encoder(None, T, T_g,
                            h_V, h_V_g, h_E, h_E_g,
                            T_ts, T_gs,
                            edge_idx, edge_idx_g,
                            batch_id, batch_id_g, h_E_0)          # [N, d]

        # === 分链 + PAD ===
        device = H_enc.device
        chain_enc = batch['chain_encoding'].to(device)
        seg_H = self._segment_by_chain(H_enc, batch_id, chain_enc)   # list of [Li, d]

        B = len(seg_H)
        L_max = max(s.shape[0] for s in seg_H)
        H_pad = pad_sequence(seg_H, batch_first=True, padding_value=0.0)  # [B, L, d]
        lengths = torch.tensor([t.size(0) for t in seg_H], device=device) # [B]
        arangeL = torch.arange(L_max, device=device).unsqueeze(0).expand(B, -1)
        valid_mask = arangeL < lengths.unsqueeze(1)                       # [B, L] bool

        aux = {
            'batch_id': batch_id,                    # 若你需要还原拼接顺序
            'chain_encoding': chain_enc,
            'tokenizer': self.tokenizer,
        }
        return H_pad, valid_mask, lengths, aux

    def _get_features(self, batch):
        return batch