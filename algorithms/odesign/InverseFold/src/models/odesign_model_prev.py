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
        self.decoder = MLPDecoder(hidden_dim,vocab=vocab_size)
        
        # self.decoder = SimpleARDecoder(hidden_dim, vocab=vocab_size,
        #                                 n_layers=4, n_heads=8,
        #                                 dropout=dropout, use_struct=True, tie_weights=True)
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
        
        log_probs, logits = self.decoder(h_V)
        print('---------------------------------------------------')
        
        return {'log_probs': log_probs, 'logits':logits}


    def _get_features(self, batch):
        return batch