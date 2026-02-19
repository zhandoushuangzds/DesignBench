# -*- coding: utf-8 -*-
import os
import json
from typing import List, Tuple, Optional, Union, Dict, Any, Iterable

import torch
import numpy as np
from torcheval.metrics.text import Perplexity
from torcheval.metrics.classification import MulticlassF1Score


# ---------------------------
# 类型与分桶（分子类型 → 不同长度桶）
# ---------------------------

TYPE_BUCKET_SPECS: Dict[str, List[Tuple[int, int, str]]] = {
    "protein": [(0, 100, "len_0_100"),
                (100, 300, "len_100_300"),
                (300, 500, "len_300_500"),
                (-1, 10**9, "len_all")],
    "rna":     [(0, 50,  "len_0_50"),
                (50, 100, "len_50_100"),
                (100, 500,"len_100_500"),
                (-1, 10**9, "len_all")],
    "dna":     [(0, 50,  "len_0_50"),
                (50, 100, "len_50_100"),
                (100, 500,"len_100_500"),
                (-1, 10**9, "len_all")],
}


def normalize_type(t) -> str:
    if t is None:
        return "protein"
    s = str(t).strip().lower()
    if s in ("protein", "prot", "p"): return "protein"
    if s in ("rna",): return "rna"
    if s in ("dna",): return "dna"
    return "protein"


# ---------------------------
# 迭代器（按链切片）——关键变化点
# ---------------------------

def _as_spans_list(spans: Union[None, torch.Tensor, np.ndarray, List[List[int]], List[Tuple[int, int]]]
                   ) -> List[Tuple[int, int]]:
    if spans is None:
        return []
    if isinstance(spans, list) and len(spans) == 1 and isinstance(spans[0], torch.Tensor):
        spans = spans[0]
    if isinstance(spans, torch.Tensor):
        arr = spans.detach().cpu().long().numpy()
    elif isinstance(spans, np.ndarray):
        arr = spans
    else:
        try:
            arr = np.asarray(spans)
        except Exception:
            return []
    if arr.ndim != 2 or arr.shape[1] != 2:
        return []
    out: List[Tuple[int, int]] = []
    for s, e in arr.tolist():
        s = int(s); e = int(e)
        if e > s >= 0:
            out.append((s, e))
    return out


def iter_chain_segments(
    log_probs: torch.Tensor,     # [N,V] 或 [B,N,V]
    target: torch.Tensor,        # [N]   或 [B,N]
    mask: torch.Tensor,          # [N]/[B,N]  float/byte
    batch: Dict[str, Any],
):
    """
    逐“链段”产出 (logits2d[T,V], tgt1d[T], L=T, tname:str)。
    - 优先使用 chain_spans 做按链切分；
    - 每个链段的分子类型优先来自 batch['type']（链级列表），
      若不可用则退回到 type_vec 在该段 mask>0 上的众数；
    - 若无 spans，则退化为样本级切分。
    """
    tv = batch.get('type_vec', None)  # [N] / [B,N]，0/1/2
    bid = batch.get('batch_id', None)
    chain_types_all = batch.get('type', None)  # 可能是 list[str] 或 list[list[str]]

    def _as_spans_list(spans):
        # 支持 None / list / torch.Tensor / np.ndarray
        if spans is None:
            return []
        if isinstance(spans, list) and len(spans) == 1 and isinstance(spans[0], torch.Tensor):
            spans = spans[0]
        if isinstance(spans, torch.Tensor):
            arr = spans.detach().cpu().long().numpy()
        elif isinstance(spans, np.ndarray):
            arr = spans
        else:
            try:
                arr = np.asarray(spans)
            except Exception:
                return []
        if arr.ndim != 2 or arr.shape[1] != 2:
            return []
        out = []
        for s, e in arr.tolist():
            s = int(s); e = int(e)
            if e > s >= 0:
                out.append((s, e))
        return out

    def _majority_type_from_slice(tv_1d: Optional[torch.Tensor], sel_1d: torch.Tensor) -> str:
        if tv_1d is None or (not torch.any(sel_1d)):
            return "protein"
        vals = tv_1d[sel_1d].detach().cpu().tolist()
        if not vals:
            return "protein"
        maj = max((0, 1, 2), key=lambda x: vals.count(x))
        return {0: "protein", 1: "rna", 2: "dna"}[maj]

    # ---------- [B,N,V] ----------
    if log_probs.dim() == 3:
        B, N, V = log_probs.shape
        spans_all = batch.get('chain_spans', None)

        # 支持 list / torch.Tensor / np.ndarray
        spans_per_sample: List[List[Tuple[int,int]]] = [[] for _ in range(B)]
        if spans_all is not None:
            if isinstance(spans_all, list) and len(spans_all) == B:
                for i in range(B):
                    spans_per_sample[i] = _as_spans_list(spans_all[i])
            elif isinstance(spans_all, torch.Tensor) and spans_all.dim() == 3 and spans_all.shape[0] == B and spans_all.shape[2] == 2:
                for i in range(B):
                    spans_per_sample[i] = _as_spans_list(spans_all[i])
            elif isinstance(spans_all, np.ndarray) and spans_all.ndim == 3 and spans_all.shape[0] == B and spans_all.shape[2] == 2:
                for i in range(B):
                    spans_per_sample[i] = _as_spans_list(spans_all[i])

        for i in range(B):
            mi = (mask[i] > 0)
            if not torch.any(mi):
                continue
            spans_i = spans_per_sample[i]
            # 尝试取链级类型列表：chain_types[i][j]
            if isinstance(chain_types_all, list) and len(chain_types_all) == B and isinstance(chain_types_all[i], (list, tuple)):
                chain_types_i = list(chain_types_all[i])
            else:
                chain_types_i = None

            if spans_i:
                for j, (s, e) in enumerate(spans_i):
                    sel = mi[s:e]
                    if not torch.any(sel):
                        continue
                    logits2d = log_probs[i, s:e][sel]
                    tgt1d    = target[i, s:e][sel]
                    # 类型优先来自链级 type
                    if chain_types_i is not None and j < len(chain_types_i):
                        tname = normalize_type(chain_types_i[j])
                    else:
                        tv2 = tv[i] if tv is not None and tv.dim() == 2 else None
                        tname = _majority_type_from_slice(tv2, sel)
                    yield (logits2d, tgt1d, int(sel.sum().item()), tname)
            else:
                logits2d = log_probs[i][mi]
                tgt1d    = target[i][mi]
                tv2 = tv[i] if tv is not None and tv.dim() == 2 else None
                tname = _majority_type_from_slice(tv2, mi)
                yield (logits2d, tgt1d, int(mi.sum().item()), tname)
        return

    # ---------- [N,V] ----------
    N, V = log_probs.shape
    m_all = (mask > 0)
    spans_all = batch.get('chain_spans', None)

    if isinstance(bid, torch.Tensor) and bid.numel() == N:
        uniq = torch.unique(bid).sort()[0].tolist()

        # 尝试把 spans 按样本分配（也要支持 ndarray）
        spans_list = None
        if isinstance(spans_all, list) and len(spans_all) == len(uniq):
            spans_list = [ _as_spans_list(x) for x in spans_all ]
        elif isinstance(spans_all, torch.Tensor) and spans_all.dim() == 3 and spans_all.shape[0] == len(uniq):
            spans_list = [ _as_spans_list(spans_all[i]) for i in range(len(uniq)) ]
        elif isinstance(spans_all, np.ndarray) and spans_all.ndim == 3 and spans_all.shape[0] == len(uniq):
            spans_list = [ _as_spans_list(spans_all[i]) for i in range(len(uniq)) ]

        for idx, b in enumerate(uniq):
            sel_b = (bid == b) & m_all
            if not torch.any(sel_b):
                continue

            # 链级 type 列表：可能是 list-of-lists
            chain_types_i = None
            if isinstance(chain_types_all, list) and len(chain_types_all) == len(uniq) and isinstance(chain_types_all[idx], (list, tuple)):
                chain_types_i = list(chain_types_all[idx])

            if spans_list and len(spans_list) > idx and spans_list[idx]:
                # 当前样本的全局起点
                idxs = torch.nonzero(bid == b, as_tuple=False).view(-1)
                g0 = idxs[0].item()
                for j, (s, e) in enumerate(spans_list[idx]):
                    rng = torch.arange(g0 + s, g0 + e, device=log_probs.device)
                    sel = (m_all.index_select(0, rng) > 0)
                    if not torch.any(sel):
                        continue
                    logits2d = log_probs.index_select(0, rng)[sel]
                    tgt1d    = target.index_select(0, rng)[sel]
                    if chain_types_i is not None and j < len(chain_types_i):
                        tname = normalize_type(chain_types_i[j])
                    else:
                        tv_seg = (tv.index_select(0, rng) if tv is not None and tv.dim() == 1 else None)
                        tname  = _majority_type_from_slice(tv_seg, sel)
                    yield (logits2d, tgt1d, int(sel.sum().item()), tname)
            else:
                logits2d = log_probs[sel_b]
                tgt1d    = target[sel_b]
                tv_b     = (tv[sel_b] if tv is not None and tv.dim() == 1 else None)
                # 如果没有 spans，就退化为样本级；类型尽量从 batch['type'][idx] 推
                if isinstance(chain_types_all, list) and len(chain_types_all) == len(uniq) and isinstance(chain_types_all[idx], str):
                    tname = normalize_type(chain_types_all[idx])
                else:
                    tname = _majority_type_from_slice(tv_b, torch.ones_like(tgt1d, dtype=torch.bool))
                yield (logits2d, tgt1d, int(sel_b.sum().item()), tname)
        return

    # 单样本 + spans（支持 ndarray）
    if spans_all is not None:
        spans = _as_spans_list(spans_all)
        if spans:
            # 链级 type：可能是 list[str]
            chain_types_i = list(chain_types_all) if isinstance(chain_types_all, (list, tuple)) else None
            for j, (s, e) in enumerate(spans):
                sel = m_all[s:e]
                if not torch.any(sel):
                    continue
                logits2d = log_probs[s:e][sel]
                tgt1d    = target[s:e][sel]
                if chain_types_i is not None and j < len(chain_types_i):
                    tname = normalize_type(chain_types_i[j])
                else:
                    tv_seg = (tv[s:e] if tv is not None and tv.dim() == 1 else None)
                    tname  = _majority_type_from_slice(tv_seg, sel)
                yield (logits2d, tgt1d, int(sel.sum().item()), tname)
            return

    # 退化：整样本
    if torch.any(m_all):
        logits2d = log_probs[m_all]
        tgt1d    = target[m_all]
        if isinstance(chain_types_all, str):
            tname = normalize_type(chain_types_all)
        else:
            tname = _majority_type_from_slice(tv if tv is not None and tv.dim() == 1 else None,
                                             torch.ones_like(tgt1d, dtype=torch.bool))
        yield (logits2d, tgt1d, int(m_all.sum().item()), tname)



# ---------------------------
# 评估累计器（overall + 每类型/长度桶）
# ---------------------------

class EvalAccumulator:
    def __init__(self, vocab_size: int, quiet_f1_warning: bool = True):
        self.vocab_size = vocab_size
        # 关掉缺类告警（可选）
        self.overall_ppl = Perplexity()
        self.overall_f1  = MulticlassF1Score(average='macro',
                                             num_classes=vocab_size)

        self.type_bucket_specs = TYPE_BUCKET_SPECS

        self.bucket_ppl: Dict[Tuple[str, str], Perplexity] = {}
        self.bucket_f1m: Dict[Tuple[str, str], MulticlassF1Score] = {}
        self.bucket_rec_num: Dict[Tuple[str, str], int] = {}
        self.bucket_rec_den: Dict[Tuple[str, str], int] = {}

        for mt, specs in self.type_bucket_specs.items():
            for _, _, name in specs:
                key = (mt, name)
                self.bucket_ppl[key] = Perplexity()
                self.bucket_f1m[key] = MulticlassF1Score(average='macro',
                                                         num_classes=vocab_size)
                self.bucket_rec_num[key] = 0
                self.bucket_rec_den[key] = 0

    def update_overall(self, log_probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if log_probs.dim() == 3:
            logits2d = log_probs.reshape(-1, log_probs.size(-1))
            tgt1d = target.reshape(-1)
            m1d = mask.reshape(-1) > 0
        else:
            logits2d = log_probs
            tgt1d = target
            m1d = mask > 0

        if torch.any(m1d):
            logits_eff = logits2d[m1d]
            tgt_eff = tgt1d[m1d]
            self.overall_ppl.update(logits_eff[None, ...].cpu(), tgt_eff[None, ...].cpu())
            self.overall_f1.update(logits_eff.argmax(-1).cpu(), tgt_eff.cpu())

    def update_sample(self, logits2d: torch.Tensor, tgt1d: torch.Tensor, L: int, tname: str):
        specs = self.type_bucket_specs.get(tname, self.type_bucket_specs["protein"])
        pred1d = logits2d.argmax(-1)
        for lo, hi, name in specs:
            if (lo < 0) or (lo <= L < hi):
                key = (tname, name)
                self.bucket_rec_num[key] += int((pred1d == tgt1d).sum().item())
                self.bucket_rec_den[key] += int(tgt1d.numel())
                self.bucket_ppl[key].update(logits2d[None, ...].cpu(), tgt1d[None, ...].cpu())
                self.bucket_f1m[key].update(pred1d.cpu(), tgt1d.cpu())

    def compute(self) -> Dict[str, Any]:
        try:
            ppl_overall = float(self.overall_ppl.compute())
        except Exception:
            ppl_overall = float('nan')
        try:
            f1m_overall = float(self.overall_f1.compute())
        except Exception:
            f1m_overall = float('nan')

        per_type: Dict[str, Dict[str, Any]] = {}
        for tname, specs in self.type_bucket_specs.items():
            tdict = {}
            for _, _, name in specs:
                key = (tname, name)
                num = self.bucket_rec_num.get(key, 0)
                den = max(1, self.bucket_rec_den.get(key, 0))
                rec = num / den
                try:
                    ppl = float(self.bucket_ppl[key].compute())
                except Exception:
                    ppl = float('nan')
                try:
                    f1m = float(self.bucket_f1m[key].compute())
                except Exception:
                    f1m = float('nan')
                tdict[name] = {
                    "recovery": rec,
                    "perplexity": ppl,
                    "f1_macro": f1m,
                    "count_tokens": den
                }
            per_type[tname] = tdict

        return {
            "overall": {"perplexity": ppl_overall, "f1_macro": f1m_overall},
            "per_type": per_type,
        }

    def reset(self):
        self.overall_ppl.reset()
        self.overall_f1.reset()
        for key in list(self.bucket_ppl.keys()):
            self.bucket_ppl[key].reset()
            self.bucket_f1m[key].reset()
            self.bucket_rec_num[key] = 0
            self.bucket_rec_den[key] = 0


# ---------------------------
# FASTA 收集与写盘
# ---------------------------

class FastaWriter:
    """
    将 (true/pred) 序列以 FASTA 形式写出：
      每个样本/链写两条：>...|true 与 >...|pred
    依赖 tokenizer.decode(list[int])。
    """
    def __init__(self, tokenizer, out_dir: str, ex_name: str):
        self.tokenizer = tokenizer
        self.out_dir = out_dir
        self.ex_name = ex_name
        os.makedirs(self.out_dir, exist_ok=True)

        self._entries: List[Tuple[str, str]] = []
        self._sample_counter = 0  # 给无名样本分配 id

    # ---- 公共入口 ----
    def collect(self, batch: Dict[str, Any], log_probs: torch.Tensor):
        S_tgt = batch['S_tgt'] if 'S_tgt' in batch else batch['S']
        mask = batch.get('loss_mask', (batch['chain_mask'].float() > 0).float())
        pred_ids = log_probs.argmax(dim=-1)

        if pred_ids.dim() == 1:
            self._collect_per_chain_single(batch, pred_ids, S_tgt, mask)
            return

        if pred_ids.dim() == 2 and 'batch_id' in batch:
            bid = batch['batch_id']
            if isinstance(bid, torch.Tensor) and bid.numel() == pred_ids.numel():
                self._collect_per_chain_single(batch,
                                               pred_ids.reshape(-1),
                                               S_tgt.reshape(-1),
                                               mask.reshape(-1).float())
                return

        if pred_ids.dim() == 2:
            self._collect_per_chain_batched(batch, pred_ids, S_tgt, mask)

    def flush(self) -> Optional[str]:
        if not self._entries:
            return None
        out_fn = os.path.join(self.out_dir, f"{self.ex_name}.fasta")
        with open(out_fn, "w") as f:
            for header, seq in self._entries:
                f.write(f"{header}\n{seq}\n")
        self._entries.clear()
        return out_fn

    # ---- 内部工具 ----
    def _decode_seq(self, ids_1d: torch.Tensor) -> str:
        return self.tokenizer.decode(ids_1d.tolist())

    def _pick_name(self, batch: Dict[str, Any], b_idx: Optional[int] = None) -> str:
        if 'title' in batch:
            val = batch['title']
            if isinstance(val, (list, tuple)):
                if b_idx is not None and b_idx < len(val):
                    return str(val[b_idx])
            elif isinstance(val, str):
                return val
        for key in ['name', 'pdb', 'pdb_id', 'uid', 'id']:
            if key in batch:
                val = batch[key]
                if isinstance(val, (list, tuple)):
                    if b_idx is not None and b_idx < len(val):
                        return str(val[b_idx])
                elif isinstance(val, str):
                    return val
        name = f"sample_{self._sample_counter}"
        self._sample_counter += 1
        return name

    def _as_spans_list(self, spans: Union[None, torch.Tensor, np.ndarray,
                                          List[List[int]], List[Tuple[int, int]]]
                       ) -> List[Tuple[int, int]]:
        if spans is None:
            return []
        if isinstance(spans, list) and len(spans) == 1 and isinstance(spans[0], torch.Tensor):
            spans = spans[0]
        if isinstance(spans, torch.Tensor):
            arr = spans.detach().cpu().long().numpy()
        elif isinstance(spans, np.ndarray):
            arr = spans
        else:
            try:
                arr = np.asarray(spans)
            except Exception:
                return []
        if arr.ndim != 2 or arr.shape[1] != 2:
            return []
        out = []
        for s, e in arr.tolist():
            s = int(s)
            e = int(e)
            if e > s >= 0:
                out.append((s, e))
        return out

    def _names_for_sample(self, batch: Dict[str, Any],
                          b_idx: Optional[int],
                          chain_names: Optional[List[str]]) -> List[str]:
        if 'title' in batch:
            t = batch['title']
            if isinstance(t, (list, tuple)) and len(t) > 0:
                if isinstance(t[0], (list, tuple)):
                    if b_idx is not None and b_idx < len(t):
                        return [str(x) for x in t[b_idx]]
                else:
                    return [str(x) for x in t]

        pdb_id = None
        for key in ['pdb_id', 'pdb', 'name', 'id', 'uid', 'title']:
            if key in batch:
                val = batch[key]
                if isinstance(val, (list, tuple)):
                    if b_idx is not None and b_idx < len(val):
                        pdb_id = os.path.splitext(str(val[b_idx]))[0]
                        break
                elif isinstance(val, str):
                    pdb_id = os.path.splitext(val)[0]
                    break
        if pdb_id is None:
            pdb_id = "sample"

        if chain_names:
            return [f"{pdb_id}_{cn}" for cn in chain_names]
        return [f"{pdb_id}_chain{i+1}" for i in range(1)]

    def _emit_chain_pair(self, name: str,
                         true_ids_1d: torch.Tensor,
                         pred_ids_1d: torch.Tensor,
                         mask_1d: torch.Tensor):
        sel = (mask_1d > 0)
        if not torch.any(sel):
            return
        L = int(sel.sum().item())
        true_seq = self._decode_seq(true_ids_1d[sel].to('cpu'))
        pred_seq = self._decode_seq(pred_ids_1d[sel].to('cpu'))
        self._entries.append((f">{name}|len={L}|true", true_seq))
        self._entries.append((f">{name}|len={L}|pred", pred_seq))

    def _collect_per_chain_single(self, batch: Dict[str, Any],
                                  pred_ids: torch.Tensor,
                                  true_ids: torch.Tensor,
                                  mask: torch.Tensor):
        spans_raw = batch.get('chain_spans', None)

        # 多样本 span（列表），同时存在 batch_id：扁平映射
        if isinstance(spans_raw, list) and len(spans_raw) > 1 and 'batch_id' in batch:
            batch_id = batch['batch_id'].to(pred_ids.device)  # [N] 0..B-1
            B = int(batch_id.max().item()) + 1

            titles_all = batch.get('title', None)
            chain_names_all = batch.get('chain_names', None)
            pdb_all = batch.get('pdb_id', None)

            for i in range(B):
                idx_i = (batch_id == i).nonzero(as_tuple=False).view(-1)
                if idx_i.numel() == 0:
                    continue
                g0 = idx_i[0].item()

                spans_i = spans_raw[i]
                spans_i_list = self._as_spans_list(spans_i)
                if not spans_i_list:
                    sel = (mask[idx_i] > 0)
                    if torch.any(sel):
                        L = int(sel.sum().item())
                        true_seq = self._decode_seq(true_ids[idx_i][sel].to('cpu'))
                        pred_seq = self._decode_seq(pred_ids[idx_i][sel].to('cpu'))
                        name = self._pick_name(batch, b_idx=i)
                        self._entries.append((f">{name}|len={L}|true", true_seq))
                        self._entries.append((f">{name}|len={L}|pred", pred_seq))
                    continue

                titles_i = None
                if isinstance(titles_all, list) and len(titles_all) == B:
                    titles_i = titles_all[i]
                elif not isinstance(titles_all, list):
                    titles_i = titles_all

                chain_names_i = None
                if isinstance(chain_names_all, list) and len(chain_names_all) == B:
                    chain_names_i = chain_names_all[i]
                elif not isinstance(chain_names_all, list):
                    chain_names_i = chain_names_all

                pdb_i = None
                if isinstance(pdb_all, list) and len(pdb_all) == B:
                    pdb_i = pdb_all[i]
                else:
                    pdb_i = pdb_all

                names_i = self._names_for_sample({'title': titles_i, 'pdb_id': pdb_i},
                                                 b_idx=None,
                                                 chain_names=(chain_names_i if isinstance(chain_names_i, (list, tuple)) else None))
                K = min(len(spans_i_list), len(names_i))
                for j in range(K):
                    s, e = spans_i_list[j]
                    self._emit_chain_pair(names_i[j],
                                          true_ids[g0+s:g0+e], pred_ids[g0+s:g0+e], mask[g0+s:g0+e].float())
                for j in range(K, len(spans_i_list)):
                    s, e = spans_i_list[j]
                    fallback = f"{self._pick_name(batch, b_idx=i)}_chain{j+1}"
                    self._emit_chain_pair(fallback,
                                          true_ids[g0+s:g0+e], pred_ids[g0+s:g0+e], mask[g0+s:g0+e].float())
            return

        # 单样本 span
        if isinstance(spans_raw, list) and len(spans_raw) == 1 and isinstance(spans_raw[0], torch.Tensor):
            spans_raw = spans_raw[0]
        spans = self._as_spans_list(spans_raw)

        chain_names = None
        if 'chain_names' in batch:
            cn = batch['chain_names']
            if isinstance(cn, list) and len(cn) == 1 and isinstance(cn[0], (list, tuple)):
                cn = cn[0]
            if isinstance(cn, (list, tuple)):
                chain_names = [str(x) for x in cn]

        titles = batch.get('title', None)
        if isinstance(titles, list) and len(titles) == 1:
            titles = titles[0]
        pdb_id = batch.get('pdb_id', None)
        if isinstance(pdb_id, list) and len(pdb_id) == 1:
            pdb_id = pdb_id[0]

        if spans:
            names = self._names_for_sample({'title': titles, 'pdb_id': pdb_id}, b_idx=None, chain_names=chain_names)
            K = min(len(spans), len(names))
            for j in range(K):
                s, e = spans[j]
                self._emit_chain_pair(names[j], true_ids[s:e], pred_ids[s:e], mask[s:e].float())
            for j in range(K, len(spans)):
                s, e = spans[j]
                fallback = f"{self._pick_name(batch)}_chain{j+1}"
                self._emit_chain_pair(fallback, true_ids[s:e], pred_ids[s:e], mask[s:e].float())
        else:
            sel = (mask > 0)
            if torch.any(sel):
                L = int(sel.sum().item())
                true_seq = self._decode_seq(true_ids[sel].to('cpu'))
                pred_seq = self._decode_seq(pred_ids[sel].to('cpu'))
                name = self._pick_name(batch)
                self._entries.append((f">{name}|len={L}|true", true_seq))
                self._entries.append((f">{name}|len={L}|pred", pred_seq))

    def _collect_per_chain_batched(self, batch: Dict[str, Any],
                                   pred_ids: torch.Tensor,
                                   true_ids: torch.Tensor,
                                   mask: torch.Tensor):
        B = pred_ids.shape[0]
        spans_all = batch.get('chain_spans', None)
        spans_list_per_sample: List[List[Tuple[int, int]]] = [[] for _ in range(B)]
        if spans_all is not None:
            if isinstance(spans_all, list) and len(spans_all) == B:
                for i in range(B):
                    spans_list_per_sample[i] = self._as_spans_list(spans_all[i])
            elif isinstance(spans_all, torch.Tensor) and spans_all.dim() == 3 and spans_all.shape[0] == B and spans_all.shape[2] == 2:
                for i in range(B):
                    spans_list_per_sample[i] = self._as_spans_list(spans_all[i])
            else:
                spans_list_per_sample = [[] for _ in range(B)]

        chain_names_all = batch.get('chain_names', None)
        titles = batch.get('title', None)

        for i in range(B):
            pi = pred_ids[i]
            ti = true_ids[i]
            mi = mask[i].float()
            spans_i = spans_list_per_sample[i]

            chain_names_i = None
            if isinstance(chain_names_all, list) and len(chain_names_all) == B and isinstance(chain_names_all[i], (list, tuple)):
                chain_names_i = [str(x) for x in chain_names_all[i]]

            if spans_i:
                names_i = self._names_for_sample(
                    {'title': titles[i] if isinstance(titles, list) and len(titles) == B else titles,
                     'pdb_id': (batch.get('pdb_id', None)[i] if isinstance(batch.get('pdb_id', None), list)
                                and len(batch.get('pdb_id', [])) == B else batch.get('pdb_id', None))},
                    b_idx=None,
                    chain_names=chain_names_i
                )
                K = min(len(spans_i), len(names_i))
                for j in range(K):
                    s, e = spans_i[j]
                    self._emit_chain_pair(names_i[j], ti[s:e], pi[s:e], mi[s:e])
                for j in range(K, len(spans_i)):
                    s, e = spans_i[j]
                    fallback = f"{self._pick_name(batch, b_idx=i)}_chain{j+1}"
                    self._emit_chain_pair(fallback, ti[s:e], pi[s:e], mi[s:e])
            else:
                sel = (mi > 0)
                if torch.any(sel):
                    L = int(sel.sum().item())
                    true_seq = self._decode_seq(ti[sel].to('cpu'))
                    pred_seq = self._decode_seq(pi[sel].to('cpu'))
                    name = self._pick_name(batch, b_idx=i)
                    self._entries.append((f">{name}|len={L}|true", true_seq))
                    self._entries.append((f">{name}|len={L}|pred", pred_seq))

if __name__ == '__main__':
    pass