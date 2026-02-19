import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  
from src.datasets.featurizer import UniTokenizer, Featurize
import os.path as osp
import gzip
import numpy as np
import torch
from typing import Optional, Dict, Any, List, Tuple, Set
from .design_interface import MInterface
from sklearn.metrics import f1_score
import math
from .inference_utils import (   
    _parse_structure_path,          
    build_cif_id_maps,              
    extract_chain_seq_and_backbone, 
)
from collections import Counter
import re

def id_to_char_global(i: int, kind: str, tok) -> str:

    if 0 <= i < 20:
        return tok.alphabet_protein[i]
    if 20 <= i < 25:
        base = tok.nuc5[i-20]  # A,C,G,T,U
        if kind == 'dna' and base == 'U': base = 'T'
        if kind == 'rna' and base == 'T': base = 'U'
        return base
    return 'X'

def diagnose_vocab_distribution(probs_cpu: torch.Tensor, kind: str, tok) -> None:
    """
    probs_cpu: [L, V] on CPU after softmax
    kind: 'dna' | 'rna' | 'protein'
    """
    L, V = probs_cpu.shape
    if kind == 'dna':
        allow = torch.tensor(tok.dna_ids)   # [20,21,22,23]
    elif kind == 'rna':
        allow = torch.tensor(tok.rna_ids)   # [20,21,22,24]
    else:
        allow = torch.arange(0, min(20, V)) 

    allow = allow[allow < V]
    if allow.numel() == 0:
        allow = torch.arange(0, V)

    mass_allowed = probs_cpu[:, allow].sum(dim=1)           # [L]
    mass_protein = probs_cpu[:, :min(20, V)].sum(dim=1)     # [L]
    top1 = probs_cpu.max(dim=-1).values                     # [L]
    ent = -(probs_cpu.clamp_min(1e-12).log() * probs_cpu).sum(dim=-1) / math.log(2.0)  # bits

    argmax_ids = probs_cpu.argmax(dim=-1)                   # [L]
    outside_frac = float((~torch.isin(argmax_ids, allow)).float().mean().item())

    cnt = Counter(int(i) for i in argmax_ids.tolist())
    allow_hist = {id_to_char_global(int(i), kind, tok): cnt.get(int(i), 0) for i in allow.tolist()}

    mass_per_token = probs_cpu.sum(dim=0) / L
    mass_dict = {id_to_char_global(i, kind, tok): float(mass_per_token[i].item()) for i in range(min(V, 25))}

    print("\n[Diag/Vocab]")
    print(f"- Allowed mass (mean/min): {float(mass_allowed.mean()):.4f} / {float(mass_allowed.min()):.4f}")
    print(f"- Protein mass leak (mean): {float(mass_protein.mean()):.4f}")
    print(f"- Argmax outside allowed: {outside_frac*100:.2f}%")
    print(f"- Top1 prob mean±std: {float(top1.mean()):.4f} ± {float(top1.std()):.4f}")
    print(f"- Entropy bits mean±std: {float(ent.mean()):.3f} ± {float(ent.std()):.3f}")
    print(f"- Argmax histogram in allowed: {allow_hist}")


def gzip_open(filename, *args, **kwargs):
    if args and "t" in args[0]:
        args = (args[0].replace("t", ""), ) + args[1:]
    if isinstance(filename, str):
        return gzip.open(filename, *args, **kwargs)
    else:
        return gzip.GzipFile(filename, *args, **kwargs)


def save_fasta(seqs, names, pred_fasta_path='pred.fasta'):
    with open(pred_fasta_path, 'w') as f:
        for seq, name in zip(seqs, names):
            f.write(f">{name}\n{seq}\n")

def reload_model(data_name, model_name, model_dir, device: str | None = None):
    
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    dev = torch.device(device)
    
    config = {}
    base = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base)
    # config['pretrained_path'] = osp.join(base, 'ckpts', f'{data_name}.ckpt')
    dn = str(data_name).strip().lower()
    if dn == 'ligand':
        ckpt = 'ligand.ckpt'
        dataset = 'Ligand'
    elif dn == 'dna':
        ckpt = 'dna.ckpt'
        dataset = 'DNA'
    elif dn == 'rna':
        ckpt = 'rna.ckpt'
        dataset = 'RNA'
    else:
        ckpt = 'mixed.ckpt'
        dataset = 'Mixed'
    

    # user defined
    config['pretrained_path'] = osp.join(model_dir, ckpt)
    # config['pretrained_path'] = osp.join(base_dir, "ckpts", ckpt)
    config['dataset'] = dataset
    config['load_memory'] = False
    config['is_colab'] = True
    config['ex_name'] = f'{model_name}'
    config['model_name'] = model_name
    config['res_dir'] = base
    config['config_path'] = osp.join(base_dir, "conf", f'{model_name}.yaml')
    # config['config_path'] = '/inspire/hdd/project/biomacromolecule/zhuangkai-25220209/DNA/ProteinInvBench-lightning_modified/src/models/configs/ODesign.yaml'
    # model = MInterface.load_from_checkpoint(checkpoint_path=config['pretrained_path'], config_path=config['config_path'])
    
    # model.eval()
    model = MInterface.load_from_checkpoint(
        checkpoint_path=config['pretrained_path'],
        config_path=config['config_path'],
        map_location=dev,
    ).to(dev).eval()

    # 4) 打印/校验
    try:
        any_param = next(model.parameters())
        print(f"[reload_model] device={dev} first_param_on={any_param.device}")
    except StopIteration:
        print(f"[reload_model] device={dev} (no params?)")
    return model, dev

# ---------------- mmCIF ----------------

# -*- coding: utf-8 -*-
# 追加到 tools.py 末尾或合适位置


# 仅当未导入 Bio.PDB 时再导入，避免你项目里重复导入报错
try:
    from Bio.PDB import StructureBuilder, Structure, Model, Chain, Residue, Atom
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.mmcifio import MMCIFIO
except Exception as _e:
    MMCIFParser = None
    MMCIFIO = None

# ------------------ 基础映射 ------------------

_AA1_TO_AA3 = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}

def _na1_to_compid(base: str, kind: str) -> str:
    """
    kind: 'rna' | 'dna'
    RNA:  A,U,G,C
    DNA:  DA,DT,DG,DC
    其他字符（含N）保持原残基不变（调用方会跳过）
    """
    b = base.upper()
    if kind == 'rna':
        if b in ('A','U','G','C'):
            return b
        return None
    else:
        mapping = {'A':'DA', 'T':'DT', 'G':'DG', 'C':'DC'}
        return mapping.get(b, None)

# ------------------ 结构裁剪与写出 ------------------

def _ensure_biopython():
    if MMCIFParser is None or MMCIFIO is None:
        raise RuntimeError("Biopython (Bio.PDB) not available. Please install biopython.")

def _load_structure_from_cif(cif_path: str):
    _ensure_biopython()
    parser = MMCIFParser(QUIET=True)
    struct_id = os.path.basename(cif_path)
    return parser.get_structure(struct_id, cif_path)

def _copy_single_chain(structure, chain_id: str):
    """
    从 structure（可能包含多模型/多链）复制出仅包含 model0 + 指定 auth chain 的新 Structure。
    """
    from Bio.PDB import Structure
    new_s = Structure.Structure("subset")
    model0 = list(structure.get_models())[0]

    from Bio.PDB import Model, Chain, Residue, Atom
    new_m = Model.Model(0)

    # 按 auth_id 选链
    for ch in model0:
        if ch.id == chain_id:
            new_ch = Chain.Chain(chain_id)
            for res in ch:
                # 仅拷贝标准残基与其原子；水与异源保留与否可按需调整
                new_res = Residue.Residue(res.id, res.resname, res.segid)
                for atom in res:
                    new_atom = Atom.Atom(
                        atom.get_name(), atom.get_coord(), atom.get_bfactor(),
                        atom.get_occupancy(), atom.get_altloc(), atom.get_fullname(),
                        atom.get_serial_number(), element=atom.element
                    )
                    new_res.add(new_atom)
                new_ch.add(new_res)
            new_m.add(new_ch)
            break
    new_s.add(new_m)
    return new_s


# ------------------ 残基名替换（按新序列） ------------------

def _mutate_chain_sequence_inplace(structure, chain_id: str, new_seq: str, kind: str) -> Tuple[int,int]:
    """
    在原 structure 中，就地把指定链的残基名重写为 new_seq 对应的 comp_id。
    kind: 'protein' | 'rna' | 'dna'
    返回 (applied, total可用位点)
    - 仅对“有主链原子”的标准多肽/多核苷酸位点进行替换
    - 长度不匹配则按最短对齐
    """
    new_seq = (new_seq or "").upper()
    if not new_seq:
        return (0, 0)

    model0 = list(structure.get_models())[0]
    chain = None
    for ch in model0:
        if ch.id == chain_id:
            chain = ch
            break
    if chain is None:
        return (0, 0)

    # 收集可替换的“主链”位置
    candidates = []
    for res in chain:
        hetflag, resseq, icode = res.id
        # 跳过水和异源（H_ 或 WAT/HOH）
        if hetflag.strip() != "":
            continue
        candidates.append(res)

    L = min(len(candidates), len(new_seq))
    applied = 0
    for i in range(L):
        res = candidates[i]
        target = new_seq[i]
        if kind == 'protein':
            aa3 = _AA1_TO_AA3.get(target, None)
            if aa3:
                res.resname = aa3
                applied += 1
        else:
            comp = _na1_to_compid(target, kind)
            if comp:
                res.resname = comp
                applied += 1
    return (applied, L)

# tools.py —— 替换这两个函数

def save_single_chain_cif(cif_src_path: str, chain_auth_id: str, out_path: str) -> None:
    """
    只导出指定 auth 链（坐标与残基名不变），写成单链 mmCIF。
    """
    _ensure_biopython()
    struct = _load_structure_from_cif(cif_src_path)
    sub_s = _copy_single_chain(struct, chain_auth_id)  # <<— 先裁剪成单链
    io = MMCIFIO()
    io.set_structure(sub_s)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    io.save(out_path)

def save_sequence_replaced_cif(cif_src_path: str,
                               chain_auth_id: str,
                               new_seq: str,
                               kind: str,
                               out_path: str) -> Tuple[int,int]:
    """
    读取源 CIF → **裁剪为单链结构** → 在该单链里把残基名重写为 new_seq（不改坐标）→ 保存。
    返回 (applied, aligned_len)
    """
    _ensure_biopython()
    # 1) 先裁剪：只保留目标链
    full_struct = _load_structure_from_cif(cif_src_path)
    sub_struct = _copy_single_chain(full_struct, chain_auth_id)

    # 2) 在“单链结构”上就地替换残基名
    applied, aligned = _mutate_chain_sequence_inplace(
        sub_struct, chain_auth_id, new_seq, kind
    )

    # 3) 写出单链 mutated CIF
    io = MMCIFIO()
    io.set_structure(sub_struct)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    io.save(out_path)
    return (applied, aligned)


# ------------------ 从 sample.title 中提取 auth 链（你 runner 里已有 title 习惯） ------------------

_CHAIN_AUTH_RE = re.compile(r"_auth([A-Za-z0-9]+)")

def parse_auth_from_title(title: str, fallback: Optional[str] = None) -> Optional[str]:
    """
    从诸如 '6H67_rna_authA_label1' 中提取 'A'。
    """
    m = _CHAIN_AUTH_RE.search(title or "")
    if m:
        return m.group(1)
    return fallback

# ---------------- 工具：判断核酸子类 ----------------
def _decide_kind(one_seq: str, prefer: str) -> str:
    """
    根据序列粗判：有 T → dna；有 U → rna；否则返回 prefer
    """
    one = (one_seq or "").upper()
    if 'T' in one: return 'dna'
    if 'U' in one: return 'rna'
    return prefer

# ---------------- 主函数：CIF -> samples ----------------
# def parse_cif_to_samples(cif_path: str,
#                          data_name: str,
#                          only_chain: Optional[str] = None) -> List[Dict[str, Any]]:
#     """
#     data_name: 'protein' | 'rna' | 'dna'
#     only_chain: 可为 auth/label 任一形式；不传则解析该类型的所有链

#     返回的 sample 结构：
#     - protein: 4 原子 'N','CA','C','O'
#     - rna/dna: 6 原子 'P','O5','C5','C4','C3','O3'   （注意：键名无撇号）
#     均为 np.ndarray(L,3)，不再外包 [ ... ]，便于直接丢进 featurizer
#     """
#     from pathlib import Path
#     prefer = data_name.strip().lower()
#     assert prefer in ('protein', 'rna', 'dna')

#     # 读结构 + 建 auth/label 映射
#     structure, fmt, cif_text = _parse_structure_path(Path(cif_path))
#     if fmt == "cif":
#         l2a, a2l, _, _, _ = build_cif_id_maps(cif_text)
#     else:
#         l2a, a2l = {}, {}

#     # 仅解析指定链（auth/label 都支持；也自动扩展到映射过来的对名）
#     only_set: Optional[Set[str]] = None
#     if only_chain:
#         u = only_chain.strip().upper()
#         cand = {u}
#         if u in l2a: cand.add(str(l2a[u]).upper())
#         if u in a2l: cand.add(str(a2l[u]).upper())
#         only_set = cand

#     model = next(structure.get_models())

#     base = os.path.basename(cif_path)
#     base = os.path.splitext(base)[0]
#     if base.endswith('.cif'):
#         base = os.path.splitext(base)[0]

#     samples: List[Dict[str, Any]] = []


#     def to_np3(name: str, coords: Dict[str, Any], L: int) -> np.ndarray:
#         """coords[name] 可能是 list[None|[x,y,z]]；转 (L,3)，缺失填 NaN"""
#         arr = (coords or {}).get(name, [])
#         out = []
#         for v in arr:
#             if v is None:
#                 out.append([np.nan, np.nan, np.nan])
#             else:
#                 out.append([float(v[0]), float(v[1]), float(v[2])])
#         a = np.asarray(out, dtype=np.float32)
#         if a.shape != (L, 3):
#             a = np.full((L, 3), np.nan, dtype=np.float32)
#         return a

#     for ch in model:
#         auth_id = ch.id
#         label_id = a2l.get(auth_id, "")
#         if only_set and (auth_id.upper() not in only_set) and (label_id.upper() not in only_set):
#             continue

#         # 提取序列与骨架
#         seq1, coords, is_na = extract_chain_seq_and_backbone(
#             ch, ndigits=3, validate=True, drop_all_none_keys=False
#         )


#         if prefer == 'protein':
#             if is_na:
#                 continue
#             # 仅 20AA
#             seq_clean = re.sub(r'[^A-Za-z]', '', (seq1 or '')).upper()
#             if not seq_clean:
#                 continue
#             # 过滤非法残基（可按需改成替换为 'X'）
#             if set(seq_clean) - set('ACDEFGHIKLMNPQRSTVWY'):
#                 continue
#             L = len(seq_clean)

#             N  = to_np3('N',  coords, L)
#             CA = to_np3('CA', coords, L)
#             C  = to_np3('C',  coords, L)
#             O  = to_np3('O',  coords, L)

#             title = f"{base}_auth{auth_id}" + (f"_label{label_id}" if label_id else "")
#             samples.append({
#                 "title": title,
#                 "type":  "protein",
#                 "seq":   seq_clean,                           # 直接 str；你的 featurizer 已兼容
#                 "N":     N,
#                 "CA":    CA,
#                 "C":     C,
#                 "O":     O,
#                 "chain_mask":     np.ones(L, dtype=np.float32),
#                 "chain_encoding": np.ones(L, dtype=np.float32),
#             })

#         else:
#             # 核酸分支（rna/dna）
#             if not is_na:
#                 continue

#             # 某些链（如 DNA/RNA 混合修饰）无法靠字符唯一判别 → 用 prefer 兜底
#             kind = _decide_kind(seq1, prefer)   # 'rna' 或 'dna'
#             if kind != prefer:
#                 continue

#             if kind == 'dna':
#                 seq_clean = (seq1 or '').upper().replace('U', 'T')
#                 seq_clean = ''.join(ch if ch in 'ATGCN' else 'N' for ch in seq_clean)
#                 # 6 原子（无撇号键名）：P,O5,C5,C4,C3,O3 ← CIF 中键带撇号
#                 # 注意：此处从 coords 读取时要用带撇号的名字
#                 key_map = {
#                     'P':  'P',
#                     'O5': "O5'",
#                     'C5': "C5'",
#                     'C4': "C4'",
#                     'C3': "C3'",
#                     'O3': "O3'",
#                 }
#             else:  # 'rna'
#                 seq_clean = (seq1 or '').upper().replace('T', 'U')
#                 seq_clean = ''.join(ch if ch in 'AUGCN' else 'N' for ch in seq_clean)
#                 # 同样 6 原子（无撇号键名）
#                 key_map = {
#                     'P':  'P',
#                     'O5': "O5'",
#                     'C5': "C5'",
#                     'C4': "C4'",
#                     'C3': "C3'",
#                     'O3': "O3'",
#                 }

#             if not seq_clean:
#                 continue
#             L = len(seq_clean)

#             # 取坐标（存成无撇号键名）
#             P  = to_np3(key_map['P'],  coords, L)
#             O5 = to_np3(key_map['O5'], coords, L)
#             C5 = to_np3(key_map['C5'], coords, L)
#             C4 = to_np3(key_map['C4'], coords, L)
#             C3 = to_np3(key_map['C3'], coords, L)
#             O3 = to_np3(key_map['O3'], coords, L)

#             title = f"{base}_{kind}_auth{auth_id}" + (f"_label{label_id}" if label_id else "")
#             samples.append({
#                 "title": title,
#                 "type":  kind,                 # 'rna' 或 'dna'
#                 "seq":   seq_clean,            # 直接 str
#                 "P":     P,
#                 "O5":    O5,
#                 "C5":    C5,
#                 "C4":    C4,
#                 "C3":    C3,
#                 "O3":    O3,
#                 "chain_mask":     np.ones(L, dtype=np.float32),
#                 "chain_encoding": np.ones(L, dtype=np.float32),
#             })

#     return samples

def parse_cif_to_samples(cif_path: str,
                         data_name: str,
                         only_chain: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    data_name: 'protein' | 'rna' | 'dna'
    only_chain: 可为 auth/label 任一形式；不传则解析该类型的所有链

    返回的 sample 结构：
    - protein: 4 原子 'N','CA','C','O'
    - rna/dna: 6 原子 'P','O5','C5','C4','C3','O3'   （注意：键名无撇号）
    均为 np.ndarray(L,3)，不再外包 [ ... ]，便于直接丢进 featurizer
    """
    from pathlib import Path
    prefer = data_name.strip().lower()
    assert prefer in ('protein', 'rna', 'dna')

    # 读结构 + 建 auth/label 映射
    structure, fmt, cif_text = _parse_structure_path(Path(cif_path))
    if fmt == "cif":
        l2a, a2l, _, _, _ = build_cif_id_maps(cif_text)
    else:
        l2a, a2l = {}, {}

    # 仅解析指定链（auth/label 都支持；也自动扩展到映射过来的对名）
    only_set: Optional[Set[str]] = None
    if only_chain:
        u = only_chain.strip().upper()
        cand = {u}
        if u in l2a: cand.add(str(l2a[u]).upper())
        if u in a2l: cand.add(str(a2l[u]).upper())
        only_set = cand

    model = next(structure.get_models())

    base = os.path.basename(cif_path)
    base = os.path.splitext(base)[0]
    if base.endswith('.cif'):
        base = os.path.splitext(base)[0]

    samples: List[Dict[str, Any]] = []

    def to_np3(name: str, coords: Dict[str, Any], L: int) -> np.ndarray:
        """coords[name] 可能是 list[None|[x,y,z]]；转 (L,3)，缺失填 NaN"""
        arr = (coords or {}).get(name, [])
        out = []
        for v in arr:
            if v is None:
                out.append([np.nan, np.nan, np.nan])
            else:
                out.append([float(v[0]), float(v[1]), float(v[2])])
        a = np.asarray(out, dtype=np.float32)
        if a.shape != (L, 3):
            a = np.full((L, 3), np.nan, dtype=np.float32)
        return a

    for ch in model:
        auth_id = ch.id
        label_id = a2l.get(auth_id, "")
        if only_set and (auth_id.upper() not in only_set) and (label_id.upper() not in only_set):
            continue

        # 提取序列与骨架
        seq1, coords, is_na = extract_chain_seq_and_backbone(
            ch, ndigits=3, validate=True, drop_all_none_keys=False
        )

        if prefer == 'protein':
            if is_na:
                continue
            # 仅 20AA
            seq_clean = re.sub(r'[^A-Za-z]', '', (seq1 or '')).upper()
            if not seq_clean:
                continue
            # 过滤非法残基（可按需改成替换为 'X'）
            if set(seq_clean) - set('ACDEFGHIKLMNPQRSTVWY'):
                continue
            L = len(seq_clean)

            N  = to_np3('N',  coords, L)
            CA = to_np3('CA', coords, L)
            C  = to_np3('C',  coords, L)
            O  = to_np3('O',  coords, L)

            title = f"{base}_auth{auth_id}" + (f"_label{label_id}" if label_id else "")
            samples.append({
                "title": base,
                "type":  "protein",
                "seq":   seq_clean,                           # 直接 str；你的 featurizer 已兼容
                "N":     N,
                "CA":    CA,
                "C":     C,
                "O":     O,
                "chain_mask":     np.ones(L, dtype=np.float32),
                "chain_encoding": np.ones(L, dtype=np.float32),
            })

        else:
            # 核酸分支（rna/dna）
            if not is_na:
                continue

            # 某些链（如 DNA/RNA 混合修饰）无法靠字符唯一判别 → 用 prefer 兜底
            kind = _decide_kind(seq1, prefer)   # 'rna' 或 'dna'
            if kind != prefer:
                continue

            if kind == 'dna':
                seq_clean = (seq1 or '').upper().replace('U', 'T')
                seq_clean = ''.join(ch if ch in 'ATGCN' else 'N' for ch in seq_clean)
                # 6 原子（无撇号键名）：P,O5,C5,C4,C3,O3 ← CIF 中键带撇号
                # 注意：此处从 coords 读取时要用带撇号的名字
                key_map = {
                    'P':  'P',
                    'O5': "O5'",
                    'C5': "C5'",
                    'C4': "C4'",
                    'C3': "C3'",
                    'O3': "O3'",
                    'N':  'N',  # 将 N 添加进来
                }
            else:  # 'rna'
                seq_clean = (seq1 or '').upper().replace('T', 'U')
                seq_clean = ''.join(ch if ch in 'AUGCN' else 'N' for ch in seq_clean)
                # 同样 6 原子（无撇号键名）
                key_map = {
                    'P':  'P',
                    'O5': "O5'",
                    'C5': "C5'",
                    'C4': "C4'",
                    'C3': "C3'",
                    'O3': "O3'",
                    'N':  'N',  # 将 N 添加进来
                }

            if not seq_clean:
                continue
            L = len(seq_clean)

            # 取坐标（存成无撇号键名）
            P  = to_np3(key_map['P'],  coords, L)
            O5 = to_np3(key_map['O5'], coords, L)
            C5 = to_np3(key_map['C5'], coords, L)
            C4 = to_np3(key_map['C4'], coords, L)
            C3 = to_np3(key_map['C3'], coords, L)
            O3 = to_np3(key_map['O3'], coords, L)

            # 处理 N 原子的坐标
            N = to_np3(key_map['N'], coords, L)

            title = f"{base}_{kind}_auth{auth_id}" + (f"_label{label_id}" if label_id else "")
            samples.append({
                "title": base,
                "type":  kind,                 # 'rna' 或 'dna'
                "seq":   seq_clean,            # 直接 str
                "P":     P,
                "O5":    O5,
                "C5":    C5,
                "C4":    C4,
                "C3":    C3,
                "O3":    O3,
                "N":     N,  # 添加 N 坐标
                "chain_mask":     np.ones(L, dtype=np.float32),
                "chain_encoding": np.ones(L, dtype=np.float32),
            })

    return samples


# ---------------- Inference & Metrics ----------------
def allowed_indices(kind: str, V: int) -> torch.Tensor:
    kind = str(kind).lower()
    if kind == 'dna':
        cand = torch.tensor([20, 21, 22, 23], dtype=torch.long)  # A,C,G,T
    elif kind == 'rna':
        cand = torch.tensor([20, 21, 22, 24], dtype=torch.long)  # A,C,G,U
    elif kind == 'protein':
        cand = torch.arange(0, min(20, V), dtype=torch.long)
    else:
        cand = torch.arange(0, V, dtype=torch.long)
    return cand[cand < V]

def beam_search_logp(logp_mat, allow_idx, beam_w, len_norm=True, alpha=0.6,
                     diversity_groups=1, gamma=0.0):
    L_, V_ = logp_mat.shape
    K_ = allow_idx.numel()
    G = max(1, int(diversity_groups))
    group_size = (beam_w + G - 1) // G
    groups = [[([], 0.0)] for _ in range(G)]
    for t in range(L_):
        chosen_per_group = []
        for g in range(G):
            beams = groups[g]
            row = logp_mat[t, allow_idx].clone()  # [K_]
            if gamma > 0 and len(chosen_per_group) > 0:
                used = torch.zeros(K_)
                for prev in chosen_per_group:
                    used[prev] += 1
                row = row - gamma * used
            cand = []
            for seq, s in beams:
                for k in range(K_):
                    cand.append((seq + [int(allow_idx[k])], s + float(row[k])))
            cand.sort(key=lambda x: x[1], reverse=True)
            groups[g] = cand[:group_size]
            top_tokens = []
            for ids, _ in groups[g]:
                pos = (allow_idx == ids[-1]).nonzero(as_tuple=False)
                top_tokens.append(int(pos[0, 0]) if pos.numel() > 0 else 0)
            chosen_per_group.append(top_tokens)
    beams_all = [item for grp in groups for item in grp]
    if len_norm:
        beams_all = [(ids, s / max(1, len(ids))) for ids, s in beams_all]
    elif alpha > 0:
        def lp_len(m): return ((5 + m) ** alpha) / ((5 + 1) ** alpha)
        beams_all = [(ids, s / lp_len(len(ids))) for ids, s in beams_all]
    beams_all.sort(key=lambda x: x[1], reverse=True)
    return beams_all[:beam_w]

def labels_4class(seq_or_ids, kind: str, tok) -> List[int]:
    kind = str(kind).lower()
    if isinstance(seq_or_ids, (list, tuple)) and len(seq_or_ids) > 0 and isinstance(seq_or_ids[0], int):
        ids = list(seq_or_ids)
    else:
        ids = tok.encode(str(seq_or_ids), kind=kind)
    if kind == 'protein':
        return [t if 0 <= t < 20 else 19 for t in ids]
    elif kind in ('dna', 'rna'):
        A, C, G = tok.nuc_to_id['A'], tok.nuc_to_id['C'], tok.nuc_to_id['G']
        if kind == 'dna':
            T = tok.nuc_to_id['T']
            id2cls = {A:0, C:1, G:2, T:3}
        else:
            U = tok.nuc_to_id['U']
            id2cls = {A:0, C:1, G:2, U:3}
        return [id2cls.get(t, 0) for t in ids]
    else:
        raise ValueError(f"Unknown kind: {kind}")

def _move_to(x, device):
    if isinstance(x, dict):
        return {k: _move_to(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_move_to(v, device) for v in x)
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x

# ==== helpers for ligand ====

def ligand_allowed_indices(tok, V: int,
                           allow_rare: bool = True,
                           allow_unk: bool = False,
                           allow_mask: bool = False) -> torch.Tensor:
    allowed = []
    inv = getattr(tok, "id_to_token", None)
    if inv is not None:
        for i, token in enumerate(inv):
            if i >= V: break
            if token == getattr(tok, "MASK_TOKEN", "*") and not allow_mask:
                continue
            if token == getattr(tok, "UNK_TOKEN", "UNK") and not allow_unk:
                continue
            if token == getattr(tok, "RARE_TOKEN", "RARE") and not allow_rare:
                continue
            allowed.append(i)
    else:
        allowed = list(range(min(V, getattr(tok, "vocab_size", V))))
    if not allowed:
        allowed = list(range(min(V, getattr(tok, "vocab_size", V))))
    return torch.tensor(allowed, dtype=torch.long)

def decode_ligand_ids(ids, tok, sep: str = " ") -> str:
    inv = getattr(tok, "id_to_token", None)
    if inv is None:
        return sep.join(str(i) for i in ids)
    out = []
    for i in ids:
        if 0 <= i < len(inv):
            out.append(inv[i])
        else:
            out.append(getattr(tok, "UNK_TOKEN", "UNK"))
    return sep.join(out)

# ========= 原有 inference（teacher-forcing）保留 =========
def inference(model, sample_input, model_name, data_name,
              topk: int = 5, temp: float = 1.0, use_beam: bool = False,
              device=None):
    """
    一次性 forward (teacher-forcing)，支持贪心或 beam（基于 log_probs 矩阵）
    返回:
      - pred_seqs:  List[str]
      - scores:     List[float]
      - true_seq:   str
      - probs_cpu:  torch.FloatTensor [L_lig, V]  # ★ 现在返回的是“配体子序列”的概率矩阵
      - metrics:    dict
    """
    kind = str(data_name).lower()
    # 1) featurizer/tokenizer
    if kind == 'ligand':
        from src.datasets.featurizer_ligand import Featurize_Ligand, LigandTokenizer
        feat = Featurize_Ligand()
        tok = LigandTokenizer()
    else:
        from src.datasets.featurizer import Featurize, UniTokenizer
        feat = Featurize(mixed=True)
        tok = UniTokenizer()

    # 2) featurize
    batch = feat.featurize([sample_input])
    if batch is None:
        raise RuntimeError("featurize defective??")

    # 3) 设备
    if device is None:
        try:
            device = next(model.model.parameters()).device
        except Exception:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = _move_to(batch, device)

    # 4) forward
    model.model.eval()
    with torch.no_grad():
        out = model.model(batch)
        log_probs = out['log_probs']         # [N_all, V]
        logits    = out['logits']            # [N_all, V]
        probs_cpu_all = torch.softmax(logits / temp, dim=-1).detach().cpu()
        logp_cpu_all  = log_probs.detach().cpu()

    # ---------- ★ 只解码“配体位置” ----------
    if kind == 'ligand':
        # 取配体掩码（优先 ligand_mask；没有则用 chain_mask）
        mask = batch.get('ligand_mask', None)
        if mask is None:
            mask = batch['chain_mask']
        mask = mask.detach().float().cpu().view(-1)  # [N_all]
        idx = (mask > 0.5).nonzero(as_tuple=False).view(-1)  # [L_lig]
        if idx.numel() == 0:
            # 没有配体节点（极端/非法样本）
            return [], [], "", torch.empty(0, logits.shape[-1]), {
                'per_candidate': [], 'top1': {'Accuracy': 0.0, 'Macro_F1': 0.0},
                'best_idx': 0, 'best': {'Accuracy': 0.0, 'Macro_F1': 0.0}
            }

        probs_cpu = probs_cpu_all.index_select(0, idx)  # [L_lig, V]
        logp_cpu  = logp_cpu_all.index_select(0, idx)   # [L_lig, V]
    else:
        probs_cpu = probs_cpu_all
        logp_cpu  = logp_cpu_all

    # 5) 允许 vocab
    L, V = probs_cpu.shape
    if kind == 'ligand':
        allow = ligand_allowed_indices(tok, V, allow_rare=True, allow_unk=False, allow_mask=False)
    else:
        allow = allowed_indices(kind, V)
        if allow.numel() == 0:
            allow = torch.arange(0, V, dtype=torch.long)

    # 6) 生成（仅配体子序列）
    pred_seqs, scores = [], []
    if not use_beam:
        sub = probs_cpu[:, allow]                 # [L, |allow|]
        ids_rel = sub.argmax(dim=-1)              # [L]
        ids_glb = allow[ids_rel].tolist()         # [L]
        if kind == 'ligand':
            seq = decode_ligand_ids(ids_glb, tok, sep=" ")
        else:
            seq = tok.decode(ids_glb, kind=kind)
        # 分数=只对配体位置求和
        lp  = float(logp_cpu[torch.arange(L), torch.tensor(ids_glb)].sum())
        pred_seqs = [seq]; scores = [lp]
    else:
        beam_width = max(1, int(topk))
        beams = beam_search_logp(
            logp_cpu, allow, beam_width,
            len_norm=True, alpha=0.6,
            diversity_groups=1, gamma=0.0
        )
        for ids_glb, logp_sum in beams:
            if kind == 'ligand':
                seq = decode_ligand_ids(ids_glb, tok, sep=" ")
            else:
                seq = tok.decode(ids_glb, kind=kind)
            pred_seqs.append(seq)
            scores.append(float(logp_sum))

    # 7) metrics（Ligand 用元素串）
    if kind == 'ligand':
        # 优先用样本里显式保存的 ligand_seq；否则从 ligand.elements 拼
        true_seq = sample_input.get('ligand_seq') \
                   or " ".join(sample_input.get('ligand', {}).get('elements', [])) \
                   or ""
        true_tokens = (true_seq or "").split()
        if pred_seqs:
            L_common = min(len(true_tokens), *(len(s.split()) for s in pred_seqs))
        else:
            L_common = 0
        if L_common == 0:
            metrics = {
                'per_candidate': [{'Accuracy': 0.0, 'Macro_F1': 0.0} for _ in pred_seqs],
                'top1': {'Accuracy': 0.0, 'Macro_F1': 0.0},
                'best_idx': 0,
                'best': {'Accuracy': 0.0, 'Macro_F1': 0.0},
            }
            return pred_seqs, scores, true_seq, probs_cpu, metrics

        true_tokens = true_tokens[:L_common]
        per_cand = []
        for seq in pred_seqs:
            pred_tokens = seq.split()[:L_common]
            acc = sum(1 for a, b in zip(true_tokens, pred_tokens) if a == b) / float(L_common)
            labels = sorted(set(true_tokens) | set(pred_tokens))
            try:
                macro_f1 = float(f1_score(true_tokens, pred_tokens, labels=labels, average='macro'))
            except Exception:
                macro_f1 = 0.0
            per_cand.append({'Accuracy': acc, 'Macro_F1': macro_f1})
        top1 = per_cand[0]
        best_idx = max(range(len(per_cand)), key=lambda i: (per_cand[i]['Accuracy'],
                                                            per_cand[i]['Macro_F1'],
                                                            scores[i]))
        best = per_cand[best_idx]
        metrics = {'per_candidate': per_cand, 'top1': top1, 'best_idx': best_idx, 'best': best}
        return pred_seqs, scores, true_seq, probs_cpu, metrics

    # protein / rna / dna（保持原逻辑）
    true_seq = sample_input['seq'][0] if isinstance(sample_input.get('seq'), list) \
               else sample_input.get('seq', '')
    L_common = min(len(true_seq), *(len(s) for s in pred_seqs)) if pred_seqs else 0
    if L_common == 0:
        metrics = {
            'per_candidate': [{'Accuracy': 0.0, 'Macro_F1': 0.0} for _ in pred_seqs],
            'top1': {'Accuracy': 0.0, 'Macro_F1': 0.0},
            'best_idx': 0,
            'best': {'Accuracy': 0.0, 'Macro_F1': 0.0},
        }
        return pred_seqs, scores, true_seq, probs_cpu, metrics
    true_seq_c = true_seq[:L_common]
    per_cand = []
    for seq in pred_seqs:
        seq_c = seq[:L_common]
        acc = sum(1 for a, b in zip(true_seq_c, seq_c) if a == b) / float(L_common)
        y_true = labels_4class(true_seq_c, kind, tok)
        y_pred = labels_4class(seq_c, kind, tok)
        try:
            macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
        except Exception:
            macro_f1 = 0.0
        per_cand.append({'Accuracy': acc, 'Macro_F1': macro_f1})
    top1 = per_cand[0]
    best_idx = max(range(len(per_cand)), key=lambda i: (per_cand[i]['Accuracy'],
                                                        per_cand[i]['Macro_F1'],
                                                        scores[i]))
    best = per_cand[best_idx]
    metrics = {'per_candidate': per_cand, 'top1': top1, 'best_idx': best_idx, 'best': best}
    return pred_seqs, scores, true_seq, probs_cpu, metrics

# def inference(model, sample_input, model_name, data_name,
#               topk: int = 5, temp: float = 1.0, use_beam: bool = False,
#               device=None):
#     """
#     原版：一次性 forward (teacher-forcing)，支持贪心或 beam（基于 log_probs 矩阵）
#     返回:
#       - pred_seqs:  List[str]
#       - scores:     List[float]
#       - true_seq:   str
#       - probs_cpu:  torch.FloatTensor [L, V]
#       - metrics:    dict
#     """
#     kind = str(data_name).lower()
#     # 1) featurizer/tokenizer
#     if kind == 'ligand':
#         from src.datasets.featurizer_ligand import Featurize_Ligand, LigandTokenizer
#         feat = Featurize_Ligand()
#         tok = LigandTokenizer()
#     else:
#         from src.datasets.featurizer import Featurize, UniTokenizer
#         feat = Featurize(mixed=True)
#         tok = UniTokenizer()

#     # 2) featurize
#     batch = feat.featurize([sample_input])
#     if batch is None:
#         raise RuntimeError("featurize defective??")

#     # 3) 设备
#     if device is None:
#         try:
#             device = next(model.model.parameters()).device
#         except Exception:
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     batch = _move_to(batch, device)

#     # 4) forward
#     model.model.eval()
#     with torch.no_grad():
#         out = model.model(batch)
#         log_probs = out['log_probs']         # [L, V]
#         logits    = out['logits']            # [L, V]
#         probs_cpu = torch.softmax(logits / temp, dim=-1).detach().cpu()
#         logp_cpu  = log_probs.detach().cpu()

#     # 5) 允许 vocab
#     L, V = probs_cpu.shape
#     if kind == 'ligand':
#         allow = ligand_allowed_indices(tok, V, allow_rare=True, allow_unk=False, allow_mask=False)
#     else:
#         allow = allowed_indices(kind, V)
#         if allow.numel() == 0:
#             allow = torch.arange(0, V, dtype=torch.long)

#     # 6) 生成
#     pred_seqs, scores = [], []
#     if not use_beam:
#         sub = probs_cpu[:, allow]                 # [L, |allow|]
#         ids_rel = sub.argmax(dim=-1)              # [L]
#         ids_glb = allow[ids_rel].tolist()         # [L]
#         if kind == 'ligand':
#             seq = decode_ligand_ids(ids_glb, tok, sep=" ")
#         else:
#             seq = tok.decode(ids_glb, kind=kind)
#         lp  = float(logp_cpu[torch.arange(L), torch.tensor(ids_glb)].sum())
#         pred_seqs = [seq]; scores = [lp]
#     else:
#         beam_width = max(1, int(topk))
#         beams = beam_search_logp(
#             logp_cpu, allow, beam_width,
#             len_norm=True, alpha=0.6,
#             diversity_groups=1, gamma=0.0
#         )
#         for ids_glb, logp_sum in beams:
#             if kind == 'ligand':
#                 seq = decode_ligand_ids(ids_glb, tok, sep=" ")
#             else:
#                 seq = tok.decode(ids_glb, kind=kind)
#             pred_seqs.append(seq)
#             scores.append(float(logp_sum))

#     # 7) metrics
#     true_seq = sample_input['seq'][0] if isinstance(sample_input.get('seq'), list) \
#                else sample_input.get('seq', '')

#     if kind == 'ligand':
#         true_tokens = (true_seq or "").split()
#         if pred_seqs:
#             L_common = min(len(true_tokens), *(len(s.split()) for s in pred_seqs))
#         else:
#             L_common = 0
#         if L_common == 0:
#             metrics = {
#                 'per_candidate': [{'Accuracy': 0.0, 'Macro_F1': 0.0} for _ in pred_seqs],
#                 'top1': {'Accuracy': 0.0, 'Macro_F1': 0.0},
#                 'best_idx': 0,
#                 'best': {'Accuracy': 0.0, 'Macro_F1': 0.0},
#             }
#             return pred_seqs, scores, true_seq, probs_cpu, metrics
#         true_tokens = true_tokens[:L_common]
#         per_cand = []
#         for seq in pred_seqs:
#             pred_tokens = seq.split()[:L_common]
#             acc = sum(1 for a, b in zip(true_tokens, pred_tokens) if a == b) / float(L_common)
#             labels = sorted(set(true_tokens) | set(pred_tokens))
#             try:
#                 macro_f1 = float(f1_score(true_tokens, pred_tokens, labels=labels, average='macro'))
#             except Exception:
#                 macro_f1 = 0.0
#             per_cand.append({'Accuracy': acc, 'Macro_F1': macro_f1})
#         top1 = per_cand[0]
#         best_idx = max(range(len(per_cand)), key=lambda i: (per_cand[i]['Accuracy'],
#                                                             per_cand[i]['Macro_F1'],
#                                                             scores[i]))
#         best = per_cand[best_idx]
#         metrics = {'per_candidate': per_cand, 'top1': top1, 'best_idx': best_idx, 'best': best}
#         return pred_seqs, scores, true_seq, probs_cpu, metrics

#     # protein / rna / dna
#     L_common = min(len(true_seq), *(len(s) for s in pred_seqs)) if pred_seqs else 0
#     if L_common == 0:
#         metrics = {
#             'per_candidate': [{'Accuracy': 0.0, 'Macro_F1': 0.0} for _ in pred_seqs],
#             'top1': {'Accuracy': 0.0, 'Macro_F1': 0.0},
#             'best_idx': 0,
#             'best': {'Accuracy': 0.0, 'Macro_F1': 0.0},
#         }
#         return pred_seqs, scores, true_seq, probs_cpu, metrics
#     true_seq_c = true_seq[:L_common]
#     per_cand = []
#     for seq in pred_seqs:
#         seq_c = seq[:L_common]
#         acc = sum(1 for a, b in zip(true_seq_c, seq_c) if a == b) / float(L_common)
#         y_true = labels_4class(true_seq_c, kind, tok)
#         y_pred = labels_4class(seq_c, kind, tok)
#         try:
#             macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
#         except Exception:
#             macro_f1 = 0.0
#         per_cand.append({'Accuracy': acc, 'Macro_F1': macro_f1})
#     top1 = per_cand[0]
#     best_idx = max(range(len(per_cand)), key=lambda i: (per_cand[i]['Accuracy'],
#                                                         per_cand[i]['Macro_F1'],
#                                                         scores[i]))
#     best = per_cand[best_idx]
#     metrics = {'per_candidate': per_cand, 'top1': top1, 'best_idx': best_idx, 'best': best}
#     return pred_seqs, scores, true_seq, probs_cpu, metrics


# ========= 新增：自回归采样版 inference =========

@torch.no_grad()
def inference_ar(model, sample_input, data_name,
                 device=None,
                 # 采样参数：
                 sample_k: int = 1,
                 temp: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 0.0,
                 # beam 参数（两者三选一：greedy / sampling / beam）
                 use_beam: bool = False,
                 beam_width: int = 5,
                 len_norm: bool = True,
                 alpha: float = 0.6):
    """
    返回:
      pred_seqs: List[str]    # 长度 = sample_k (采样) 或 beam_width (beam) 或 1 (greedy)
      scores:    List[float]  # 对应的对数似然（含长度归一/惩罚）
      true_seq:  str
      step_probs: Optional[List[torch.Tensor]]  # 每条生成路径的逐步分布（可选）
    """
    kind = str(data_name).lower()
    # 1) featurize
    from src.datasets.featurizer import Featurize, UniTokenizer
    feat = Featurize(mixed=True)
    tok = UniTokenizer()

    batch = feat.featurize([sample_input])
    if batch is None:
        raise RuntimeError("featurize defective??")

    # 2) 设备
    if device is None:
        try:
            device = next(model.model.parameters()).device
        except Exception:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = _move_to(batch, device)
    mdl = model.model
    mdl.eval()

    # 3) 编码 + PAD
    H_pad, valid_mask, lengths, aux = mdl.encode_and_pad(batch)   # [B,L,d], [B,L], [B]
    B, L, V = H_pad.size(0), H_pad.size(1), mdl.vocab_size

    # 4) 合法 token（全局一种；如果你想每链不同，可自行扩展成 per-batch）
    allow = allowed_indices(kind, V).to(device)
    if allow.numel() == 0:
        allow = torch.arange(0, V, device=device, dtype=torch.long)

    # --- 工具函数 ---
    def mask_to_allowed_logits(logits_step):   # logits_step: [B, V]
        # 只保留 allow，其余设为 -inf
        out = logits_step.clone()
        mask = torch.ones_like(out, dtype=torch.bool)
        mask[:, allow] = False
        out = out.masked_fill(mask, -1e9)
        return out

    def sample_from_logits(logits_step):       # [B, V]
        ls = logits_step / max(1e-6, float(temp))
        if top_k > 0:
            # top-k
            tk = min(top_k, ls.size(-1))
            vals, idx = torch.topk(ls, tk, dim=-1)
            probs = torch.softmax(vals, dim=-1)
            choice = torch.multinomial(probs, 1)           # [B,1]
            picked = idx.gather(-1, choice).squeeze(-1)    # [B]
            return picked
        if top_p and top_p > 0.0:
            # nucleus
            sorted_logits, sorted_idx = torch.sort(ls, dim=-1, descending=True)
            cprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            cutoff = (cprobs > top_p).float().argmax(dim=-1)  # [B]
            rows = torch.arange(ls.size(0), device=ls.device)
            ks = (cutoff + 1).clamp_max(ls.size(-1))
            # 对每行取前 ks[i]
            picked = []
            for i in range(ls.size(0)):
                k_i = int(ks[i].item())
                probs_i = torch.softmax(sorted_logits[i, :k_i], dim=-1)
                j = torch.multinomial(probs_i, 1)
                picked.append(sorted_idx[i, j])
            return torch.stack(picked, dim=0).squeeze(-1)
        # 普通采样
        return torch.multinomial(torch.softmax(ls, dim=-1), 1).squeeze(-1)

    # 5A) —— 逐步“贪心/采样”解码（支持 sample_k 次）——
    def decode_sampling(k_times: int):
        """
        向量化并行采样：把同一个样本复制 k 次并行推断。
        - B 原本是“链条数”；并行后 batch 变成 B*k。
        - 返回按分数从高到低排序的 k 条合并序列（每条把该样本的所有链拼起来）。
        """
        k = max(1, int(k_times))

        # 1) 扩展 batch 维： [B,L,d] -> [B*k,L,d]
        H_rep = H_pad.repeat_interleave(k, dim=0)         # [B*k, L, d]
        VM_rep = valid_mask.repeat_interleave(k, dim=0)   # [B*k, L]
        LEN_rep = lengths.repeat_interleave(k, dim=0)     # [B*k]
        Bk, L_tot = H_rep.size(0), H_rep.size(1)

        # 2) 预先缓存因果掩码，避免每步重建
        causal = mdl.decoder._causal_mask(L_tot, H_rep.device)

        # 3) 自回归逐步生成（但一次前向覆盖 B*k 条）
        y = torch.zeros(Bk, L_tot, dtype=torch.long, device=device)
        logp_sum = torch.zeros(Bk, dtype=torch.float, device=device)

        for t in range(L_tot):
            active = (t < LEN_rep)
            if not active.any():
                break
            _, logits = mdl.decoder(H_rep, y, VM_rep)  # [Bk,L,V]
            step_logits = logits[:, t, :]                                 # [Bk,V]
            step_logits = mask_to_allowed_logits(step_logits)

            # 贪心 or 采样
            if k == 1 and top_k == 0 and (not top_p) and abs(temp - 1.0) < 1e-6:
                picked = step_logits.argmax(dim=-1)                       # [Bk]
            else:
                picked = sample_from_logits(step_logits)                  # [Bk]

            active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            y[active_idx, t] = picked[active_idx]
            logp_sum[active_idx] += torch.log_softmax(
                step_logits[active_idx], dim=-1
            ).gather(-1, picked[active_idx].unsqueeze(-1)).squeeze(-1)

        # 4) 把 B*k 行按“每 k 行属于同一条链位的不同采样”还原成 k 份结果
        #    我们需要把同一采样编号的所有链拼在一起。
        results = []
        for sample_idx in range(k):
            seqs_one = []
            # 这 k 次是 repeat_interleave 形成的：行号 sample_idx::k
            rows = torch.arange(sample_idx, Bk, k, device=device)
            for r in rows.tolist():
                ids = y[r, :int(LEN_rep[r])].tolist()
                seqs_one.append(tok.decode(ids, kind=kind))
            merged = "".join(seqs_one)
            # 简单长度归一
            score = float(logp_sum[rows].sum().item()) / max(1, int(lengths.sum().item()))
            results.append((merged, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [s for s,_ in results], [float(sc) for _,sc in results]

    # 5B) —— 真·AR beam search（逐步展开；这里按“每条样本只有一条链”最常见场景实现）——
    def decode_beam():
        assert B == 1, "示例实现的 AR beam 先支持单链样本，若多链可按链分别跑再拼接。"
        L1 = int(lengths[0].item())
        beams = [([], 0.0)]  # (ids, score)
        for t in range(L1):
            cand = []
            for ids, score in beams:
                # 构造当前前缀 y
                y = torch.zeros(1, L, dtype=torch.long, device=device)
                if ids:
                    y[0, :len(ids)] = torch.tensor(ids, device=device, dtype=torch.long)
                _, logits = mdl.decoder(H_pad[:1, :, :], y, valid_mask[:1, :])  # [1,L,V]
                step_logits = logits[0, t, :]           # [V]
                step_logits = mask_to_allowed_logits(step_logits.unsqueeze(0))[0]  # [V]
                step_logp = torch.log_softmax(step_logits, dim=-1)
                # 取前 beam_width 个
                topv, topi = torch.topk(step_logp, k=min(beam_width, allow.numel()))
                for v, i in zip(topv.tolist(), topi.tolist()):
                    cand.append((ids + [i], score + float(v)))
            # 选 beam_width
            beams = sorted(cand, key=lambda x: x[1], reverse=True)[:beam_width]

        # 长度惩罚
        if len_norm:
            beams = [(ids, s / max(1, len(ids))) for ids, s in beams]
        elif alpha > 0:
            def lp_len(m): return ((5 + m) ** alpha) / ((5 + 1) ** alpha)
            beams = [(ids, s / lp_len(len(ids))) for ids, s in beams]
        beams.sort(key=lambda x: x[1], reverse=True)

        seqs, scores = [], []
        for ids, s in beams:
            seqs.append(tok.decode(ids, kind=kind))
            scores.append(float(s))
        return seqs, scores

    # 选择路径
    if use_beam:
        pred_seqs, scores = decode_beam()
    else:
        pred_seqs, scores = decode_sampling(sample_k)

    # 6) 真值与简单评估（维持你原逻辑）
    true_seq = sample_input['seq'][0] if isinstance(sample_input.get('seq'), list) \
               else sample_input.get('seq', '')

    # 真值（仅为了保持返回签名一致；你可以忽略）
    true_seq = sample_input['seq'][0] if isinstance(sample_input.get('seq'), list) \
               else sample_input.get('seq', '')

    # === 只打印预测序列（不算任何 metrics）===
    if pred_seqs:
        print("[AR] Generated sequences:")
        for i, (seq, sc) in enumerate(zip(pred_seqs, scores)):
            # 按需保留/去掉 score 展示；只想看序列就 print(seq)
            print(f"{i:02d}: score={sc:.4f}  seq={seq}")
    else:
        print("[AR] No sequences generated.")

    # 保持 5 元组返回，但不再提供 probs/metrics
    return pred_seqs, scores, true_seq, None, None