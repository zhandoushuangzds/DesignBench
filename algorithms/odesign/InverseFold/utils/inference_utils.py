#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip
import io
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

# -------- Biopython imports (统一 & 正确导入类，而不是模块) --------
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.StructureBuilder import StructureBuilder

# ===================== 常量 & 映射 =====================
WATER_CODES = {"HOH", "WAT", "DOD", "H2O"}
_WATER_NAMES = {"HOH", "WAT"}

# 主链槽位（核酸/蛋白）
NA_BACKBONE_ORDER = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "O4'", "C1'", "C2'", "N"]  # 含 N (N9/N1)
PROT_BACKBONE_ORDER = ["N", "CA", "C", "O"]

# 无撇短名 -> 带撇规范名
key_map = {
    "P": "P",
    "O5": "O5'",
    "C5": "C5'",
    "C4": "C4'",
    "C3": "C3'",
    "O3": "O3'",
    "N": "N",
}

# 核酸别名（容错 O5* 等）
NA_BACKBONE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "P": ("P",),
    "O5'": ("O5'", "O5*"),
    "C5'": ("C5'", "C5*"),
    "C4'": ("C4'", "C4*"),
    "C3'": ("C3'", "C3*"),
    "O3'": ("O3'", "O3*"),
    "O4'": ("O4'", "O4*"),
    "C1'": ("C1'", "C1*"),
    "C2'": ("C2'", "C2*"),
}

NA3_TO_1 = {
    "DA": "A",
    "DT": "T",
    "DG": "G",
    "DC": "C",
    "DI": "I",
    "A": "A",
    "T": "T",
    "G": "G",
    "C": "C",
    "U": "U",
    "I": "I",
    "DU": "U",
    "URA": "U",
    "ADE": "A",
    "THY": "T",
    "GUA": "G",
    "CYT": "C",
}

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",
    "ASX": "B",
    "GLX": "Z",
    "XLE": "J",
    "UNK": "X",
    "XXX": "X",
}

# ===================== 基础小工具 =====================
def _sniff_fmt_from_text(txt: str) -> str:
    t = txt[:4000]
    return "cif" if ("_atom_site." in t or "data_" in t) else "pdb"

def _read_text_maybe_gz(path: Path) -> str:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return path.read_text(encoding="utf-8", errors="ignore")

def _load_cif_dict_from_text(cif_text) -> dict:
    if cif_text is None:
        raise ValueError("Empty cif_text passed to _load_cif_dict_from_text")
    handle = io.BytesIO(cif_text) if isinstance(cif_text, bytes) else io.StringIO(str(cif_text))
    return MMCIF2Dict(handle)

def _ensure_biopython():
    return True

def _infer_pdbid_from_filename(path: str) -> str:
    name = Path(path).name.lower()
    m = re.match(r"([0-9a-z]{4})", name)
    return (m.group(1) if m else Path(path).stem.split(".")[0]).upper()

def _canon_elem_symbol(sym: str) -> Optional[str]:
    if sym is None:
        return None
    s = str(sym).strip()
    if not s or s == "*":
        return None
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:].lower()

def _biopy_get_element(atom: Any) -> str:
    e = getattr(atom, "element", None)
    if isinstance(e, str) and e.strip():
        return _canon_elem_symbol(e) or "X"
    name = atom.get_name().strip()
    guess = re.sub(r"[^A-Za-z]", "", name)[:2]
    return _canon_elem_symbol(guess) or "X"

def canon_atom_name(name: str) -> str:
    s = name.strip()
    s = s.replace("’", "'").replace("′", "'").replace("`", "'")
    s = s.replace("*", "'")
    return s

def sanitize_coord(atom, ndigits: int) -> Optional[List[float]]:
    try:
        coord = atom.coord
        if coord is None or len(coord) != 3:
            return None
        x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return None
        return [round(x, ndigits), round(y, ndigits), round(z, ndigits)]
    except Exception:
        return None

# ===================== CIF 解析/映射 =====================
def _parse_structure_path(path: Path) -> Tuple[Any, str, str]:
    """
    读取本地 .cif/.cif.gz/.pdb/.pdb.gz 并解析。
    返回: (structure, fmt, text)；fmt in {"cif", "pdb"}。
    """
    txt = _read_text_maybe_gz(path)
    fmt = _sniff_fmt_from_text(txt)
    parser = MMCIFParser(QUIET=True) if fmt == "cif" else PDBParser(QUIET=True)

    parse_path = path.with_suffix(".cif" if fmt == "cif" else ".pdb")
    if (not parse_path.exists()) or (parse_path.read_text(encoding="utf-8", errors="ignore") != txt):
        parse_path.write_text(txt, encoding="utf-8")

    st = parser.get_structure("st", str(parse_path))
    return st, fmt, txt

def build_cif_id_maps(cif_text: str):
    """
    从 CIF 文本构建：
      - label2auth, auth2label
      - label2entity, entity2label, entity2auth
    """
    d = _load_cif_dict_from_text(cif_text)

    # _atom_site: label_asym_id <-> auth_asym_id
    lab = d.get("_atom_site.label_asym_id", [])
    aut = d.get("_atom_site.auth_asym_id", [])
    if isinstance(lab, str):
        lab = [lab]
    if isinstance(aut, str):
        aut = [aut]
    label2auth: Dict[str, str] = {}
    auth2label: Dict[str, str] = {}
    for L, A in zip(lab, aut):
        if L and L != "?":
            label2auth.setdefault(L, A)
        if A and A != "?":
            auth2label.setdefault(A, L)

    # _struct_asym: label_asym_id <-> entity_id
    sa_id = d.get("_struct_asym.id", [])
    sa_ent = d.get("_struct_asym.entity_id", [])
    if isinstance(sa_id, str):
        sa_id = [sa_id]
    if isinstance(sa_ent, str):
        sa_ent = [sa_ent]
    label2entity: Dict[str, str] = {}
    entity2label: Dict[str, set] = {}
    for L, E in zip(sa_id, sa_ent):
        if not L or L == "?" or not E or E == "?":
            continue
        label2entity[L] = E
        entity2label.setdefault(E, set()).add(L)

    # _entity_poly.pdbx_strand_id: entity -> {auth_asym_id}
    ep_eid = d.get("_entity_poly.entity_id", [])
    ep_str = d.get("_entity_poly.pdbx_strand_id", [])
    if isinstance(ep_eid, str):
        ep_eid = [ep_eid]
    if isinstance(ep_str, str):
        ep_str = [ep_str]
    entity2auth: Dict[str, set] = {}
    for E, S in zip(ep_eid, ep_str):
        if not E or E == "?" or not S or S == "?":
            continue
        parts = [s.strip() for s in re.split(r"[,; ]+", S) if s.strip()]
        if parts:
            entity2auth.setdefault(E, set()).update(parts)

    return label2auth, auth2label, label2entity, entity2label, entity2auth

# ===================== 序列/主链提取（推理前用） =====================
def build_residue_atom_index(res) -> Dict[str, Any]:
    index: Dict[str, Any] = {}
    for atom in res.get_atoms():
        nm = canon_atom_name(atom.get_name())
        try:
            if atom.is_disordered():
                alts = atom.disordered_get_list()
                atom = sorted(alts, key=lambda a: (a.occupancy or 0.0), reverse=True)[0]
        except Exception:
            pass
        if nm not in index:
            index[nm] = atom
    return index

def guess_is_nucleic(chain) -> bool:
    total = 0
    na_like = 0
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        total += 1
        rn = res.get_resname().strip().upper()
        if rn in NA3_TO_1 or (rn.startswith("D") and rn[1:] in ("A", "T", "G", "C", "U", "I")):
            na_like += 1
    return total > 0 and na_like / max(total, 1) >= 0.5

def na3_to1_safe(resname: str) -> str:
    rn = resname.strip().upper()
    if rn in NA3_TO_1:
        return NA3_TO_1[rn]
    if rn.startswith("D") and rn[1:] in ("A", "T", "G", "C", "U", "I"):
        return rn[1:]
    return "N"

def prot3_to1_safe(resname: str) -> str:
    rn = re.sub(r"[^A-Z]", "", resname.strip().upper())
    return AA3_TO_1.get(rn, "X")

def extract_chain_seq_and_backbone(
    chain,
    ndigits: int = 3,
    validate: bool = True,
    drop_all_none_keys: bool = False,
) -> Tuple[str, Dict[str, List[Optional[List[float]]]], bool]:
    is_na = guess_is_nucleic(chain)
    order = NA_BACKBONE_ORDER if is_na else PROT_BACKBONE_ORDER

    residues = [res for res in chain.get_residues() if res.id[0] == " "]
    letters = [na3_to1_safe(r.get_resname()) if is_na else prot3_to1_safe(r.get_resname()) for r in residues]
    seq = "".join(letters)

    coords: Dict[str, List[Optional[List[float]]]] = {a: [] for a in order}
    for res in residues:
        atom_index = build_residue_atom_index(res)
        for atom_name in order:
            atom = atom_index.get(canon_atom_name(atom_name))
            if atom_name == "N":
                # N9/N1 选择
                base = na3_to1_safe(res.get_resname()) if is_na else None
                if base in {"A", "G"}:
                    atom = atom_index.get(canon_atom_name("N9")) or atom
                elif base in {"C", "T", "U"}:
                    atom = atom_index.get(canon_atom_name("N1")) or atom
            if atom is None and is_na:
                for alias in NA_BACKBONE_ALIASES.get(atom_name, (atom_name,)):
                    atom = atom_index.get(canon_atom_name(alias))
                    if atom is not None:
                        break
            coords[atom_name].append(sanitize_coord(atom, ndigits) if atom is not None else None)

    if drop_all_none_keys:
        coords = {k: v for k, v in coords.items() if any(item is not None for item in v)}

    if validate:
        L = len(seq)
        for k, arr in coords.items():
            if len(arr) != L:
                raise ValueError(f"[Invariant violated] {k} 长度={len(arr)} != 序列长度={L}")
            for idx, item in enumerate(arr):
                if item is not None and not (isinstance(item, list) and len(item) == 3):
                    raise ValueError(f"[Bad coord] {k}[{idx}] 非 3 维坐标: {item}")

    return seq, coords, is_na

# ===================== 提供给 featurizer 的规整坐标 =====================
def _atom_xyz_if_exists(residue, atom_name: str):
    try:
        if residue.has_id(atom_name):
            at = residue[atom_name]
            x, y, z = at.get_coord()
            return [round(float(x), 3), round(float(y), 3), round(float(z), 3)]
    except Exception:
        pass
    return None

def _get_backbone_atom(residue, canonical_key: str):
    canon_name = key_map.get(canonical_key, canonical_key)
    aliases = NA_BACKBONE_ALIASES.get(canon_name, (canon_name,))
    for cand in aliases:
        try:
            if residue.has_id(cand):
                at = residue[cand]
                x, y, z = at.get_coord()
                return [round(float(x), 3), round(float(y), 3), round(float(z), 3)]
        except Exception:
            continue
    return None

def _detect_chain_kind(chain) -> Optional[str]:
    _AA3 = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "SEC", "PYL", "MSE", "ASX", "GLX", "XLE", "UNK", "XXX", "HID", "HIE", "HIP", "HSD", "HSE", "HSP",
    }
    _NA = {"A", "C", "G", "U", "DA", "DC", "DG", "DT", "I", "DI", "DU"}
    aa_cnt = na_cnt = dna_cnt = rna_cnt = 0
    for res in chain:
        name = str(res.get_resname()).upper().strip()
        if name in _AA3:
            aa_cnt += 1
        elif name in _NA:
            na_cnt += 1
            if name.startswith("D") or name == "DT":
                dna_cnt += 1
            elif name in {"A", "C", "G", "U"}:
                rna_cnt += 1
    if aa_cnt > 0:
        return "protein"
    if na_cnt > 0:
        return "dna" if dna_cnt >= rna_cnt else "rna"
    return None

def _coords_list_to_array(coords_col: List[Optional[List[float]]], L: int) -> np.ndarray:
    out = np.full((L, 3), np.nan, dtype=np.float32)
    if not isinstance(coords_col, (list, tuple)):
        return out
    n = min(L, len(coords_col))
    for i in range(n):
        v = coords_col[i]
        if v is None:
            continue
        try:
            x, y, z = float(v[0]), float(v[1]), float(v[2])
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                out[i] = (x, y, z)
        except Exception:
            try:
                x, y, z = float(v["x"]), float(v["y"]), float(v["z"])
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    out[i] = (x, y, z)
            except Exception:
                continue
    return out

def _extract_chain_seq_and_coords(chain, kind: str):
    """
    返回: (seq_str, coords_dict, rep_xyz)
      - protein: coords_dict 包含 'N','CA','C','O'（每个是 (L,3) ndarray，缺失=NaN）
      - nucleic: coords_dict 包含 'P','O5','C5','C4','C3','O3'（键为无撇短名；值为 (L,3) ndarray，缺失=NaN）
    rep 使用 protein=CA, NA=O5，形状均为 (L,3) ndarray。
    """
    residues = [r for r in chain]

    if kind == "protein":
        seq_chars = []
        N_list, CA_list, C_list, O_list = [], [], [], []
        for r in residues:
            name3 = str(r.get_resname()).upper()
            seq_chars.append(AA3_TO_1.get(name3, "X"))
            N_list.append(_atom_xyz_if_exists(r, "N"))
            CA_list.append(_atom_xyz_if_exists(r, "CA"))
            C_list.append(_atom_xyz_if_exists(r, "C"))
            O_list.append(_atom_xyz_if_exists(r, "O"))
        L = len(seq_chars)
        coords_dict = {
            "N": _coords_list_to_array(N_list, L),
            "CA": _coords_list_to_array(CA_list, L),
            "C": _coords_list_to_array(C_list, L),
            "O": _coords_list_to_array(O_list, L),
        }
        rep = coords_dict["CA"].copy()
        return "".join(seq_chars), coords_dict, rep

    # nucleic acids
    seq_chars = []
    slot_lists = {k: [] for k in ("P", "O5", "C5", "C4", "C3", "O3")}
    for r in residues:
        name3 = str(r.get_resname()).upper()
        seq_chars.append(NA3_TO_1.get(name3, "N"))
        slot_lists["P"].append(_get_backbone_atom(r, "P"))
        slot_lists["O5"].append(_get_backbone_atom(r, "O5"))
        slot_lists["C5"].append(_get_backbone_atom(r, "C5"))
        slot_lists["C4"].append(_get_backbone_atom(r, "C4"))
        slot_lists["C3"].append(_get_backbone_atom(r, "C3"))
        slot_lists["O3"].append(_get_backbone_atom(r, "O3"))

    L = len(seq_chars)
    coords_dict = {k: _coords_list_to_array(v, L) for k, v in slot_lists.items()}
    rep = coords_dict["O5"].copy()
    return "".join(seq_chars), coords_dict, rep

# ===================== Ligand 样本抽取（支持 ligand_only） =====================
def extract_ligand_samples_from_cif(
    cif_path: str,
    only_chain: Optional[str] = None,  # 受体+配对模式：限定受体链
    ligand_only: bool = False,  # True=仅配体；False=受体+配体
    skip_waters: bool = True,
) -> List[Dict[str, Any]]:
    _ensure_biopython()
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", str(cif_path))
    if structure is None or len(structure) == 0:
        return []

    model = structure[0]
    pdb_id = _infer_pdbid_from_filename(cif_path)
    only_chain_u = only_chain.upper() if only_chain else None

    outs: List[Dict[str, Any]] = []

    # -------- 仅配体模式 --------
    if ligand_only:
        for ch in model:
            lig_chain_id = ch.id
            for res in ch:
                hetflag, resseq, icode = res.id
                is_het = (hetflag is not None) and (hetflag.strip() != "")
                if not is_het:
                    continue
                resname = str(res.get_resname()).upper()
                if skip_waters and (resname in WATER_CODES):
                    continue
                elements, coords = [], []
                for at in res:
                    elements.append(_biopy_get_element(at))
                    x, y, z = at.get_coord()
                    coords.append([round(float(x), 3), round(float(y), 3), round(float(z), 3)])
                if len(coords) < 1:
                    continue
                lig_xyz = np.asarray(coords, dtype=np.float32)
                if not np.isfinite(lig_xyz).all():
                    continue

                norm_elems = [_canon_elem_symbol(e) or "UNK" for e in elements]
                lig_seq_str = " ".join(norm_elems)
                res_id = int(resseq) if isinstance(resseq, int) else None

                title = f"{pdb_id}_LIG_chain{lig_chain_id}"
                if res_id is not None:
                    title += f"_res{res_id}"
                outs.append(
                    {
                        "title": pdb_id,
                        "pdb_id": pdb_id,
                        "seq": lig_seq_str,  # 日志用途
                        "ligand": {
                            "elements": norm_elems,
                            "coords": lig_xyz.tolist(),
                            "chain_mask": [[1.0] * len(norm_elems)],
                            "chain_encoding": [[1.0] * len(norm_elems)],
                        },
                        "lig_chain_id": lig_chain_id,
                        "res_id": res_id,
                    }
                )
        return outs

    # -------- 成对模式：预扫聚合链 --------
    polymer_chains = []
    for ch in model:
        cid = ch.id
        kind = _detect_chain_kind(ch)
        if kind is None:
            continue
        seq_str, coords_dict, rep_xyz = _extract_chain_seq_and_coords(ch, kind)
        if not seq_str or coords_dict is None:
            continue
        L = len(seq_str)
        if L == 0:
            continue
        polymer_chains.append(
            {
                "kind": kind,
                "chain_id": cid,
                "L": L,
                "seq": seq_str,
                "coords": coords_dict,  # (L,3) ndarray
                "rep": rep_xyz,  # (L,3) ndarray
            }
        )
    if not polymer_chains:
        return []

    # -------- 遍历 ligand，与受体配对 --------
    for ch in model:
        lig_chain_id = ch.id
        for res in ch:
            hetflag, resseq, icode = res.id
            resname = str(res.get_resname()).upper()
            is_het = (hetflag is not None) and (hetflag.strip() != "")
            if not is_het:
                continue
            if skip_waters and (resname in WATER_CODES):
                continue

            elements, coords = [], []
            for at in res:
                elements.append(_biopy_get_element(at))
                x, y, z = at.get_coord()
                coords.append([round(float(x), 3), round(float(y), 3), round(float(z), 3)])
            if len(coords) < 1:
                continue
            lig_xyz = np.asarray(coords, dtype=np.float32)
            if not np.isfinite(lig_xyz).all():
                continue

            rec_cands = (
                [p for p in polymer_chains if p["chain_id"].upper() == only_chain_u]
                if only_chain_u is not None
                else polymer_chains
            )
            if not rec_cands:
                continue

            res_id = int(resseq) if isinstance(resseq, int) else None
            norm_elems = [_canon_elem_symbol(e) or "UNK" for e in elements]
            lig_seq_str = " ".join(norm_elems)

            for rec in rec_cands:
                title = f"{pdb_id}_chain{lig_chain_id}_with_chain{rec['chain_id']}"
                if res_id is not None:
                    title += f"_res{res_id}"
                outs.append(
                    {
                        "title": pdb_id,
                        "pdb_id": pdb_id,
                        "type": [rec["kind"]],
                        "seq": [rec["seq"]],
                        "coords": rec["coords"],  # (L,3) ndarray
                        "chain_mask": [np.ones(rec["L"], np.float32)],
                        "chain_encoding": [np.ones(rec["L"], np.float32)],
                        "chain_names": [rec["chain_id"]],
                        "ligand": {
                            "elements": norm_elems,
                            "coords": lig_xyz.tolist(),
                            "chain_mask": [[1.0] * len(norm_elems)],
                            "chain_encoding": [[1.0] * len(norm_elems)],
                        },
                        "lig_chain_id": lig_chain_id,
                        "rec_chain_id": rec["chain_id"],
                        "res_id": res_id,
                        "ligand_seq": lig_seq_str,
                    }
                )
    return outs

# ===================== 写 CIF（修复 StructureBuilder 导入/调用） =====================
from functools import lru_cache
from typing import Optional, List, Tuple
from pathlib import Path

from Bio.PDB import MMCIFParser, MMCIFIO
from Bio.PDB.StructureBuilder import StructureBuilder as BPStructureBuilder

# 若你在本文件上方已定义 _biopy_get_element / canon_atom_name，可复用；
# 这里做兜底以防未导入（不会覆盖你已有实现）。
try:
    _biopy_get_element
except NameError:
    import re as _re
    def _biopy_get_element(atom) -> str:
        e = getattr(atom, "element", None)
        if isinstance(e, str) and e.strip():
            s = e.strip()
            return (s[0].upper() + s[1:].lower()) if len(s) > 1 else s.upper()
        nm = str(getattr(atom, "name", "") or getattr(atom, "get_name", lambda: "")()).strip()
        guess = _re.sub(r"[^A-Za-z]", "", nm)[:2]
        if not guess:
            return "X"
        return (guess[0].upper() + guess[1:].lower()) if len(guess) > 1 else guess.upper()

try:
    canon_atom_name
except NameError:
    def canon_atom_name(name: str) -> str:
        s = name.strip()
        s = s.replace('’', "'").replace('′', "'").replace('`', "'").replace('*', "'")
        return s

# 水分子别名
_WATER_NAMES = {"HOH", "WAT", "DOD", "H2O"}

def _pdb_fullname(name: str) -> str:
    """PDB 风格 4 字符 fullname（右对齐），过长截断。"""
    n = str(name or "").strip()
    if len(n) > 4:
        n = n[:4]
    return f"{n:>4s}"

def _canon_element_for_biopy(elem: str) -> str:
    if not elem:
        return "X"
    s = str(elem).strip()
    if not s:
        return "X"
    s = "".join(ch for ch in s if ch.isalpha())[:2].upper()
    return s if s else "X"

def _mk_atom_names(element_uc: str, idx: int) -> tuple[str, str]:
    """用于配体统一命名（无论来源），生成 C01/O02 等 4 字符名。"""
    el2 = f"{element_uc:>2}"[:2]
    n2 = f"{(idx % 100):02d}"
    name = el2[0] + el2[1] + n2   # 长度恰好 4
    fullname = name
    return name, fullname

@lru_cache(maxsize=8)
def _load_structure_cached(cif_src_path: str):
    parser = MMCIFParser(QUIET=True)
    st = parser.get_structure("src", str(cif_src_path))
    if st is None or len(st) == 0:
        raise RuntimeError(f"Empty structure: {cif_src_path}")
    return st

def _find_chain(model, chain_id: str):
    for ch in model:
        if ch.id == str(chain_id):
            return ch
    return None

def _pick_ligand_residue_flex(chain, res_id: Optional[int] = None):
    het_candidates = []
    any_candidates = []
    for res in chain.get_residues():
        hetflag, resseq, icode = res.id
        resname = getattr(res, "resname", "").upper() if hasattr(res, "resname") else ""
        is_water = resname in _WATER_NAMES
        is_het = (hetflag != " ")
        if res_id is not None:
            if int(resseq) == int(res_id):
                any_candidates.append(res)
                if is_het and not is_water:
                    het_candidates.append(res)
        else:
            if is_het and not is_water:
                het_candidates.append(res)
    if res_id is not None:
        if het_candidates:
            het_candidates.sort(key=lambda r: len(list(r.get_atoms())), reverse=True)
            return het_candidates[0]
        if any_candidates:
            any_candidates.sort(key=lambda r: len(list(r.get_atoms())), reverse=True)
            return any_candidates[0]
        return None
    if het_candidates:
        het_candidates.sort(key=lambda r: len(list(r.get_atoms())), reverse=True)
        return het_candidates[0]
    return None

def _copy_chain_to_builder(sb: BPStructureBuilder, chain, *, dst_chain_id: Optional[str] = None):
    """
    将 Biopython 的 receptor chain 内容复制到 StructureBuilder。
    ★ 只复制聚合物残基（hetflag == ' '），不带水和其他 HET。
    ★ 受体保留原子名。
    """
    target_cid = str(dst_chain_id) if dst_chain_id is not None else str(chain.id)
    sb.init_chain(target_cid)
    for res in chain:
        hetflag, resseq, icode = res.id
        # 仅保留聚合物（蛋白/核酸）残基
        if hetflag != ' ':
            continue
        hf = ' '
        ic = (icode if icode else ' ')
        rs = int(resseq) if isinstance(resseq, int) else 1
        sb.init_residue(str(res.get_resname()), hf, rs, ic)
        for idx, at in enumerate(res, 1):
            x, y, z = (float(c) for c in at.get_coord())
            # 原子名与 fullname（受体保留原子名）
            raw_name = canon_atom_name(at.get_name())
            fullname = _pdb_fullname(raw_name)
            # 元素优先来自 at.element；否则从原子名首字母
            raw_el = getattr(at, "element", None)
            if isinstance(raw_el, str) and raw_el.strip():
                el = _canon_element_for_biopy(raw_el)
            else:
                el = _canon_element_for_biopy(raw_name[:1])
            # 其它字段
            occ = float(getattr(at, "occupancy", 1.0))
            bf  = float(getattr(at, "bfactor", 0.0))
            alt = getattr(at, "altloc", " ") or " "
            try:
                serial = int(at.serial_number) if getattr(at, "serial_number", None) else idx
            except Exception:
                serial = idx
            sb.init_atom(raw_name, (x, y, z), bf, occ, alt, fullname, serial, el)

def _normalize_new_elems(new_elems: List[str]) -> List[Optional[str]]:
    norm: List[Optional[str]] = []
    for t in new_elems:
        if t is None:
            norm.append(None); continue
        s = str(t).strip()
        if not s or s.upper() in ("RARE", "UNK", "*"):
            norm.append(None)
        else:
            norm.append(_canon_element_for_biopy(s))
    return norm

# -------------------- 写 ligand-only（直接坐标，合成名字） --------------------
def save_single_ligand_cif(out_path: str, chain_id: str, res_id: Optional[int], atoms_xyz, elements) -> None:
    """
    直接根据传入的坐标/元素写出一个单残基的 ligand CIF（ligand-only 场景用）。
    ★ 配体统一使用合成原子名（C01/O02…）。
    """
    sb = BPStructureBuilder()
    sb.init_structure("lig_single")
    sb.init_model(0)
    sb.init_chain(str(chain_id))
    resseq = int(res_id) if (res_id is not None and str(res_id).strip().isdigit()) else 1
    sb.init_residue("LIG", "H_", resseq, " ")
    for idx, (xyz, elem) in enumerate(zip(atoms_xyz, elements), 1):
        x, y, z = map(float, xyz)
        element_uc = _canon_element_for_biopy(elem)
        name, fullname = _mk_atom_names(element_uc, idx)  # 合成名
        sb.init_atom(name, (x, y, z), 0.0, 1.0, " ", fullname, idx, element_uc)
    structure = sb.get_structure()
    io = MMCIFIO(); io.set_structure(structure)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    io.save(out_path)

def save_ligand_elements_replaced_cif_direct(
    atoms_xyz, new_elems: List[str], out_path: str, chain_id: str = "L", res_id: Optional[int] = None
) -> None:
    """
    直接把预测元素写到给定坐标上（ligand-only 场景用）。
    ★ 配体统一使用合成原子名（C01/O02…）。
    """
    sb = BPStructureBuilder()
    sb.init_structure("lig_pred")
    sb.init_model(0)
    sb.init_chain(str(chain_id))
    resseq = int(res_id) if (res_id is not None and str(res_id).strip().isdigit()) else 1
    sb.init_residue("LIG", "H_", resseq, " ")
    aligned = min(len(atoms_xyz), len(new_elems))
    for idx in range(aligned):
        x, y, z = map(float, atoms_xyz[idx])
        element_uc = _canon_element_for_biopy(new_elems[idx])
        name, fullname = _mk_atom_names(element_uc, idx + 1)  # 合成名
        sb.init_atom(name, (x, y, z), 0.0, 1.0, " ", fullname, idx + 1, element_uc)
    structure = sb.get_structure()
    io = MMCIFIO(); io.set_structure(structure)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    io.save(out_path)

# -------------------- 从源 CIF 导出 ligand（合成名字） --------------------
def save_single_ligand_cif_from_cif(cif_src_path: str, chain_id: str, res_id: Optional[int], out_path: str) -> None:
    """
    从源 CIF 中（仅根据 chain/res_id）找配体残基，单独写出。
    ★ 配体统一使用合成原子名（C01/O02…），不保留原子名。
    """
    st = _load_structure_cached(str(cif_src_path))
    model = st[0]
    chain = _find_chain(model, chain_id)
    if chain is None:
        raise RuntimeError(f"Chain not found: {chain_id}")
    tgt_res = _pick_ligand_residue_flex(chain, res_id=res_id)
    if tgt_res is None:
        raise RuntimeError(f"Ligand not found on chain={chain_id} (res_id={res_id})")

    sb = BPStructureBuilder()
    sb.init_structure("lig_single")
    sb.init_model(0)
    sb.init_chain(str(chain_id))
    resseq = tgt_res.id[1] if isinstance(tgt_res.id[1], int) else (res_id if res_id is not None else 1)
    sb.init_residue("LIG", "H_", int(resseq), " ")

    for idx, at in enumerate(tgt_res, 1):
        x, y, z = (float(c) for c in at.get_coord())
        # 元素：优先原始，否则推断
        raw_el = getattr(at, "element", None)
        el_uc = _canon_element_for_biopy(raw_el if (isinstance(raw_el, str) and raw_el.strip()) else _biopy_get_element(at))
        name, fullname = _mk_atom_names(el_uc, idx)  # 合成名
        occ = float(getattr(at, "occupancy", 1.0))
        bf  = float(getattr(at, "bfactor", 0.0))
        alt = getattr(at, "altloc", " ") or " "
        try:
            serial = int(at.serial_number) if getattr(at, "serial_number", None) else idx
        except Exception:
            serial = idx
        sb.init_atom(name, (x, y, z), bf, occ, alt, fullname, serial, el_uc)

    structure = sb.get_structure()
    io = MMCIFIO(); io.set_structure(structure)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    io.save(out_path)

# -------------------- 在不改变原子名的前提下替换配体元素 -> 改为合成名字 --------------------
def save_ligand_elements_replaced_cif(
    cif_src_path: str, chain_id: str, res_id: Optional[int], new_elems: List[str], out_path: str
) -> Tuple[int, int]:
    """
    将预测元素 token 替换到指定 ligand 残基上（仅替换 element 字段，坐标/编号不变）。
    ★ 输出时配体统一使用合成原子名（C01/O02…），不保留原子名；对齐长度=min(原子数, 预测长度)。
    返回 (applied, aligned)。
    """
    st = _load_structure_cached(str(cif_src_path))
    model = st[0]
    chain = _find_chain(model, chain_id)
    if chain is None:
        raise RuntimeError(f"Chain not found: {chain_id}")
    tgt = _pick_ligand_residue_flex(chain, res_id=res_id)
    if tgt is None:
        raise RuntimeError(f"Ligand not found on chain={chain_id} (res_id={res_id})")

    norm_tokens = _normalize_new_elems(new_elems)
    aligned = min(len(list(tgt.get_atoms())), len(norm_tokens))

    sb = BPStructureBuilder()
    sb.init_structure("lig_mut")
    sb.init_model(0)
    sb.init_chain(str(chain_id))
    resseq = tgt.id[1] if isinstance(tgt.id[1], int) else (res_id if res_id is not None else 1)
    sb.init_residue("LIG", "H_", int(resseq), " ")

    applied = 0
    for idx, at in enumerate(tgt, 1):
        x, y, z = (float(c) for c in at.get_coord())
        if idx <= aligned and norm_tokens[idx - 1] is not None:
            el_uc = norm_tokens[idx - 1]; applied += 1
        else:
            raw_el = getattr(at, "element", None)
            el_uc = _canon_element_for_biopy(raw_el if (isinstance(raw_el, str) and raw_el.strip()) else _biopy_get_element(at))
        name, fullname = _mk_atom_names(el_uc, idx)  # 合成名
        occ = float(getattr(at, "occupancy", 1.0))
        bf  = float(getattr(at, "bfactor", 0.0))
        alt = getattr(at, "altloc", " ") or " "
        try:
            serial = int(at.serial_number) if getattr(at, "serial_number", None) else idx
        except Exception:
            serial = idx
        sb.init_atom(name, (x, y, z), bf, occ, alt, fullname, serial, el_uc)

    structure = sb.get_structure()
    io = MMCIFIO(); io.set_structure(structure)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    io.save(out_path)
    return applied, aligned

# -------------------- 保存“受体整链 + 单个配体”（受体保留名；配体合成名） --------------------
def save_pair_cif_from_cif(
    cif_src_path: str, rec_chain_id: str, lig_chain_id: str, lig_res_id: Optional[int], out_path: str
) -> None:
    """
    保存“受体整链 + 单个 ligand 残基”的 pair CIF。
    ★ 受体：保留原子名，仅聚合物残基（无水/其他 HET）。
    ★ 配体：统一使用合成原子名（C01/O02…）。
    """
    st = _load_structure_cached(str(cif_src_path))
    model = st[0]

    rec_ch = _find_chain(model, rec_chain_id)
    if rec_ch is None:
        raise RuntimeError(f"Receptor chain not found: {rec_chain_id}")

    lig_ch = _find_chain(model, lig_chain_id)
    if lig_ch is None:
        raise RuntimeError(f"Ligand chain not found: {lig_chain_id}")
    lig_res = _pick_ligand_residue_flex(lig_ch, res_id=lig_res_id)
    if lig_res is None:
        raise RuntimeError(f"Ligand not found on chain={lig_chain_id} (res_id={lig_res_id})")

    sb = BPStructureBuilder()
    sb.init_structure("pair")
    sb.init_model(0)

    # 受体：保留原子名，且仅聚合物残基
    _copy_chain_to_builder(sb, rec_ch, dst_chain_id=str(rec_chain_id))

    # 配体：合成原子名
    sb.init_chain(str(lig_chain_id))
    hetflag, resseq, icode = lig_res.id
    sb.init_residue("LIG", "H_", int(resseq) if isinstance(resseq, int) else 1, " ")
    for idx, at in enumerate(lig_res, 1):
        x, y, z = (float(c) for c in at.get_coord())
        raw_el = getattr(at, "element", None)
        el_uc = _canon_element_for_biopy(raw_el if (isinstance(raw_el, str) and raw_el.strip()) else _biopy_get_element(at))
        name, fullname = _mk_atom_names(el_uc, idx)  # 合成名
        occ = float(getattr(at, "occupancy", 1.0))
        bf  = float(getattr(at, "bfactor", 0.0))
        alt = getattr(at, "altloc", " ") or " "
        try:
            serial = int(at.serial_number) if getattr(at, "serial_number", None) else idx
        except Exception:
            serial = idx
        sb.init_atom(name, (x, y, z), bf, occ, alt, fullname, serial, el_uc)

    structure = sb.get_structure()
    io = MMCIFIO(); io.set_structure(structure)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    io.save(out_path)

# -------------------- 保存“受体整链 + 预测元素配体”（受体保留名；配体合成名） --------------------
def save_pair_with_predicted_ligand_cif(
    cif_src_path: str,
    rec_chain_id: str,
    lig_chain_id: str,
    lig_res_id: Optional[int],
    new_elems: List[str],
    out_path: str,
) -> Tuple[int, int]:
    """
    保存“受体整链 + 预测元素替换后的 ligand”的 pair CIF。
    ★ 受体保留原子名，仅聚合物残基；配体使用合成原子名，仅替换 element。
    返回: (applied, aligned)
    """
    st = _load_structure_cached(str(cif_src_path))
    model = st[0]

    rec_ch = _find_chain(model, rec_chain_id)
    if rec_ch is None:
        raise RuntimeError(f"Receptor chain not found: {rec_chain_id}")

    lig_ch = _find_chain(model, lig_chain_id)
    if lig_ch is None:
        raise RuntimeError(f"Ligand chain not found: {lig_chain_id}")
    lig_res = _pick_ligand_residue_flex(lig_ch, res_id=lig_res_id)
    if lig_res is None:
        raise RuntimeError(f"Ligand not found on chain={lig_chain_id} (res_id={lig_res_id})")

    norm_tokens = _normalize_new_elems(new_elems)
    aligned = min(len(list(lig_res.get_atoms())), len(norm_tokens))

    sb = BPStructureBuilder()
    sb.init_structure("pair_pred")
    sb.init_model(0)

    # 受体：保留原子名，且仅聚合物残基
    _copy_chain_to_builder(sb, rec_ch, dst_chain_id=str(rec_chain_id))

    # 配体：合成原子名 + 元素替换
    sb.init_chain(str(lig_chain_id))
    hetflag, resseq, icode = lig_res.id
    sb.init_residue("LIG", "H_", int(resseq) if isinstance(resseq, int) else 1, " ")
    applied = 0
    for idx, at in enumerate(lig_res, 1):
        x, y, z = (float(c) for c in at.get_coord())
        if idx <= aligned and norm_tokens[idx - 1] is not None:
            el_uc = norm_tokens[idx - 1]; applied += 1
        else:
            raw_el = getattr(at, "element", None)
            el_uc = _canon_element_for_biopy(raw_el if (isinstance(raw_el, str) and raw_el.strip()) else _biopy_get_element(at))
        name, fullname = _mk_atom_names(el_uc, idx)  # 合成名
        occ = float(getattr(at, "occupancy", 1.0))
        bf  = float(getattr(at, "bfactor", 0.0))
        alt = getattr(at, "altloc", " ") or " "
        try:
            serial = int(at.serial_number) if getattr(at, "serial_number", None) else idx
        except Exception:
            serial = idx
        sb.init_atom(name, (x, y, z), bf, occ, alt, fullname, serial, el_uc)

    structure = sb.get_structure()
    io = MMCIFIO(); io.set_structure(structure)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    io.save(out_path)
    return applied, aligned




# def _new_empty_structure(structure_id: str = "lig") -> "StructureBuilder.Structure":
#     # 用 StructureBuilder 构造最简结构：Structure -> Model -> Chain
#     sb = StructureBuilder.StructureBuilder()
#     sb.init_structure(structure_id)
#     sb.init_model(0)
#     return sb

# def _canon_element_for_biopy(elem: str) -> str:
#     """Biopython 写 CIF/PDB 时用的大写元素码（1-2 个字符）。非法就用 'X'。"""
#     if not elem:
#         return "X"
#     s = str(elem).strip()
#     if not s:
#         return "X"
#     # 只保留字母，最多两位，转大写
#     s = ''.join(ch for ch in s if ch.isalpha())[:2].upper()
#     return s if s else "X"

# def _mk_atom_names(element_uc: str, idx: int) -> tuple[str, str]:
#     """
#     生成 Biopython 需要的 name / fullname（4 字符）。
#     规则：元素右对齐到 2 位，后跟两位序号。fullname 必须 4 字符。
#     """
#     # 元素两位右对齐
#     el2 = f"{element_uc:>2}"[:2]            # ' C' / 'FE'
#     # 两位递增编号（避免超长）
#     n2  = f"{(idx % 100):02d}"              # '01'..'99'
#     name = el2[0] + el2[1] + n2             # 长度恰好 4
#     fullname = name                         # fullname 也必须 4 字符
#     return name, fullname


# def save_single_ligand_cif(out_path, comp_id, chain_id, res_id, atoms_xyz, elements):
#     sb = StructureBuilder.StructureBuilder()
#     sb.init_structure("lig_single")
#     sb.init_model(0)
#     sb.init_chain(str(chain_id))

#     # HET 残基：hetfield 用 'H_'
#     resseq = int(res_id) if (res_id is not None and str(res_id).strip().isdigit()) else 1
#     sb.init_residue(str(comp_id), "H_", resseq, " ")

#     for idx, (xyz, elem) in enumerate(zip(atoms_xyz, elements), 1):
#         x, y, z = map(float, xyz)
#         element_uc = _canon_element_for_biopy(elem)          # -> 'FE' / 'C' / ...
#         name, fullname = _mk_atom_names(element_uc, idx)     # 4 字符

#         bfactor   = 0.0
#         occupancy = 1.0
#         altloc    = " "

#         # 位置参数顺序：name, coord, bfactor, occupancy, altloc, fullname, serial_number=None, element=None
#         sb.init_atom(name, (x, y, z), bfactor, occupancy, altloc, fullname, idx, element_uc)

#     structure = sb.get_structure()
#     io = MMCIFIO()
#     io.set_structure(structure)
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     io.save(out_path)



# def save_single_ligand_cif_from_cif(
#     cif_src_path: str, chain_id: str, comp_id: str, res_id: Optional[int], out_path: str
# ) -> None:
#     """
#     从源 CIF 中找指定 ligand，单独写出（Biopython版）。
#     """
#     _ensure_biopython()
#     parser = MMCIFParser(QUIET=True)
#     st = parser.get_structure("src", str(cif_src_path))
#     if st is None or len(st) == 0:
#         raise RuntimeError("empty structure")

#     model = st[0]
#     tgt_res = None
#     for ch in model:
#         if ch.id != chain_id:
#             continue
#         for res in ch:
#             hetflag, resseq, icode = res.id
#             is_het = (hetflag is not None) and (hetflag.strip() != "")
#             if not is_het:
#                 continue
#             if str(res.get_resname()).upper() != str(comp_id).upper():
#                 continue
#             if res_id is not None:
#                 if not isinstance(resseq, int) or int(resseq) != int(res_id):
#                     continue
#             tgt_res = res
#             break
#         if tgt_res is not None:
#             break
#     if tgt_res is None:
#         raise RuntimeError(f"ligand not found (chain={chain_id}, comp={comp_id}, res_id={res_id})")

#     atoms_xyz, elements = [], []
#     for at in tgt_res:
#         x, y, z = at.get_coord()
#         atoms_xyz.append((float(x), float(y), float(z)))
#         elements.append(_biopy_get_element(at))

#     save_single_ligand_cif(
#         out_path=out_path,
#         comp_id=comp_id,
#         chain_id=chain_id,
#         res_id=res_id,
#         atoms_xyz=atoms_xyz,
#         elements=elements,
#     )


# def save_ligand_elements_replaced_cif(
#     cif_src_path: str, chain_id: str, comp_id: str, res_id: Optional[int],
#     new_elems: List[str], out_path: str
# ) -> Tuple[int, int]:
#     parser = MMCIFParser(QUIET=True)
#     st = parser.get_structure("src", str(cif_src_path))
#     if st is None or len(st) == 0:
#         raise RuntimeError("Empty structure")

#     model = st[0]
#     tgt = None
#     for ch in model:
#         if ch.id != str(chain_id):
#             continue
#         for res in ch:
#             hetflag, resseq, icode = res.id
#             is_het = (hetflag is not None) and (hetflag.strip() != "")
#             if not is_het:
#                 continue
#             if str(res.get_resname()).upper() != str(comp_id).upper():
#                 continue
#             if res_id is not None and (not isinstance(resseq, int) or int(resseq) != int(res_id)):
#                 continue
#             tgt = res
#             break
#         if tgt is not None:
#             break
#     if tgt is None:
#         raise RuntimeError(f"Ligand not found: chain={chain_id}, comp={comp_id}, res_id={res_id}")

#     # 规范化 token（None/空/RARE/UNK/* -> 不替换）
#     norm_tokens: List[Optional[str]] = []
#     for t in new_elems:
#         if t is None:
#             norm_tokens.append(None)
#         else:
#             s = str(t).strip()
#             if not s or s.upper() in ("RARE", "UNK", "*"):
#                 norm_tokens.append(None)
#             else:
#                 norm_tokens.append(_canon_element_for_biopy(s))  # 直接转成大写 1-2 位
#     aligned = min(len(tgt), len(norm_tokens))

#     sb = StructureBuilder.StructureBuilder()
#     sb.init_structure("lig_mut")
#     sb.init_model(0)
#     sb.init_chain(str(chain_id))
#     resseq = tgt.id[1] if isinstance(tgt.id[1], int) else 1
#     sb.init_residue(str(comp_id), "H_", int(resseq), " ")

#     applied = 0
#     for idx, at in enumerate(tgt, 1):
#         x, y, z = (float(c) for c in at.get_coord())

#         if idx <= aligned and norm_tokens[idx-1] is not None:
#             element_uc = norm_tokens[idx-1]
#             applied += 1
#         else:
#             # 从原原子里还原 element（尽量 1-2 位大写）
#             raw_el = getattr(at, "element", None)
#             raw_el = str(raw_el) if raw_el is not None else ""
#             element_uc = _canon_element_for_biopy(raw_el)

#         name, fullname = _mk_atom_names(element_uc, idx)

#         occupancy = float(getattr(at, "occupancy", 1.0))
#         bfactor   = float(getattr(at, "bfactor", 0.0))
#         altloc    = " "

#         sb.init_atom(name, (x, y, z), bfactor, occupancy, altloc, fullname, idx, element_uc)

#     structure = sb.get_structure()
#     io = MMCIFIO()
#     io.set_structure(structure)
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     io.save(out_path)

#     return applied, aligned


# def save_pair_cif_from_cif(
#     cif_src_path: str,
#     rec_chain_id: str,
#     lig_chain_id: str,
#     lig_comp_id: str,
#     lig_res_id: Optional[int],
#     out_path: str,
# ) -> None:
#     _ensure_biopython()
#     parser = MMCIFParser(QUIET=True)
#     st = parser.get_structure("src", str(cif_src_path))
#     if st is None or len(st) == 0:
#         raise RuntimeError("Empty structure")
#     model = st[0]

#     # 找 receptor 链
#     rec_ch = None
#     for ch in model:
#         if ch.id == str(rec_chain_id):
#             rec_ch = ch
#             break
#     if rec_ch is None:
#         raise RuntimeError(f"Receptor chain not found: {rec_chain_id}")

#     # 找 ligand 残基
#     lig_res = None
#     for ch in model:
#         if ch.id != str(lig_chain_id):
#             continue
#         for res in ch:
#             hetflag, resseq, icode = res.id
#             is_het = (hetflag is not None) and (hetflag.strip() != "")
#             if not is_het:
#                 continue
#             if str(res.get_resname()).upper() != str(lig_comp_id).upper():
#                 continue
#             if lig_res_id is not None and (not isinstance(resseq, int) or int(resseq) != int(lig_res_id)):
#                 continue
#             lig_res = res
#             break
#         if lig_res is not None:
#             break
#     if lig_res is None:
#         raise RuntimeError(f"Ligand not found: chain={lig_chain_id}, comp={lig_comp_id}, res_id={lig_res_id}")

#     # 写出：受体整链 + ligand 单残基
#     sb = StructureBuilder.StructureBuilder()
#     sb.init_structure("pair")
#     sb.init_model(0)

#     # 受体链（原样复制）
#     sb.init_chain(str(rec_chain_id))
#     for res in rec_ch:
#         hetflag, resseq, icode = res.id
#         sb.init_residue(str(res.get_resname()), (hetflag if hetflag else " "), int(resseq) if isinstance(resseq,int) else 1, (icode if icode else " "))
#         for at in res:
#             x, y, z = (float(c) for c in at.get_coord())
#             el = _canon_element_for_biopy(at.element if getattr(at, "element", None) else _biopy_get_element(at))
#             name, fullname = _mk_atom_names(el, at.serial_number if getattr(at, "serial_number", None) else 1)
#             occ = float(getattr(at, "occupancy", 1.0)); bf = float(getattr(at, "bfactor", 0.0))
#             sb.init_atom(name, (x,y,z), bf, occ, " ", fullname, None, el)

#     # Ligand（只复制该残基）
#     sb.init_chain(str(lig_chain_id))
#     hetflag, resseq, icode = lig_res.id
#     sb.init_residue(str(lig_comp_id), "H_", int(resseq) if isinstance(resseq,int) else 1, " ")
#     for idx, at in enumerate(lig_res, 1):
#         x, y, z = (float(c) for c in at.get_coord())
#         el = _canon_element_for_biopy(at.element if getattr(at, "element", None) else _biopy_get_element(at))
#         name, fullname = _mk_atom_names(el, idx)
#         occ = float(getattr(at, "occupancy", 1.0)); bf = float(getattr(at, "bfactor", 0.0))
#         sb.init_atom(name, (x,y,z), bf, occ, " ", fullname, idx, el)

#     structure = sb.get_structure()
#     io = MMCIFIO(); io.set_structure(structure)
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     io.save(out_path)


# def save_pair_with_predicted_ligand_cif(
#     cif_src_path: str,
#     rec_chain_id: str,
#     lig_chain_id: str,
#     lig_comp_id: str,
#     lig_res_id: Optional[int],
#     new_elems: List[str],
#     out_path: str,
# ) -> Tuple[int, int]:
#     """
#     返回: (applied, aligned) —— 成功替换的原子数 / 对齐的配体原子数
#     """
#     _ensure_biopython()
#     parser = MMCIFParser(QUIET=True)
#     st = parser.get_structure("src", str(cif_src_path))
#     if st is None or len(st) == 0:
#         raise RuntimeError("Empty structure")
#     model = st[0]

#     # receptor 链
#     rec_ch = None
#     for ch in model:
#         if ch.id == str(rec_chain_id):
#             rec_ch = ch
#             break
#     if rec_ch is None:
#         raise RuntimeError(f"Receptor chain not found: {rec_chain_id}")

#     # ligand 残基
#     lig_res = None
#     for ch in model:
#         if ch.id != str(lig_chain_id):
#             continue
#         for res in ch:
#             hetflag, resseq, icode = res.id
#             is_het = (hetflag is not None) and (hetflag.strip() != "")
#             if not is_het:
#                 continue
#             if str(res.get_resname()).upper() != str(lig_comp_id).upper():
#                 continue
#             if lig_res_id is not None and (not isinstance(resseq, int) or int(resseq) != int(lig_res_id)):
#                 continue
#             lig_res = res
#             break
#         if lig_res is not None:
#             break
#     if lig_res is None:
#         raise RuntimeError(f"Ligand not found: chain={lig_chain_id}, comp={lig_comp_id}, res_id={lig_res_id}")

#     # 规范化新元素
#     norm_tokens: List[Optional[str]] = []
#     for t in new_elems:
#         if t is None:
#             norm_tokens.append(None)
#         else:
#             s = str(t).strip()
#             if not s or s.upper() in ("RARE", "UNK", "*"):
#                 norm_tokens.append(None)
#             else:
#                 norm_tokens.append(_canon_element_for_biopy(s))  # 1-2 位大写
#     aligned = min(len(lig_res), len(norm_tokens))

#     # 写出 pair（受体原样 + 替换元素后的 ligand）
#     sb = StructureBuilder.StructureBuilder()
#     sb.init_structure("pair_pred")
#     sb.init_model(0)

#     # 受体
#     sb.init_chain(str(rec_chain_id))
#     for res in rec_ch:
#         hetflag, resseq, icode = res.id
#         sb.init_residue(str(res.get_resname()), (hetflag if hetflag else " "), int(resseq) if isinstance(resseq,int) else 1, (icode if icode else " "))
#         for at in res:
#             x, y, z = (float(c) for c in at.get_coord())
#             el = _canon_element_for_biopy(at.element if getattr(at, "element", None) else _biopy_get_element(at))
#             name, fullname = _mk_atom_names(el, at.serial_number if getattr(at, "serial_number", None) else 1)
#             occ = float(getattr(at, "occupancy", 1.0)); bf = float(getattr(at, "bfactor", 0.0))
#             sb.init_atom(name, (x,y,z), bf, occ, " ", fullname, None, el)

#     # 配体（按 new_elems 替换）
#     sb.init_chain(str(lig_chain_id))
#     hetflag, resseq, icode = lig_res.id
#     sb.init_residue(str(lig_comp_id), "H_", int(resseq) if isinstance(resseq,int) else 1, " ")
#     applied = 0
#     for idx, at in enumerate(lig_res, 1):
#         x, y, z = (float(c) for c in at.get_coord())
#         if idx <= aligned and norm_tokens[idx-1] is not None:
#             el = norm_tokens[idx-1]; applied += 1
#         else:
#             raw_el = getattr(at, "element", None)
#             raw_el = str(raw_el) if raw_el is not None else ""
#             el = _canon_element_for_biopy(raw_el)
#         name, fullname = _mk_atom_names(el, idx)
#         occ = float(getattr(at, "occupancy", 1.0)); bf = float(getattr(at, "bfactor", 0.0))
#         sb.init_atom(name, (x,y,z), bf, occ, " ", fullname, idx, el)

#     structure = sb.get_structure()
#     io = MMCIFIO(); io.set_structure(structure)
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     io.save(out_path)
#     return applied, aligned

if __name__ == "__main":
    pass