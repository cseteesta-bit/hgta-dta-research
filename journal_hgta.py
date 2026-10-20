#!/usr/bin/env python3
"""Leakage-free HGTA training for sequence-based drug--target affinity.

The official test set is never evaluated until training and checkpoint selection
have finished. Protein language-model weights are frozen and cached; the trainable
model performs atom--residue rather than graph--protein-token interaction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, rdBase
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version

SMI, SEQ, Y = "compound_iso_smiles", "target_sequence", "affinity"


@dataclass
class Config:
    train_csv: str = "/home/cse1000/Downloads/HGTA/data/davis_train.csv"
    test_csv: str = "/home/cse1000/Downloads/HGTA/data/davis_test.csv"
    out_dir: str = "results_hgta_journal"
    protein_model: str = "facebook/esm2_t12_35M_UR50D"
    seeds: Tuple[int, ...] = (4221, 17, 71, 314, 2025)
    val_fraction: float = 0.10
    val_split: str = "pair"  # pair, drug, target, both
    batch_size: int = 16
    epochs: int = 300
    patience: int = 35
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    dropout: float = 0.20
    grad_clip: float = 1.0
    num_workers: int = 0
    hidden_dim: int = 256
    gnn_layers: int = 3
    attention_heads: int = 8
    attention_blocks: int = 2
    max_protein_tokens: int = 512
    esm_chunk_residues: int = 1022
    cache_dtype: str = "float16"
    use_amp: bool = True
    deterministic: bool = True


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=False)


def canonical_smiles(s: str) -> str | None:
    mol = Chem.MolFromSmiles(str(s))
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_and_audit(path: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (SMI, SEQ, Y) if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    df = df[[SMI, SEQ, Y]].copy()
    df[SEQ] = df[SEQ].astype(str).str.strip().str.upper()
    df[Y] = pd.to_numeric(df[Y], errors="coerce")
    if df[Y].isna().any() or not np.isfinite(df[Y]).all():
        raise ValueError(f"{name}: non-finite affinity values found")
    df["canonical_smiles"] = [canonical_smiles(x) for x in df[SMI]]
    bad = int(df["canonical_smiles"].isna().sum())
    if bad:
        raise ValueError(f"{name}: {bad} invalid SMILES; correct them rather than using zero graphs")
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")
    bad_seq = [i for i, s in enumerate(df[SEQ]) if not s or not set(s) <= valid_aa]
    if bad_seq:
        raise ValueError(f"{name}: {len(bad_seq)} empty/invalid protein sequences")
    sizes = df.groupby(["canonical_smiles", SEQ], dropna=False).size()
    duplicate_groups = int((sizes > 1).sum())
    if duplicate_groups:
        conflicting = int(df.groupby(["canonical_smiles", SEQ], dropna=False)[Y].nunique().gt(1).sum())
        print(f"{name}: aggregating {duplicate_groups} duplicate pair groups with the median "
              f"({conflicting} have differing labels).")
        df = (df.groupby(["canonical_smiles", SEQ], as_index=False, dropna=False)
                .agg(affinity=(Y, "median"), affinity_mean=(Y, "mean"),
                     affinity_std=(Y, "std"), affinity_min=(Y, "min"),
                     affinity_max=(Y, "max"), measurement_count=(Y, "size")))
        df[SMI] = df["canonical_smiles"]
        df["affinity_std"] = df["affinity_std"].fillna(0.0)
    else:
        df["affinity_mean"] = df[Y]
        df["affinity_std"] = 0.0
        df["affinity_min"] = df[Y]
        df["affinity_max"] = df[Y]
        df["measurement_count"] = 1
    return df


def overlap_report(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, int]:
    td, vd = set(train.canonical_smiles), set(test.canonical_smiles)
    tp, vp = set(train[SEQ]), set(test[SEQ])
    train_pairs = set(zip(train.canonical_smiles, train[SEQ]))
    test_pairs = set(zip(test.canonical_smiles, test[SEQ]))
    return {
        "train_rows": len(train), "test_rows": len(test),
        "shared_drugs": len(td & vd), "shared_targets": len(tp & vp),
        "duplicate_pairs_across_train_test": len(train_pairs & test_pairs),
    }


def split_train_val(df: pd.DataFrame, fraction: float, mode: str, seed: int):
    if not 0 < fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    rng = np.random.default_rng(seed)
    if mode == "pair":
        order = rng.permutation(len(df)); n = max(1, round(len(df) * fraction))
        va = np.zeros(len(df), bool); va[order[:n]] = True; tr = ~va
    elif mode in {"drug", "target"}:
        col = "canonical_smiles" if mode == "drug" else SEQ
        groups = np.array(sorted(df[col].unique()), dtype=object); rng.shuffle(groups)
        chosen = set(groups[:max(1, round(len(groups) * fraction))])
        va = df[col].isin(chosen).to_numpy(); tr = ~va
    elif mode == "both":
        drugs = np.array(sorted(df.canonical_smiles.unique()), dtype=object)
        prots = np.array(sorted(df[SEQ].unique()), dtype=object)
        rng.shuffle(drugs); rng.shuffle(prots)
        vd = set(drugs[:max(1, round(len(drugs) * math.sqrt(fraction)))])
        vp = set(prots[:max(1, round(len(prots) * math.sqrt(fraction)))])
        is_d, is_p = df.canonical_smiles.isin(vd).to_numpy(), df[SEQ].isin(vp).to_numpy()
        va, tr = is_d & is_p, ~is_d & ~is_p  # cross-partition rows are intentionally excluded
    else:
        raise ValueError("val_split must be pair, drug, target, or both")
    if va.sum() < 2 or tr.sum() < 2:
        raise ValueError(f"{mode} split produced too few rows: train={tr.sum()}, val={va.sum()}")
    return df.loc[tr].reset_index(drop=True), df.loc[va].reset_index(drop=True)


def atom_features(a: Chem.Atom) -> List[float]:
    hyb = a.GetHybridization()
    vals = [
        a.GetAtomicNum() / 100, a.GetDegree() / 6, a.GetFormalCharge() / 5,
        a.GetTotalNumHs() / 4, a.GetTotalValence() / 8, a.GetMass() / 250,
        float(a.GetIsAromatic()), float(a.IsInRing()),
        float(hyb == Chem.HybridizationType.SP), float(hyb == Chem.HybridizationType.SP2),
        float(hyb == Chem.HybridizationType.SP3), float(a.GetChiralTag()) / 3,
    ]
    return [float(x) for x in vals]


def bond_features(b: Chem.Bond) -> List[float]:
    t = b.GetBondType()
    return [float(t == Chem.BondType.SINGLE), float(t == Chem.BondType.DOUBLE),
            float(t == Chem.BondType.TRIPLE), float(t == Chem.BondType.AROMATIC),
            float(b.GetIsConjugated()), float(b.IsInRing()), float(b.GetStereo()) / 6]


def mol_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    edges, attrs = [], []
    for b in mol.GetBonds():
        i, j, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)
        edges.extend(((i, j), (j, i))); attrs.extend((e, e))
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).T.contiguous()
        ea = torch.tensor(attrs, dtype=torch.float32)
    else:
        ei, ea = torch.empty((2, 0), dtype=torch.long), torch.empty((0, 7))
    return Data(x=x, edge_index=ei, edge_attr=ea)


class ProteinCache:
    def __init__(self, cfg: Config, device: torch.device, cache_dir: Path):
        self.cfg, self.device, self.cache_dir = cfg, device, cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = None; self.model = None

    def _path(self, seq: str) -> Path:
        tag = hashlib.sha256((self.cfg.protein_model + "\0" + seq).encode()).hexdigest()
        return self.cache_dir / f"{tag}.pt"

    @torch.inference_mode()
    def get(self, seq: str) -> torch.Tensor:
        path = self._path(seq)
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.protein_model)
            self.model = AutoModel.from_pretrained(self.cfg.protein_model, add_pooling_layer=False).to(self.device).eval()
        pieces = []
        amp = self.cfg.use_amp and self.device.type == "cuda"
        for start in range(0, len(seq), self.cfg.esm_chunk_residues):
            chunk = seq[start:start + self.cfg.esm_chunk_residues]
            tok = self.tokenizer(chunk, return_tensors="pt", add_special_tokens=True).to(self.device)
            with torch.amp.autocast(device_type=self.device.type, enabled=amp):
                h = self.model(**tok).last_hidden_state[0]
            special = self.tokenizer.get_special_tokens_mask(tok["input_ids"][0].tolist(), already_has_special_tokens=True)
            keep = torch.tensor([not x for x in special], dtype=torch.bool, device=h.device)
            pieces.append(h[keep].float().cpu())
        emb = torch.cat(pieces)
        if len(emb) > self.cfg.max_protein_tokens:
            emb = F.adaptive_avg_pool1d(emb.T.unsqueeze(0), self.cfg.max_protein_tokens)[0].T
        if self.cfg.cache_dtype == "float16": emb = emb.half()
        torch.save(emb, path)
        return emb


class DTADataset(Dataset):
    def __init__(self, df: pd.DataFrame): self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return i, r.canonical_smiles, r[SEQ], float(r[Y])


def collator(cache: ProteinCache):
    def fn(rows):
        idx, smi, seq, y = zip(*rows)
        proteins = [cache.get(s).float() for s in seq]
        length = max(map(len, proteins)); dim = proteins[0].shape[-1]
        p = torch.zeros(len(rows), length, dim); pm = torch.zeros(len(rows), length, dtype=torch.bool)
        for i, e in enumerate(proteins): p[i, :len(e)] = e; pm[i, :len(e)] = True
        return list(idx), Batch.from_data_list([mol_graph(s) for s in smi]), p, pm, torch.tensor(y)
    return fn


class FusionBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.dp = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.pd = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.nd, self.np = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fd = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d, d))
        self.fp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d, d))
        self.nd2, self.np2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)

    def forward(self, d, dm, p, pm):
        do = self.dp(d, p, p, key_padding_mask=~pm, need_weights=False)[0]
        po = self.pd(p, d, d, key_padding_mask=~dm, need_weights=False)[0]
        d, p = self.nd(d + self.drop(do)), self.np(p + self.drop(po))
        return self.nd2(d + self.drop(self.fd(d))), self.np2(p + self.drop(self.fp(p)))


class JournalHGTA(nn.Module):
    def __init__(self, cfg: Config, protein_dim: int):
        super().__init__(); d = cfg.hidden_dim
        self.atom_in, self.edge_in = nn.Linear(12, d), nn.Linear(7, d)
        self.gnn, self.gnorm = nn.ModuleList(), nn.ModuleList()
        for _ in range(cfg.gnn_layers):
            mlp = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(2*d, d))
            self.gnn.append(GINEConv(mlp, edge_dim=d)); self.gnorm.append(nn.LayerNorm(d))
        self.protein_in = nn.Sequential(nn.Linear(protein_dim, d), nn.LayerNorm(d))
        self.fusion = nn.ModuleList([FusionBlock(d, cfg.attention_heads, cfg.dropout) for _ in range(cfg.attention_blocks)])
        self.head = nn.Sequential(nn.Linear(4*d, 2*d), nn.LayerNorm(2*d), nn.GELU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(2*d, d//2), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d//2, 1))

    def forward(self, g: Batch, p: torch.Tensor, pm: torch.Tensor):
        x, e = self.atom_in(g.x), self.edge_in(g.edge_attr)
        for conv, norm in zip(self.gnn, self.gnorm): x = norm(x + conv(x, g.edge_index, e))
        d, dm = to_dense_batch(x, g.batch); p = self.protein_in(p)
        for block in self.fusion: d, p = block(d, dm, p, pm)
        dv = (d * dm.unsqueeze(-1)).sum(1) / dm.sum(1, keepdim=True).clamp_min(1)
        pv = (p * pm.unsqueeze(-1)).sum(1) / pm.sum(1, keepdim=True).clamp_min(1)
        return self.head(torch.cat((dv, pv, dv*pv, (dv-pv).abs()), -1)).squeeze(-1)


def concordance(y: np.ndarray, pred: np.ndarray) -> float:
    # Exact O(n log n) implementation with target ties excluded and prediction ties worth 0.5.
    order = np.argsort(y, kind="mergesort"); y, pred = y[order], pred[order]
    ranks = np.searchsorted(np.unique(pred), pred) + 1; bit = np.zeros(ranks.max()+1, np.int64)
    def query(i):
        s = 0
        while i: s += bit[i]; i -= i & -i
        return s
    def add(i):
        while i < len(bit): bit[i] += 1; i += i & -i
    score = pairs = previous = 0.0; i = 0
    while i < len(y):
        j = i
        while j < len(y) and y[j] == y[i]: j += 1
        for k in range(i, j):
            less, upto = query(ranks[k]-1), query(ranks[k])
            score += less + .5*(upto-less); pairs += previous
        for k in range(i, j): add(ranks[k]); previous += 1
        i = j
    return float(score/pairs) if pairs else float("nan")


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    err = p-y; mse = float(np.mean(err**2)); ss = float(np.sum((y-y.mean())**2))
    pearson = float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else float("nan")
    return {"MSE": mse, "RMSE": math.sqrt(mse), "MAE": float(np.mean(abs(err))),
            "R2": float(1-np.sum(err**2)/ss) if ss else float("nan"),
            "Pearson": pearson, "CI": concordance(y, p),
            "Within0.5": float(100*np.mean(abs(err) <= .5)), "Within1.0": float(100*np.mean(abs(err) <= 1))}


@torch.inference_mode()
def predict(model, loader, device, amp):
    model.eval(); ps, ys, ids = [], [], []
    for idx, g, p, pm, y in loader:
        g, p, pm = g.to(device), p.to(device), pm.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=amp): out = model(g, p, pm)
        ids.extend(idx); ps.extend(out.float().cpu().tolist()); ys.extend(y.tolist())
    return np.asarray(ids), np.asarray(ys), np.asarray(ps)


def train_one(cfg, seed, train_df, val_df, test_df, cache, protein_dim, device, root: Path):
    seed_all(seed, cfg.deterministic); run = root / f"seed_{seed}"; run.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed); fn = collator(cache)
    train_ld = DataLoader(DTADataset(train_df), cfg.batch_size, shuffle=True, generator=gen, collate_fn=fn,
                          num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    val_ld = DataLoader(DTADataset(val_df), cfg.batch_size, shuffle=False, collate_fn=fn, num_workers=cfg.num_workers)
    test_ld = DataLoader(DTADataset(test_df), cfg.batch_size, shuffle=False, collate_fn=fn, num_workers=cfg.num_workers)
    model = JournalHGTA(cfg, protein_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=.5, patience=10, min_lr=1e-6)
    amp = cfg.use_amp and device.type == "cuda"; scaler = torch.amp.GradScaler(enabled=amp)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, cfg.epochs+1):
        model.train(); losses, count = 0.0, 0
        for _, g, p, pm, y in train_ld:
            g, p, pm, y = g.to(device), p.to(device), pm.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp): loss = F.mse_loss(model(g, p, pm), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); scaler.step(opt); scaler.update()
            losses += loss.item()*len(y); count += len(y)
        _, vy, vp = predict(model, val_ld, device, amp); vm = metrics(vy, vp); sched.step(vm["MSE"])
        row = {"epoch": epoch, "train_mse": losses/count, "lr": opt.param_groups[0]["lr"], **{f"val_{k}":v for k,v in vm.items()}}
        history.append(row)
        if vm["MSE"] < best - 1e-8:
            best, stale = vm["MSE"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": vm}, run/"best.pt")
        else: stale += 1
        if epoch == 1 or epoch % 10 == 0: print(f"seed={seed} epoch={epoch} train={losses/count:.4f} val={vm['MSE']:.4f}")
        if stale >= cfg.patience: break
    pd.DataFrame(history).to_csv(run/"history.csv", index=False)
    ckpt = torch.load(run/"best.pt", map_location=device, weights_only=True); model.load_state_dict(ckpt["model"])
    # First and only test evaluation for this seed.
    ids, ty, tp = predict(model, test_ld, device, amp); tm = metrics(ty, tp)
    out = test_df.iloc[ids].copy(); out["prediction"] = tp; out["absolute_error"] = abs(tp-ty)
    out.to_csv(run/"test_predictions.csv", index=False)
    with open(run/"metrics.json", "w") as f: json.dump({"seed": seed, "best_epoch": ckpt["epoch"], "validation": ckpt["val_metrics"], "test": tm}, f, indent=2)
    return tm


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--train-csv", default=Config.train_csv); ap.add_argument("--test-csv", default=Config.test_csv)
    ap.add_argument("--out-dir", default="results_hgta_journal"); ap.add_argument("--seeds", nargs="+", type=int, default=[4221,17,71,314,2025])
    ap.add_argument("--val-split", choices=["pair","drug","target","both"], default="pair")
    ap.add_argument("--val-fraction", type=float, default=.1); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--patience", type=int, default=35)
    args = ap.parse_args(); cfg = Config(args.train_csv, args.test_csv, args.out_dir, seeds=tuple(args.seeds),
                                         val_split=args.val_split, val_fraction=args.val_fraction,
                                         epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
    root = Path(cfg.out_dir); root.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = load_and_audit(cfg.train_csv, "train"), load_and_audit(cfg.test_csv, "test")
    audit_before = overlap_report(train, test)
    test_pairs = set(zip(test["canonical_smiles"], test[SEQ]))
    leaked = train[["canonical_smiles", SEQ]].apply(tuple, axis=1).isin(test_pairs)
    if leaked.any():
        excluded = train.loc[leaked].copy()
        excluded["exclusion_reason"] = "canonical_pair_also_present_in_test"
        excluded.to_csv(root / "excluded_train_test_overlaps.csv", index=False)
        train = train.loc[~leaked].reset_index(drop=True)
        print(f"Removed {len(excluded)} leaked canonical pairs from training; test set was preserved.")
    audit_after = overlap_report(train, test)
    if audit_after["duplicate_pairs_across_train_test"]:
        raise RuntimeError(f"failed to remove train/test leakage: {audit_after}")
    train, val = split_train_val(train, cfg.val_fraction, cfg.val_split, cfg.seeds[0])
    cache = ProteinCache(cfg, device, root/"protein_cache")
    first = cache.get(pd.concat([train[SEQ], val[SEQ], test[SEQ]]).iloc[0]); protein_dim = first.shape[1]
    manifest = {"config": asdict(cfg), "audit_before_exclusion": audit_before, "audit_after_exclusion": audit_after, "actual_train_rows":len(train), "actual_val_rows":len(val),
                "sha256":{"train_csv":sha256_file(cfg.train_csv), "test_csv":sha256_file(cfg.test_csv)},
                "environment":{"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,
                               "cuda":torch.version.cuda,"rdkit":rdBase.rdkitVersion,"transformers":transformers_version,
                               "device":str(device)}, "created_unix":time.time()}
    with open(root/"manifest.json", "w") as f: json.dump(manifest, f, indent=2)
    results = [train_one(cfg, s, train, val, test, cache, protein_dim, device, root) for s in cfg.seeds]
    frame = pd.DataFrame(results); summary = {c:{"mean":float(frame[c].mean()),"std":float(frame[c].std(ddof=1))} for c in frame}
    frame.insert(0, "seed", cfg.seeds); frame.to_csv(root/"all_seed_metrics.csv", index=False)
    with open(root/"summary.json", "w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
