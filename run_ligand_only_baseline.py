#!/usr/bin/env python3
"""Frozen-split ligand-only DTA control baseline.

This is intentionally a negative/control model: it receives an RDKit Morgan
fingerprint of the ligand and never receives a protein sequence.  It is useful
for detecting how much performance a pair split can obtain from drug identity
or chemistry alone.  Validation selects checkpoints; test labels are read only
after selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, TensorDataset


SMILES_CANDIDATES = ("canonical_smiles", "compound_iso_smiles", "smiles")
TARGET_CANDIDATES = ("affinity", "label", "y")


def choose_column(frame, candidates, description):
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(f"Missing {description}; expected one of {candidates}, got {list(frame.columns)}")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fingerprint(smiles, n_bits, radius):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    bits = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    out = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bits, out)
    return out


def metrics(y, p):
    e = p - y; mse = float(np.mean(e * e)); denom = float(np.sum((y - y.mean()) ** 2))
    pearson = float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else float("nan")
    return {"MSE": mse, "RMSE": math.sqrt(mse), "MAE": float(np.mean(np.abs(e))),
            "R2": float(1 - np.sum(e * e) / denom) if denom else float("nan"),
            "Pearson": pearson,
            "Within0.5": float(100 * np.mean(np.abs(e) <= 0.5)),
            "Within1.0": float(100 * np.mean(np.abs(e) <= 1.0))}


class FingerprintMLP(nn.Module):
    def __init__(self, n_bits, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.inference_mode()
def predict(model, loader, device, y_mean, y_std):
    model.eval(); ys, ps = [], []
    for x, y in loader:
        # Model outputs are in training-label standardized units; convert them
        # back before comparing to raw validation/test affinity labels.
        ps.append((model(x.to(device)).cpu().numpy() * y_std) + y_mean); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def run_seed(seed, train_x, train_y, val_x, val_y, test_x, test_y, test_frame, args, root, device):
    seed_everything(seed)
    run = root / f"seed_{seed}"; run.mkdir()
    # Standardize features using training values only.
    mean = train_x.mean(0); std = train_x.std(0); std[std < 1e-6] = 1.0
    tx, vx, qx = (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std
    # Standardize regression target using training values only.
    y_mean, y_std = float(train_y.mean()), float(train_y.std())
    if y_std < 1e-8: raise ValueError("Training labels have zero variance")
    ty = (train_y - y_mean) / y_std
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty)),
                              batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(vx), torch.from_numpy(val_y)), batch_size=1024)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(qx), torch.from_numpy(test_y)), batch_size=1024)
    model = FingerprintMLP(args.n_bits, args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True); loss = nn.functional.mse_loss(model(x), y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item() * len(y); n += len(y)
        vy, vp = predict(model, val_loader, device, y_mean, y_std); vm = metrics(vy, vp)
        scheduler.step(vm["MSE"])
        history.append({"epoch": epoch, "train_mse_standardized": total / n, "lr": opt.param_groups[0]["lr"],
                        **{f"val_{k}": v for k, v in vm.items()}})
        if vm["MSE"] < best - 1e-8:
            best, stale = vm["MSE"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation": vm}, run / "best.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"seed={seed} epoch={epoch} train={total/n:.4f} val={vm['MSE']:.4f}")
        if stale >= args.patience: break
    pd.DataFrame(history).to_csv(run / "history.csv", index=False)
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    # The test labels are only evaluated after validation-based checkpoint selection.
    test_y_out, test_pred = predict(model, test_loader, device, y_mean, y_std); tm = metrics(test_y_out, test_pred)
    out = test_frame.copy(); out["prediction"] = test_pred; out["absolute_error"] = np.abs(test_y_out - test_pred)
    out.to_csv(run / "test_predictions.csv", index=False)
    (run / "metrics.json").write_text(json.dumps({"seed": seed, "best_epoch": checkpoint["epoch"],
        "validation": checkpoint["validation"], "test": tm}, indent=2))
    return tm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Directory containing locked train.csv, validation.csv, test.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[4221, 17, 71, 314, 2025])
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5); ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--n-bits", type=int, default=2048); ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--hidden-dim", type=int, default=512)
    args = ap.parse_args()
    root = Path(args.out_dir)
    if root.exists(): raise ValueError(f"Output directory exists: {root}. Choose a new name.")
    data = Path(args.data_dir)
    paths = {name: data / f"{name}.csv" for name in ("train", "validation", "test")}
    if any(not p.exists() for p in paths.values()): raise FileNotFoundError(paths)
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    smi = choose_column(frames["train"], SMILES_CANDIDATES, "SMILES column")
    label = choose_column(frames["train"], TARGET_CANDIDATES, "affinity label column")
    for name, frame in frames.items():
        if smi not in frame or label not in frame: raise ValueError(f"{name} has inconsistent columns")
        frame[label] = pd.to_numeric(frame[label], errors="raise")
        if not np.isfinite(frame[label]).all(): raise ValueError(f"{name} has non-finite labels")
    cache = {}
    def make_x(frame):
        return np.stack([cache.setdefault(s, fingerprint(s, args.n_bits, args.radius)) for s in frame[smi].astype(str)])
    xs = {name: make_x(frame) for name, frame in frames.items()}
    ys = {name: frames[name][label].to_numpy(np.float32) for name in frames}
    root.mkdir(parents=True)
    manifest = {"model": "ligand_only_morgan_mlp", "input": "canonical molecular fingerprint only; no protein sequence",
                "data_sha256": {name: sha256(path) for name, path in paths.items()}, "args": vars(args),
                "n_unique_ligands": len(cache)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [run_seed(seed, xs["train"], ys["train"], xs["validation"], ys["validation"],
                        xs["test"], ys["test"], frames["test"], args, root, device) for seed in args.seeds]
    table = pd.DataFrame(results); table.insert(0, "seed", args.seeds); table.to_csv(root / "all_seed_metrics.csv", index=False)
    summary = {col: {"mean": float(table[col].mean()), "std": float(table[col].std(ddof=1))} for col in table.columns if col != "seed"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
