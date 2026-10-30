#!/usr/bin/env python3
"""Validation-selected DeepDTA-style sequence-CNN comparator.

This is a source-controlled *DeepDTA-style* baseline, not a claim that it is
bit-identical to a historical Keras implementation.  It reads the same raw
GraphDTA CSV folds, selects checkpoints only with a seeded validation split,
and evaluates the supplied test fold only after selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


SMILES = "#%()+-./0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ\\abcdefghijklmnopqrstuvwxyz"
PROTEIN = "ABCDEFGHIKLMNOPQRSTUVWXYZ"


def args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=["kiba", "davis"], required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=4221)
    p.add_argument("--val-fraction", type=float, default=.20)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--max-drug-length", type=int, default=100)
    p.add_argument("--max-protein-length", type=int, default=1000)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--finalize-only", action="store_true",
                   help="Write metrics from an existing best.pt without retraining.")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SequenceDataset(Dataset):
    def __init__(self, frame, drug_length, protein_length):
        required = {"compound_iso_smiles", "target_sequence", "affinity"}
        if required - set(frame): raise ValueError(f"Missing columns {sorted(required-set(frame))}")
        self.drugs = frame.compound_iso_smiles.astype(str).tolist()
        self.proteins = frame.target_sequence.astype(str).str.upper().tolist()
        self.y = frame.affinity.astype(float).to_numpy(np.float32)
        self.drug_length, self.protein_length = drug_length, protein_length
        self.drug_vocab = {c: i + 1 for i, c in enumerate(SMILES)}
        self.protein_vocab = {c: i + 1 for i, c in enumerate(PROTEIN)}

    @staticmethod
    def encode(text, vocab, length):
        output = np.zeros(length, dtype=np.int64)
        for i, char in enumerate(text[:length]):
            output[i] = vocab.get(char, 0)
        return output

    def __len__(self): return len(self.y)

    def __getitem__(self, index):
        return (torch.from_numpy(self.encode(self.drugs[index], self.drug_vocab, self.drug_length)),
                torch.from_numpy(self.encode(self.proteins[index], self.protein_vocab, self.protein_length)),
                torch.tensor(self.y[index]))


class Encoder(nn.Module):
    def __init__(self, vocab_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, 128, padding_idx=0)
        self.convs = nn.Sequential(
            nn.Conv1d(128, 32, 4), nn.ReLU(),
            nn.Conv1d(32, 64, 6), nn.ReLU(),
            nn.Conv1d(64, 96, 8), nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        encoded = self.embedding(sequence).transpose(1, 2)
        return self.dropout(self.convs(encoded).amax(dim=-1))


class DeepDTAStyle(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.drug = Encoder(len(SMILES), dropout)
        self.protein = Encoder(len(PROTEIN), dropout)
        self.head = nn.Sequential(nn.Linear(192, 1024), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, drug, protein):
        return self.head(torch.cat((self.drug(drug), self.protein(protein)), dim=1)).squeeze(1)


def ci(y, p):
    order = np.argsort(y, kind="mergesort"); y, p = y[order], p[order]
    ranks = np.searchsorted(np.unique(p), p) + 1; bit = np.zeros(int(ranks.max()) + 2)
    def add(i):
        while i < len(bit): bit[i] += 1; i += i & -i
    def query(i):
        total = 0.
        while i: total += bit[i]; i -= i & -i
        return total
    seen = score = pairs = 0.; start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and y[end] == y[start]: end += 1
        for i in range(start, end):
            lower = query(ranks[i]-1); equal = query(ranks[i])-lower
            score += lower + .5*equal; pairs += seen
        for i in range(start, end): add(ranks[i])
        seen += end-start; start = end
    return float(score/pairs) if pairs else float("nan")


def metrics(y, p):
    error = p-y; mse = float(np.mean(error**2)); denominator = float(np.sum((y-y.mean())**2))
    return {"RMSE": float(np.sqrt(mse)), "MSE": mse, "MAE": float(np.mean(np.abs(error))),
            "R2": float(1-np.sum(error**2)/denominator) if denominator else float("nan"),
            "Pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else float("nan"), "CI": ci(y, p)}


@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); ys, ps = [], []
    for drug, protein, y in loader:
        ps.append(model(drug.to(device), protein.to(device)).float().cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    a = args_parser()
    if not 0 < a.val_fraction < .5: raise ValueError("--val-fraction must be between 0 and 0.5")
    if a.out_dir.exists() and not a.finalize_only:
        raise ValueError(f"Output already exists: {a.out_dir}")
    train_csv, test_csv = Path("data") / f"{a.dataset}_train.csv", Path("data") / f"{a.dataset}_test.csv"
    if not train_csv.exists() or not test_csv.exists(): raise FileNotFoundError("Expected data/<dataset>_train.csv and data/<dataset>_test.csv")
    a.out_dir.mkdir(parents=True, exist_ok=True); seed_all(a.seed)
    device = torch.device(f"cuda:{a.cuda}" if torch.cuda.is_available() else "cpu")
    train_all = SequenceDataset(pd.read_csv(train_csv), a.max_drug_length, a.max_protein_length)
    test_set = SequenceDataset(pd.read_csv(test_csv), a.max_drug_length, a.max_protein_length)
    n_val = round(len(train_all)*a.val_fraction); n_fit = len(train_all)-n_val
    fit, validation = random_split(train_all, [n_fit, n_val], generator=torch.Generator().manual_seed(a.seed))
    fit_loader = DataLoader(fit, a.batch_size, shuffle=True)
    val_loader = DataLoader(validation, a.batch_size); test_loader = DataLoader(test_set, a.batch_size)
    model = DeepDTAStyle(a.dropout).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    if not a.finalize_only:
        best, stale, history = float("inf"), 0, []
        for epoch in range(1, a.epochs+1):
            model.train(); total = count = 0
            for drug, protein, y in fit_loader:
                optimizer.zero_grad(set_to_none=True); prediction = model(drug.to(device), protein.to(device)); loss = nn.functional.mse_loss(prediction, y.to(device))
                loss.backward(); optimizer.step(); total += loss.item()*len(y); count += len(y)
            vy, vp = predict(model, val_loader, device); val_mse = float(np.mean((vy-vp)**2))
            history.append({"epoch": epoch, "train_mse": total/count, "validation_mse": val_mse})
            if val_mse < best:
                best, stale = val_mse, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "validation_mse": val_mse}, a.out_dir/"best.pt")
            else: stale += 1
            if epoch == 1 or epoch % 10 == 0: print(f"seed={a.seed} epoch={epoch} train_mse={total/count:.4f} val_mse={val_mse:.4f}")
            if stale >= a.patience: break
        pd.DataFrame(history).to_csv(a.out_dir/"history.csv", index=False)
    elif not (a.out_dir / "best.pt").exists():
        raise FileNotFoundError(f"Cannot finalize: missing {a.out_dir / 'best.pt'}")
    checkpoint = torch.load(a.out_dir/"best.pt", map_location=device, weights_only=True); model.load_state_dict(checkpoint["model"])
    y, p = predict(model, test_loader, device)
    np.savetxt(a.out_dir/"test_predictions.csv", np.column_stack((y, p)), delimiter=",", header="y_true,y_pred", comments="")
    report = {"protocol": "DeepDTA_style_validation_selected_clean_v1", "dataset": a.dataset, "seed": a.seed,
              "best_epoch": checkpoint["epoch"], "validation_mse": checkpoint["validation_mse"], "test": metrics(y, p),
              "fit_rows": n_fit, "validation_rows": n_val, "test_rows": len(test_set), "device": str(device),
              "csv_sha256": {"train": sha256(train_csv), "test": sha256(test_csv)},
              "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}}
    (a.out_dir/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); print(f"Saved: {a.out_dir.resolve()}")


if __name__ == "__main__": main()
