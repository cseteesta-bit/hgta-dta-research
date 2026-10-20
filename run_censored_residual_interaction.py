#!/usr/bin/env python3
"""Exploratory floor-aware gated residual DTA model for development splits.

This script is deliberately separate from ``run_residual_interaction.py``.
It tests a data hypothesis: observations at a specified lower floor may be
left-censored rather than exact labels.  For floor observations, the training
loss penalizes only predictions above the floor; ordinary squared error is used
for observations above the floor.  Checkpoints are still selected by ordinary
validation MSE, and test data are only evaluated after selection.

Use only on an explicitly designated development split.  The floor assumption
must be verified against raw assay provenance before being a paper claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader, TensorDataset

from run_residual_interaction import (
    ESMMeanCache,
    ResidualInteraction,
    hash_file,
    metrics,
    seed_all,
)


def floor_aware_loss(pred, target, floor_standardized, censored_weight):
    """Balanced one-sided loss for labels at an assumed left-censoring floor."""
    censored = target <= floor_standardized + 1e-6
    terms = []
    if (~censored).any():
        terms.append((1.0 - censored_weight) * (pred[~censored] - target[~censored]).square().mean())
    if censored.any():
        # A latent value at or below the floor is compatible with pred <= floor.
        terms.append(censored_weight * torch.relu(pred[censored] - floor_standardized).square().mean())
    if not terms:
        raise RuntimeError("Empty batch")
    return sum(terms)


@torch.inference_mode()
def predict(model, loader, device, y_mean, y_std):
    model.eval(); yy=[]; pp=[]; gates=[]
    for drug, protein, y in loader:
        pred, _, _, _, gate = model(drug.to(device), protein.to(device), return_parts=True)
        pp.append(pred.cpu().numpy() * y_std + y_mean)
        yy.append(y.numpy()); gates.append(gate.cpu().numpy())
    return np.concatenate(yy), np.concatenate(pp), np.concatenate(gates)


def train_seed(seed, data, labels, test_frame, args, root, device):
    seed_all(seed); run=root/f"seed_{seed}"; run.mkdir()
    dx_train, px_train = data["train"]
    dx_val, px_val = data["validation"]
    dx_test, px_test = data["test"]
    y_train, y_val, y_test = labels["train"], labels["validation"], labels["test"]
    dm, ds = dx_train.mean(0), dx_train.std(0); ds[ds < 1e-6] = 1
    pm, ps = px_train.mean(0), px_train.std(0); ps[ps < 1e-6] = 1
    dx_train, dx_val, dx_test = (dx_train-dm)/ds, (dx_val-dm)/ds, (dx_test-dm)/ds
    px_train, px_val, px_test = (px_train-pm)/ps, (px_val-pm)/ps, (px_test-pm)/ps
    ym, ys = float(y_train.mean()), float(y_train.std())
    if ys <= 0: raise ValueError("Training labels have zero variance")
    floor_standardized = (args.floor - ym) / ys
    train = DataLoader(TensorDataset(torch.from_numpy(dx_train), torch.from_numpy(px_train),
                                     torch.from_numpy((y_train-ym)/ys)), batch_size=args.batch_size,
                       shuffle=True, generator=torch.Generator().manual_seed(seed))
    val = DataLoader(TensorDataset(torch.from_numpy(dx_val), torch.from_numpy(px_val),
                                   torch.from_numpy(y_val)), batch_size=1024)
    test = DataLoader(TensorDataset(torch.from_numpy(dx_test), torch.from_numpy(px_test),
                                    torch.from_numpy(y_test)), batch_size=1024)
    model = ResidualInteraction(dx_train.shape[1], px_train.shape[1], args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=.5, patience=10, min_lr=1e-6)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs+1):
        model.train(); total=n=0
        for drug, protein, y in train:
            drug, protein, y = drug.to(device), protein.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred, _, _, residual, _ = model(drug, protein, return_parts=True)
            loss = floor_aware_loss(pred, y, floor_standardized, args.censored_weight)
            loss = loss + args.residual_l2 * residual.square().mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item()*len(y); n += len(y)
        vy, vp, vg = predict(model, val, device, ym, ys); vm = metrics(vy, vp)
        schedule.step(vm["MSE"])
        history.append({"epoch": epoch, "train_objective": total/n, "lr": opt.param_groups[0]["lr"],
                        "val_mean_gate": float(vg.mean()), **{f"val_{k}": v for k,v in vm.items()}})
        if vm["MSE"] < best - 1e-8:
            best, stale = vm["MSE"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation": vm,
                        "validation_mean_gate": float(vg.mean())}, run/"best.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"seed={seed} epoch={epoch} train_floor_loss={total/n:.4f} val_mse={vm['MSE']:.4f} gate={vg.mean():.3f}")
        if stale >= args.patience: break
    pd.DataFrame(history).to_csv(run/"history.csv", index=False)
    ckpt = torch.load(run/"best.pt", map_location=device, weights_only=True); model.load_state_dict(ckpt["model"])
    ty, tp, tg = predict(model, test, device, ym, ys); tm = metrics(ty, tp)
    tm["MeanInteractionGate"] = float(tg.mean())
    out = test_frame.copy(); out["prediction"] = tp; out["absolute_error"] = np.abs(tp-ty); out["interaction_gate"] = tg
    out.to_csv(run/"test_predictions.csv", index=False)
    (run/"metrics.json").write_text(json.dumps({"seed":seed, "best_epoch":ckpt["epoch"],
        "selection_metric":"validation_MSE", "validation":ckpt["validation"], "test":tm}, indent=2))
    return tm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[4221,17,71,314,2025])
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5); ap.add_argument("--dropout", type=float, default=.2)
    ap.add_argument("--hidden-dim", type=int, default=512); ap.add_argument("--residual-l2", type=float, default=.001)
    ap.add_argument("--protein-model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--floor", type=float, default=5.0, help="Hypothesized left-censoring floor in affinity units.")
    ap.add_argument("--censored-weight", type=float, default=.5, help="Relative loss weight for floor observations.")
    args = ap.parse_args()
    if not 0 < args.censored_weight < 1: raise ValueError("--censored-weight must be between 0 and 1")
    root = Path(args.out_dir)
    if root.exists(): raise ValueError(f"Output directory exists: {root}")
    source = Path(args.data_dir); paths = {n:source/f"{n}.csv" for n in ("train","validation","test")}
    frames = {n:pd.read_csv(p) for n,p in paths.items()}; needed = {"canonical_smiles","target_sequence","affinity"}
    for n,f in frames.items():
        if needed-set(f): raise ValueError(f"{n} missing {sorted(needed-set(f))}")
        f.target_sequence = f.target_sequence.astype(str).str.strip().str.upper()
        f.affinity = pd.to_numeric(f.affinity, errors="raise")
    train_floor_fraction = float(np.mean(np.isclose(frames["train"].affinity.to_numpy(), args.floor)))
    if train_floor_fraction == 0: raise ValueError(f"No training labels equal --floor={args.floor}; censoring hypothesis is inapplicable")
    root.mkdir(parents=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); drug_cache = {}
    def fp(s):
        if s not in drug_cache:
            mol = Chem.MolFromSmiles(s)
            if mol is None: raise ValueError(f"Invalid SMILES: {s}")
            a = np.zeros(2048, dtype=np.float32); DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), a); drug_cache[s] = a
        return drug_cache[s]
    esm = ESMMeanCache(args.protein_model, root/"protein_cache", device)
    sequences = sorted(set().union(*(set(f.target_sequence) for f in frames.values())))
    protein_cache = {s:esm.get(s).numpy().astype(np.float32) for s in sequences}
    data = {n:(np.stack([fp(s) for s in f.canonical_smiles]), np.stack([protein_cache[s] for s in f.target_sequence])) for n,f in frames.items()}
    labels = {n:frames[n].affinity.to_numpy(np.float32) for n in frames}
    manifest = {"model":"floor_aware_gated_residual_interaction_morgan_esm2", "formula":"drug_main + protein_main + sigmoid(gate)*interaction_residual",
        "floor_hypothesis":{"floor":args.floor,"training_floor_fraction":train_floor_fraction,"loss":"one-sided_squared_error_above_floor"},
        "args":vars(args), "source_sha256":{n:hash_file(p) for n,p in paths.items()}, "n_unique_drugs":len(drug_cache),"n_unique_proteins":len(protein_cache)}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2))
    results = [train_seed(s, data, labels, frames["test"], args, root, device) for s in args.seeds]
    table = pd.DataFrame(results); table.insert(0,"seed",args.seeds); table.to_csv(root/"all_seed_metrics.csv", index=False)
    summary = {c:{"mean":float(table[c].mean()),"std":float(table[c].std(ddof=1))} for c in table.columns if c != "seed"}
    (root/"summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
