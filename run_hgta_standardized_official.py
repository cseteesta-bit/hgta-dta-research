#!/usr/bin/env python3
"""HGTA-v2 stabilization ablation on an unchanged official train/test fold.

Change versus ``run_hgta_official_folds.py``: affinity is z-scored using the
fit portion of the supplied training fold only.  Optimization and checkpoint
selection use that scale; reported predictions and metrics are transformed back
to the original affinity scale.  This is deliberately a single, auditable
change, so it can be compared fairly with the HGTA-v1 official-fold baseline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import journal_hgta as hgta
from run_hgta_official_folds import load_raw_fold


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", default="official_esm2_cache")
    p.add_argument("--seeds", nargs="+", type=int, default=[4221, 17, 71, 314, 2025])
    p.add_argument("--validation-seed", type=int, default=20261020)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--hidden-dim", type=int, default=256)
    return p.parse_args()


def normalise(frame: pd.DataFrame, mean: float, std: float) -> pd.DataFrame:
    out = frame.copy()
    out["raw_affinity"] = out[hgta.Y].astype(float)
    out[hgta.Y] = (out["raw_affinity"] - mean) / std
    return out


@torch.inference_mode()
def predict(model, loader, device, amp):
    model.eval(); indices, ys, predictions = [], [], []
    # journal_hgta's point-prediction collator returns five values.  The UQ
    # trainer has an additional assay-variance field, hence do not assume it
    # here.
    for idx, graph, protein, mask, y in loader:
        graph, protein, mask = graph.to(device), protein.to(device), mask.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            mu = model(graph, protein, mask)
        indices.extend(idx); ys.extend(y.tolist()); predictions.extend(mu.float().cpu().tolist())
    return np.asarray(indices), np.asarray(ys), np.asarray(predictions)


def raw_metrics(y_normalised, prediction_normalised, mean, std):
    return hgta.metrics(y_normalised * std + mean, prediction_normalised * std + mean)


def train_one(cfg, seed, train, validation, test, raw_test, cache, protein_dim, device, root, mean, std):
    hgta.seed_all(seed, cfg.deterministic)
    run = root / f"seed_{seed}"; run.mkdir(parents=True)
    generator = torch.Generator().manual_seed(seed)
    collate = hgta.collator(cache)
    train_loader = DataLoader(hgta.DTADataset(train), cfg.batch_size, shuffle=True, generator=generator,
                              collate_fn=collate, num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(hgta.DTADataset(validation), cfg.batch_size, shuffle=False,
                                   collate_fn=collate, num_workers=cfg.num_workers)
    test_loader = DataLoader(hgta.DTADataset(test), cfg.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=cfg.num_workers)
    model = hgta.JournalHGTA(cfg, protein_dim).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=.5, patience=10, min_lr=1e-6)
    amp = cfg.use_amp and device.type == "cuda"; scaler = torch.amp.GradScaler(enabled=amp)
    best, stale, history = float("inf"), 0, []

    for epoch in range(1, cfg.epochs + 1):
        model.train(); total, n = 0.0, 0
        for _, graph, protein, mask, y in train_loader:
            graph, protein, mask, y = graph.to(device), protein.to(device), mask.to(device), y.to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                mu = model(graph, protein, mask)
                loss = F.mse_loss(mu, y)
            scaler.scale(loss).backward(); scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimiser); scaler.update()
            total += loss.item() * len(y); n += len(y)
        _, vy, vp = predict(model, validation_loader, device, amp)
        validation_metrics = raw_metrics(vy, vp, mean, std)
        validation_norm_mse = float(np.mean((vy - vp) ** 2))
        scheduler.step(validation_norm_mse)
        history.append({"epoch": epoch, "train_normalized_mse": total / n,
                        "validation_normalized_mse": validation_norm_mse,
                        **{f"val_{k}": v for k, v in validation_metrics.items()},
                        "lr": optimiser.param_groups[0]["lr"]})
        if validation_norm_mse < best - 1e-8:
            best, stale = validation_norm_mse, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "validation": validation_metrics, "validation_normalized_mse": validation_norm_mse}, run / "best.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"seed={seed} epoch={epoch} train_zmse={total/n:.4f} val_mse={validation_metrics['MSE']:.4f}")
        if stale >= cfg.patience:
            break

    pd.DataFrame(history).to_csv(run / "history.csv", index=False)
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    ids, ty, tp = predict(model, test_loader, device, amp)
    test_metrics = raw_metrics(ty, tp, mean, std)
    out = raw_test.iloc[ids].copy()
    out["prediction"] = tp * std + mean
    out["absolute_error"] = np.abs(out["prediction"].to_numpy() - out[hgta.Y].to_numpy())
    out.to_csv(run / "test_predictions.csv", index=False)
    with open(run / "metrics.json", "w") as f:
        json.dump({"seed": seed, "best_epoch": checkpoint["epoch"],
                   "selection_metric": "validation_normalized_mse",
                   "label_normalization": {"fit_mean": mean, "fit_std": std},
                   "validation": checkpoint["validation"], "test": test_metrics}, f, indent=2)
    return test_metrics


def main():
    args = arguments()
    if len(set(args.seeds)) != len(args.seeds): raise ValueError("Repeated seeds")
    root = Path(args.out_dir)
    if root.exists(): raise ValueError(f"Output already exists: {root}")
    train_path, test_path = Path(args.train_csv).resolve(), Path(args.test_csv).resolve()
    all_train, raw_test = load_raw_fold(train_path, "train"), load_raw_fold(test_path, "test")
    raw_train, raw_validation = hgta.split_train_val(all_train, args.val_fraction, "pair", args.validation_seed)
    fit_mean, fit_std = float(raw_train[hgta.Y].mean()), float(raw_train[hgta.Y].std(ddof=0))
    if not np.isfinite(fit_std) or fit_std <= 0: raise ValueError("Training affinity standard deviation must be positive")
    train, validation, test = (normalise(frame, fit_mean, fit_std) for frame in (raw_train, raw_validation, raw_test))
    cfg = hgta.Config(train_csv=str(train_path), test_csv=str(test_path), out_dir=str(root), seeds=tuple(args.seeds),
                      val_fraction=args.val_fraction, val_split="pair", batch_size=args.batch_size, epochs=args.epochs,
                      patience=args.patience, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                      dropout=args.dropout, hidden_dim=args.hidden_dim)
    root.mkdir(parents=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = hgta.ProteinCache(cfg, device, Path(args.cache_dir).resolve())
    protein_dim = cache.get(pd.concat([train[hgta.SEQ], validation[hgta.SEQ], test[hgta.SEQ]], ignore_index=True).iloc[0]).shape[1]
    manifest = {"protocol": "official_fold_hgta_v2_train_label_standardisation_v1", "dataset_name": args.dataset_name,
                "source": {"train_csv": str(train_path), "test_csv": str(test_path),
                           "train_sha256": hgta.sha256_file(str(train_path)), "test_sha256": hgta.sha256_file(str(test_path)),
                           "raw_train_rows": len(all_train), "raw_test_rows": len(raw_test)},
                "validation": {"seed": args.validation_seed, "fraction": args.val_fraction,
                               "fit_rows": len(raw_train), "validation_rows": len(raw_validation)},
                "label_normalization": {"method": "fit_partition_z_score", "fit_mean": fit_mean, "fit_std": fit_std,
                                        "test_labels_used_for_fit": False}, "config": asdict(cfg)}
    (root / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2))
    results = [train_one(cfg, s, train, validation, test, raw_test, cache, protein_dim, device, root, fit_mean, fit_std) for s in args.seeds]
    frame = pd.DataFrame(results); frame.insert(0, "seed", args.seeds); frame.to_csv(root / "all_seed_metrics.csv", index=False)
    summary = {c: {"mean": float(frame[c].mean()), "std": float(frame[c].std(ddof=1))} for c in frame.columns if c != "seed"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
