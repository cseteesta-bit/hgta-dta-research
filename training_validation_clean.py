#!/usr/bin/env python3
"""Reproducible, validation-selected GraphDTA reference benchmark.

This file is intended to be copied into a clean clone of
https://github.com/thinng/GraphDTA.  It uses that repository's unmodified
processed ``data/processed/<dataset>_{train,test}.pt`` files and model classes.

Differences from GraphDTA's training_validation.py:
  * an explicit seed controls train/validation partitioning and initialization;
  * model selection is solely by validation MSE;
  * the official test fold is evaluated exactly once, after selection;
  * all metrics/configuration/history are written to a new output directory.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import DataLoader, TestbedDataset, ci, mse, pearson, rmse, spearman


MODELS = {"gin": GINConvNet, "gat": GATNet, "gat_gcn": GAT_GCN, "gcn": GCNNet}


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=["kiba", "davis"], required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default="gin")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=4221)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--cuda", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def predict(model, device, loader):
    model.eval(); labels, predictions = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data).view(-1)
            labels.append(data.y.view(-1).float().cpu())
            predictions.append(output.float().cpu())
    return torch.cat(labels).numpy(), torch.cat(predictions).numpy()


def all_metrics(y, prediction):
    return {"RMSE": float(rmse(y, prediction)), "MSE": float(mse(y, prediction)),
            "Pearson": float(pearson(y, prediction)), "Spearman": float(spearman(y, prediction)),
            "CI": float(ci(y, prediction)), "MAE": float(np.abs(y - prediction).mean())}


def main():
    args = parse_args()
    if not 0 < args.val_fraction < .5: raise ValueError("--val-fraction must be between 0 and 0.5")
    root = Path(args.out_dir)
    if root.exists(): raise ValueError(f"Output already exists: {root}")
    root.mkdir(parents=True)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    train_name, test_name = f"{args.dataset}_train", f"{args.dataset}_test"
    train_all = TestbedDataset(root="data", dataset=train_name)
    test_data = TestbedDataset(root="data", dataset=test_name)
    if not len(train_all) or not len(test_data): raise RuntimeError("Empty processed dataset")
    validation_size = round(len(train_all) * args.val_fraction)
    fit_size = len(train_all) - validation_size
    generator = torch.Generator().manual_seed(args.seed)
    fit_data, validation_data = torch.utils.data.random_split(train_all, [fit_size, validation_size], generator=generator)
    # Global seeding above controls this shuffle; omit DataLoader's generator
    # argument for compatibility with the older PyG DataLoader used by GraphDTA.
    train_loader = DataLoader(fit_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    model = MODELS[args.model]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    best_validation, best_epoch, stale, history = float("inf"), -1, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train(); losses, observations = 0.0, 0
        for data in train_loader:
            data = data.to(device); optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float())
            loss.backward(); optimizer.step()
            losses += loss.item() * data.y.numel(); observations += data.y.numel()
        vy, vp = predict(model, device, validation_loader)
        validation_mse = float(mse(vy, vp))
        history.append({"epoch": epoch, "train_mse": losses / observations, "validation_mse": validation_mse})
        if validation_mse < best_validation:
            best_validation, best_epoch, stale = validation_mse, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_mse": validation_mse}, root / "best.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} train_mse={losses/observations:.4f} val_mse={validation_mse:.4f}")
        if stale >= args.patience: break

    pd.DataFrame(history).to_csv(root / "history.csv", index=False)
    checkpoint = torch.load(root / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    # This is the first and only test-fold evaluation.
    y, prediction = predict(model, device, test_loader)
    result = {"protocol": "GraphDTA_validation_selected_clean_v1", "dataset": args.dataset,
              "model": args.model, "seed": args.seed, "best_epoch": checkpoint["epoch"],
              "validation_mse": checkpoint["validation_mse"], "test": all_metrics(y, prediction),
              "fit_rows": fit_size, "validation_rows": validation_size, "test_rows": len(test_data),
              "device": str(device), "args": vars(args)}
    np.savetxt(root / "test_predictions.csv", np.column_stack([y, prediction]), delimiter=",",
               header="y_true,y_pred", comments="")
    (root / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)); print(f"Saved: {root.resolve()}")


if __name__ == "__main__": main()
