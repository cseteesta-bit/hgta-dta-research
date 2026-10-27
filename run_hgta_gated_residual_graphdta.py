#!/usr/bin/env python3
"""Validation-selected gated HGTA residual on a frozen GraphDTA-GIN backbone.

Copy this file into a GraphDTA checkout after ``training_validation_clean.py``
has created a seed-matched ``best.pt``.  The base GIN/CNN prediction is frozen
and reproduced exactly at epoch zero.  A small atom--protein attention module
learns only a gated residual, selected by validation MSE.  Thus the official
test fold is evaluated once only after selection, and the new interaction layer
cannot be credited unless it outperforms the validated base checkpoint.
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
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_batch

from models.ginconv import GINConvNet
from utils import DataLoader, TestbedDataset, ci, mse, pearson, rmse, spearman


def args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=["kiba", "davis"], default="kiba")
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--val-fraction", type=float, default=.20)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=.10)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--cuda", type=int, default=0)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def metric_dict(y, p):
    return {"RMSE": float(rmse(y, p)), "MSE": float(mse(y, p)), "MAE": float(np.abs(y-p).mean()),
            "Pearson": float(pearson(y, p)), "Spearman": float(spearman(y, p)), "CI": float(ci(y, p))}


class GatedAtomProteinResidual(nn.Module):
    """Frozen GraphDTA GIN/CNN prediction plus trainable atom--motif residual."""
    def __init__(self, base_state, hidden_dim=64, dropout=.10):
        super().__init__()
        self.base = GINConvNet()
        self.base.load_state_dict(base_state)
        for parameter in self.base.parameters(): parameter.requires_grad_(False)
        self.atom_projection = nn.Sequential(nn.Linear(32, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.motif_projection = nn.Sequential(nn.Linear(32, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.interaction_norm = nn.LayerNorm(hidden_dim)
        width = 128 + 128 + hidden_dim + 1
        self.residual = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))
        self.gate = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        # At epoch zero the output must equal the frozen base prediction exactly.
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.gate[-1].weight); nn.init.constant_(self.gate[-1].bias, -5.0)

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen BatchNorm/dropout in reference-inference mode at all times.
        self.base.eval()
        return self

    @torch.no_grad()
    def _base_features(self, data):
        b = self.base
        x, edge_index, batch, target = data.x, data.edge_index, data.batch, data.target
        x = torch.relu(b.conv1(x, edge_index)); x = b.bn1(x)
        x = torch.relu(b.conv2(x, edge_index)); x = b.bn2(x)
        x = torch.relu(b.conv3(x, edge_index)); x = b.bn3(x)
        x = torch.relu(b.conv4(x, edge_index)); x = b.bn4(x)
        atom_nodes = torch.relu(b.conv5(x, edge_index)); atom_nodes = b.bn5(atom_nodes)
        drug_vector = torch.relu(b.fc1_xd(global_add_pool(atom_nodes, batch)))
        embedded = b.embedding_xt(target)
        motif_tokens = b.conv_xt_1(embedded)                 # [batch, 32, 121]
        protein_vector = b.fc1_xt(motif_tokens.reshape(-1, 32 * 121))
        joined = torch.cat((drug_vector, protein_vector), 1)
        joined = b.relu(b.fc1(joined)); joined = b.dropout(joined)
        joined = b.relu(b.fc2(joined)); joined = b.dropout(joined)
        base_prediction = b.out(joined).view(-1, 1)
        atom_dense, atom_mask = to_dense_batch(atom_nodes, batch)
        return atom_dense, atom_mask, motif_tokens.transpose(1, 2), drug_vector, protein_vector, base_prediction

    def forward(self, data):
        atom_dense, atom_mask, motifs, drug, protein, base_prediction = self._base_features(data)
        atoms = self.atom_projection(atom_dense)
        motifs = self.motif_projection(motifs)
        attended, _ = self.cross_attention(atoms, motifs, motifs, need_weights=False)
        atoms = self.interaction_norm(atoms + attended)
        interaction = (atoms * atom_mask.unsqueeze(-1)).sum(1) / atom_mask.sum(1, keepdim=True).clamp_min(1)
        features = torch.cat((drug, protein, interaction, base_prediction), dim=1)
        residual = self.residual(features)
        gate = torch.sigmoid(self.gate(features))
        return (base_prediction + gate * residual).view(-1), base_prediction.view(-1), gate.view(-1)


@torch.inference_mode()
def predict(model, device, loader):
    model.eval(); ys, ps, bases, gates = [], [], [], []
    for data in loader:
        data = data.to(device); prediction, base, gate = model(data)
        ys.append(data.y.view(-1).float().cpu()); ps.append(prediction.cpu()); bases.append(base.cpu()); gates.append(gate.cpu())
    return tuple(torch.cat(items).numpy() for items in (ys, ps, bases, gates))


def main():
    a = args()
    if not 0 < a.val_fraction < .5: raise ValueError("--val-fraction must be between 0 and 0.5")
    root, base_path = Path(a.out_dir), Path(a.base_checkpoint)
    if root.exists(): raise ValueError(f"Output already exists: {root}")
    if not base_path.is_file(): raise FileNotFoundError(base_path)
    seed_all(a.seed); root.mkdir(parents=True)
    device = torch.device(f"cuda:{a.cuda}" if torch.cuda.is_available() else "cpu")
    full_train = TestbedDataset(root="data", dataset=f"{a.dataset}_train")
    test_data = TestbedDataset(root="data", dataset=f"{a.dataset}_test")
    validation_rows = round(len(full_train) * a.val_fraction); fit_rows = len(full_train) - validation_rows
    fit, validation = torch.utils.data.random_split(full_train, [fit_rows, validation_rows], generator=torch.Generator().manual_seed(a.seed))
    train_loader = DataLoader(fit, batch_size=a.batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=a.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=a.batch_size, shuffle=False)
    checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
    model = GatedAtomProteinResidual(checkpoint["model"], a.hidden_dim, a.dropout).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=a.learning_rate, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # Epoch zero is the frozen base model: a valid validation candidate.
    vy, vp, vb, vg = predict(model, device, validation_loader)
    best, best_epoch, stale = float(mse(vy, vp)), 0, 0
    torch.save({"model": model.state_dict(), "epoch": 0, "validation_mse": best}, root / "best.pt")
    history = [{"epoch": 0, "train_mse": None, "validation_mse": best, "mean_gate": float(vg.mean())}]
    print(f"epoch=0 base_val_mse={best:.4f} gate={vg.mean():.4f}")
    for epoch in range(1, a.epochs + 1):
        model.train(); total = count = 0
        for data in train_loader:
            data = data.to(device); optimizer.zero_grad(set_to_none=True)
            prediction, _, _ = model(data); loss = loss_fn(prediction, data.y.view(-1).float())
            loss.backward(); optimizer.step(); total += loss.item() * data.y.numel(); count += data.y.numel()
        vy, vp, vb, vg = predict(model, device, validation_loader); validation_mse = float(mse(vy, vp))
        history.append({"epoch": epoch, "train_mse": total/count, "validation_mse": validation_mse, "mean_gate": float(vg.mean())})
        if validation_mse < best - 1e-10:
            best, best_epoch, stale = validation_mse, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_mse": best}, root / "best.pt")
        else: stale += 1
        if epoch == 1 or epoch % 10 == 0: print(f"epoch={epoch} train_mse={total/count:.4f} val_mse={validation_mse:.4f} gate={vg.mean():.4f}")
        if stale >= a.patience: break
    pd.DataFrame(history).to_csv(root / "history.csv", index=False)
    selected = torch.load(root / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(selected["model"])
    # First and only test evaluation for this residual model.
    y, prediction, base_prediction, gate = predict(model, device, test_loader)
    result = {"protocol": "frozen_GraphDTA_GIN_gated_atom_protein_residual_v1", "dataset": a.dataset, "seed": a.seed,
              "base_checkpoint": str(base_path.resolve()), "best_epoch": selected["epoch"], "validation_mse": selected["validation_mse"],
              "test": metric_dict(y, prediction), "frozen_base_test": metric_dict(y, base_prediction),
              "mean_test_gate": float(gate.mean()), "fit_rows": fit_rows, "validation_rows": validation_rows,
              "test_rows": len(test_data), "device": str(device), "args": vars(a)}
    np.savetxt(root / "test_predictions.csv", np.column_stack([y, prediction, base_prediction, gate]), delimiter=",",
               header="y_true,y_pred,base_pred,interaction_gate", comments="")
    (root / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)); print(f"Saved: {root.resolve()}")


if __name__ == "__main__": main()
