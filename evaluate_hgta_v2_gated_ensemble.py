#!/usr/bin/env python3
"""Evaluate HGTA-v2 gated-residual runs and their matched frozen GIN bases.

Each input directory is one run produced by run_hgta_gated_residual_graphdta.py.
The script verifies that every run contains exactly the same test examples,
then averages the five HGTA-v2 predictions.  The frozen base predictions are
also averaged, so the reported comparison is paired and like-for-like.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def concordance(y, p):
    order = np.argsort(y, kind="mergesort")
    y, p = y[order], p[order]
    ranks = np.searchsorted(np.unique(p), p) + 1
    bit = np.zeros(int(ranks.max()) + 2)

    def add(i):
        while i < len(bit):
            bit[i] += 1
            i += i & -i

    def query(i):
        total = 0.0
        while i:
            total += bit[i]
            i -= i & -i
        return total

    seen = score = pairs = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and y[end] == y[start]:
            end += 1
        for i in range(start, end):
            lower = query(ranks[i] - 1)
            equal = query(ranks[i]) - lower
            score += lower + 0.5 * equal
            pairs += seen
        for i in range(start, end):
            add(ranks[i])
        seen += end - start
        start = end
    return float(score / pairs) if pairs else float("nan")


def metrics(y, p):
    err = p - y
    mse = float(np.mean(err ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(err))),
        "R2": float(1 - np.sum(err ** 2) / total) if total else float("nan"),
        "Pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else float("nan"),
        "CI": concordance(y, p),
    }


def load_run(path):
    pred_path = path / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}")
    table = pd.read_csv(pred_path)
    needed = {"y_true", "y_pred", "base_pred"}
    missing = needed - set(table)
    if missing:
        raise ValueError(f"{pred_path}: missing columns {sorted(missing)}")
    # Test rows are generated from one fixed DataLoader order; row index is its stable key.
    table = table[["y_true", "y_pred", "base_pred"]].copy()
    meta = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    return table, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True, type=Path,
                    help="Five (or more) HGTA-v2 run directories, one per seed.")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    tables, per_seed = [], {}
    expected_y = None
    for path in args.run_dirs:
        table, meta = load_run(path)
        seed = str(meta["seed"])
        y = table.y_true.to_numpy(float)
        if expected_y is None:
            expected_y = y
        elif len(y) != len(expected_y) or not np.array_equal(y, expected_y):
            raise ValueError(f"{path}: test labels/order do not match the other runs")
        tables.append(table)
        per_seed[seed] = {
            "best_epoch": meta["best_epoch"],
            "validation_mse": meta["validation_mse"],
            "test": meta["test"],
            "frozen_base_test": meta["frozen_base_test"],
            "mean_test_gate": meta["mean_test_gate"],
        }

    y = expected_y
    hgta = np.mean([t.y_pred.to_numpy(float) for t in tables], axis=0)
    base = np.mean([t.base_pred.to_numpy(float) for t in tables], axis=0)
    report = {
        "protocol": "frozen_GraphDTA_GIN_gated_atom_protein_residual_v1",
        "seeds": [int(json.loads((p / "metrics.json").read_text())["seed"]) for p in args.run_dirs],
        "n_pairs": int(len(y)),
        "ensemble": metrics(y, hgta),
        "frozen_base_ensemble": metrics(y, base),
        "ensemble_difference_hgta_minus_base": {
            "MSE": float(np.mean((hgta - y) ** 2) - np.mean((base - y) ** 2)),
            "MAE": float(np.mean(np.abs(hgta - y)) - np.mean(np.abs(base - y))),
        },
        "per_seed": per_seed,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y, "hgta_v2_prediction": hgta, "frozen_base_prediction": base}).to_csv(
        args.out_dir / "ensemble_predictions.csv", index=False)
    (args.out_dir / "ensemble_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {(args.out_dir / 'ensemble_report.json').resolve()}")


if __name__ == "__main__":
    main()
