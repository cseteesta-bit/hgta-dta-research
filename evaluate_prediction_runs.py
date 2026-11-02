#!/usr/bin/env python3
"""Create a verified mean-prediction ensemble from independent run folders."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
            low = query(ranks[i]-1); equal = query(ranks[i])-low
            score += low + .5*equal; pairs += seen
        for i in range(start, end): add(ranks[i])
        seen += end-start; start = end
    return float(score/pairs) if pairs else float("nan")


def metrics(y, p):
    error = p-y; mse = float(np.mean(error**2)); denominator = float(np.sum((y-y.mean())**2))
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(abs(error))),
            "R2": float(1-np.sum(error**2)/denominator) if denominator else float("nan"),
            "Pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else float("nan"), "CI": ci(y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prediction-column", default="y_pred")
    ap.add_argument("--truth-column", default="y_true")
    args = ap.parse_args()
    labels, predictions, per_seed = None, [], {}
    for directory in args.run_dirs:
        table = pd.read_csv(directory / "test_predictions.csv")
        if not {args.truth_column, args.prediction_column} <= set(table):
            raise ValueError(f"{directory}: expected {args.truth_column} and {args.prediction_column}")
        y = table[args.truth_column].to_numpy(float); p = table[args.prediction_column].to_numpy(float)
        if labels is None: labels = y
        elif len(y) != len(labels) or not np.array_equal(y, labels):
            raise ValueError(f"{directory}: test labels/order do not match")
        meta = json.loads((directory / "metrics.json").read_text())
        per_seed[str(meta.get("seed", directory.name))] = meta.get("test", metrics(y, p))
        predictions.append(p)
    ensemble = np.mean(predictions, axis=0)
    report = {"n_pairs": int(len(labels)), "runs": [str(path) for path in args.run_dirs],
              "ensemble": metrics(labels, ensemble), "per_seed": per_seed}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": labels, "ensemble_prediction": ensemble}).to_csv(args.out_dir / "ensemble_predictions.csv", index=False)
    (args.out_dir / "ensemble_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)); print(f"Saved: {(args.out_dir/'ensemble_report.json').resolve()}")


if __name__ == "__main__": main()
