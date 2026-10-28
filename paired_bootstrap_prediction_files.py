#!/usr/bin/env python3
"""Paired row-bootstrap comparison for two prediction columns in one CSV.

This is appropriate for a fixed benchmark test set when entity identifiers are
not available in the stored predictions.  It estimates uncertainty over test
rows, not over new drugs or targets; report that limitation explicitly.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(samples, observed):
    lo, hi = np.quantile(samples, [0.025, 0.975])
    # Two-sided bootstrap sign probability, with a finite-sample floor.
    positive = (np.count_nonzero(samples >= 0) + 1) / (len(samples) + 1)
    negative = (np.count_nonzero(samples <= 0) + 1) / (len(samples) + 1)
    return {"observed_candidate_minus_base": float(observed),
            "ci95": [float(lo), float(hi)],
            "p_two_sided": float(min(1.0, 2 * min(positive, negative)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--candidate-column", default="hgta_v2_prediction")
    ap.add_argument("--base-column", default="frozen_base_prediction")
    ap.add_argument("--truth-column", default="y_true")
    ap.add_argument("--repetitions", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.repetitions < 1000:
        raise ValueError("Use at least 1,000 bootstrap repetitions")

    frame = pd.read_csv(args.predictions)
    required = {args.truth_column, args.candidate_column, args.base_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {args.predictions}")
    y = frame[args.truth_column].to_numpy(float)
    candidate = frame[args.candidate_column].to_numpy(float)
    base = frame[args.base_column].to_numpy(float)
    if not (np.isfinite(y).all() and np.isfinite(candidate).all() and np.isfinite(base).all()):
        raise ValueError("Predictions and labels must be finite")

    squared_delta = (candidate - y) ** 2 - (base - y) ** 2
    absolute_delta = np.abs(candidate - y) - np.abs(base - y)
    rng = np.random.default_rng(args.seed)
    n = len(y)
    mse_samples = np.empty(args.repetitions)
    mae_samples = np.empty(args.repetitions)
    for i in range(args.repetitions):
        index = rng.integers(0, n, size=n)
        mse_samples[i] = squared_delta[index].mean()
        mae_samples[i] = absolute_delta[index].mean()

    report = {
        "predictions": str(args.predictions),
        "n_pairs": int(n),
        "repetitions": args.repetitions,
        "rng_seed": args.seed,
        "direction": "Negative candidate-minus-base differences favor the candidate.",
        "limitation": "Row bootstrap only: predictions do not contain drug/target identifiers, so this does not quantify entity-cluster uncertainty.",
        "MSE": summarize(mse_samples, squared_delta.mean()),
        "MAE": summarize(mae_samples, absolute_delta.mean()),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "paired_bootstrap_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
