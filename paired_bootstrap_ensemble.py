#!/usr/bin/env python3
"""Paired, cluster-aware bootstrap comparison of two saved DTA ensembles.

This script never trains models and never fits anything using labels.  It only
compares pre-existing test predictions, using the same examples in both arms.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_ensemble(results_dir: Path):
    files = sorted(results_dir.glob("seed_*/test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No seed_*/test_predictions.csv below {results_dir}")
    frames = [pd.read_csv(p) for p in files]
    required = {"y_true", "y_pred"}
    for p, df in zip(files, frames):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p}: missing columns {sorted(missing)}")
    key_candidates = ["drug_id", "target_id"]
    if not set(key_candidates).issubset(frames[0].columns):
        key_candidates = ["canonical_smiles", "target_sequence"]
    if not set(key_candidates).issubset(frames[0].columns):
        raise ValueError("Prediction files need drug_id/target_id or canonical_smiles/target_sequence keys.")
    keys = key_candidates
    base = frames[0][keys + ["y_true"]].copy()
    if base.duplicated(keys).any():
        raise ValueError("Duplicate prediction keys found; each test pair must appear once per seed.")
    pred_columns = []
    for i, df in enumerate(frames):
        part = df[keys + ["y_true", "y_pred"]].copy()
        if part.duplicated(keys).any():
            raise ValueError(f"Duplicate prediction keys in seed file {files[i]}")
        part = part.rename(columns={"y_true": f"y_true_{i}", "y_pred": f"pred_{i}"})
        base = base.merge(part, on=keys, how="outer", validate="one_to_one")
        pred_columns.append(f"pred_{i}")
    y_columns = [c for c in base if c.startswith("y_true_")]
    if base[keys + ["y_true"] + y_columns + pred_columns].isna().any().any():
        raise ValueError("Seed prediction files do not contain exactly the same test pairs.")
    for c in y_columns:
        if not np.allclose(base["y_true"], base[c], rtol=0, atol=1e-8):
            raise ValueError("Inconsistent labels across saved prediction files.")
    return base[keys + ["y_true"]].assign(y_pred=base[pred_columns].mean(axis=1)), keys


def metric_differences(y, uq, baseline, indices):
    e_uq = (uq[indices] - y[indices]) ** 2
    e_base = (baseline[indices] - y[indices]) ** 2
    return float(e_uq.mean() - e_base.mean()), float(np.abs(uq[indices] - y[indices]).mean() - np.abs(baseline[indices] - y[indices]).mean())


def bootstrap(y, uq, baseline, groups, repetitions, rng):
    n = len(y)
    unique_groups = np.unique(groups)
    group_rows = {g: np.flatnonzero(groups == g) for g in unique_groups}
    output = np.empty((repetitions, 2), dtype=float)
    for b in range(repetitions):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_rows[g] for g in sampled])
        output[b] = metric_differences(y, uq, baseline, idx)
    return output


def summarize(values, observed):
    lo, hi = np.quantile(values, [0.025, 0.975])
    # Two-sided sign-based bootstrap p-value; ties are irrelevant for continuous losses.
    p = min(1.0, 2.0 * min(np.mean(values >= 0), np.mean(values <= 0)))
    return {"observed_uq_minus_baseline": float(observed), "ci95": [float(lo), float(hi)], "p_two_sided": float(p)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uq-dir", required=True, type=Path, help="UQ results directory containing seed_*/test_predictions.csv")
    ap.add_argument("--baseline-dir", required=True, type=Path, help="Baseline results directory containing seed_*/test_predictions.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("subset_bootstrap_comparison"))
    ap.add_argument("--repetitions", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    if args.repetitions < 1000:
        raise ValueError("Use at least 1000 bootstrap repetitions.")
    uq, keys = read_ensemble(args.uq_dir)
    base, base_keys = read_ensemble(args.baseline_dir)
    if keys != base_keys:
        raise ValueError(f"Key scheme differs: UQ={keys}, baseline={base_keys}")
    df = uq.merge(base[keys + ["y_true", "y_pred"]], on=keys, suffixes=("_uq", "_baseline"), validate="one_to_one")
    if len(df) == 0 or not np.allclose(df.y_true_uq, df.y_true_baseline, rtol=0, atol=1e-8):
        raise ValueError("Ensembles have different test labels or no common pairs.")
    y, p_uq, p_base = (df.y_true_uq.to_numpy(float), df.y_pred_uq.to_numpy(float), df.y_pred_baseline.to_numpy(float))
    observed = metric_differences(y, p_uq, p_base, np.arange(len(df)))
    rng = np.random.default_rng(args.seed)
    schemes = {"pair": np.arange(len(df)), "drug": df[keys[0]].astype(str).to_numpy(), "target": df[keys[1]].astype(str).to_numpy()}
    report = {"n_pairs": int(len(df)), "keys": keys, "repetitions": args.repetitions, "rng_seed": args.seed,
              "direction": "Negative UQ-minus-baseline differences favor UQ.", "results": {}}
    for name, groups in schemes.items():
        values = bootstrap(y, p_uq, p_base, groups, args.repetitions, rng)
        report["results"][name] = {"n_clusters": int(len(np.unique(groups))),
            "MSE": summarize(values[:, 0], observed[0]), "MAE": summarize(values[:, 1], observed[1])}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "paired_bootstrap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.out_dir.resolve() / 'paired_bootstrap_report.json'}")


if __name__ == "__main__":
    main()
