#!/usr/bin/env python3
"""Paired, cluster-aware bootstrap comparison of saved DTA ensembles."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


TRUTH = ("y_true", "true", "label", "affinity", "y", "Y")
PRED = ("y_pred", "pred", "prediction", "pred_mean", "mean", "mu")


def read_ensemble(directory):
    files = sorted(Path(directory).glob("seed_*/test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No seed_*/test_predictions.csv below {directory}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        truth = next((c for c in TRUTH if c in df.columns), None)
        pred = next((c for c in PRED if c in df.columns), None)
        if truth is None or pred is None:
            raise ValueError(f"{path}: could not identify truth/prediction columns. Available: {list(df.columns)}")
        frames.append(df.rename(columns={truth: "y_true", pred: "y_pred"}))
    candidates = (("drug_id", "target_id"), ("canonical_smiles", "target_sequence"), ("smiles", "sequence"))
    keys = next((list(k) for k in candidates if set(k).issubset(frames[0].columns)), None)
    if keys is None:
        raise ValueError(f"No supported pair keys in {files[0]}. Available: {list(frames[0].columns)}")
    merged = frames[0][keys + ["y_true"]].copy()
    if merged.duplicated(keys).any():
        raise ValueError("Duplicate test-pair keys in a seed prediction file.")
    pred_cols = []
    for i, frame in enumerate(frames):
        part = frame[keys + ["y_true", "y_pred"]].rename(columns={"y_true": f"truth_{i}", "y_pred": f"prediction_{i}"})
        if part.duplicated(keys).any():
            raise ValueError(f"Duplicate test-pair keys in {files[i]}")
        merged = merged.merge(part, on=keys, how="outer", validate="one_to_one")
        pred_cols.append(f"prediction_{i}")
    check_cols = ["y_true"] + [f"truth_{i}" for i in range(len(frames))] + pred_cols
    if merged[check_cols].isna().any().any():
        raise ValueError("Seed files do not contain exactly the same test pairs.")
    for i in range(len(frames)):
        if not np.allclose(merged.y_true, merged[f"truth_{i}"], rtol=0, atol=1e-8):
            raise ValueError("Labels differ between seed prediction files.")
    return merged[keys + ["y_true"]].assign(y_pred=merged[pred_cols].mean(axis=1)), keys


def difference(y, uq, base, idx):
    return [float(np.mean((uq[idx]-y[idx])**2)-np.mean((base[idx]-y[idx])**2)),
            float(np.mean(np.abs(uq[idx]-y[idx]))-np.mean(np.abs(base[idx]-y[idx])))]


def bootstrap(y, uq, base, groups, reps, rng):
    unique = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in unique}
    samples = np.empty((reps, 2))
    for i in range(reps):
        idx = np.concatenate([rows[g] for g in rng.choice(unique, len(unique), replace=True)])
        samples[i] = difference(y, uq, base, idx)
    return samples


def stats(values, observed):
    lo, hi = np.quantile(values, [0.025, .975])
    p = min(1., 2 * min(np.mean(values >= 0), np.mean(values <= 0)))
    return {"observed_uq_minus_baseline": float(observed), "ci95": [float(lo), float(hi)], "p_two_sided": float(p)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uq-dir", required=True, type=Path)
    ap.add_argument("--baseline-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("subset_bootstrap_comparison"))
    ap.add_argument("--repetitions", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    if args.repetitions < 1000: raise ValueError("Use at least 1000 repetitions.")
    uq, keys = read_ensemble(args.uq_dir)
    base, base_keys = read_ensemble(args.baseline_dir)
    if keys != base_keys: raise ValueError(f"Different key schemes: {keys} versus {base_keys}")
    df = uq.merge(base[keys + ["y_true", "y_pred"]], on=keys, suffixes=("_uq", "_baseline"), validate="one_to_one")
    if len(df) == 0 or not np.allclose(df.y_true_uq, df.y_true_baseline, rtol=0, atol=1e-8):
        raise ValueError("UQ and baseline have different test labels or no shared pairs.")
    y, pu, pb = df.y_true_uq.to_numpy(float), df.y_pred_uq.to_numpy(float), df.y_pred_baseline.to_numpy(float)
    observed = difference(y, pu, pb, np.arange(len(df)))
    schemes = {"pair": np.arange(len(df)), "drug": df[keys[0]].astype(str).to_numpy(), "target": df[keys[1]].astype(str).to_numpy()}
    rng = np.random.default_rng(args.seed)
    report = {"n_pairs": len(df), "keys": keys, "repetitions": args.repetitions, "rng_seed": args.seed,
              "direction": "Negative UQ-minus-baseline differences favor UQ.", "results": {}}
    for name, groups in schemes.items():
        values = bootstrap(y, pu, pb, groups, args.repetitions, rng)
        report["results"][name] = {"n_clusters": len(np.unique(groups)), "MSE": stats(values[:,0], observed[0]), "MAE": stats(values[:,1], observed[1])}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "paired_bootstrap_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); print(f"Saved: {output.resolve()}")


if __name__ == "__main__": main()
