#!/usr/bin/env python3
"""Create reproducible cold-drug, cold-target, or double-cold DTA partitions.

Input is an existing directory with train.csv, validation.csv, and test.csv.
All rows are concatenated and repartitioned without consulting affinity labels.
For cold-drug, canonical SMILES are disjoint across partitions.  For
cold-target, protein sequences are disjoint.  For double-cold, both molecular
and sequence groups are disjoint; cross-assigned pairs are excluded rather
than leaking an entity between splits.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""): h.update(part)
    return h.hexdigest()


def require(frame, columns):
    missing = sorted(set(columns) - set(frame.columns))
    if missing: raise ValueError(f"Missing required columns: {missing}")


def assign_groups(frame, column, fractions, seed):
    """Assign whole groups while approximately matching row-count fractions."""
    counts = frame.groupby(column, sort=False).size()
    # copy=True is essential: shuffling a view can mutate the pandas Index and
    # detach group labels from their counts.
    groups = counts.index.to_numpy(dtype=object, copy=True)
    rng = np.random.default_rng(seed); rng.shuffle(groups)
    # Large groups first, with shuffled ties, limits row-count imbalance.
    groups = sorted(groups, key=lambda g: -int(counts[g]))
    goals = np.asarray(fractions, dtype=float) * len(frame)
    totals = np.zeros(3, dtype=float); labels = {}
    for group in groups:
        size = int(counts[group])
        # Choose the assignment that minimizes the normalized squared distance
        # from requested row totals after this whole group is placed.
        candidate_errors = []
        for split in range(3):
            trial = totals.copy(); trial[split] += size
            candidate_errors.append(float(np.sum(np.square((trial - goals) / goals))))
        choice = int(np.argmin(candidate_errors))
        labels[group] = choice; totals[choice] += size
    return labels, totals.astype(int)


def audit(parts, mode):
    drug_sets = [set(p["canonical_smiles"]) for p in parts]
    target_sets = [set(p["target_sequence"]) for p in parts]
    names = ("train", "validation", "test")
    for i in range(3):
        if parts[i].empty: raise ValueError(f"{names[i]} is empty")
        for j in range(i+1, 3):
            if mode in ("drug", "both") and drug_sets[i] & drug_sets[j]:
                raise RuntimeError("Drug overlap after split")
            if mode in ("target", "both") and target_sets[i] & target_sets[j]:
                raise RuntimeError("Target overlap after split")
    inputs = [set(zip(p["canonical_smiles"], p["target_sequence"])) for p in parts]
    for i in range(3):
        for j in range(i+1, 3):
            if inputs[i] & inputs[j]: raise RuntimeError("Exact model-input overlap after split")
    return {"rows": {n: len(p) for n, p in zip(names, parts)},
            "unique_drugs": {n: int(p["canonical_smiles"].nunique()) for n, p in zip(names, parts)},
            "unique_targets": {n: int(p["target_sequence"].nunique()) for n, p in zip(names, parts)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=("drug", "target", "both"), required=True)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--train-fraction", type=float, default=.70)
    ap.add_argument("--validation-fraction", type=float, default=.10)
    args = ap.parse_args()
    if not 0 < args.train_fraction < 1 or not 0 < args.validation_fraction < 1:
        raise ValueError("Fractions must be between zero and one")
    test_fraction = 1 - args.train_fraction - args.validation_fraction
    if test_fraction <= 0: raise ValueError("train_fraction + validation_fraction must be < 1")
    source, output = Path(args.input_dir), Path(args.out_dir)
    if output.exists(): raise ValueError(f"Output already exists: {output}")
    source_paths = {n: source / f"{n}.csv" for n in ("train", "validation", "test")}
    if any(not p.exists() for p in source_paths.values()): raise FileNotFoundError(source_paths)
    frames = []
    for split, path in source_paths.items():
        frame = pd.read_csv(path)
        require(frame, ("canonical_smiles", "target_sequence", "affinity"))
        frame["_source_split"] = split
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    if all_rows.duplicated(["canonical_smiles", "target_sequence"]).any():
        raise ValueError("Input contains repeated exact model inputs; use the input-unambiguous dataset first.")
    fractions = (args.train_fraction, args.validation_fraction, test_fraction)
    if args.mode == "drug":
        labels, expected = assign_groups(all_rows, "canonical_smiles", fractions, args.seed)
        assigned = all_rows["canonical_smiles"].map(labels).to_numpy()
        parts = [all_rows.loc[assigned == i].drop(columns="_source_split").reset_index(drop=True) for i in range(3)]
        excluded = all_rows.iloc[0:0].copy()
    elif args.mode == "target":
        labels, expected = assign_groups(all_rows, "target_sequence", fractions, args.seed)
        assigned = all_rows["target_sequence"].map(labels).to_numpy()
        parts = [all_rows.loc[assigned == i].drop(columns="_source_split").reset_index(drop=True) for i in range(3)]
        excluded = all_rows.iloc[0:0].copy()
    else:
        # Independent entity assignments retain only diagonal drug/target cells.
        # If their allocation probabilities were (0.7, 0.1, 0.2), retained
        # rows would scale roughly with their squares and be badly imbalanced.
        # Allocate each entity family proportional to sqrt(desired fraction),
        # so diagonal retained rows target the requested 70/10/20 proportions.
        latent_fractions = np.sqrt(np.asarray(fractions, dtype=float))
        latent_fractions = tuple((latent_fractions / latent_fractions.sum()).tolist())
        drug_labels, _ = assign_groups(all_rows, "canonical_smiles", latent_fractions, args.seed)
        target_labels, _ = assign_groups(all_rows, "target_sequence", latent_fractions, args.seed + 1)
        d = all_rows["canonical_smiles"].map(drug_labels).to_numpy()
        t = all_rows["target_sequence"].map(target_labels).to_numpy()
        keep = d == t
        parts = [all_rows.loc[keep & (d == i)].drop(columns="_source_split").reset_index(drop=True) for i in range(3)]
        excluded = all_rows.loc[~keep].copy()
        expected = np.array([len(p) for p in parts])
    actual = np.asarray([len(p) for p in parts])
    if not np.array_equal(expected, actual):
        raise RuntimeError(f"Internal group-assignment error: expected {expected.tolist()}, got {actual.tolist()}")
    summary = audit(parts, args.mode)
    output.mkdir(parents=True)
    names = ("train", "validation", "test")
    for name, frame in zip(names, parts): frame.to_csv(output / f"{name}.csv", index=False)
    if len(excluded): excluded.to_csv(output / "excluded_cross_partition_pairs.csv", index=False)
    hashes = {f"{name}.csv": sha256(output / f"{name}.csv") for name in names}
    manifest = {"protocol": f"cold_{args.mode}_v1", "mode": args.mode, "seed": args.seed,
                "fractions": {"train": args.train_fraction, "validation": args.validation_fraction, "test": test_fraction},
                "source_files": {name: sha256(path) for name, path in source_paths.items()},
                "sha256": hashes, "excluded_cross_partition_pairs": int(len(excluded)),
                "row_targets_before_double_cold_filter": expected.tolist(),
                "double_cold_entity_allocation_fractions": latent_fractions if args.mode == "both" else None,
                **summary}
    (output / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__": main()
