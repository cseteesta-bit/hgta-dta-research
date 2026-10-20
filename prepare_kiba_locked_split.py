#!/usr/bin/env python3
"""Create an auditable, fixed KIBA train/validation/test partition.

Input files must be GraphDTA-style kiba_train.csv and kiba_test.csv.  Unlike
the earlier exploratory Davis workflow, this script NEVER median-aggregates
different labels: it stops when it detects them.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except ImportError as e:
    raise SystemExit("RDKit is required. In the Sho2 environment run: conda install -c conda-forge rdkit") from e


def choose(df, choices, name):
    col = next((c for c in choices if c in df.columns), None)
    if col is None: raise ValueError(f"Cannot find {name}; columns are {list(df.columns)}")
    return col


def normalise(path, split):
    raw = pd.read_csv(path)
    smi = choose(raw, ("compound_iso_smiles", "smiles", "SMILES"), "SMILES")
    seq = choose(raw, ("target_sequence", "sequence", "target"), "target sequence")
    label = choose(raw, ("affinity", "label", "y", "Y"), "affinity label")
    out = raw[[smi, seq, label]].copy(); out.columns = ["smiles", "target_sequence", "affinity"]
    out = out.dropna().reset_index(drop=True)
    out["smiles"] = out.smiles.astype(str).str.strip()
    out["target_sequence"] = out.target_sequence.astype(str).str.strip().str.upper()
    out["affinity"] = pd.to_numeric(out.affinity, errors="raise")
    def canonical(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None: raise ValueError(f"Invalid SMILES in {path}: {s!r}")
        return Chem.MolToSmiles(mol, canonical=True)
    out["canonical_smiles"] = out.smiles.map(canonical)
    out["drug_id"] = out.canonical_smiles.map(lambda x: "drug_" + hashlib.sha256(x.encode()).hexdigest()[:16])
    out["target_id"] = out.target_sequence.map(lambda x: "target_" + hashlib.sha256(x.encode()).hexdigest()[:16])
    out["source_split"] = split
    return out[["drug_id", "target_id", "canonical_smiles", "target_sequence", "affinity", "source_split"]]


def deduplicate_exact(df, name, audit):
    keys = ["drug_id", "target_id"]
    duplicated = df.duplicated(keys, keep=False)
    groups = df.loc[duplicated].groupby(keys, sort=False)
    conflicting = sum(g.affinity.nunique() > 1 for _, g in groups)
    audit[f"{name}_duplicate_pair_groups"] = int(len(groups))
    audit[f"{name}_conflicting_duplicate_groups"] = int(conflicting)
    if conflicting:
        raise ValueError(f"{name} has {conflicting} duplicate pair groups with different labels. Stop: reconstruct target/drug identifiers from the original KIBA source before training.")
    return df.drop_duplicates(keys, keep="first").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--test-csv", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--val-fraction", type=float, default=.15)
    p.add_argument("--seed", type=int, default=20260915)
    a = p.parse_args()
    if not 0 < a.val_fraction < 0.5: raise ValueError("--val-fraction must be between 0 and 0.5")
    audit = {"input_train": str(a.train_csv), "input_test": str(a.test_csv), "seed": a.seed, "val_fraction": a.val_fraction}
    train, test = normalise(a.train_csv, "train"), normalise(a.test_csv, "test")
    audit["raw_train_rows"], audit["raw_test_rows"] = len(train), len(test)
    train, test = deduplicate_exact(train, "train", audit), deduplicate_exact(test, "test", audit)
    keys = ["drug_id", "target_id"]
    leaked = train.merge(test[keys], on=keys, how="inner")
    audit["cross_split_duplicate_pairs_removed_from_train"] = int(len(leaked))
    if len(leaked): train = train.merge(test[keys], on=keys, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge").reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    val_n = int(round(len(train) * a.val_fraction))
    val_positions = set(rng.choice(len(train), size=val_n, replace=False).tolist())
    mask = np.fromiter((i in val_positions for i in range(len(train))), dtype=bool, count=len(train))
    val, final_train = train.loc[mask].drop(columns="source_split"), train.loc[~mask].drop(columns="source_split")
    test = test.drop(columns="source_split")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", final_train), ("val", val), ("test", test)):
        frame.to_csv(a.out_dir / f"{name}.csv", index=False)
        audit[f"{name}_rows"] = len(frame)
        audit[f"{name}_sha256"] = hashlib.sha256((a.out_dir / f"{name}.csv").read_bytes()).hexdigest()
    audit["unique_drugs_test"], audit["unique_targets_test"] = int(test.drug_id.nunique()), int(test.target_id.nunique())
    (a.out_dir / "audit_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2)); print(f"Saved locked KIBA split: {a.out_dir.resolve()}")

if __name__ == "__main__": main()
