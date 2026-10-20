#!/usr/bin/env python3
"""Export KIBA fold-setting-1 CSVs directly from GraphDTA/DeepDTA raw files.

This intentionally avoids GraphDTA's PyTorch-Geometric conversion and retains
the original matrix IDs, which are essential for a leakage/ambiguity audit.
"""
import argparse, json, pickle
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path, help="GraphDTA/data/kiba directory")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    raw = args.raw_dir
    needed = [raw / "Y", raw / "ligands_can.txt", raw / "proteins.txt", raw / "folds/train_fold_setting1.txt", raw / "folds/test_fold_setting1.txt"]
    missing = [str(x) for x in needed if not x.exists()]
    if missing: raise FileNotFoundError("Missing required GraphDTA KIBA files: " + ", ".join(missing))
    ligands, proteins = load_json(raw / "ligands_can.txt"), load_json(raw / "proteins.txt")
    with open(raw / "Y", "rb") as f: affinity = np.asarray(pickle.load(f, encoding="latin1"), dtype=float)
    drug_ids, target_ids = list(ligands), list(proteins)
    if affinity.shape != (len(drug_ids), len(target_ids)):
        raise ValueError(f"Y shape {affinity.shape} does not match {len(drug_ids)} drugs x {len(target_ids)} targets")
    observed_rows, observed_cols = np.where(~np.isnan(affinity))
    train_fold = [i for block in load_json(raw / "folds/train_fold_setting1.txt") for i in block]
    test_fold = load_json(raw / "folds/test_fold_setting1.txt")
    def records(indices):
        rows = []
        for index in indices:
            r, c = int(observed_rows[index]), int(observed_cols[index])
            molecule = Chem.MolFromSmiles(ligands[drug_ids[r]])
            if molecule is None: raise ValueError(f"Invalid source SMILES for {drug_ids[r]}")
            rows.append({"drug_id": drug_ids[r], "target_id": target_ids[c],
                         "compound_iso_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
                         "target_sequence": proteins[target_ids[c]], "affinity": float(affinity[r, c]),
                         "matrix_row": r, "matrix_col": c, "fold_index": int(index)})
        return pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train, test = records(train_fold), records(test_fold)
    train.to_csv(args.out_dir / "kiba_train.csv", index=False)
    test.to_csv(args.out_dir / "kiba_test.csv", index=False)
    manifest = {"raw_dir": str(raw.resolve()), "affinity_shape": list(affinity.shape), "observed_pairs": int(len(observed_rows)),
                "train_rows": len(train), "test_rows": len(test), "train_file": "kiba_train.csv", "test_file": "kiba_test.csv"}
    (args.out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2)); print(f"Saved: {args.out_dir.resolve()}")

if __name__ == "__main__": main()
