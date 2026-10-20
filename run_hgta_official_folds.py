#!/usr/bin/env python3
"""Train the current HGTA model on an *unchanged* official train/test fold.

This is a benchmark-protocol runner, deliberately separate from the strict
cold-split experiments.  It preserves every supplied CSV row and label: no
deduplication, median aggregation, overlap removal, or split rebuilding is
performed.  The validation partition is drawn only from the supplied training
CSV and is used solely for checkpoint selection; the supplied test CSV is read
only after that selection has completed.

The model implementation is imported from ``journal_hgta.py``.  Therefore this
script is an official-fold HGTA baseline, not a claimed reproduction of GraphDTA
or a SOTA claim.  Its purpose is to establish a fair, auditable starting point
before making HGTA-v2 architectural changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import journal_hgta as hgta


def stable_id(prefix: str, value: str) -> str:
    """Create a deterministic output identifier when a source CSV lacks one."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def load_raw_fold(path: Path, split: str) -> pd.DataFrame:
    """Validate a raw fold while retaining every original row and label."""
    frame = pd.read_csv(path)
    needed = [hgta.SMI, hgta.SEQ, hgta.Y]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(f"{split}: missing required columns {missing} in {path}")
    if frame.empty:
        raise ValueError(f"{split}: empty CSV")

    # Work on a copy, retaining all source columns and all source rows.
    frame = frame.copy()
    frame[hgta.SMI] = frame[hgta.SMI].astype(str).str.strip()
    frame[hgta.SEQ] = frame[hgta.SEQ].astype(str).str.strip().str.upper()
    frame[hgta.Y] = pd.to_numeric(frame[hgta.Y], errors="coerce")
    if frame[[hgta.SMI, hgta.SEQ, hgta.Y]].isna().any().any():
        raise ValueError(f"{split}: missing SMILES, sequence, or affinity value")
    if not np.isfinite(frame[hgta.Y].to_numpy(dtype=float)).all():
        raise ValueError(f"{split}: non-finite affinity value")

    frame["canonical_smiles"] = [hgta.canonical_smiles(s) for s in frame[hgta.SMI]]
    invalid = int(frame["canonical_smiles"].isna().sum())
    if invalid:
        raise ValueError(f"{split}: {invalid} invalid SMILES; correct the raw source first")

    valid_aa = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")
    bad_sequence = frame[hgta.SEQ].map(lambda s: not s or not set(s) <= valid_aa)
    if bad_sequence.any():
        raise ValueError(f"{split}: {int(bad_sequence.sum())} empty or invalid sequences")

    # These are derived output keys only; they never affect fitting or splitting.
    if "drug_id" not in frame:
        frame["drug_id"] = frame["canonical_smiles"].map(lambda s: stable_id("drug", s))
    else:
        frame["drug_id"] = frame["drug_id"].astype(str)
    if "target_id" not in frame:
        frame["target_id"] = frame[hgta.SEQ].map(lambda s: stable_id("target", s))
    else:
        frame["target_id"] = frame["target_id"].astype(str)

    # train_one() uses these through its generic dataset interface.  They are
    # constant for point-prediction HGTA and do not alter the affinity label.
    frame["affinity_std"] = 0.0
    frame["measurement_count"] = 1
    return frame.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official-fold, row-preserving HGTA benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-name", required=True, help="Metadata label, e.g. kiba_graphdta_fold1")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default="official_esm2_cache",
                        help="Shared frozen ESM embedding cache; contains no labels")
    parser.add_argument("--seeds", nargs="+", type=int, default=[4221, 17, 71, 314, 2025])
    parser.add_argument("--validation-seed", type=int, default=20261020)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--protein-model", default="facebook/esm2_t12_35M_UR50D")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be unique")
    root = Path(args.out_dir)
    if root.exists():
        raise ValueError(f"Output directory already exists; choose a new name: {root}")
    train_path, test_path = Path(args.train_csv).resolve(), Path(args.test_csv).resolve()
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Both --train-csv and --test-csv must exist")

    train_all = load_raw_fold(train_path, "train")
    test = load_raw_fold(test_path, "test")
    train, validation = hgta.split_train_val(
        train_all, args.val_fraction, "pair", args.validation_seed
    )

    cfg = hgta.Config(
        train_csv=str(train_path), test_csv=str(test_path), out_dir=str(root),
        protein_model=args.protein_model, seeds=tuple(args.seeds),
        val_fraction=args.val_fraction, val_split="pair", batch_size=args.batch_size,
        epochs=args.epochs, patience=args.patience, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, dropout=args.dropout, hidden_dim=args.hidden_dim,
    )
    root.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = hgta.ProteinCache(cfg, device, Path(args.cache_dir).resolve())
    first_sequence = pd.concat([train[hgta.SEQ], validation[hgta.SEQ], test[hgta.SEQ]], ignore_index=True).iloc[0]
    protein_dim = cache.get(first_sequence).shape[1]

    manifest = {
        "protocol": "official_fold_preserved_rows_validation_selected_v1",
        "dataset_name": args.dataset_name,
        "statement": (
            "Every supplied source row and affinity label is retained. Derived columns "
            "support graph construction/output keys only. Validation is sampled only from "
            "the provided training CSV; test prediction occurs once after checkpoint selection."
        ),
        "source": {
            "train_csv": str(train_path), "test_csv": str(test_path),
            "train_sha256": hgta.sha256_file(str(train_path)),
            "test_sha256": hgta.sha256_file(str(test_path)),
            "raw_train_rows": len(train_all), "raw_test_rows": len(test),
            "raw_train_columns": list(pd.read_csv(train_path, nrows=0).columns),
            "raw_test_columns": list(pd.read_csv(test_path, nrows=0).columns),
        },
        "validation": {
            "method": "pair_random_split_of_provided_training_rows_only",
            "seed": args.validation_seed, "fraction": args.val_fraction,
            "fit_rows": len(train), "validation_rows": len(validation),
        },
        "config": asdict(cfg), "device": str(device),
        "derived_columns": ["canonical_smiles", "drug_id", "target_id", "affinity_std", "measurement_count"],
    }
    (root / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2))

    results = [
        hgta.train_one(cfg, seed, train, validation, test, cache, protein_dim, device, root)
        for seed in args.seeds
    ]
    metrics_frame = pd.DataFrame(results)
    metrics_frame.insert(0, "seed", args.seeds)
    metrics_frame.to_csv(root / "all_seed_metrics.csv", index=False)
    summary = {
        column: {"mean": float(metrics_frame[column].mean()),
                 "std": float(metrics_frame[column].std(ddof=1))}
        for column in metrics_frame.columns if column != "seed"
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved: {root.resolve()}")


if __name__ == "__main__":
    main()
