#!/usr/bin/env python3
"""Convert audited KIBA locked splits into the frozen HGTA runner schema."""
import argparse, hashlib, importlib, json
from pathlib import Path
import pandas as pd

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path, help="Directory from prepare_kiba_locked_split_v2.py")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--trainer-module", default="journal_hgta")
    args=ap.parse_args()
    if args.out_dir.exists(): raise ValueError("Output directory already exists; choose a new name.")
    trainer=importlib.import_module(args.trainer_module)
    smi, seq, label = trainer.SMI, trainer.SEQ, trainer.Y
    source={"train":"train.csv", "validation":"val.csv", "test":"test.csv"}
    frames={}; next_index=0
    for split, name in source.items():
        path=args.input_dir/name
        if not path.exists(): raise FileNotFoundError(path)
        df=pd.read_csv(path, dtype={"drug_id":str,"target_id":str})
        basic={"drug_id","target_id","canonical_smiles","target_sequence","affinity"}
        if not basic <= set(df): raise ValueError(f"{path} missing {sorted(basic-set(df))}")
        if smi not in df: df[smi]=df["canonical_smiles"]
        if seq not in df: df[seq]=df["target_sequence"]
        if label not in df: df[label]=df["affinity"]
        df["pair_index"]=range(next_index,next_index+len(df)); next_index+=len(df)
        frames[split]=df
    # IDs and feature inputs must be unique across the frozen partitions.
    all_rows=pd.concat(frames.values(),ignore_index=True)
    if all_rows.duplicated(["drug_id","target_id"]).any(): raise ValueError("Duplicate raw ID pairs across partitions")
    if all_rows.duplicated(["canonical_smiles",seq]).any(): raise ValueError("Duplicate model inputs across partitions")
    args.out_dir.mkdir(parents=True)
    hashes={}
    for split,df in frames.items():
        path=args.out_dir/f"{split}.csv"; df.to_csv(path,index=False); hashes[path.name]=digest(path)
    provenance=json.loads((args.input_dir/"audit_manifest.json").read_text()) if (args.input_dir/"audit_manifest.json").exists() else {}
    manifest={"dataset":"KIBA", "schema":"frozen_hgta_runner_v1", "source_directory":str(args.input_dir.resolve()),
              "source_audit":provenance, "trainer_module":args.trainer_module,
              "trainer_columns":{"SMI":smi,"SEQ":seq,"Y":label}, "sha256":hashes,
              "rows":{split:len(df) for split,df in frames.items()}}
    (args.out_dir/"subset_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(args.out_dir.resolve()),"rows":manifest["rows"],"trainer_columns":manifest["trainer_columns"]},indent=2))

if __name__=="__main__": main()
