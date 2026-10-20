#!/usr/bin/env python3
"""Create a leakage-free KIBA subset with one raw record per model input.

Rows sharing (canonical_smiles, target_sequence) anywhere in train/validation/
test are excluded.  This is intentionally conservative: no labels are merged.
"""
import argparse, hashlib, importlib, json
from pathlib import Path
import pandas as pd

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-dir",required=True,type=Path); ap.add_argument("--out-dir",required=True,type=Path); ap.add_argument("--trainer-module",default="journal_hgta"); a=ap.parse_args()
    if a.out_dir.exists(): raise ValueError("Output directory already exists; choose a new name.")
    trainer=importlib.import_module(a.trainer_module); smi,seq,y=trainer.SMI,trainer.SEQ,trainer.Y
    source={"train":"train.csv","validation":"val.csv","test":"test.csv"}; frames=[]
    for split,file in source.items():
        df=pd.read_csv(a.input_dir/file,dtype={"drug_id":str,"target_id":str}); df["_split"]=split; frames.append(df)
    all_rows=pd.concat(frames,ignore_index=True); inputs=["canonical_smiles","target_sequence"]
    required={"drug_id","target_id","canonical_smiles","target_sequence","affinity"}
    if not required <= set(all_rows): raise ValueError("Input files lack required audited columns.")
    counts=all_rows.groupby(inputs,sort=False).size().rename("input_count")
    all_rows=all_rows.join(counts,on=inputs)
    retained=all_rows.query("input_count == 1").drop(columns="input_count")
    if retained.empty: raise ValueError("All rows were excluded; inspect the KIBA source.")
    audit={"rule":"Retain only model inputs occurring exactly once globally; never aggregate labels.","input_rows":len(all_rows),"excluded_repeated_model_inputs":int(len(all_rows)-len(retained)),"retained_rows":len(retained),"per_split":{}}
    a.out_dir.mkdir(parents=True); hashes={}; index=0
    for split in ("train","validation","test"):
        df=retained.query("_split == @split").drop(columns="_split").copy()
        if smi not in df: df[smi]=df["canonical_smiles"]
        if seq not in df: df[seq]=df["target_sequence"]
        if y not in df: df[y]=df["affinity"]
        df["pair_index"]=range(index,index+len(df)); index+=len(df)
        path=a.out_dir/f"{split}.csv"; df.to_csv(path,index=False); hashes[path.name]=sha(path); audit["per_split"][split]=len(df)
    manifest={"dataset":"KIBA", "schema":"input_unambiguous_hgta_v1", "source_directory":str(a.input_dir.resolve()),"audit":audit,"trainer_columns":{"SMI":smi,"SEQ":seq,"Y":y},"sha256":hashes}
    (a.out_dir/"subset_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2)); print(f"Saved: {a.out_dir.resolve()}")
if __name__=="__main__": main()
