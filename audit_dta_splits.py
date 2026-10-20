#!/usr/bin/env python3
"""Audit a locked DTA split for label shift, shortcut baselines, and similarity.

All baseline parameters are fit on train.csv only.  Validation and test labels
are used solely to report metrics.  Protein similarity is explicitly a simple
3-mer Jaccard proxy, not sequence alignment identity.
"""
from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def concordance(y, p):
    order=np.argsort(y,kind="mergesort"); y,p=y[order],p[order]
    ranks=np.searchsorted(np.unique(p),p)+1; bit=np.zeros(ranks.max()+1,dtype=np.int64)
    def query(i):
        total=0
        while i: total+=bit[i]; i-=i&-i
        return total
    def add(i):
        while i<len(bit): bit[i]+=1; i+=i&-i
    score=pairs=previous=0.; i=0
    while i<len(y):
        j=i
        while j<len(y) and y[j]==y[i]: j+=1
        for k in range(i,j):
            less,upto=query(ranks[k]-1),query(ranks[k]); score+=less+.5*(upto-less); pairs+=previous
        for k in range(i,j): add(ranks[k]); previous+=1
        i=j
    return float(score/pairs) if pairs else float("nan")


def metrics(y, p):
    e=p-y; mse=float(np.mean(e*e)); denom=float(np.sum((y-y.mean())**2))
    pearson=float(np.corrcoef(y,p)[0,1]) if np.std(y) and np.std(p) else float("nan")
    return {"MSE":mse,"RMSE":math.sqrt(mse),"MAE":float(np.mean(np.abs(e))),
            "R2":float(1-np.sum(e*e)/denom) if denom else float("nan"),"Pearson":pearson,"CI":concordance(y,p)}


def label_summary(frame, split):
    y=frame["affinity"].to_numpy(float)
    return {"split":split,"rows":len(frame),"unique_drugs":int(frame["canonical_smiles"].nunique()),
            "unique_targets":int(frame["target_sequence"].nunique()),"mean":float(y.mean()),"std":float(y.std(ddof=1)),
            "min":float(y.min()),"p05":float(np.quantile(y,.05)),"p25":float(np.quantile(y,.25)),
            "median":float(np.median(y)),"p75":float(np.quantile(y,.75)),"p95":float(np.quantile(y,.95)),"max":float(y.max())}


def kmers(sequence, k=3):
    return {sequence[i:i+k] for i in range(max(0,len(sequence)-k+1))} or {sequence}


def protein_similarity(train_sequences, eval_sequences):
    train=[kmers(s) for s in train_sequences]
    rows=[]
    for seq in eval_sequences:
        value=kmers(seq); best=max(len(value&t)/len(value|t) for t in train)
        rows.append({"target_sequence":seq,"max_train_3mer_jaccard":float(best),"length":len(seq)})
    return pd.DataFrame(rows)


def drug_similarity(train_smiles, eval_smiles):
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    train_fps=[]
    for smi in train_smiles:
        mol=Chem.MolFromSmiles(smi)
        if mol is None: raise ValueError(f"Invalid SMILES: {smi}")
        train_fps.append(generator.GetFingerprint(mol))
    rows=[]
    for smi in eval_smiles:
        mol=Chem.MolFromSmiles(smi)
        if mol is None: raise ValueError(f"Invalid SMILES: {smi}")
        rows.append({"canonical_smiles":smi,"max_train_morgan_tanimoto":float(max(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(mol),train_fps)))})
    return pd.DataFrame(rows)


def summarize_similarity(frame, column, value):
    x=frame[value].to_numpy(float)
    return {"entity":column,"n":len(frame),"mean":float(x.mean()),"median":float(np.median(x)),
            "p05":float(np.quantile(x,.05)),"p25":float(np.quantile(x,.25)),"p75":float(np.quantile(x,.75)),"p95":float(np.quantile(x,.95)),"min":float(x.min()),"max":float(x.max())}


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir",required=True); ap.add_argument("--out-dir",required=True)
    args=ap.parse_args(); data=Path(args.data_dir); out=Path(args.out_dir)
    if out.exists(): raise ValueError(f"Output already exists: {out}")
    paths={n:data/f"{n}.csv" for n in ("train","validation","test")}
    frames={n:pd.read_csv(p) for n,p in paths.items()}
    required={"canonical_smiles","target_sequence","affinity"}
    for name,frame in frames.items():
        missing=required-set(frame.columns)
        if missing: raise ValueError(f"{name} missing {sorted(missing)}")
        frame["target_sequence"]=frame["target_sequence"].astype(str).str.strip().str.upper()
        frame["affinity"]=pd.to_numeric(frame["affinity"],errors="raise")
        if not np.isfinite(frame["affinity"]).all(): raise ValueError(f"{name} has nonfinite labels")
    train=frames["train"]; out.mkdir(parents=True)
    label_rows=[label_summary(frames[n],n) for n in ("train","validation","test")]
    pd.DataFrame(label_rows).to_csv(out/"label_distribution.csv",index=False)
    global_mean=float(train.affinity.mean()); drug_mean=train.groupby("canonical_smiles").affinity.mean(); target_mean=train.groupby("target_sequence").affinity.mean()
    baseline_rows=[]
    for split in ("validation","test"):
        frame=frames[split]; y=frame.affinity.to_numpy(float)
        predictions={"train_global_mean":np.full(len(frame),global_mean),
          "train_drug_mean":frame.canonical_smiles.map(drug_mean).fillna(global_mean).to_numpy(float),
          "train_target_mean":frame.target_sequence.map(target_mean).fillna(global_mean).to_numpy(float)}
        for method,pred in predictions.items():
            row={"split":split,"baseline":method,"unseen_drug_fraction":float((~frame.canonical_smiles.isin(drug_mean.index)).mean()),
                 "unseen_target_fraction":float((~frame.target_sequence.isin(target_mean.index)).mean()),**metrics(y,pred)}
            baseline_rows.append(row)
    pd.DataFrame(baseline_rows).to_csv(out/"train_only_mean_baselines.csv",index=False)
    report={"source_sha256":{n:sha256(p) for n,p in paths.items()},"label_distribution":label_rows,
            "train_only_mean_baselines":baseline_rows,"entity_overlap":{}}
    for entity in ("canonical_smiles","target_sequence"):
        sets={n:set(frames[n][entity]) for n in frames}
        report["entity_overlap"][entity]={"train_validation":len(sets["train"]&sets["validation"]),"train_test":len(sets["train"]&sets["test"]),"validation_test":len(sets["validation"]&sets["test"])}
    sim_report={}
    for split in ("validation","test"):
        drug=drug_similarity(sorted(set(train.canonical_smiles)),sorted(set(frames[split].canonical_smiles)))
        protein=protein_similarity(sorted(set(train.target_sequence)),sorted(set(frames[split].target_sequence)))
        drug.to_csv(out/f"{split}_drug_similarity.csv",index=False); protein.to_csv(out/f"{split}_protein_similarity.csv",index=False)
        sim_report[split]={"drug_to_train":summarize_similarity(drug,"drug","max_train_morgan_tanimoto"),"protein_to_train":summarize_similarity(protein,"target","max_train_3mer_jaccard")}
    report["similarity_to_train"]=sim_report
    (out/"audit_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2)); print(f"Saved: {out.resolve()}")


if __name__=="__main__": main()
