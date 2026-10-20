#!/usr/bin/env python3
"""Build fixed sequence-unambiguous Davis partitions from a reconstruction ZIP."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', default='davis_reconstruction_20260909_130657.zip')
    ap.add_argument('--out-dir', default='davis_unambiguous_v1')
    ap.add_argument('--validation-fold', type=int, default=0)
    args = ap.parse_args()
    with zipfile.ZipFile(args.archive) as z:
        rows = list(csv.DictReader(io.StringIO(z.read('davis_id_preserving.csv').decode('utf-8-sig'))))
        source = json.loads(z.read('summary.json'))
    if not rows: raise ValueError('Empty reconstruction')
    folds = {int(r['official_train_fold']) for r in rows if r['official_split']=='train'}
    if args.validation_fold not in folds: raise ValueError('Unknown validation fold')
    seq_targets = defaultdict(set)
    for r in rows: seq_targets[r['target_sequence']].add(r['target_id'])
    kept = [r for r in rows if len(seq_targets[r['target_sequence']])==1]
    excluded = [r for r in rows if len(seq_targets[r['target_sequence']])>1]
    parts = {'train':[], 'validation':[], 'test':[]}
    for r in kept:
        split = 'test' if r['official_split']=='test' else ('validation' if int(r['official_train_fold'])==args.validation_fold else 'train')
        parts[split].append(r)
    seen = set(); seen_inputs = set()
    for name, data in parts.items():
        if not data: raise ValueError('Empty '+name)
        for r in data:
            key = (r['drug_id'],r['target_id'])
            inp = (r['canonical_smiles'],r['target_sequence'])
            if key in seen or inp in seen_inputs: raise ValueError('Repeated pair or representation')
            seen.add(key); seen_inputs.add(inp)
    out = Path(args.out_dir); out.mkdir(parents=True,exist_ok=False)
    hashes = {}
    for name,data in dict(parts, excluded_shared_sequence_targets=excluded).items():
        path=out/(name+'.csv')
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(data)
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'description':'Restricted sequence-unambiguous Davis benchmark; not full Davis and not a cold-start benchmark.',
        'source_revision':source['revision'], 'archive_sha256':hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),
        'validation_fold':args.validation_fold, 'counts':{k:len(v) for k,v in parts.items()},
        'retained_target_ids':len({r['target_id'] for r in kept}), 'excluded_rows':len(excluded),
        'sha256':hashes, 'selection':'Sequence maps to exactly one source target ID; no label-based selection or aggregation.'}
    (out/'subset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2)); print('Prepared:',out.resolve())


if __name__=='__main__': main()
