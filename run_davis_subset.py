#!/usr/bin/env python3
"""Run existing HGTA trainers with frozen ID-preserving partitions."""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=['baseline','uq'],required=True)
    ap.add_argument('--data-dir',default='davis_unambiguous_v1')
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=[4221,17,71,314,2025])
    ap.add_argument('--epochs',type=int,default=300)
    ap.add_argument('--patience',type=int,default=35)
    ap.add_argument('--batch-size',type=int,default=16)
    args=ap.parse_args()
    if len(set(args.seeds))!=len(args.seeds): raise ValueError('Repeated seeds')
    root=Path(args.out_dir)
    if root.exists(): raise ValueError('Output directory already exists; choose a new name')
    data=Path(args.data_dir).resolve()
    manifest=json.loads((data/'subset_manifest.json').read_text())
    for name,sha in manifest['sha256'].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest()!=sha: raise ValueError('Changed split file: '+name)
    module=importlib.import_module('journal_hgta' if args.model=='baseline' else 'uq_hgta')
    pd=module.pd; np=module.np
    partitions={}
    all_pairs=set(); all_inputs=set(); all_indices=set()
    for split in ('train','validation','test'):
        frame=pd.read_csv(data/(split+'.csv'),dtype={'drug_id':str,'target_id':str})
        needed={'pair_index','drug_id','target_id','canonical_smiles',module.SMI,module.SEQ,module.Y}
        if not needed <= set(frame): raise ValueError('Missing ID/feature fields in '+split)
        if frame.empty or frame[list(needed)].isna().any().any(): raise ValueError('Empty or missing values')
        if not np.isfinite(frame[module.Y].to_numpy(dtype=float)).all(): raise ValueError('Nonfinite labels')
        for r in frame.to_dict('records'):
            pair=(r['drug_id'],r['target_id']); inp=(r['canonical_smiles'],r[module.SEQ]); idx=r['pair_index']
            if pair in all_pairs or inp in all_inputs or idx in all_indices: raise ValueError('Overlapping/duplicate partitions')
            all_pairs.add(pair); all_inputs.add(inp); all_indices.add(idx)
        # Legacy dataset API expects these fields; no dispersion is inferred.
        frame['affinity_std']=0.0; frame['measurement_count']=1
        partitions[split]=frame
    # Adapt only data entry points. Existing model/training implementation is retained.
    def load_ids(path,name):
        return partitions['train' if name=='train' else 'test'].copy()
    def frozen_split(frame,fraction,mode,seed):
        return partitions['train'].copy(),partitions['validation'].copy()
    module.load_and_audit=load_ids
    module.split_train_val=frozen_split
    root.mkdir(parents=True)
    source_path=Path(module.__file__)
    metadata={'model':args.model,'data_manifest':manifest,'trainer_file':source_path.name,
              'trainer_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
              'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'actual_partition_policy':'Saved train/validation/test files; legacy val_fraction is ignored.',
              'note':'This trainer evaluates test once per seed. Freeze settings before full run.'}
    (root/'fixed_split_manifest.json').write_text(json.dumps(metadata,indent=2))
    for split,frame in partitions.items():
        frame[['pair_index','drug_id','target_id']].to_csv(root/(split+'_membership.csv'),index=False)
    sys.argv=[source_path.name,'--train-csv',str(data/'train.csv'),'--test-csv',str(data/'test.csv'),
              '--out-dir',str(root),'--epochs',str(args.epochs),'--patience',str(args.patience),
              '--batch-size',str(args.batch_size),'--seeds']+list(map(str,args.seeds))
    module.main()


if __name__=='__main__': main()
