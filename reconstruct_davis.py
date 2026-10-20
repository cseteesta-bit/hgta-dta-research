#!/usr/bin/env python3
"""Reconstruct ID-preserving Davis records and audit existing CSVs. Python 3.8+."""
import argparse
import csv
import hashlib
import io
import json
import math
import pickle
import re
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from rdkit import Chem

REPO = 'https://api.github.com/repos/thinng/GraphDTA/commits/master'
FILES = ['ligands_can.txt', 'proteins.txt', 'Y',
         'folds/train_fold_setting1.txt', 'folds/test_fold_setting1.txt']
FIELDS = ['pair_index', 'drug_index', 'target_index', 'drug_id', 'target_id',
          'compound_iso_smiles', 'canonical_smiles', 'target_sequence',
          'sequence_sha256', 'kd_nM', 'affinity', 'official_split',
          'official_train_fold', 'sequence_target_count']


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Davis-provenance-audit'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


class MatrixUnpickler(pickle.Unpickler):
    """Only permit the NumPy constructors required for an ndarray pickle."""
    def find_class(self, module, name):
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == '_reconstruct':
            return np.core.multiarray._reconstruct
        raise pickle.UnpicklingError('Unapproved pickle global: %s.%s' % (module, name))


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonicalizer():
    cache = {}
    def canonical(s):
        if s not in cache:
            mol = Chem.MolFromSmiles(s)
            if mol is None or not mol.GetNumAtoms():
                raise ValueError('Invalid SMILES: ' + s)
            cache[s] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return cache[s]
    return canonical


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-csv', default='/home/cse1000/Downloads/HGTA/data/davis_train.csv')
    ap.add_argument('--test-csv', default='/home/cse1000/Downloads/HGTA/data/davis_test.csv')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--revision', default=None, help='Optional exact GraphDTA commit SHA')
    ap.add_argument('--label-tolerance', type=float, default=1e-6)
    args = ap.parse_args()
    if not math.isfinite(args.label_tolerance) or args.label_tolerance < 0:
        raise ValueError('Label tolerance must be finite and nonnegative')
    out = Path(args.out_dir or ('davis_reconstruction_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    out.mkdir(parents=True, exist_ok=False)
    revision = args.revision or json.loads(fetch(REPO))['sha']
    if not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('--revision must be a full 40-character commit SHA')
    base = 'https://raw.githubusercontent.com/thinng/GraphDTA/%s/data/davis/' % revision
    raw = {}; provenance = {}
    for name in FILES:
        print('Downloading', name, flush=True)
        data = fetch(base + name)
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        raw[name] = data
        provenance[name] = {'url': base + name, 'sha256': hashlib.sha256(data).hexdigest()}
    # Python 3.8 dictionaries retain insertion order. Never sort these axes.
    drugs = json.loads(raw['ligands_can.txt'])
    targets = json.loads(raw['proteins.txt'])
    matrix = MatrixUnpickler(io.BytesIO(raw['Y']), encoding='latin1').load()
    if not isinstance(matrix, np.ndarray) or matrix.dtype.kind not in 'fiu':
        raise ValueError('Y must be a numeric NumPy matrix')
    if matrix.shape != (len(drugs), len(targets)):
        raise ValueError('Matrix dimensions do not match dictionary axes')
    if np.isinf(matrix).any() or np.any(matrix[np.isfinite(matrix)] <= 0):
        raise ValueError('Invalid Kd values in source matrix')
    ri, ci = np.where(~np.isnan(matrix))
    train_folds = json.loads(raw['folds/train_fold_setting1.txt'])
    test_indices = json.loads(raw['folds/test_fold_setting1.txt'])
    flat = [x for fold in train_folds for x in fold] + test_indices
    if len(flat) != len(ri) or set(flat) != set(range(len(ri))):
        raise ValueError('Official folds do not partition the observed matrix entries exactly')
    fold_for = {idx: fold for fold, indices in enumerate(train_folds) for idx in indices}
    test_set = set(test_indices)
    canonical = canonicalizer()
    seq_ids = defaultdict(list)
    for tid, seq in targets.items():
        seq_ids[seq.strip().upper()].append(tid)
    ditems, titems = list(drugs.items()), list(targets.items())
    records = []; by_input = defaultdict(list)
    for idx, (i, j) in enumerate(zip(ri, ci)):
        did, smi = ditems[int(i)]; tid, seq = titems[int(j)]
        seq = seq.strip().upper(); kd = float(matrix[i, j])
        r = dict(zip(FIELDS, [idx, int(i), int(j), did, tid, smi, canonical(smi), seq,
            hashlib.sha256(seq.encode()).hexdigest(), kd, -math.log10(kd / 1e9),
            'test' if idx in test_set else 'train', fold_for.get(idx, ''), len(seq_ids[seq])]))
        records.append(r); by_input[(r['canonical_smiles'], seq)].append(r)
    write_csv(out/'davis_id_preserving.csv', FIELDS, records)
    for split in ('train', 'test'):
        write_csv(out/('official_%s_id_preserving.csv' % split), FIELDS,
                  [r for r in records if r['official_split'] == split])
    target_report = [{'target_id':tid, 'sequence_sha256':hashlib.sha256(seq.strip().upper().encode()).hexdigest(),
        'sequence_length':len(seq.strip()), 'shared_target_ids':json.dumps(seq_ids[seq.strip().upper()]),
        'requires_representation_review':len(seq_ids[seq.strip().upper()]) > 1}
        for tid,seq in titems]
    write_csv(out/'target_representation_audit.csv', list(target_report[0]), target_report)
    reconciliation = []; sources = {}; unique_matched = Counter(); all_possible = set()
    for split, filename in [('train', args.train_csv), ('test', args.test_csv)]:
        path = Path(filename)
        if not path.is_file():
            sources[split] = {'path':str(path), 'status':'missing; reconciliation skipped'}
            continue
        sources[split] = {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        with path.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if not {'compound_iso_smiles','target_sequence','affinity'} <= set(reader.fieldnames or []):
                raise ValueError('Missing input columns in '+str(path))
            for line, row in enumerate(reader, 2):
                try:
                    smi = canonical(row['compound_iso_smiles'])
                    seq = row['target_sequence'].strip().upper()
                    y = float(row['affinity'])
                    if not math.isfinite(y): raise ValueError('Nonfinite label')
                    candidates = by_input.get((smi, seq), [])
                    matches = [r for r in candidates if abs(r['affinity']-y) <= args.label_tolerance]
                    status = 'unique_label_input_match' if len(matches)==1 else ('ambiguous_label_input_match' if matches else ('label_mismatch' if candidates else 'input_not_found'))
                except (ValueError, TypeError, KeyError):
                    matches = []; candidates = []; status='invalid_input'
                if len(matches)==1: unique_matched[matches[0]['pair_index']]+=1
                all_possible.update(r['pair_index'] for r in matches)
                reconciliation.append({'csv_split':split,'csv_record':line,'status':status,
                    'n_input_candidates':len(candidates),'n_label_input_candidates':len(matches),
                    'candidate_pair_indices':json.dumps([r['pair_index'] for r in matches]),
                    'candidate_target_ids':json.dumps([r['target_id'] for r in matches]),
                    'candidate_official_splits':json.dumps([r['official_split'] for r in matches])})
    if reconciliation:
        write_csv(out/'csv_reconciliation.csv',list(reconciliation[0]),reconciliation)
    summary = {'revision':revision,'source_files':provenance,'existing_csvs':sources,
        'matrix_shape':list(matrix.shape),'observed_records':len(records),'drug_ids':len(drugs),
        'target_ids':len(targets),'unique_sequences':len(seq_ids),
        'sequence_groups_with_multiple_ids':sum(len(v)>1 for v in seq_ids.values()),
        'reconciliation_status_counts':dict(Counter(r['status'] for r in reconciliation)),
        'repeated_uniquely_matched_source_entries':{str(k):v for k,v in unique_matched.items() if v>1},
        'source_entries_with_no_possible_csv_match':len(records)-len(all_possible) if reconciliation else None,
        'limitations':['Matching inputs and labels is a consistency check, not definitive provenance.',
            'Ambiguous target identities are not resolved by choosing the first match.',
            'Official folds preserved; they are not claimed to be free of sequence overlap.',
            'No mutation, deletion, domain, or state annotation was converted into a sequence.',
            'Do not feed outputs into the old loader: it drops target IDs and aggregates distinct targets.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    archive=out.with_name(out.name+'.zip')
    with ZipFile(archive,'x',ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file(): z.write(path,arcname=str(path.relative_to(out)))
    print(json.dumps(summary,indent=2))
    print('\nUpload this ZIP:',archive.resolve())


if __name__ == '__main__': main()
