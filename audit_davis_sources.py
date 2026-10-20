#!/usr/bin/env python3
"""Read-only Davis source audit. Requires RDKit in the existing Sho2 environment."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from rdkit import Chem

SMI, SEQ, Y = 'compound_iso_smiles', 'target_sequence', 'affinity'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', default='/home/cse1000/Downloads/HGTA/data/davis_train.csv')
    parser.add_argument('--test-csv', default='/home/cse1000/Downloads/HGTA/data/davis_test.csv')
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()
    out = Path(args.out_dir or ('davis_source_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    out.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    summary = {'sources': {}, 'interpretation': 'Conflicting labels are not assumed to be replicates. No records or labels modified.'}
    originals = []; fields = []; invalid = []; cache = {}
    for split, filename in [('train', args.train_csv), ('test', args.test_csv)]:
        path = Path(filename)
        with path.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if not {SMI, SEQ, Y} <= set(cols):
                raise ValueError('Missing required columns: ' + str(path))
            for col in cols:
                if col not in fields: fields.append(col)
            n = 0
            for line, row in enumerate(reader, start=2):
                n += 1
                s = row.get(SMI, '') or ''
                if s not in cache:
                    mol = Chem.MolFromSmiles(s)
                    cache[s] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None and mol.GetNumAtoms() else None
                canonical = cache[s]
                seq = (row.get(SEQ, '') or '').strip().upper()
                try: y = float(row[Y])
                except (ValueError, TypeError): y = float('nan')
                record = {'audit_split': split, 'audit_csv_record': line, **{'source_' + k: v for k,v in row.items()}}
                originals.append(record)
                if not canonical or not seq or not math.isfinite(y):
                    invalid.append(record); continue
                groups[(canonical, seq)].append((record, y, s))
        summary['sources'][split] = {'path': str(path), 'rows': n, 'columns': cols,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'additional_columns': [c for c in cols if c not in (SMI,SEQ,Y)]}
    reports = []; conflict_rows = []
    for (canonical, seq), entries in groups.items():
        labels = [x[1] for x in entries]
        if len(entries) < 2: continue
        gid = hashlib.sha256((canonical+'\0'+seq).encode()).hexdigest()
        differing = []
        for col in fields:
            if col not in (SMI,SEQ,Y) and len({x[0].get('source_'+col,'') for x in entries})>1:
                differing.append(col)
        report = {'pair_sha256':gid, 'canonical_smiles':canonical, 'sequence_sha256':hashlib.sha256(seq.encode()).hexdigest(),
            'sequence_length':len(seq), 'n_records':len(entries), 'n_distinct_labels':len(set(labels)),
            'label_min':min(labels), 'label_max':max(labels), 'label_range':max(labels)-min(labels),
            'n_raw_smiles':len({x[2] for x in entries}), 'splits':','.join(sorted({x[0]['audit_split'] for x in entries})),
            'differing_metadata_columns':','.join(differing)}
        reports.append(report)
        if len(set(labels))>1:
            for record, _, _ in entries: conflict_rows.append({'pair_sha256':gid, **record})
    def write(name, records, columns):
        with (out/name).open('w', newline='', encoding='utf-8') as f:
            writer=csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(records)
    reports.sort(key=lambda r:r['label_range'], reverse=True)
    reportcols=['pair_sha256','canonical_smiles','sequence_sha256','sequence_length','n_records','n_distinct_labels','label_min','label_max','label_range','n_raw_smiles','splits','differing_metadata_columns']
    rawcols=['audit_split','audit_csv_record']+['source_'+k for k in fields]
    write('duplicate_groups.csv', reports, reportcols)
    write('conflicting_original_records.csv',conflict_rows,['pair_sha256']+rawcols)
    write('invalid_records.csv',invalid,rawcols)
    summary.update({'unique_valid_canonical_pairs':len(groups), 'invalid_records':len(invalid),
        'duplicate_groups_across_combined_sources':len(reports),
        'conflicting_groups_across_combined_sources':sum(r['n_distinct_labels']>1 for r in reports),
        'cross_split_pair_groups':sum(r['splits']=='test,train' for r in reports),
        'top_10_conflicts':reports[:10],
        'next_evidence_needed':'Original dataset release/URL, preprocessing script and target/compound ID mappings; assay units/types/constructs if available. Three input columns alone cannot establish why labels differ.'})
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    archive = out.with_name(out.name+'.zip')
    with ZipFile(archive,'x',ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()): z.write(path,arcname=path.name)
    print('Audit complete. Originals unchanged.')
    print('Conflicting combined groups:',summary['conflicting_groups_across_combined_sources'])
    print('Additional source columns:',{k:v['additional_columns'] for k,v in summary['sources'].items()})
    print('Upload this ZIP:',archive.resolve())

if __name__ == '__main__': main()
