#!/usr/bin/env python3
"""Replay independently inferred, per-sender constant position offsets on clean
InTAS receiver logs and verify the 2-D positions against an existing NVBD CSV.

Publication-readiness Comment #16, RECONSTRUCTED REPLAY (not original generator).
The offset manifest is inferred from NVBD, so exact replay is *not* independent
proof of the original PRNG seed, attacker selection, or full 52-column generator.
Input sources are read-only. New output directory must not already exist.

Example:
  py NAGT_Comment16_Replay_Offsets.py --dataset dataset.csv \
    --manifest comment16_source_audit_fixed/inferred_sender_offsets.csv \
    --train-source Train/InTAS_highway_2.zip \
    --validation-source Validation/InTAS_highway_2.zip \
    --test-source Test/InTAS_highway_2.zip \
    --outdir comment16_replay_verified
"""
from __future__ import annotations
import argparse
from pathlib import Path
from zipfile import ZipFile
import json
import hashlib
import sys
import numpy as np
import pandas as pd

SPLITS = ('Train', 'Validation', 'Test')
KEY = ['split', 'source_vehicle_file', 'rcvTime', 'sender_id', 'messageID']

def load_messages(raw: bytes, label: str):
    text = raw.decode('utf-8-sig')
    try:
        x = json.loads(text)
    except json.JSONDecodeError:
        x = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(x, dict) and isinstance(x.get('messages'), list):
        x = x['messages']
    if not isinstance(x, list):
        raise ValueError(f'{label}: expected list of messages or JSONL')
    return x

def coords(v, label):
    if isinstance(v, str):
        v = [s.strip() for s in v.split(',')]
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        raise ValueError(f'{label}: invalid position format')
    xy = float(v[0]), float(v[1])
    if not all(np.isfinite(xy)):
        raise ValueError(f'{label}: non-finite coordinates')
    return xy

def source_rows(path: Path, split: str):
    if not path.is_file() or path.suffix.lower() != '.zip':
        raise ValueError(f'Missing ZIP input for {split}: {path}')
    rows = []
    with ZipFile(path) as z:
        files = [n for n in z.namelist()
                 if Path(n).name.startswith('veh_') and n.lower().endswith(('.json', '.jsonl'))]
        if not files:
            raise ValueError(f'No veh_*.json entries in {path}')
        for member in files:
            name = Path(member).name
            for m in load_messages(z.read(member), member):
                sx, sy = coords(m['sender']['pos'], member)
                rows.append((split, name, int(m['rcvTime']), str(m['sender_id']),
                             int(m['messageID']), sx, sy))
    return pd.DataFrame(rows, columns=KEY + ['source_x', 'source_y'])

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--train-source', type=Path, required=True)
    p.add_argument('--validation-source', type=Path, required=True)
    p.add_argument('--test-source', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--tolerance', type=float, default=1e-5)
    args = p.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f'Output directory already exists: {args.outdir}')
    if args.tolerance <= 0:
        raise ValueError('Tolerance must be strictly positive')
    m = pd.read_csv(args.manifest)
    required = ['split','sender_id','offset_x','offset_y']
    missing = set(required) - set(m.columns)
    if missing:
        raise ValueError(f'Manifest missing {sorted(missing)}')
    m = m[required].copy()
    m['sender_id'] = m['sender_id'].astype(str)
    if m.duplicated(['split', 'sender_id']).any():
        raise ValueError('Manifest contains duplicate sender/split groups')
    if m[['offset_x','offset_y']].isna().any().any():
        raise ValueError('Manifest has missing offset coordinates')
    if not np.isfinite(m[['offset_x','offset_y']].to_numpy(dtype=float)).all():
        raise ValueError('Manifest contains non-finite offsets')
    sources = [args.train_source,args.validation_source,args.test_source]
    raw = pd.concat([source_rows(path, split) for path,split in zip(sources,SPLITS)], ignore_index=True)
    if raw.duplicated(KEY).any():
        raise ValueError('Duplicate reception keys in the input source logs')
    raw = raw.merge(m, how='left', on=['split','sender_id'], validate='many_to_one')
    raw['offset_x'] = raw['offset_x'].fillna(0.0)
    raw['offset_y'] = raw['offset_y'].fillna(0.0)
    raw['replayed_sender_pos_x'] = raw['source_x'] + raw['offset_x']
    raw['replayed_sender_pos_y'] = raw['source_y'] + raw['offset_y']
    d = pd.read_csv(args.dataset, low_memory=False)
    needed = KEY + ['attacker', 'sender_pos_x', 'sender_pos_y']
    if set(needed) - set(d.columns):
        raise ValueError(f'Dataset missing {sorted(set(needed)-set(d.columns))}')
    d = d[needed].copy()
    d['source_vehicle_file'] = d['source_vehicle_file'].astype(str).map(lambda s: Path(s).name)
    d['sender_id'] = d['sender_id'].astype(str)
    for col in ['rcvTime','messageID']:
        d[col] = pd.to_numeric(d[col], errors='raise').astype('int64')
    if d.duplicated(KEY).any():
        raise ValueError('Duplicate reception keys in the reference dataset')
    combined = raw.merge(d, on=KEY, how='outer', validate='one_to_one', indicator=True)
    unpaired = int((combined['_merge'] != 'both').sum())
    if unpaired:
        raise ValueError(f'{unpaired} unmatched reception rows: incorrect source or dataset')
    tol = args.tolerance
    dx = (combined['replayed_sender_pos_x'] - combined['sender_pos_x']).abs()
    dy = (combined['replayed_sender_pos_y'] - combined['sender_pos_y']).abs()
    failures = int(((dx > tol) | (dy > tol)).sum())
    wrong_label = int((combined['attacker'].astype(int) !=
                       ((combined['offset_x'].abs() > tol) | (combined['offset_y'].abs() > tol)).astype(int)).sum())
    if failures or wrong_label:
        raise ValueError(f'FAILED: {failures} coordinate mismatches and {wrong_label} attacker-label mismatches')
    if len(m) != int((combined[combined['attacker'].eq(1)][['split','sender_id']].drop_duplicates()).shape[0]):
        raise ValueError('Manifest does not have exactly one entry per attacker sender/split group')
    lines = [
        'Comment #16: independent replay using INFERRED offset manifest',
        'Important: reconstructed replay, not the original offset-generation program.',
        f'Source log rows matched: {len(combined):,}',
        f'Attacker sender/split groups replayed: {len(m)}',
        f'Position mismatches at tolerance {tol}: {failures}',
        f'Attacker label mismatches: {wrong_label}',
        f'Max absolute position replay error: {max(dx.max(),dy.max()):.12g}',
        'Result: PASSED',
        'Limitations: offsets were recovered from NVBD; original PRNG seed and',
        'full 52-column construction pipeline are not established by this test.',
        'Licensing/attribution must be verified separately against exact upstream release.',
    ]
    args.outdir.mkdir(parents=True)
    view = combined[KEY + ['source_x','source_y','offset_x','offset_y',
                           'replayed_sender_pos_x','replayed_sender_pos_y']].copy()
    view.to_csv(args.outdir/'replayed_positions.csv', index=False, float_format='%.12g')
    (args.outdir/'replay_report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'Files saved under: {args.outdir}')

if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, OSError) as exc:
        print('REPLAY STOPPED:', exc, file=sys.stderr)
        sys.exit(2)
