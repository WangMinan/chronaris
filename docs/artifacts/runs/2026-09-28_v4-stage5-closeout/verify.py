"""Verify the original exported tables without training code or server paths."""
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev

root = Path(__file__).resolve().parent
evidence = json.loads((root/'evidence.json').read_text())
for name, digest in evidence['data_files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, name
with (root/'data/setting_metrics.csv').open(encoding='utf-8') as handle: rows=list(csv.DictReader(handle))
with (root/'data/summary.csv').open(encoding='utf-8') as handle: summaries=list(csv.DictReader(handle))
assert len(rows)==288 and len(summaries)==32
keys=('domain','method','route','consumer','task','metric'); grouped=defaultdict(list)
for row in rows:grouped[tuple(row[k] for k in keys)].append(float(row['value']))
for row in summaries:
    values=grouped[tuple(row[k] for k in keys)];assert len(values)==9
    expected=dict(mean=mean(values),std_across_settings=pstdev(values),minimum=min(values),maximum=max(values))
    assert all(abs(float(row[k])-v)<1e-12 for k,v in expected.items())
with (root/'data/training.csv').open(encoding='utf-8') as handle: training=list(csv.DictReader(handle))
assert len(training)==72 and sum(int(r['new_updates']) for r in training)==12900
print('PASS: 36 units, 72 routes, 288 metric rows, 32 summaries, 12,900 new updates; original data hashes match.')
