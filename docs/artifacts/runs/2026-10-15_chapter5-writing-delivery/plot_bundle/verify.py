"""Validate portable data hashes, aggregate values, scope and generated image coverage."""
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean,pstdev


def verify(root):
    evidence=json.loads((root/'evidence.json').read_text())
    for name,digest in evidence['data_files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('nonportable data path')
        with path.open('rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=digest:raise ValueError('data changed: '+name)
    def rows(name):
        with (root/'data'/name).open(newline='',encoding='utf-8') as handle:return list(csv.DictReader(handle))
    for raw,summary,keys in (
        ('native_metrics.csv','native_summary.csv',('domain','method','variant','route','consumer','task','metric','error_units','role')),
        ('simulation_groups.csv','simulation_summary.csv',('method','variant','route','consumer','task','metric','condition','role'))):
        groups=defaultdict(list)
        for r in rows(raw):groups[tuple(r[k] for k in keys)].append(r['value'])
        output=rows(summary)
        if len(output)!=len(groups):raise ValueError('aggregate coverage mismatch')
        for r in output:
            all_values=groups[tuple(r[k] for k in keys)];values=[float(v) for v in all_values if v!='']
            if len(all_values)!=int(r['settings']) or len(values)!=int(r['available_settings']):raise ValueError('aggregation count changed')
            for k,v in {'mean':mean(values),'std':pstdev(values)}.items() if values else []:
                if not math.isclose(float(r[k]),v,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('aggregation mismatch')
    figures=json.loads((root/'figures/figure_inputs.json').read_text())
    if not figures:raise ValueError('no figures')
    for fig in figures:
        for name in fig['inputs']+[fig['script'],fig['style']]:
            if not (root/name).is_file():raise ValueError('missing figure input')
        for ext in ('png','svg'):
            if not (root/'figures'/f"{fig['figure']}.{ext}").is_file():raise ValueError('missing figure')
    print('PASS',len(evidence['data_files']),'data files and',len(figures),'figures')


if __name__=='__main__':verify(Path(__file__).resolve().parent)
