"""Select fixed windows, independently audit remote CSVs, verify local cached data."""
import json
import shlex
import subprocess
from collections import defaultdict

import numpy as np
import torch

from common import HERE, PREP, ASSETS, manifest, save_json, save_csv, sha
from raw_remote import digest_array


def selection(domain):
    grouped = defaultdict(list)
    for r in manifest(domain):
        task = 'window' if '::event' not in r['sample_id'] else 'event'
        grouped[(task, r['subject_id'], r['record_id'])].append(r)
    result = {}
    # First/last records per subject, first/last windows and a fixed hashed interior.
    by_subject = defaultdict(list)
    for key in grouped:
        by_subject[key[:2]].append(key)
    import hashlib
    for keys in by_subject.values():
        keys = sorted(keys)
        for key in {keys[0], keys[-1]}:
            rows = sorted(grouped[key], key=lambda r: r['context_start_s'])
            chosen = [rows[0], rows[-1], min(rows, key=lambda r: hashlib.sha256(r['sample_id'].encode()).hexdigest())]
            for r in chosen:
                result[r['sample_id']] = r | dict(domain=domain)
    assert len(result) >= 20
    if domain == 'cogpilot':
        assert min(sum(('::event' in s) == event for s in result) for event in (False, True)) >= 20
    return sorted(result.values(), key=lambda r: r['sample_id'])


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    request = selection('clare') + selection('cogpilot')
    save_json(HERE/'raw_selection.json', request)
    output = ASSETS/'raw_response.json'
    if not output.exists():
        remote = 'exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c ' + shlex.quote((HERE/'raw_remote.py').read_text())
        p = subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                            'wangminan@10.69.120.91', remote], input=json.dumps(request), capture_output=True, text=True, timeout=7200)
        if p.returncode:
            raise RuntimeError(p.stderr)
        save_json(output, json.loads(p.stdout))
    response = json.loads(output.read_text())
    assert [r['sample_id'] for r in response] == [r['sample_id'] for r in request]
    targets = {d: torch.load(PREP/d/'task_targets.pt', weights_only=True) for d in ('clare', 'cogpilot')}
    indices = {d: {s: i for i, s in enumerate(t['sample_ids'])} for d, t in targets.items()}
    rows, field_rows, sources = [], [], {}
    for expected, actual in zip(request, response, strict=True):
        domain, sid = expected['domain'], expected['sample_id']
        cache = next((PREP/domain/'native_cache').glob('*_'+expected['source_sample_hash'][:12]+'.npz'))
        sources[str(cache)] = sha(cache)
        with np.load(cache) as data:
            for field in actual['fields']:
                stream, j = field['stream'], field['index']
                mask = data[stream+'_feature_mask'][:, j]
                times = data[stream+'_timestamps_s'][mask]
                values = data[stream+'_values'][mask, j]
                field_rows.append(dict(domain=domain, sample_id=sid, **field,
                    cache_count=int(mask.sum()), time_matches=digest_array(times)==field['time_sha256'],
                    values_match=digest_array(values)==field['value_sha256']))
        start = (actual['start_native']-expected['record_time_origin_native'])*expected['native_time_scale_s']
        time_error = abs(start-expected['context_start_s'])
        for task, raw_value in actual['targets'].items():
            stored = float(targets[domain]['values'][task][indices[domain][sid]].reshape(-1)[0])
            rows.append(dict(domain=domain, sample_id=sid, subject=expected['subject_id'], task=task,
                raw_target=raw_value, cached_target=stored, target_matches=float(np.float32(raw_value))==stored,
                context_start_error_s=time_error, tolerance_s=actual['time_tolerance_s'],
                time_matches=time_error <= actual['time_tolerance_s'],
                source_identity_matches=expected['source_sample_hash']==actual['rebuilt_source_sample_hash']))
    save_csv(HERE/'raw_targets.csv', rows)
    save_csv(HERE/'raw_fields.csv', field_rows)
    passed = all(r['target_matches'] and r['time_matches'] and r['source_identity_matches'] for r in rows)
    passed = passed and all(r['time_matches'] and r['values_match'] and r['cache_count']==r['count'] for r in field_rows)
    save_json(HERE/'raw_verification.json', dict(status='passed' if passed else 'failed', windows=len(request),
        target_checks=len(rows), field_checks=len(field_rows), raw_response_sha256=sha(output),
        raw_response_path=str(output), source_files={s['path']:s for r in response for s in r['sources']},
        label_files={r['label_file']['path']:r['label_file'] for r in response if 'label_file' in r},
        cache_files=sources, script_sha256=sha(HERE/'raw_remote.py'), encoder_updates=0))
    print('raw audit', passed, len(request), len(rows), len(field_rows), flush=True)
    assert passed, 'Raw audit mismatch: retain evidence and diagnose before readout fits'


if __name__ == '__main__':
    main()
