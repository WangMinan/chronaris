"""Run frozen-model or source-native diagnostics without any training entrypoint."""
import argparse
import inspect
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np
from chronaris.evaluation.application_tasks.mult_diagnosis import frozen_interventions, native_clip_audit
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json, save_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('frozen', 'native'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sentinel = args.output/('frozen_replay.json' if args.mode == 'frozen' else 'native_clip_receipt.json')
    if sentinel.exists():
        raise FileExistsError(sentinel)
    if args.mode == 'frozen':
        frozen_interventions(args.report, args.output)
    else:
        root = Path(__file__).resolve().parents[3]
        receipt = json.loads((args.report/'qualification.json').read_text())
        with np.load(receipt['snapshot_path'], allow_pickle=False) as raw:
            meta = json.loads(str(raw['metadata']))
            windows = [dict(r, final_features=x[-1].tolist()) for r, x in zip(meta['windows'], raw['physiology'])]
        assert all(r['accepted'] for r in windows)
        code = (root/'src/chronaris/dataset/cogpilot_performance.py').read_text()+'\n'
        code += (root/'src/chronaris/dataset/cogpilot_history_features.py').read_text()+'\n'
        code += inspect.getsource(native_clip_audit)+'\nprint(json.dumps(native_clip_audit(request), allow_nan=False))\n'
        request = dict(windows=windows, sources=meta['sources'], code=code)
        command = 'exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote(
            'import json,sys;request=json.load(sys.stdin);exec(request["code"])')
        result = subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe', '-o', 'BatchMode=yes',
                                 '-o', 'ConnectTimeout=10', 'wangminan@10.69.120.91', command],
                                input=json.dumps(request), capture_output=True, text=True, timeout=1800)
        if result.returncode:
            raise RuntimeError(result.stderr)
        payload = json.loads(result.stdout)
        save_rows(args.output/'native_clip_windows.csv', payload.pop('rows'))
        save_json(sentinel, payload)
        print('native audit completed', flush=True)
