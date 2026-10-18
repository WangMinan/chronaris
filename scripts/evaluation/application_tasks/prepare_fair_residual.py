"""Run the preregistered, read-only input extraction on the source host."""
import argparse
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import numpy as np

from chronaris.dataset.cogpilot_fair_residual import extract_fair_residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[3]
    old = root/'docs/review/stage/performance-pilot-20261018'
    with np.load(root/'artifacts/application_evaluation/2026-10-18-performance-pilot/observations-v2.npz') as data:
        meta = json.loads(str(data['metadata']))
    windows = {}
    for row in meta['windows']:
        if row['accepted']:
            windows.setdefault(row['record_id'], []).append(row)
    request = dict(windows=windows, records=meta['records'], source_hashes=json.loads((old/'sources.json').read_text()))
    code = (root/'src/chronaris/dataset/cogpilot_performance.py').read_text()
    code += '\n'+(root/'src/chronaris/dataset/cogpilot_history_features.py').read_text()
    code += '\n'+inspect.getsource(extract_fair_residual)
    code += '\nimport io,sys\nb=io.BytesIO()\nnp.savez_compressed(b,**extract_fair_residual(request))\nsys.stdout.buffer.write(b.getvalue())\n'
    request['code'] = code
    command = 'exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote(
        'import json,sys;request=json.load(sys.stdin);exec(request["code"])')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.partial')
    with partial.open('xb') as out:
        result = subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe', '-o', 'BatchMode=yes',
                                 '-o', 'ConnectTimeout=10', 'wangminan@10.69.120.91', command],
                                input=json.dumps(request).encode(), stdout=out, stderr=subprocess.PIPE, timeout=3600)
    args.output.with_suffix('.log').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    partial.rename(args.output)
    print(args.output, args.output.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
