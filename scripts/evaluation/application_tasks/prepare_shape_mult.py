"""Transport the registered source-only native extraction, preserving failures."""
import argparse
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import numpy as np
from chronaris.dataset.cogpilot_shape_extract import extract_shape_native


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.original, allow_pickle=False) as raw:
        meta = json.loads(str(raw['metadata']))
    root = Path(__file__).resolve().parents[3]
    code = '\n'.join((root/'src/chronaris/dataset'/name).read_text() for name in
                     ('cogpilot_performance.py', 'cogpilot_history_features.py', 'cogpilot_shape_features.py'))
    code += '\n'+inspect.getsource(extract_shape_native)
    code += '\nimport io,sys\nb=io.BytesIO()\nnp.savez_compressed(b,**extract_shape_native(request))\nsys.stdout.buffer.write(b.getvalue())\n'
    request = dict(code=code, windows=meta['windows'], sources=meta['sources'])
    command = 'exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote(
        'import json,sys;request=json.load(sys.stdin);exec(request["code"])')
    with args.output.with_suffix('.partial').open('xb') as handle:
        result = subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                 'wangminan@10.69.120.91', command], input=json.dumps(request).encode(),
                                stdout=handle, stderr=subprocess.PIPE, timeout=3600)
    args.output.with_suffix('.log').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    args.output.with_suffix('.partial').rename(args.output)
    print(args.output, args.output.stat().st_size)
