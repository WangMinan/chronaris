"""Read-only source-host extraction for the preregistered performance pilot."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.output.exists():
        raise FileExistsError(args.output)
    records = list(csv.DictReader((root/'docs/review/stage/task-anchor-recovery-20261018/performance_sources.csv').open()))
    code = (root/'src/chronaris/dataset/cogpilot_performance.py').read_text()
    code += '\nimport io,sys\nbuf=io.BytesIO()\nnp.savez_compressed(buf,**extract_performance_windows(request["records"]))\nsys.stdout.buffer.write(buf.getvalue())\n'
    command = 'exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c ' + shlex.quote(
        'import json,sys;request=json.load(sys.stdin);exec(request["code"])')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.partial')
    with partial.open('xb') as output:
        result = subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe', '-o', 'BatchMode=yes',
                                 '-o', 'ConnectTimeout=10', 'wangminan@10.69.120.91', command],
                                input=json.dumps(dict(records=records, code=code)).encode(),
                                stdout=output, stderr=subprocess.PIPE, timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    partial.rename(args.output)
    print(args.output, args.output.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
