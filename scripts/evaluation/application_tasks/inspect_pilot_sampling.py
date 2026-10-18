"""Transport the fixed read-only source sampling check to the source host."""
import argparse
import inspect
import json
from pathlib import Path
import shlex
import subprocess

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.performance_pilot_diagnosis import inspect_source_sampling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = json.loads((args.original/'record_inventory.json').read_text())
    first = {}
    for row in sorted(records, key=lambda r: r['record_id']):
        first.setdefault(row['subject'], row)
    code = 'import json,hashlib\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\n'
    code += inspect.getsource(file_digest)+'\n'+inspect.getsource(inspect_source_sampling)
    code += '\nprint(json.dumps(inspect_source_sampling(request)))\n'
    command = 'exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote(
        'import json,sys;request=json.load(sys.stdin);exec(request["code"])')
    request = dict(records=list(first.values()), source_hashes=json.loads((args.original/'sources.json').read_text()), code=code)
    result = subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe', '-o', 'BatchMode=yes',
                             '-o', 'ConnectTimeout=10', 'wangminan@10.69.120.91', command],
                            input=json.dumps(request), capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise RuntimeError(result.stderr)
    rows = json.loads(result.stdout)
    args.output.write_text(json.dumps(rows, indent=2)+'\n')
    print('verified source sampling files:', len(rows))


if __name__ == '__main__':
    main()
