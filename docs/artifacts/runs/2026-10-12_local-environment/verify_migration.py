import concurrent.futures
import hashlib
import json
from pathlib import Path

manifest = Path('/mnt/e/chronaris-migration-20261012/inventory.jsonl')
rows = [json.loads(line) for line in manifest.read_text().splitlines()]

def check(row):
    path = Path(row['source'])
    assert path.is_file() and not path.is_symlink(), str(path)
    assert path.stat().st_size == row['bytes'], str(path)
    digest = row.get('expected_sha256')
    if digest:
        with path.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == digest, str(path)
    return bool(digest)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    checked = sum(pool.map(check, rows))
result = dict(status='passed', files=len(rows), bytes=sum(r['bytes'] for r in rows), frozen_hashes_checked=checked)
Path('/home/wangminan/chronaris-migration-20261012/restored-verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
