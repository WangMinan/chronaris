"""Restore only missing, hash-pinned external assets; never replace an existing file."""
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
from http.client import IncompleteRead
from concurrent.futures import ThreadPoolExecutor
import shutil

from chronaris.evaluation.application_tasks.chapter5_assets import read, sha


def missing_assets(manifests):
    jobs = {}
    for manifest in manifests:
        record = read(manifest)
        for name, digest in record['files'].items():
            path = Path(name)
            if path.exists():
                if sha(path) != digest:
                    raise ValueError(f'existing asset changed: {path}')
                continue
            matches = [v for v in record.values() if isinstance(v, dict) and 'revision' in v
                       and 'source' in v and path.is_relative_to(v['path'])]
            if len(matches) != 1:
                raise ValueError(f'missing asset has no pinned upstream: {path}')
            source = matches[0]
            url = f"https://huggingface.co/{source['source']}/resolve/{source['revision']}/{path.relative_to(source['path'])}"
            job = dict(path=name, sha256=digest, url=url)
            if name in jobs and jobs[name] != job:
                raise ValueError('conflicting asset manifests')
            jobs[name] = job
    return list(jobs.values())


def restore_one(job):
    path = Path(job['path'])
    if path.exists():
        if sha(path) != job['sha256']:
            raise ValueError(f'existing asset changed: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.download')
    for attempt in range(3):
        offset=temporary.stat().st_size if temporary.exists() else 0
        request=Request(job['url'],headers={'Range':f'bytes={offset}-'} if offset else {})
        try:
            with urlopen(request, timeout=120) as source:
                partial=getattr(source,'status',200)==206
                if partial and not source.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('download range does not match the preserved partial file')
                with temporary.open('ab' if partial else 'wb') as target:
                    shutil.copyfileobj(source,target,length=8*1024*1024)
            break
        except (TimeoutError,ConnectionError,URLError,IncompleteRead):
            if attempt==2:raise
    if sha(temporary) != job['sha256']:
        raise ValueError(f'download hash mismatch, retained for diagnosis: {temporary}')
    temporary.replace(path)


def restore_assets(manifests):
    jobs = missing_assets(manifests)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(restore_one, jobs))
    assert not missing_assets(manifests)
    return dict(status='completed', restored_files=len(jobs))
