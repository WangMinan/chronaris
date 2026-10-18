"""Paths and immutable evidence helpers for this bounded audit."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PREP = REPO / 'artifacts/application_evaluation/2026-09-06_v4-public-development'
REGISTRY = REPO / 'docs/requirements/thesis-v4-public-subjects.json'
ASSETS = REPO / 'artifacts/application_evaluation/2026-10-18-controlled-readouts'
INVENTORY = REPO / 'docs/artifacts/runs/2026-10-18_chapter5-materials/data/inventory.csv'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save_csv(path, rows):
    assert rows
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def manifest(domain):
    return [json.loads(line) for line in (PREP/domain/'sample_manifest.jsonl').read_text().splitlines()]
