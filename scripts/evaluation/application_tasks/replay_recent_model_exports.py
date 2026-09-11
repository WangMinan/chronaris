"""Revalidate recorded recent-model exports without additional neural updates."""
import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT/'src'))
from chronaris.evaluation.application_tasks.recent_model_smoke import revalidate_recent_exports


def main():
    parser = argparse.ArgumentParser()
    for name in ('source-root', 'evidence-path', 'source-inventory', 'output-root'):
        parser.add_argument('--'+name, required=True)
    result = revalidate_recent_exports(**vars(parser.parse_args()))
    print(json.dumps({'status': result['status'], 'units': len(result['results']),
        'prior_files_unchanged': result['prior_files_unchanged']}))


if __name__ == '__main__':
    main()
