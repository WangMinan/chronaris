"""Run only after an exclusive GPU window is authorized; at most 30 minutes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))
from chronaris.evaluation.application_tasks.chapter5_budget_probe import probe_domain, probe_pair

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--domain', choices=('clare', 'cogpilot'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(json.dumps(probe_domain(args.domain, args.root) if args.domain else probe_pair(args.root), indent=2))
