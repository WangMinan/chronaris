"""Prepare an explicit public readout revision without touching frozen data."""
import argparse
import json
from chronaris.evaluation.application_tasks.public_readout_revision import prepare_readout_revision


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', required=True, choices=('clare', 'cogpilot'))
    p.add_argument('--source-root', required=True)
    p.add_argument('--registry-path', required=True)
    p.add_argument('--output-root', required=True)
    print(json.dumps(prepare_readout_revision(**vars(p.parse_args())), indent=2))


if __name__ == '__main__':
    main()
