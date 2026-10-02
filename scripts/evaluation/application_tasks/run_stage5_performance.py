"""Run a bounded formal-shape diagnostic; output is never a production checkpoint."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))

from chronaris.evaluation.application_tasks.stage5_performance import production_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    print(production_probe(parent=args.parent, output=args.output, resume=args.resume))


if __name__ == '__main__':
    main()
