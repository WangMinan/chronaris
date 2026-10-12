"""Replay every frozen public sequence model on its complete internal validation role."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))

if __name__ == '__main__':
    from chronaris.evaluation.application_tasks.device_replay import run_replay
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-root', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    run_replay(args.main_root, args.output_root)
