"""Pause an authorized pipeline after its next resumable checkpoint."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))
from chronaris.evaluation.application_tasks.pipeline_pause import pause_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    print(pause_pipeline(**vars(args)))
