"""Check fixed historical-performance and physiology inputs before training."""
import argparse
import json
from chronaris.evaluation.application_tasks.fair_residual_inputs import qualify_fair_inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(qualify_fair_inputs(args.snapshot, args.output), indent=2))
