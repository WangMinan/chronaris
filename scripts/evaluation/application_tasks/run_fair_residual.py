"""Execute the two registered fair residual folds only after input qualification."""
import argparse
import json
from chronaris.evaluation.application_tasks.fair_residual_run import run_fair_residual

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--assets', required=True)
    args = parser.parse_args()
    print(json.dumps(run_fair_residual(args.snapshot, args.report, args.assets), indent=2))
