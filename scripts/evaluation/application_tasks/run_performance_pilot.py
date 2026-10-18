"""Execute only the qualified single performance-task comparison unit."""
import argparse
import json

from chronaris.evaluation.application_tasks.performance_pilot_run import run_performance_pilot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--assets', required=True)
    args = parser.parse_args()
    print(json.dumps(run_performance_pilot(args.snapshot, args.report, args.assets), indent=2))


if __name__ == '__main__':
    main()
