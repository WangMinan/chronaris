"""Run the registered no-training performance-task qualification."""
import argparse
import json

from chronaris.evaluation.application_tasks.performance_pilot_audit import qualify_performance_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(qualify_performance_task(args.snapshot, args.output), indent=2))


if __name__ == '__main__':
    main()
