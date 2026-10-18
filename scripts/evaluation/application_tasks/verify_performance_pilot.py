"""Recompute the bounded pilot's metrics and verify immutable sources."""
import argparse
import json

from chronaris.evaluation.application_tasks.performance_pilot_verify import verify_performance_pilot, replay_performance_pilot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_performance_pilot(args.report), indent=2))
    if args.replay:
        print(json.dumps(replay_performance_pilot(args.report), indent=2))


if __name__ == '__main__':
    main()
