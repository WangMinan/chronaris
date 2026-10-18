"""Inspect the frozen pilot without optimizer updates or new readout fitting."""
import argparse
import json
from chronaris.evaluation.application_tasks.performance_pilot_diagnosis import run_pilot_diagnosis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run_pilot_diagnosis(args.original, args.output), indent=2))


if __name__ == '__main__':
    main()
