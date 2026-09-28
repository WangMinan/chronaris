"""Export fixed stage-5 evidence; rendering is a separate standalone command."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))
from chronaris.evaluation.application_tasks.stage5_closeout import export_stage5_closeout

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    print(export_stage5_closeout(args.run_root, args.output_root))
