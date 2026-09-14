"""Generate an independent stage-4 acceptance report without rerunning training."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'src'))
from chronaris.evaluation.application_tasks.comparison_closeout import closeout
from chronaris.evaluation.application_tasks.comparison_report import render

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    result = closeout(args.source_root, args.output_root)
    render(result, args.output_root)
    print({k:result[k] for k in ('status','model_units','representation_routes','consumer_routes')})
