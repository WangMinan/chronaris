"""CLI for the source-qualified bounded shape-preserving comparison."""
import argparse
from chronaris.evaluation.application_tasks.shape_mult_run import run_shape_mult

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    args=p.parse_args();print(run_shape_mult(args.report,args.assets))
