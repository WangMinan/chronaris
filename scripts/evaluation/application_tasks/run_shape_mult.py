"""CLI for the source-qualified bounded shape-preserving comparison."""
import argparse
from chronaris.evaluation.application_tasks.shape_mult_run import run_shape_mult

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    p.add_argument('--selection-policy',choices=('task_head','common_readout'),default='task_head')
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--history-aux-weight',type=float,choices=(0.,.5),default=0.)
    args=p.parse_args();print(run_shape_mult(args.report,args.assets,args.selection_policy,args.seed,args.history_aux_weight))
