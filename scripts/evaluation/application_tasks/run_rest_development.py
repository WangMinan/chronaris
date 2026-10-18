import argparse
from chronaris.evaluation.application_tasks.shape_mult_run import run_shape_mult

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();print(run_shape_mult(a.report,a.assets,selection_policy='common_readout',rest_calibration=True))
