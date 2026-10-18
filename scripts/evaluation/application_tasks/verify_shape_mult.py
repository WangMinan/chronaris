"""Independently verify receipts, arithmetic and fresh frozen predictions."""
import argparse
from chronaris.evaluation.application_tasks.shape_mult_verify import verify_shape_mult

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();print(verify_shape_mult(a.report,a.assets))
