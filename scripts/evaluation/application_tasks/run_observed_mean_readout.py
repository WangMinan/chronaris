"""Run the preregistered frozen historical-mean readout audit."""
import argparse
from chronaris.evaluation.application_tasks.observed_mean_readout import run_observed_mean
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();print(run_observed_mean(a.report,a.assets),flush=True)
