import argparse
from chronaris.evaluation.application_tasks.observed_mean_verify import verify_observed_mean
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();print(verify_observed_mean(a.report,a.assets),flush=True)
