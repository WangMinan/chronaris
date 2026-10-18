import argparse
from chronaris.evaluation.application_tasks.chronos_numeric_check import check_chronos_precision

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();print(check_chronos_precision(a.report,a.assets))
