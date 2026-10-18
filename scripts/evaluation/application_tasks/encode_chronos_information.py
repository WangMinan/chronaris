import argparse
from chronaris.evaluation.application_tasks.chronos_information_run import encode_chronos_information

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();print(encode_chronos_information(a.report,a.assets))
