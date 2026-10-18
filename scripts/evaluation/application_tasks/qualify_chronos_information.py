import argparse
from chronaris.evaluation.application_tasks.chronos_information_qualification import qualify_chronos_information

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args()
    print(qualify_chronos_information(a.report))
