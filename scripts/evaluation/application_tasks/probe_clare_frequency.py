import argparse
from chronaris.evaluation.application_tasks.clare_frequency_probe import probe_clare_frequency

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();print(probe_clare_frequency(a.snapshot,a.report,a.assets))
