import argparse
from chronaris.evaluation.application_tasks.expanded_fit_inputs import qualify_expanded_fit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--additional',required=True);p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();result=qualify_expanded_fit(a.additional,a.report,a.assets)
    print({k:result[k] for k in ('status','windows','source_files','encoder_updates')})
