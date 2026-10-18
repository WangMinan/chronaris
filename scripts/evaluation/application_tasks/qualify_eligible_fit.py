import argparse
from chronaris.evaluation.application_tasks.eligible_fit_inputs import qualify_eligible_fit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();result=qualify_eligible_fit(a.source,a.report,a.assets)
    print({k:result[k] for k in ('status','windows','source_files','encoder_updates')})
