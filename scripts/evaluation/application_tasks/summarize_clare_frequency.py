import argparse
from chronaris.evaluation.application_tasks.clare_frequency_summary import summarize_clare_frequency

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();r=summarize_clare_frequency(a.report,a.assets);print({k:r[k] for k in ('status','readiness','qualified_windows','encoder_updates')})
