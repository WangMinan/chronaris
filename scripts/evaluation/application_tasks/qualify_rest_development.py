import argparse
from chronaris.evaluation.application_tasks.rest_development_inputs import qualify_rest_development

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();r=qualify_rest_development(a.source,a.report,a.assets);print({k:r[k] for k in ('status','windows','subjects','encoder_updates')})
