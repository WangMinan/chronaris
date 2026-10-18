import argparse
from chronaris.evaluation.application_tasks.development_exposure import audit_expanded_exposure

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args()
    r=audit_expanded_exposure(a.report);print(r['status'],r['new_results_scope'])
