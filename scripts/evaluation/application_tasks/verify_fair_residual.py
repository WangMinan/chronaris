"""Independent arithmetic and optional new-process replay of the fair study."""
import argparse
import json
from chronaris.evaluation.application_tasks.fair_residual_verify import verify_fair_residual

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();print(json.dumps(verify_fair_residual(args.report,args.replay),indent=2))
