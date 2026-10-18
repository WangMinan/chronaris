import argparse
import datetime
from chronaris.evaluation.application_tasks import chronos_information_readout as score
from chronaris.evaluation.application_tasks import chronos_information_run as run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    p.add_argument('--deadline',help='Explicit new resource deadline with UTC offset')
    a=p.parse_args()
    if a.deadline:
        deadline=datetime.datetime.fromisoformat(a.deadline)
        if deadline.tzinfo is None:raise ValueError('deadline requires an explicit UTC offset')
        score.NIGHT_DEADLINE=run.NIGHT_DEADLINE=deadline.timestamp()
    print(score.score_chronos_information(a.report,a.assets))
