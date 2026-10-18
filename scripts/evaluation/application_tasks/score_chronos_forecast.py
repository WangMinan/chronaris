import argparse,datetime
from chronaris.evaluation.application_tasks.chronos_forecast_results import score_chronos_forecast
from chronaris.evaluation.application_tasks.chronos_information_run import NIGHT_DEADLINE

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True);p.add_argument('--deadline')
    a=p.parse_args();deadline=NIGHT_DEADLINE
    if a.deadline:
        stamp=datetime.datetime.fromisoformat(a.deadline)
        if stamp.tzinfo is None:raise ValueError('deadline requires UTC offset')
        deadline=stamp.timestamp()
    print(score_chronos_forecast(a.report,a.assets,deadline))
