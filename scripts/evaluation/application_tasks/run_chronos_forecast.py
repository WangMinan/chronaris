import argparse,datetime
from chronaris.evaluation.application_tasks.chronos_forecast_run import run_chronos_forecast
from chronaris.evaluation.application_tasks.chronos_information_run import NIGHT_DEADLINE

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    p.add_argument('--deadline',help='Explicit resource extension, with UTC offset');a=p.parse_args();deadline=NIGHT_DEADLINE
    if a.deadline:
        stamp=datetime.datetime.fromisoformat(a.deadline)
        if stamp.tzinfo is None:raise ValueError('deadline requires UTC offset')
        deadline=stamp.timestamp()
    r=run_chronos_forecast(a.report,a.assets,deadline);print({k:r[k] for k in ('status','completed_models','updates','windows_seen')})
