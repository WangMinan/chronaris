import argparse
from chronaris.evaluation.application_tasks.chronos_forecast_delivery import deliver_chronos_forecast
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();deliver_chronos_forecast(a.report,a.assets)
