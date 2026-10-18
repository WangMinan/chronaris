import argparse
from chronaris.evaluation.application_tasks.chronos_forecast_qualification import qualify_forecast_inputs,qualify_forecast_hardware

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--hardware',action='store_true');a=p.parse_args()
    r=qualify_forecast_hardware(a.report) if a.hardware else qualify_forecast_inputs(a.report)
    print({k:r[k] for k in ('status','optimizer_updates')})
