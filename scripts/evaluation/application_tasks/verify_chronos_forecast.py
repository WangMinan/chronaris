import argparse,datetime
from pathlib import Path
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.chronos_forecast_verify import verify_chronos_forecast
from chronaris.evaluation.application_tasks.chronos_information_run import NIGHT_DEADLINE
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True);p.add_argument('--deadline');a=p.parse_args()
    deadline=NIGHT_DEADLINE
    if a.deadline:
        stamp=datetime.datetime.fromisoformat(a.deadline)
        if stamp.tzinfo is None:raise ValueError('deadline requires UTC offset')
        deadline=stamp.timestamp()
    r=verify_chronos_forecast(a.report,a.assets,deadline)
    print({k:r[k] for k in ('status','models','updates','predictions','maximum_metric_delta')},flush=True)
    root=Path(a.assets)
    save_json(Path(a.report)/'asset_manifest.json',dict(root=str(root.resolve()),files={str(p.relative_to(root)):file_digest(p) for p in sorted(root.rglob('*')) if p.is_file()}))
