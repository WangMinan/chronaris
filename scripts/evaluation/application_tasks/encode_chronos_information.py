import argparse
import datetime
import json
from pathlib import Path
from chronaris.evaluation.application_tasks import chronos_information_run as run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    p.add_argument('--deadline',help='Explicit new resource deadline with UTC offset; scientific settings unchanged')
    a=p.parse_args();status=Path(a.report)/'encoding_status.json'
    if status.exists() and json.loads(status.read_text())['status']=='completed':
        print('Encoding already completed; frozen receipts retained.');raise SystemExit(0)
    if a.deadline:
        deadline=datetime.datetime.fromisoformat(a.deadline)
        if deadline.tzinfo is None:raise ValueError('deadline requires an explicit UTC offset')
        run.NIGHT_DEADLINE=deadline.timestamp()
        record=Path(a.report)/('budget_override_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'.json')
        record.write_text(json.dumps({'deadline':a.deadline,'settings_changed':False},indent=2)+'\n')
    print(run.encode_chronos_information(a.report,a.assets))
