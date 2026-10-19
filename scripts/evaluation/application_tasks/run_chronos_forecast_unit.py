"""One bounded native unit: fit, score, replay, deliver; retain every stage log."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    p.add_argument('--deadline',help='Explicit resumed resource deadline with UTC offset')
    a=p.parse_args();report,assets=Path(a.report),Path(a.assets);assets.mkdir(parents=True,exist_ok=True)
    if a.deadline:
        stamp=datetime.datetime.fromisoformat(a.deadline)
        if stamp.tzinfo is None or stamp.timestamp()<=time.time():raise ValueError('deadline must be future and timezone-aware')
    receipt=report/'pipeline_receipt.json'
    if receipt.exists():raise FileExistsError('pipeline receipt already exists')
    state=dict(status='running',pid=os.getpid(),started_epoch=time.time(),stages=[])
    if a.deadline:state['deadline']=a.deadline
    def save():
        temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(receipt)
    save();base=Path(__file__).parent
    for stage in ('run','score','verify','deliver'):
        log=assets/(stage+'_unit.log')
        if log.exists():raise FileExistsError(log)
        state['stage']=stage;save()
        command=[sys.executable,str(base/(stage+'_chronos_forecast.py')),'--report',str(report),'--assets',str(assets)]
        if a.deadline and stage!='deliver':command+=['--deadline',a.deadline]
        with log.open('w') as output:
            result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
        state['stages'].append(dict(stage=stage,returncode=result.returncode,log=str(log.resolve()),finished_epoch=time.time()));save()
        if result.returncode:
            state.update(status='stopped',finished_epoch=time.time());save()
            print(json.dumps(state),flush=True);return result.returncode
    state.update(status='completed',finished_epoch=time.time());save();print(json.dumps(state),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
