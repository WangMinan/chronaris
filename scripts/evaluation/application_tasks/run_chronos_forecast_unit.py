"""One bounded native unit: fit, score, replay, deliver; retain every stage log."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--assets',required=True)
    a=p.parse_args();report,assets=Path(a.report),Path(a.assets);assets.mkdir(parents=True,exist_ok=True)
    receipt=report/'pipeline_receipt.json'
    if receipt.exists():raise FileExistsError('pipeline receipt already exists')
    state=dict(status='running',pid=os.getpid(),started_epoch=time.time(),stages=[])
    def save():
        temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(receipt)
    save();base=Path(__file__).parent
    for stage in ('run','score','verify','deliver'):
        log=assets/(stage+'_unit.log')
        if log.exists():raise FileExistsError(log)
        state['stage']=stage;save()
        with log.open('w') as output:
            result=subprocess.run([sys.executable,str(base/(stage+'_chronos_forecast.py')),'--report',str(report),'--assets',str(assets)],stdout=output,stderr=subprocess.STDOUT)
        state['stages'].append(dict(stage=stage,returncode=result.returncode,log=str(log.resolve()),finished_epoch=time.time()));save()
        if result.returncode:
            state.update(status='stopped',finished_epoch=time.time());save()
            print(json.dumps(state),flush=True);return result.returncode
    state.update(status='completed',finished_epoch=time.time());save();print(json.dumps(state),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
