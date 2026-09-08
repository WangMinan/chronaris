import json
from pathlib import Path
import runpy
import subprocess
import time


def test_formal_chain_freezes_before_parallel_native_queues_and_resumes(tmp_path,monkeypatch):
    script=Path(__file__).parents[3]/'scripts/evaluation/application_tasks/run_v4_confirmation_chain.py'
    monkeypatch.chdir(tmp_path)
    base=tmp_path/'artifacts/application_evaluation';formal=base/'2026-09-08_v4-confirmation'
    upstream=base/'2026-09-08_v4-development-chain/chain_state.json';upstream.parent.mkdir(parents=True)
    upstream.write_text(json.dumps(dict(status='development_complete_ready_for_freeze',
        source_code_sha256='e82738fa97687d482f9b0b3d5df66cfd0254ed86886ebea073768a8166884d86')))
    calls=[];active=[]
    class Child:
        def __init__(self,command,stdout,stderr,env=None):
            stage=command[2];self.stage=stage;self.pid=100+len(calls);calls.append(stage)
            if stage=='configuration-cuda-validation':reply={'status':'completed'}
            elif stage=='freeze-configuration':
                formal.mkdir(parents=True);(formal/'frozen_configuration.json').write_text('{}')
                reply={'status':'frozen'}
            else:
                assert calls[:2]==['configuration-cuda-validation','freeze-configuration']
                self.backend=command[command.index('--backend')+1]
                if self.backend=='nonparametric':assert env['CUDA_VISIBLE_DEVICES']==''
                active.append(self.backend);reply={'status':'completed'}
                (formal/f'native_{self.backend}_queue.json').write_text(json.dumps(reply))
                (formal/'native_confirmation_plan.json').write_text(json.dumps({'blocked_domains':[{'domain':'dingxin'}]}))
            stdout.write(json.dumps(reply)+'\n');stdout.flush()
        def wait(self,timeout):return 0
        def poll(self):
            assert set(active)=={'neural','nonparametric'}
            return 0
    monkeypatch.setattr(subprocess,'Popen',Child)
    monkeypatch.setattr(time,'sleep',lambda seconds:None)
    runpy.run_path(str(script),run_name='__main__')
    state=json.loads((base/'2026-09-08_v4-confirmation-chain/chain_state.json').read_text())
    assert state['status']=='native_confirmation_completed' and state['blocked_domains']==[{'domain':'dingxin'}]
    assert len(calls)==4
    runpy.run_path(str(script),run_name='__main__')
    assert len(calls)==4
