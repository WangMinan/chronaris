import fcntl
import json
from pathlib import Path
import runpy
import subprocess
import time


def test_development_chain_retries_gpu_wait_and_does_not_repeat_completed_stages(tmp_path,monkeypatch):
    script=Path(__file__).parents[3]/'scripts/evaluation/application_tasks/run_v4_development_chain.py'
    monkeypatch.chdir(tmp_path)
    base=tmp_path/'artifacts/application_evaluation'
    root=base/'2026-09-08_v4-development-chain';root.mkdir(parents=True)
    upstream=base/'2026-09-08_v4-candidate-pressure/queue_state.json';upstream.parent.mkdir()
    upstream.write_text(json.dumps(dict(status='completed',current_child_pid=None,
        source_code_sha256='9bcaeaec8a4f06502ed9e6309ff34832cf61f7e25e504a4d1c2d4534a2ff9b43')))
    public=base/'2026-09-08_v4-public-screen';review=base/'2026-09-08_v4-candidate-review';pressure=base/'2026-09-08_v4-review-pressure'
    steps={'public-screen':(public/'run_state.json','completed'),
        'public-screen-results':(public/'results_summary.json','public_scores_verified_not_final_adoption'),
        'candidate-review-cohort':(review/'run_state.json','completed'),
        'public-review-results':(review/'public_results_summary.json','public_review_verified_not_final_selection'),
        'review-pressure-cohort':(pressure/'queue_state.json','completed'),
        'simulation-review-results':(review/'simulation_results_summary.json','simulation_review_verified_not_final_selection'),
        'candidate-adoption':(review/'adoption_decisions.json','single_factor_decisions_ready_not_frozen')}
    calls=[]
    class Child:
        pid=123
        def __init__(self,command,stdout,stderr):
            stage='full_cuda_tests' if '-m' in command else command[2];calls.append(stage)
            if stage=='full_cuda_tests':return
            status='waiting_gpu' if stage=='public-screen' and calls.count(stage)==1 else steps[stage][1]
            stdout.write(json.dumps({'status':status})+'\n');stdout.flush()
            if status!='waiting_gpu':
                path,_=steps[stage];path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'status':status}))
        def wait(self,timeout):return 0
    monkeypatch.setattr(subprocess,'Popen',Child)
    monkeypatch.setattr(fcntl,'flock',lambda *args:None)
    monkeypatch.setattr(time,'sleep',lambda seconds:None)
    runpy.run_path(str(script),run_name='__main__')
    state=json.loads((root/'chain_state.json').read_text())
    assert state['status']=='development_complete_ready_for_freeze' and not state['confirmation_started']
    assert len(state['completed_steps'])==8 and len(calls)==9
    runpy.run_path(str(script),run_name='__main__')
    assert len(calls)==9
