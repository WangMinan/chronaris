"""Four native forecasting fits, sealed scoring, and source-bound restart."""
import json
from pathlib import Path
import subprocess
import time
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.chronos_forecast_inputs import forecast_data,forecast_folds,VIEWS
from chronaris.evaluation.application_tasks.chronos_forecast_training import fit_forecast
from chronaris.evaluation.application_tasks.chronos_information_run import NIGHT_DEADLINE,atomic_json
from chronaris.evaluation.application_tasks.development_comparison import ASSETS
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.recent_models import verify_assets


def require_forecast_predecessors():
    prior=Path('docs/review/stage/eligible-fit-20261019')
    result=json.loads((prior/'run_status.json').read_text())
    assert result['status']=='completed' and json.loads((prior/'verification.json').read_text())['status']=='passed'
    if not (result['representation_success'] and result['pairing_success']):
        rest=Path('docs/review/stage/rest-development-20261019')
        if json.loads((rest/'qualification.json').read_text())['status']=='passed':
            assert json.loads((rest/'run_status.json').read_text())['status']=='completed'
            assert json.loads((rest/'verification.json').read_text())['status']=='passed'


def run_chronos_forecast(report,root,deadline=NIGHT_DEADLINE):
    report,root=Path(report),Path(root);root.mkdir(parents=True,exist_ok=True)
    require_forecast_predecessors()
    if (report/'run_status.json').exists():
        old=json.loads((report/'run_status.json').read_text())
        if old['status']=='completed':return old
        attempts=report/'attempts';attempts.mkdir(exist_ok=True)
        save_json(attempts/(str(time.time_ns())+'.json'),old)
    q=json.loads((report/'qualification.json').read_text());assert q['status']=='passed'
    protected=json.loads((report/'protected_sources.json').read_text());assert all(file_digest(p)==h for p,h in protected.items())
    assets=verify_assets(ASSETS);torch.set_num_threads(1)
    code=[Path(__file__),Path(__file__).with_name('chronos_forecast_inputs.py'),Path(__file__).with_name('chronos_forecast_training.py'),
          Path(__file__).parents[2]/'modeling/fusion_encoders/chronos_forecast.py']
    binding=dict(plan_sha256=file_digest(report/'PLAN.md'),qualification_sha256=file_digest(report/'qualification.json'),
                 input_qualification_sha256=file_digest(report/'input_qualification.json'),assets_sha256=file_digest(ASSETS),seed=17,
                 folds=forecast_folds(),source_files={str(p.resolve()):file_digest(p) for p in code})
    path=report/'training_binding.json'
    if path.exists():
        saved=json.loads(path.read_text());assert {k:saved[k] for k in binding}==binding;binding=saved
    else:
        binding['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();save_json(path,binding)
    deadline=min(deadline,time.time()+14400)
    budgets=report/'resource_budgets.json';history=json.loads(budgets.read_text()) if budgets.exists() else []
    history.append(dict(started_epoch=time.time(),deadline_epoch=deadline));save_json(budgets,history)
    started=time.monotonic();units=[]
    with development_gpu_lock() as acquired:
        if not acquired:raise BlockingIOError('GPU busy')
        try:
            for spec in forecast_folds():
                for view in VIEWS:
                    if time.time()>=deadline:raise TimeoutError('native unit deadline')
                    data,x,mask,proof=forecast_data(spec,view);name=spec['name']+'_'+view
                    atomic_json(report/'run_status.json',dict(status='running',stage=name,completed_units=units))
                    r=fit_forecast(assets,data,x,mask,proof,root/'training'/name,report,name,binding,deadline)
                    if not r['complete']:raise TimeoutError('native fit saved before completion')
                    receipt=report/(name+'_receipt.json')
                    units.append(dict(name=name,fold=spec['name'],view=view,updates=r['updates'],epochs=r['epochs'],best_epoch=r['best_epoch'],
                                      windows_seen=r['windows_seen'],receipt_path=str(receipt.resolve()),receipt_sha256=file_digest(receipt),
                                      checkpoint_path=r['checkpoint_path'],checkpoint_sha256=r['checkpoint_sha256']))
                    del data,x,mask;torch.cuda.empty_cache()
            assert all(file_digest(p)==h for p,h in protected.items());verify_assets(ASSETS)
            save_json(report/'locked_models.json',dict(binding=binding,units=units,outer_scored=False))
            result=dict(status='completed',completed_models=len(units),units=units,updates=sum(u['updates'] for u in units),
                        windows_seen=sum(u['windows_seen'] for u in units),elapsed_s=time.monotonic()-started,binding=binding,outer_scored=False)
        except BaseException as exc:
            result=dict(status='budget_saved' if isinstance(exc,TimeoutError) else 'failed',error=repr(exc),completed_units=units,
                        elapsed_s=time.monotonic()-started,binding=binding)
            atomic_json(report/'run_status.json',result);raise
    atomic_json(report/'run_status.json',result);return result
