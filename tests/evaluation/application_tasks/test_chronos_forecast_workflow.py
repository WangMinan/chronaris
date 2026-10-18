import json
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks import chronos_forecast_training as training
from chronaris.evaluation.application_tasks import chronos_forecast_results as scoring
from chronaris.evaluation.application_tasks import chronos_forecast_verify as verification
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json


def test_locked_native_workflow_replays_and_checks_actual_training_lineage(tmp_path,monkeypatch):
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model
    torch.set_num_threads(1);monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    def pinned(_):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            config=Chronos2CoreConfig(d_model=16,d_kv=8,d_ff=32,num_layers=1,num_heads=2,dropout_rate=.1,
                chronos_config=dict(context_length=96,input_patch_size=16,input_patch_stride=16,output_patch_size=16,
                                    quantiles=[.1,.5,.9],use_reg_token=True,use_arcsinh=True,max_output_patches=1,time_encoding_scale=96))
            return Chronos2Model(config)
    @contextmanager
    def lock():yield True
    folds=[dict(name=f'fold0{i}',fit=['a'],select=['b'],score=['c','d']) for i in (1,2)]
    class Data:
        def __init__(self,spec):
            self.fold=spec;self.roles=dict(fit=np.arange(8),select=np.arange(8,10),score=np.arange(10,14))
            self.score_open=False;self.target_scale=1.;self.weights=torch.ones(14);self.y=np.zeros(14)
            self.subjects=np.array(['a']*8+['b']*2+['c']*2+['d']*2)
            self.rows=[dict(sample_id=str(i),subject=self.subjects[i]) for i in range(14)]
            self.normalizer=dict(vehicle=dict(center=[0.]*12,scale=[1.]*12))
        def indices(self,role):
            if role=='score' and not self.score_open:raise PermissionError('sealed')
            return self.roles[role]
    def inputs(spec,view):
        data=Data(spec);g=torch.Generator().manual_seed(7 if view=='aircraft' else 9)
        x=torch.randn(14,96,3,generator=g)
        return data,x,torch.ones_like(x,dtype=torch.bool),{}
    monkeypatch.setattr(training,'load_forecast_model',pinned)
    monkeypatch.setattr(scoring,'load_forecast_model',pinned)
    for module in (scoring,verification):
        monkeypatch.setattr(module,'forecast_data',inputs);monkeypatch.setattr(module,'forecast_folds',lambda:folds)
        monkeypatch.setattr(module,'verify_assets',lambda _:{});monkeypatch.setattr(module,'development_gpu_lock',lock)
    report=tmp_path/'report';report.mkdir();assets=tmp_path/'assets';assets.mkdir()
    (report/'PLAN.md').write_text('CPU end-to-end check');save_json(report/'qualification.json',dict(status='passed'))
    save_json(report/'protected_sources.json',{})
    binding=dict(plan_sha256=file_digest(report/'PLAN.md'),qualification_sha256=file_digest(report/'qualification.json'),source_files={})
    units=[]
    for spec in folds:
        for view in scoring.VIEWS:
            data,x,mask,proof=inputs(spec,view);name=spec['name']+'_'+view
            r=training.fit_forecast({},data,x,mask,proof,assets/'training'/name,report,name,binding,float('inf'))
            path=report/(name+'_receipt.json')
            units.append(dict(name=name,fold=spec['name'],view=view,updates=r['updates'],epochs=r['epochs'],best_epoch=r['best_epoch'],
                              receipt_path=str(path),receipt_sha256=file_digest(path),checkpoint_path=r['checkpoint_path'],checkpoint_sha256=r['checkpoint_sha256']))
    save_json(report/'locked_models.json',dict(binding=binding,units=units,outer_scored=False))
    save_json(report/'run_status.json',dict(status='completed',updates=sum(u['updates'] for u in units),
                                          windows_seen=sum(u['epochs']*8 for u in units)))
    references=[]
    for spec in folds:
        data=Data(spec)
        for i in data.roles['score']:references.append(dict(fold=spec['name'],method='strong',sample_id=str(i),
                                                          subject=str(data.subjects[i]),truth=0.,prediction=1e-6))
    path=tmp_path/'references.csv';pd.DataFrame(references).to_csv(path,index=False)
    save_json(report/'reference_reports.json',[dict(path=str(path),sha256=file_digest(path),prefix='reference')])
    result=scoring.score_chronos_forecast(report,assets,float('inf'));assert not result['qualified_routes']
    check=verification.verify_chronos_forecast(report,assets,float('inf'))
    assert check['status']=='passed' and check['models']==4 and len(check['replay'])==8
    assert check['maximum_metric_delta']<1e-10 and check['optimizer_updates_in_verification']==0
