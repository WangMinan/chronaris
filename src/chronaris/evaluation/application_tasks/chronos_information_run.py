"""Resumable pinned-model exports and one-pass, history-only adaptation."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.chronos_information_qualification import information_data,VIEW_COLUMNS
from chronaris.evaluation.application_tasks.development_comparison import ASSETS
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json,save_rows
from chronaris.evaluation.application_tasks.recent_model_training import short_fit,load_encoder_checkpoint
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.recent_models import verify_assets
from chronaris.modeling.fusion_encoders.chronos_information import ChronosInformationEncoder
from chronaris.modeling.training.rng import isolated_training_rng,canonical_training_state_sha256


NIGHT_DEADLINE=datetime.datetime.fromisoformat('2026-10-19T09:30:00+08:00').timestamp()


def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    save_json(temp,value);temp.replace(path)


class TrainingProgress(dict):
    def __init__(self,path,stage):
        super().__init__(optimizer_updates=0)
        self.path=path;self.stage=stage;self.last=0.

    def update(self,*args,**kwargs):
        super().update(*args,**kwargs)
        if time.monotonic()-self.last>=10:
            atomic_json(self.path,dict(status='running',stage=self.stage,phase='history_adaptation',
                                      progress=dict(self),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            self.last=time.monotonic()


def export_information(encoder,values,mask,path,binding,status_path,stage):
    path=Path(path);receipt=path.with_suffix('.json');partial=path.with_suffix('.partial')
    meta=dict(binding=binding,shape=[len(values),encoder.feature_dim],completed=False,rows=0,chunks=[])
    if receipt.exists():
        meta=json.loads(receipt.read_text());assert meta['binding']==binding
        if meta['completed']:
            assert file_digest(path)==meta['sha256'];return meta
    if partial.exists():
        cache=np.lib.format.open_memmap(partial,mode='r+')
        assert list(cache.shape)==meta['shape']
        for chunk in meta['chunks']:
            assert hashlib.sha256(cache[chunk['start']:chunk['stop']].tobytes()).hexdigest()==chunk['sha256']
    else:
        if path.exists():raise FileExistsError('unbound final cache')
        cache=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=tuple(meta['shape']))
    encoder.eval()
    for off in range(meta['rows'],len(values),64):
        if time.time()>=NIGHT_DEADLINE:raise TimeoutError('night budget saved at export boundary')
        stop=min(off+64,len(values));chunks=[]
        for start in range(off,stop,8):
            chunks.append(encoder(values[start:min(start+8,stop)],mask[start:min(start+8,stop)]).cpu().numpy())
        result=np.concatenate(chunks)
        assert np.isfinite(result).all() and result.shape==(stop-off,encoder.feature_dim)
        cache[off:stop]=result;cache.flush()
        meta['chunks'].append(dict(start=off,stop=stop,sha256=hashlib.sha256(result.tobytes()).hexdigest()))
        meta['rows']=stop;atomic_json(receipt,meta)
        atomic_json(status_path,dict(status='running',stage=stage,phase='export',rows=stop,total=len(values)))
    del cache;partial.replace(path)
    meta.update(completed=True,sha256=file_digest(path));atomic_json(receipt,meta)
    print(stage,'exported',len(values),'rows',flush=True)
    return meta


def encode_chronos_information(report,assets_root):
    report,root=Path(report),Path(assets_root)
    q=json.loads((report/'qualification.json').read_text());assert q['status']=='passed'
    protected=json.loads((report/'protected_sources.json').read_text())
    assert all(file_digest(p)==h for p,h in protected.items())
    assets=verify_assets(ASSETS);torch.set_num_threads(1)
    code_paths=[Path(__file__),Path(__file__).with_name('chronos_information_qualification.py'),
                Path(__file__).with_name('recent_model_training.py'),
                Path(__file__).parents[2]/'modeling/fusion_encoders/chronos_information.py',
                Path(__file__).parents[2]/'modeling/fusion_encoders/recent_models.py']
    binding=dict(plan_sha256=file_digest(report/'PLAN.md'),qualification_sha256=file_digest(report/'qualification.json'),
                 assets_sha256=file_digest(ASSETS),seed=17,source_files={str(p.resolve()):file_digest(p) for p in code_paths})
    binding_path=report/'encoding_binding.json'
    if binding_path.exists():
        saved=json.loads(binding_path.read_text());assert {k:saved[k] for k in binding}==binding
        binding=saved
    else:
        binding['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        save_json(binding_path,binding)
    status_path=report/'encoding_status.json';units=[];start=time.monotonic()
    with development_gpu_lock() as acquired:
        if not acquired:raise BlockingIOError('GPU busy')
        try:
            for spec in FOLDS:
                data,x,m=information_data(spec)
                fit=data.indices('fit');order=np.random.default_rng(17).permutation(fit)
                ids=[data.rows[i]['sample_id'] for i in order]
                for view,columns in VIEW_COLUMNS.items():
                    name=spec['name']+'_'+view;values=x[:,:,columns];mask=m[:,:,columns]
                    input_sha=canonical_training_state_sha256(values,mask,[r['sample_id'] for r in data.rows],data.normalizer)
                    with isolated_training_rng(17):
                        encoder=ChronosInformationEncoder(assets,len(columns)).eval()
                        base_binding=dict(experiment=binding,fold=spec['name'],view=view,input_sha256=input_sha,
                                          model_sha256=assets['chronos2']['weights_sha256'],route='frozen')
                        frozen=export_information(encoder,values,mask,root/(name+'_frozen.npy'),base_binding,status_path,name+'_frozen')
                        train_root=root/'training'/name;train_root.mkdir(parents=True,exist_ok=True)
                        metadata=dict(experiment=binding,fold=spec['name'],view=view,input_sha256=input_sha,seed=17,
                                      method_name='chronos2',target_supervision='history_only',full_training_role=True,
                                      checkpoint_interval=250,train_ids=ids)
                        checkpoint=train_root/'training.pt';cursor=0
                        if checkpoint.exists():
                            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                            assert saved['metadata']==metadata;cursor=saved['update']
                            load_encoder_checkpoint(encoder,saved['encoder']);del saved
                        progress=TrainingProgress(status_path,name);adapt_start=time.monotonic()
                        while cursor<len(order):
                            if time.time()>=NIGHT_DEADLINE:raise TimeoutError('night budget saved at optimizer checkpoint')
                            end=min(cursor+250,len(order))
                            receipt=short_fit(encoder,values=values[order],mask=mask[order],prompts=None,
                                              targets={},definitions=(),train_ids=ids,metadata=metadata,root=train_root,
                                              stop_after=end,progress=progress)
                            cursor=receipt['updates']
                            save_rows(report/(name+'_history.csv'),receipt['history'])
                            print(name,'adapted',cursor,'/',len(order),flush=True)
                        checkpoint_sha=file_digest(checkpoint)
                        adapted_binding=dict(base_binding,model_sha256=checkpoint_sha,route='history_adapted')
                        adapted=export_information(encoder,values,mask,root/(name+'_adapted.npy'),adapted_binding,status_path,name+'_adapted')
                        unit=dict(fold=spec['name'],view=view,channels=len(columns),features=encoder.feature_dim,
                                  parameters=sum(p.numel() for p in encoder.parameters()),updates=cursor,
                                  fit_subjects=list(spec['fit']),fit_sample_ids=ids,
                                  checkpoint_path=str(checkpoint.resolve()),checkpoint_sha256=checkpoint_sha,
                                  frozen_path=str((root/(name+'_frozen.npy')).resolve()),frozen_sha256=frozen['sha256'],
                                  adapted_path=str((root/(name+'_adapted.npy')).resolve()),adapted_sha256=adapted['sha256'],
                                  adaptation_and_export_seconds=time.monotonic()-adapt_start)
                        save_json(report/(name+'_receipt.json'),unit);units.append(unit)
                        del encoder;torch.cuda.empty_cache()
                del data,x,m
            verify_assets(ASSETS)
            assert all(file_digest(p)==h for p,h in protected.items())
            result=dict(status='completed',units=units,updates=sum(u['updates'] for u in units),
                        elapsed_s=time.monotonic()-start,binding=binding,outer_scores_computed=False)
            atomic_json(status_path,result);return result
        except BaseException as exc:
            result=dict(status='budget_saved' if isinstance(exc,TimeoutError) else 'failed',error=repr(exc),
                        completed_units=units,elapsed_s=time.monotonic()-start,binding=binding)
            atomic_json(status_path,result);raise
