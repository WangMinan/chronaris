from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chronaris.evaluation.application_tasks import chapter5_delivery as delivery
from chronaris.evaluation.application_tasks import chapter5_delivery_assets as assets
from chronaris.evaluation.application_tasks.chapter5_assets import read,sha,write
from chronaris.evaluation.application_tasks.chapter5_delivery_export import aggregate,archive_csv


def test_delivery_order_never_opens_outer_before_training_replay_and_development_grid():
    stages=delivery.steps({'native':[{'key':'new_0'}],'replay':['new_0']})
    assert stages.index('sensitivity') < stages.index('freeze') < stages.index('generate') < stages.index('native:new_0')
    assert [s for s in stages if s.startswith('train:')]==['train:24','train:27','train:30','train:33']
    assert stages[-1]=='materials' and len([s for s in stages if s.startswith('simulation:')])==11


def test_outer_gate_cannot_be_opened_by_unfrozen_plan(tmp_path):
    with pytest.raises(ValueError,match='all extra models'):
        delivery.require_freeze(tmp_path,{})
    assert not list(tmp_path.iterdir())


def test_plan_source_and_content_mutation_rejected(tmp_path,monkeypatch):
    from chronaris.evaluation.application_tasks import v4_development_data as data
    monkeypatch.setattr(data,'v4_workflow_source_sha256',lambda:'frozen')
    source=tmp_path/'model';source.write_text('original')
    monkeypatch.setattr(delivery,'execution_files',lambda:{str(source):sha(source)})
    value=delivery.seal_development_plan(dict(root=str(tmp_path),source_code_sha256='frozen',input_files={str(source):sha(source)}))
    write(tmp_path/'delivery_plan.json',value);assert delivery.plan_at(tmp_path)==value
    value['source_code_sha256']='other';write(tmp_path/'delivery_plan.json',value)
    with pytest.raises(ValueError,match='implementation changed'):delivery.plan_at(tmp_path)
    value['source_code_sha256']='frozen';write(tmp_path/'delivery_plan.json',value);source.write_text('modified')
    with pytest.raises(ValueError,match='source changed'):delivery.plan_at(tmp_path)


def test_missing_native_key_rejected_before_reading_labels():
    with pytest.raises(ValueError,match='outside declared'):delivery.native_record({'native':[]},'new_0')


def test_asset_restore_keeps_existing_bytes_and_rejects_corrupt_download(tmp_path,monkeypatch):
    import io,hashlib
    path=tmp_path/'weight';expected=hashlib.sha256(b'correct').hexdigest()
    job={'path':str(path),'sha256':expected,'url':'https://example.invalid/pinned'}
    monkeypatch.setattr(assets,'urlopen',lambda *a,**k:io.BytesIO(b'wrong'))
    with pytest.raises(ValueError,match='download hash'):assets.restore_one(job)
    assert not path.exists() and path.with_suffix('.download').read_bytes()==b'wrong'
    monkeypatch.setattr(assets,'urlopen',lambda *a,**k:io.BytesIO(b'correct'))
    assets.restore_one(job);assert path.read_bytes()==b'correct'
    path.write_bytes(b'local-original')
    with pytest.raises(ValueError,match='existing asset'):assets.restore_one(job)
    assert path.read_bytes()==b'local-original'


def test_asset_resolution_requires_revision_and_hash(tmp_path):
    manifest=tmp_path/'manifest.json';model=tmp_path/'weights/model.bin'
    write(manifest,{'files':{str(model):'hash'},'model':{'path':str(model.parent),'source':'org/model','revision':'fixed'}})
    jobs=assets.missing_assets([manifest,manifest]);assert len(jobs)==1
    assert jobs[0]['url']=='https://huggingface.co/org/model/resolve/fixed/model.bin'
    write(manifest,{'files':{str(model):'hash'}})
    with pytest.raises(ValueError,match='pinned upstream'):assets.missing_assets([manifest])


def test_fold_mean_is_not_weighted_by_window_count_and_missing_values_visible():
    result=aggregate([{'method':'a','value':1.,'windows':100},{'method':'a','value':3.,'windows':1},
                      {'method':'a','value':None,'windows':10}],('method',))[0]
    assert result=={'method':'a','mean':2.,'std':1.,'settings':3,'available_settings':2}
    with pytest.raises(ValueError,match='nonfinite'):aggregate([{'method':'a','value':float('nan')}],('method',))


def test_archive_keeps_full_sample_sequence_and_role(tmp_path):
    import csv,gzip
    source=tmp_path/'prediction.npz'
    np.savez(source,held_out_sample_ids=np.array(['a','b']),held_out_state=np.array([[1,2],[2,1]]),validation_sample_ids=np.array(['c']))
    archive_csv(source,tmp_path/'data.csv.gz')
    with gzip.open(tmp_path/'data.csv.gz','rt') as handle:rows=list(csv.DictReader(handle))
    assert [r['sample_id'] for r in rows]==['a','b'] and json.loads(rows[1]['state'])==[2,1]
    np.savez(source,held_out_sample_ids=np.array(['a','b']),held_out_state=np.array([[1,2]]))
    with pytest.raises(ValueError,match='count mismatch'):archive_csv(source,tmp_path/'other.csv.gz')


def test_sensor_offload_loads_projection_before_dispatch_and_keeps_dtype(monkeypatch):
    import torch
    from chronaris.modeling.fusion_encoders.sensorllm_adapter import SensorLLMWindowEncoder
    encoder=SensorLLMWindowEncoder.__new__(SensorLLMWindowEncoder);torch.nn.Module.__init__(encoder)
    encoder.core=torch.nn.Module();encoder.core.ts_proj=torch.nn.Linear(2,2,dtype=torch.bfloat16)
    encoder.core.layers=torch.nn.ModuleList([torch.nn.Linear(1,1) for _ in range(32)])
    encoder.inference_offload=True
    saved={'weight':torch.full((2,2),3.,dtype=torch.bfloat16),'bias':torch.full((2,),4.,dtype=torch.bfloat16)}
    def dispatch(core,**kwargs):
        assert torch.equal(core.ts_proj.weight,saved['weight'])
        assert kwargs['device_map']=={'':'cuda',**{f'layers.{i}':'cpu' for i in range(8,32)}}
        return core
    monkeypatch.setitem(sys.modules,'accelerate',SimpleNamespace(dispatch_model=dispatch))
    encoder.load_checkpoint_state(saved)
    assert encoder.core.ts_proj.weight.dtype==torch.bfloat16
    assert not any(p.requires_grad for p in encoder.parameters())


def test_portable_plot_bundle_renders_without_repository_imports(tmp_path):
    import shutil,subprocess
    from chronaris.evaluation.application_tasks.chapter5_export import csv_file
    source=Path(__file__).parents[3]/'docs/artifacts/runs/2026-10-15_chapter5-writing-delivery/plot_bundle'
    root=tmp_path/'bundle';shutil.copytree(source,root);data=root/'data';data.mkdir()
    native=[dict(domain='clare',method='chronaris',variant='thesis_reference',route='self_supervised',consumer='linear',
                 task='workload_classification',metric='macro_f1',error_units='native',role='held_out',value=v) for v in (.4,.6)]
    csv_file(data/'native_metrics.csv',native)
    csv_file(data/'native_summary.csv',aggregate(native,('domain','method','variant','route','consumer','task','metric','error_units','role')))
    sim=[];sensitivity=[];causality=[];costs=[];timing=[]
    for route in ('self_supervised','task_guided'):
        for condition in ('clean_asynchronous','random_missing_00pct'):
            for task,metric,consumer in [('workload_classification','macro_f1','linear'),('workload_regression','rmse','linear'),
                    ('workload_regression','rmse','minirocket'),('maneuver_segmentation','segmental_f1_iou_0.50','causal_tcn_duration')]:
                sim.append(dict(method='chronaris',variant='reference',route=route,consumer=consumer,task=task,metric=metric,condition=condition,role='held_out',value=.5))
        for weight in (0.,.025,.05,.1):
            for task,metric in [('simulated_future_workload_classification','macro_f1'),('simulated_future_workload_regression','rmse')]:
                for consumer in ('linear','minirocket'):
                    sensitivity.append(dict(weight=weight,route=route,task=task,metric=metric,consumer=consumer,value=.5))
        for cutoff in (5.,15.,25.):
            for perturbation in ('repeat','future_values','future_missing'):
                causality.append(dict(route=route,cutoff_s=cutoff,perturbation=perturbation,max_delta=1e-6))
        costs.append(dict(method='chronaris',variant='reference',route=route,mean=.1,std=.001))
        for target in ('relative_clock_offset_magnitude_s','signed_relative_clock_offset_s','primary_physiology_response_lag_s'):
            timing.append(dict(route=route,condition='clean_asynchronous',method='chronaris',unit='chronaris/reference/seed17/'+route,target=target,rmse=1.))
    csv_file(data/'simulation_groups.csv',sim)
    csv_file(data/'simulation_summary.csv',aggregate(sim,('method','variant','route','consumer','task','metric','condition','role')))
    for name,rows in [('sensitivity',sensitivity),('causality_summary',causality),('inference_summary',costs),('timing_metrics',timing)]:csv_file(data/(name+'.csv'),rows)
    write(root/'evidence.json',dict(data_files={str(p.relative_to(root)):sha(p) for p in data.iterdir()}))
    subprocess.run([sys.executable,'-I',str(root/'plot.py')],cwd=tmp_path,check=True,capture_output=True)
    subprocess.run([sys.executable,'-I',str(root/'verify.py')],cwd=tmp_path,check=True,capture_output=True)
    assert len(list((root/'figures').glob('*.png')))==17
    assert len(list((root/'figures').glob('*.svg')))==17


def test_asset_resume_checks_server_range_and_retries_transient_network(tmp_path,monkeypatch):
    import io,hashlib
    path=tmp_path/'model.bin';path.with_suffix('.bin.download').write_bytes(b'abc')
    job=dict(path=str(path),sha256=hashlib.sha256(b'abcdef').hexdigest(),url='https://example.invalid/pinned')
    calls=[]
    def download(request,**kwargs):
        calls.append(request.get_header('Range'))
        if len(calls)==1:raise TimeoutError('transient')
        response=io.BytesIO(b'def');response.status=206;response.headers={'Content-Range':'bytes 3-5/6'};return response
    monkeypatch.setattr(assets,'urlopen',download);assets.restore_one(job)
    assert path.read_bytes()==b'abcdef' and calls==['bytes=3-','bytes=3-']
