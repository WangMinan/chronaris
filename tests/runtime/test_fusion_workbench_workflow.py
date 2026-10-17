"""Task identity, data ingestion and recovery contracts for the workbench."""
from copy import deepcopy
import time

import pytest

from tests.runtime.test_fusion_workbench import case, fake_adapter
from chronaris.serving.fusion_workbench import inference
from chronaris.serving.fusion_workbench.catalog import Catalog
from chronaris.serving.fusion_workbench.cases import read_json
from chronaris.serving.fusion_workbench.server import DemoStore


def payload(case, count=2):
    sample = case.sample(0)
    arrays = {k:getattr(sample,k).tolist() for k in (
        'physiology_values','physiology_timestamps_s','physiology_feature_mask',
        'vehicle_values','vehicle_timestamps_s','vehicle_feature_mask')}
    return {'format':'chronaris.observations.v1','title':'新观测记录',
        'model_id':case.manifest['checkpoint_sha256'],'schema':case.schema.to_dict(),
        'windows':[{'sample_id':f'new-{i}','start_s':float(i),'duration_s':30.,**deepcopy(arrays)} for i in range(count)]}


def wait(store, job):
    for _ in range(300):
        saved=read_json(store.job_root(job['id'])/'job.json')
        if saved['status'] in ('completed','failed') and store.active is None:
            return saved
        time.sleep(.01)
    raise AssertionError('job did not finish')


def test_import_append_deduplicate_and_validate(case):
    catalog=Catalog(case.root.parent);catalog.refresh()
    p=payload(case);record=catalog.import_record(p)
    assert len(record['windows'])==2
    assert record['appendable']
    assert len(catalog.cases)==2
    p['record_id']=record['id']
    assert len(catalog.import_record(p)['windows'])==2
    p['windows'][0]['physiology_values'][0][0]=100
    with pytest.raises(ValueError,match='内容发生变化'):
        catalog.import_record(p)
    assert len(catalog.cases[record['id']].manifest['windows'])==2
    p=payload(case);p['schema']['physiology_feature_names']=['other']
    with pytest.raises(ValueError,match='字段顺序'):
        catalog.import_record(p)
    p=payload(case);p['windows'][0]['physiology_feature_mask']=[[1],[1]]
    with pytest.raises(ValueError,match='布尔'):
        catalog.import_record(p)
    assert not list(case.root.parent.glob('.import-*'))


def test_long_selection_incremental_restart_and_delivery(case,fake_adapter,tmp_path):
    store=DemoStore(case.root.parent,tmp_path/'runs')
    try:
        record=store.import_record(payload(case,12))
        job=store.submit({'case_id':record['id'],'start':0,'stop':10,'incremental':True})
        saved=wait(store,job)
        assert saved['status']=='completed',saved
        assert len(saved['window_keys'])==10
        assert saved['request']['stop']==10
        result=store.deliver(job['id'])
        assert result['windows']==10 and result['valid_points']==960
        assert len((store.job_root(job['id'])/'pooled_features.csv').read_text().splitlines())==11
        store.close()
        store=DemoStore(case.root.parent,tmp_path/'runs')
        job=store.submit({'case_id':record['id'],'start':0,'stop':12,'incremental':True})
        assert job['selected_indices']==[10,11]
        assert wait(store,job)['status']=='completed'
        with pytest.raises(ValueError,match='没有新增窗口'):
            store.submit({'case_id':record['id'],'start':0,'stop':12,'incremental':True})
    finally:
        store.close()


def test_retry_uses_snapshot_and_reuses_finished_chunks(case,fake_adapter,tmp_path,monkeypatch):
    store=DemoStore(case.root.parent,tmp_path/'runs')
    load=inference.load_adapter
    calls=[]
    def failing_load(case):
        adapter,p=load(case)
        def encode(batch):
            calls.extend(batch.sample_ids)
            if batch.sample_ids[0]=='sample1':
                raise ValueError('输入读取中断')
            return adapter(batch)
        return encode,p
    monkeypatch.setattr(inference,'load_adapter',failing_load)
    try:
        failed=wait(store,store.submit({'case_id':'test-case','start':0,'stop':2}))
        assert failed['status']=='failed' and failed['done']==1
        # Changing catalog order must not change the original retry selection.
        store.cases['test-case'].manifest['windows'].reverse()
        calls.clear()
        def counting_load(case):
            adapter,p=load(case)
            def encode(batch):
                calls.extend(batch.sample_ids)
                return adapter(batch)
            return encode,p
        monkeypatch.setattr(inference,'load_adapter',counting_load)
        recovered=wait(store,store.retry(failed['id']))
        assert recovered['status']=='completed',recovered
        assert recovered['reused_windows']==1
        assert calls==['sample1']
        assert recovered['retry_of']==failed['id']
        assert read_json(store.job_root(failed['id'])/'job.json')['status']=='failed'
    finally:
        store.close()


def test_input_mutation_rejects_retry_cache(case,fake_adapter,tmp_path):
    inference.run_case(case,[0],tmp_path/'first')
    (case.root/'window.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='输入窗口校验'):
        inference.run_case(case,[0],tmp_path/'second',resume_root=tmp_path/'first')


def test_incremental_recomputes_after_model_or_implementation_change(case,fake_adapter,tmp_path,monkeypatch):
    from chronaris.serving.fusion_workbench import server
    from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
    store = DemoStore(case.root.parent, tmp_path/'runs')
    try:
        request = {'case_id':'test-case','start':0,'stop':2,'incremental':True}
        assert wait(store, store.submit(request))['status'] == 'completed'
        other = tmp_path/'different.pt'; other.write_bytes(b'different weights')
        digest = sha256_file(other)
        store.catalog.models[digest] = (other, case.schema)
        changed = wait(store, store.submit(dict(request, model_id=digest)))
        assert changed['total'] == 2 and changed['reused_windows'] == 0
        monkeypatch.setattr(server, 'implementation_digest', lambda: 'new implementation')
        changed = wait(store, store.submit(request))
        assert changed['total'] == 2 and changed['reused_windows'] == 0
    finally:
        store.close()


@pytest.mark.parametrize('times', [[1., 0.], [0., 31.], [0., float('nan')]])
def test_import_rejects_invalid_observation_times(case, times):
    catalog = Catalog(case.root.parent); catalog.refresh()
    p = payload(case, 1)
    p['windows'][0]['physiology_timestamps_s'] = times
    with pytest.raises(ValueError):
        catalog.import_record(p)


def test_empty_observations_rejected(case):
    catalog=Catalog(case.root.parent);catalog.refresh()
    p=payload(case,1)
    for stream in ('physiology','vehicle'):
        p['windows'][0][stream+'_feature_mask']=[[False],[False]]
    with pytest.raises(ValueError,match='有效观测'):
        catalog.import_record(p)


def test_schema_review_is_persistent_and_scoped(case,tmp_path):
    from chronaris.serving.fusion_workbench.fields import FieldReviews
    path=tmp_path/'reviews.json';reviews=FieldReviews(path)
    rows=reviews.rows(case)
    reviews.review(case,{'ids':[rows[0]['id']],'decision':'approved'})
    assert FieldReviews(path).rows(case)[0]['status']=='approved'
    with pytest.raises(ValueError):
        reviews.review(case,{'ids':['unknown'],'decision':'approved'})


def test_registered_model_metadata_and_mismatch(case,tmp_path,monkeypatch):
    from chronaris.serving.fusion_workbench import catalog as module
    p={'physiology_feature_names':case.schema.physiology_feature_names,
       'vehicle_feature_names':case.schema.vehicle_feature_names,'fold':{'train_sample_ids':['a'],'validation_sample_ids':['b']},
       'encoder_manifest':{'backbone_config':{'semantic_event_enabled':True}}}
    monkeypatch.setattr(module,'inspect_model',lambda path:p)
    monkeypatch.setattr(module,'load_common_pretraining_checkpoint',lambda *a,**k:None)
    catalog=Catalog(case.root.parent);catalog.refresh()
    model=tmp_path/'new.pt';model.write_bytes(b'another-model')
    digest=catalog.register_model(model,case.schema)
    listed=next(m for m in catalog.model_list() if m['id']==digest)
    assert listed['semantic_event_enabled'] and listed['training_windows']==1
    assert listed['compatible_cases']==['test-case']
    p['vehicle_feature_names']=['different']
    with pytest.raises(ValueError,match='不兼容'):
        catalog.register_model(model,case.schema)


def test_import_rejects_spreadsheet_formula_identifier(case):
    catalog=Catalog(case.root.parent);catalog.refresh()
    p=payload(case,1);p['windows'][0]['sample_id']='=1+1'
    with pytest.raises(ValueError,match='标识不合法'):
        catalog.import_record(p)


def test_interrupted_job_can_resume_after_service_restart(case,fake_adapter,tmp_path):
    from chronaris.serving.fusion_workbench.cases import write_json
    store=DemoStore(case.root.parent,tmp_path/'runs')
    done=wait(store,store.submit({'case_id':'test-case','start':0,'stop':2}))
    store.close()
    root=store.job_root(done['id'])
    done.update(status='exporting');write_json(root/'job.json',done)
    restored=DemoStore(case.root.parent,tmp_path/'runs')
    try:
        assert restored.history()[0]['retryable']
        result=wait(restored,restored.retry(done['id']))
        assert result['status']=='completed' and result['reused_windows']==2
    finally:
        restored.close()


def test_http_import_fields_and_boundary_errors(case,tmp_path,monkeypatch):
    import json
    import threading
    from urllib.request import Request,urlopen
    from urllib.error import HTTPError
    from chronaris.serving.fusion_workbench.http import make_server
    from chronaris.serving.fusion_workbench import catalog as module
    monkeypatch.setattr(module,'inspect_model',lambda path:{
        'physiology_feature_names':case.schema.physiology_feature_names,
        'vehicle_feature_names':case.schema.vehicle_feature_names})
    store=DemoStore(case.root.parent,tmp_path/'runs');server=make_server(store,0)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{server.server_port}'
    def request(path,body=None):
        data=None if body is None else json.dumps(body).encode()
        return json.load(urlopen(Request(base+path,data=data,headers={'Content-Type':'application/json'}),timeout=5))
    try:
        assert request('/api/sources')==[]
        assert len(request('/api/models'))==1
        record=request('/api/cases/import',payload(case))
        assert len(request('/api/cases'))==2
        rows=request('/api/fields/'+record['id'])
        request('/api/fields/'+record['id'],{'ids':[rows[0]['id']],'decision':'approved'})
        assert request('/api/fields/'+record['id'])[0]['status']=='approved'
        invalid=payload(case);invalid['windows'][0]['physiology_values'][0][0]=float('nan')
        for path,body in (('/api/cases/import',invalid),('/api/runs',{'case_id':[],'start':0,'stop':1}),('/api/sources/prepare',{'source_id':'unknown'})):
            with pytest.raises(HTTPError) as e:request(path,body)
            assert e.value.code==400
    finally:
        server.shutdown();server.server_close();store.close()


def test_device_change_invalidates_incremental_and_resume(case, fake_adapter, tmp_path, monkeypatch):
    from chronaris.serving.fusion_workbench import device
    from chronaris.serving.fusion_workbench.cases import write_json
    monkeypatch.setattr(device.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(device.torch.cuda, 'get_device_name', lambda i: 'test GPU')
    cpu = inference.window_key(case, 0, device='cpu')
    gpu = inference.window_key(case, 0, device='cuda')
    assert cpu != gpu
    store = DemoStore(case.root.parent, tmp_path/'runs')
    try:
        job = wait(store, store.submit({'case_id':'test-case','start':0,'stop':2,'incremental':True}))
        assert job['device'] == 'cpu'
        # A previous GPU result cannot satisfy a CPU incremental request.
        job['window_keys'] = [inference.window_key(case, i, device='cuda') for i in (0, 1)]
        write_json(store.job_root(job['id'])/'job.json', job)
        repeat = wait(store, store.submit({'case_id':'test-case','start':0,'stop':2,'incremental':True}))
        assert repeat['total'] == 2 and repeat['reused_windows'] == 0
    finally:
        store.close()


def test_requested_unavailable_gpu_fails_without_cpu_fallback(case, tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='未回退'):
        DemoStore(case.root.parent, tmp_path/'runs', device='cuda')


def test_out_of_memory_preserves_failed_job(case, tmp_path, monkeypatch):
    import torch
    def failed(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError('GPU显存不足')
    monkeypatch.setattr(inference, 'load_adapter', failed)
    store = DemoStore(case.root.parent, tmp_path/'runs')
    try:
        job = wait(store, store.submit({'case_id':'test-case','start':0,'stop':1}))
        assert job['status'] == 'failed' and '显存不足' in job['error']
        assert job['request']['model_id'] == case.manifest['checkpoint_sha256']
    finally:
        store.close()
