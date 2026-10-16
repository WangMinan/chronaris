import pytest
import torch
from chronaris.evaluation.application_tasks import chapter5_delivery_sensor as sensor


def test_all_sensor_fold_gates_precede_formal_training_and_outer_evaluation():
    from chronaris.evaluation.application_tasks.chapter5_delivery import steps
    keys=[f'reference_{i}' for i in (8,35,62,89,116)]
    stages=steps(dict(replay=keys,native=[dict(key=k) for k in keys]))
    last_replay=max(stages.index('replay:'+k) for k in keys)
    assert last_replay < min(stages.index(s) for s in stages if s.startswith('train:'))
    assert last_replay < stages.index('freeze') < min(stages.index('native:'+k) for k in keys)


@pytest.mark.parametrize('method,domain,dimension', [('sensorllm_deepseek','clare',4096),('chronos2','dingxin',768)])
def test_reference_requires_exact_identity_original_device_and_full_finite_features(method,domain,dimension):
    request={'sample_ids':['a','b'],'role':'validation','method':method,'domain':domain,'feature_dim':dimension}
    good=dict(request=request,device='NVIDIA GeForce RTX 4090',source_code_sha256=sensor.SOURCE,optimizer_updates=0)
    features=torch.zeros(2,dimension)
    sensor.validate_output(good,request,features)
    for changes in ({'device':'NVIDIA GeForce RTX 4070 SUPER'},{'source_code_sha256':'modified'},
                    {'optimizer_updates':1},{'request':dict(request,role='held_out')}):
        with pytest.raises(ValueError):sensor.validate_output(good|changes,request,features)
    for invalid in (features[:1],features[:,:-1],features.double(),features+float('nan')):
        with pytest.raises(ValueError):sensor.validate_output(good,request,invalid)


def test_reference_rejects_unplanned_domain_before_loading_any_asset(tmp_path):
    for spec in ({'method':'chronaris','domain':'clare'},{'method':'sensorllm_deepseek','domain':'dingxin'},
                 {'method':'chronos2','domain':'clare'},{'method':'chronos2','domain':'cogpilot'}):
        with pytest.raises(ValueError,match='limited to frozen'):
            sensor.encode_remote(spec,{}, {}, {},None,None,tmp_path,'held_out')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('method,domain', [('sensorllm_deepseek','clare'),('chronos2','dingxin')])
def test_delivery_installs_reference_encoder_for_both_device_sensitive_methods(monkeypatch,method,domain):
    from chronaris.evaluation.application_tasks import chapter5_delivery as delivery,chapter5_native as native
    spec={'method':method,'domain':domain}
    row={'key':'reference_262','index':262,'origin':'reference','unit':spec}
    monkeypatch.setattr(native,'unit_record',lambda *a:{'unit':spec})
    monkeypatch.setattr(native,'_encode',None)
    delivery.native_record({'main_root':'unused','native':[row]},row['key'])
    assert native._encode.__name__=='encode_remote'


def test_chronos_replay_failure_still_stops_delivery_under_original_tolerance(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from contextlib import nullcontext
    from chronaris.evaluation.application_tasks import chapter5_delivery as delivery,v4_public_screen
    from chronaris.evaluation.application_tasks.device_replay import prediction_changes
    expected=[dict(sample_id='a',task='regression',field=4,target_valid=True,target=0.,group_id='g',
                   no_observation=False,sample_weight=1.,prediction=0.,probability=None)]
    delta=prediction_changes(expected,[expected[0]|{'prediction':2e-5}])
    assert not delta['strict']['passed'] and delta['strict']['atol']==delta['strict']['rtol']==1e-5
    row={'key':'reference_262','index':262,'unit':{'method':'chronos2','domain':'dingxin'}}
    monkeypatch.setattr(delivery,'plan_at',lambda root:dict(plan_sha256='fixed',data_root='data',registry_path='registry'))
    monkeypatch.setattr(delivery,'steps',lambda plan:['replay:reference_262'])
    monkeypatch.setattr(delivery,'native_record',lambda *a:(row,{}))
    monkeypatch.setattr(v4_public_screen,'development_gpu_lock',lambda:nullcontext(True))
    replay=SimpleNamespace(replay_record=lambda *a,**k:{'rows':[delta]})
    reference=SimpleNamespace(supports=sensor.supports,bind_evidence=lambda result,root:result)
    monkeypatch.setattr(delivery,'helper',lambda name:replay if name=='device_replay' else reference)
    with pytest.raises(ValueError,match='frozen inference qualification failed'):
        delivery.worker(tmp_path,'replay:reference_262')
    assert not (tmp_path/'delivery_models.json').exists()


def test_reference_receipts_are_mandatory_and_transferred_bytes_bound(tmp_path):
    from chronaris.evaluation.application_tasks.chapter5_assets import write,sha
    result={'status':'completed','rows':[{}],'files':{}}
    with pytest.raises(ValueError,match='incomplete'):sensor.bind_evidence(result,tmp_path)
    p=tmp_path/'summary_adapted/representation/reference_executor';p.mkdir(parents=True)
    data=p/'features.pt';data.write_bytes(b'original')
    write(p/'receipt.json',{'files':{str(data):sha(data)}})
    bound=sensor.bind_evidence(result,tmp_path)
    assert str(data) in bound['files'] and str(p/'receipt.json') in bound['files']
    data.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):sensor.bind_evidence(result,tmp_path)
