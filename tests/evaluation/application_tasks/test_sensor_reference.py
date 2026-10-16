from copy import deepcopy
import pytest
import torch
from chronaris.evaluation.application_tasks import chapter5_delivery_sensor as sensor


def test_reference_requires_exact_identity_original_device_and_full_finite_features():
    request={'sample_ids':['a','b'],'role':'validation'}
    good=dict(request=request,device='NVIDIA GeForce RTX 4090',source_code_sha256=sensor.SOURCE,optimizer_updates=0)
    features=torch.zeros(2,4096)
    sensor.validate_output(good,request,features)
    for changes in ({'device':'NVIDIA GeForce RTX 4070 SUPER'},{'source_code_sha256':'modified'},
                    {'optimizer_updates':1},{'request':dict(request,role='held_out')}):
        with pytest.raises(ValueError):sensor.validate_output(good|changes,request,features)
    for invalid in (features[:1],features.double(),features+float('nan')):
        with pytest.raises(ValueError):sensor.validate_output(good,request,invalid)


def test_reference_rejects_unplanned_domain_before_loading_any_asset(tmp_path):
    for spec in ({'method':'chronaris','domain':'clare'},{'method':'sensorllm_deepseek','domain':'dingxin'}):
        with pytest.raises(ValueError,match='limited to frozen'):
            sensor.encode_remote(spec,{}, {}, {},None,None,tmp_path,'held_out')
    assert not list(tmp_path.iterdir())


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
