from dataclasses import make_dataclass
import numpy as np
import pytest

from chronaris.evaluation.application_tasks.public_readout_revision import event_endpoint_s, revise_public_observation
from chronaris.evaluation.application_tasks.grouped_readout_repair import nested_readout, transformed


def test_native_endpoint_excludes_boundary_without_mutating_original():
    row = dict(sample_id='event', source_sample_hash='a'*64, task_valid_mask={'event_response': True},
               native_time_scale_s=86400., record_time_origin_native=738000., context_start_s=0.)
    cutoff = event_endpoint_s(row)
    assert cutoff < 12.
    fields = [s+'_'+f for s in ('physiology','vehicle') for f in ('timestamps_s','values','feature_mask')]
    Sample = make_dataclass('Sample', fields+['sample_id','source_sample_hash','context_duration_s'])
    t=np.array([0.,cutoff,12.]);v=np.arange(3.)[:,None];mask=np.ones((3,1),bool)
    sample=Sample(t,v,mask,t,v,mask,'event','a'*64,12.)
    fixed=revise_public_observation(sample,row)
    assert len(fixed.physiology_timestamps_s)==len(fixed.vehicle_timestamps_s)==1
    assert len(sample.physiology_timestamps_s)==3 and fixed.source_sample_hash!=sample.source_sample_hash
    with pytest.raises(ValueError,match='identity'):
        revise_public_observation(sample,row|{'source_sample_hash':'b'*64})


def test_nested_selection_ignores_scoring_labels_and_bounds_are_single_factor():
    rng=np.random.default_rng(1);x=rng.normal(size=(80,64));y=rng.normal(size=80)
    groups=np.repeat(['a','b','c','d'],20);tr=np.arange(60);te=np.arange(60,80)
    candidates=[dict(parameter=a) for a in (.1,10.)]
    a,trace=nested_readout(x,y,groups,tr,te,candidates,bounds=(1.,9.))
    y[te]=1000.
    b,other=nested_readout(x,y,groups,tr,te,candidates,bounds=(1.,9.))
    assert trace==other
    for key in a:
        np.testing.assert_array_equal(a[key]['prediction'],b[key]['prediction'])
    np.testing.assert_array_equal(a['selected_clipped']['prediction'],np.clip(a['selected']['prediction'],1.,9.))
    assert a['selected']['candidate']==a['selected_clipped']['candidate']
    with pytest.raises(ValueError,match='overlapping'):
        nested_readout(x,y,groups,tr,np.arange(50,70),candidates)


def test_block_scaling_survives_standardization_and_missing_class_rejected():
    x=np.random.default_rng(2).normal(size=(80,64))
    a,_=transformed(x[:60],x[60:],dict(parameter=1.))
    b,_=transformed(x[:60],x[60:],dict(parameter=1.,weights=(.1,1.,.1)))
    np.testing.assert_allclose(b[:,:24],a[:,:24]*.1)
    np.testing.assert_array_equal(b[:,24:48],a[:,24:48])
    y=np.zeros(80);y[:20]=1;g=np.repeat(['a','b','c','d'],20)
    with pytest.raises(ValueError,match='lacks'):
        nested_readout(x,y,g,np.arange(60),np.arange(60,80),[dict(parameter=1.)],classes=[0,1])


def test_versioned_targets_preserve_legacy_and_reject_tampering(tmp_path,monkeypatch):
    import torch
    from chronaris.dataset.clare_native import CLARE_SCHEMA
    from chronaris.dataset.lazy_observed import LazyObservedDataset,NativeSampleRecord
    from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskTargets
    from chronaris.evaluation.application_tasks.v4_public_data import V4PublicData,PUBLIC_TASKS
    from chronaris.evaluation.application_tasks import public_readout_revision as m
    source=tmp_path/'source';(source/'clare').mkdir(parents=True)
    for name in ('summary.json','sample_manifest.jsonl','task_targets.pt','prepared_cache_index.json'):
        (source/'clare'/name).write_text('original')
    registry=tmp_path/'registry.json';registry.write_text('{}')
    ids=('a','b','c');scores=torch.tensor([4.9,5.,7.]);valid=torch.tensor([True,True,True])
    target=ApplicationTaskTargets(ids,dict(workload_regression=scores,workload_classification=(scores>=7).long()),
        dict(workload_regression=valid,workload_classification=valid),{})
    records=[NativeSampleRecord(s,s,float(v),10.,'a'*64) for s,v in zip(ids,scores)]
    dataset=LazyObservedDataset(records,schema=CLARE_SCHEMA,loader=lambda _:None)
    rows=tuple(dict(sample_id=s,subject_id=s,source_sample_hash='a'*64,
        task_valid_mask=dict(workload_classification=True,workload_regression=True)) for s in ids)
    data=V4PublicData('clare',dataset,target,PUBLIC_TASKS['clare'],rows)
    monkeypatch.setattr(m,'load_prepared_public_development',lambda *a,**kw:data)
    output=tmp_path/'revised'
    m.prepare_readout_revision('clare',source_root=source,registry_path=registry,output_root=output)
    revised=m.load_readout_revision(output/'clare')
    assert revised.targets.values[m.STANDARD_TASK].tolist()==[0,1,1]
    assert target.values['workload_classification'].tolist()==[0,0,1]
    assert m.STANDARD_TASK not in target.values
    with pytest.raises(FileExistsError):
        m.prepare_readout_revision('clare',source_root=source,registry_path=registry,output_root=output)
    with pytest.raises(ValueError,match='outside'):
        m.prepare_readout_revision('clare',source_root=source,registry_path=registry,output_root=source)
    (output/'clare'/'task_targets.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        m.load_readout_revision(output/'clare')
