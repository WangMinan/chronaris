import torch
from chronaris.modeling.fusion_encoders.chronos_information import pack_observed_states


def test_packing_preserves_observed_level_scale_and_masks_missing_channels():
    values=torch.arange(2*6*5,dtype=torch.float32).reshape(2,6,5)
    mask=torch.ones_like(values,dtype=torch.bool);mask[0,:,0]=False
    embeds=[torch.ones(5,8,3),torch.ones(5,8,3)*2]
    scales=[(torch.arange(5.)[:,None],torch.ones(5,1)*3)]*2
    out=pack_observed_states(embeds,scales,values,mask)
    assert out.shape==(2,29)
    assert torch.equal(out[0,:5],torch.zeros(5))
    torch.testing.assert_close(out[:,-4:],values[:,-1,-4:])
    torch.testing.assert_close(out[1,:25].reshape(5,5)[:,-2],torch.arange(5.))
    torch.testing.assert_close(out[1,:25].reshape(5,5)[:,-1],torch.ones(5)*3)


def test_export_resumes_committed_chunks_and_rejects_modified_completed_cache(tmp_path,monkeypatch):
    import time
    import numpy as np
    import pytest
    from chronaris.evaluation.application_tasks import chronos_information_run as run
    monkeypatch.setattr(run,'NIGHT_DEADLINE',time.time()+60)
    class Encoder:
        feature_dim=4
        fail=True
        def eval(self):return self
        def __call__(self,x,mask):
            if self.fail and x[0,0,0]>=64:raise RuntimeError('interrupted')
            return x[:,-1]
    x=torch.arange(70.).reshape(70,1,1).expand(-1,96,4);mask=torch.ones_like(x,dtype=torch.bool)
    path=tmp_path/'features.npy';encoder=Encoder();binding={'source':'fixed'}
    with pytest.raises(RuntimeError):run.export_information(encoder,x,mask,path,binding,tmp_path/'status.json','test')
    encoder.fail=False
    result=run.export_information(encoder,x,mask,path,binding,tmp_path/'status.json','test')
    assert result['rows']==70 and result['completed']
    np.testing.assert_array_equal(np.load(path),x[:,-1].numpy())
    bad=np.load(path,mmap_mode='r+');bad[0,0]+=1;bad.flush();del bad
    with pytest.raises(AssertionError):run.export_information(encoder,x,mask,path,binding,tmp_path/'status.json','test')


def test_independent_features_keep_sample_order_and_all_predeclared_blocks(tmp_path):
    import numpy as np
    from chronaris.evaluation.application_tasks.chronos_information_readout import feature_rows
    units=[]
    for view,n in [('aircraft',3),('physiology_history',5)]:
        path=tmp_path/(view+'.npy');np.save(path,np.arange(4*n).reshape(4,n))
        units.append(dict(fold='one',view=view,frozen_path=str(path)))
    actual=feature_rows(units,'one','frozen','independent',[3,1])
    np.testing.assert_array_equal(actual,np.c_[np.arange(20).reshape(4,5)[[3,1]],np.arange(12).reshape(4,3)[[3,1]]])


def test_pretrained_gate_keeps_stronger_prior_references():
    from chronaris.evaluation.application_tasks.chronos_information_readout import information_gates
    from chronaris.evaluation.application_tasks.shape_mult_results import metric_tables
    rows=[]
    for fold,people in [('fold01',['a','b','c']),('fold02',['d','e'])]:
        for subject in people:
            methods={'reference_strong':.8}
            for route in ('frozen','adapted'):
                for view in ('aircraft','physiology_history','independent','joint'):
                    methods['chronos_'+route+'_'+view]=(.9 if route=='frozen' else .75) if view=='joint' else 1.
            for method,value in methods.items():rows.append(dict(fold=fold,method=method,subject=subject,sample_id=subject,truth=0.,prediction=value))
    people,summary=metric_tables(rows);gates=information_gates(people,summary)
    assert all(not g['passed'] for g in gates if g['route']=='frozen')
    assert all(g['passed'] for g in gates if g['route']=='adapted')
