import torch
from chronaris.modeling.fusion_encoders.fair_residual import FairResidualModel
from chronaris.modeling.fusion_encoders.causal_query import CausalQueryStream
from chronaris.evaluation.application_tasks.performance_pilot_data import future_query_variants


def test_matched_mult_capacity_and_isolated_aircraft_path():
    torch.set_num_threads(1); torch.manual_seed(17)
    dual = FairResidualModel('mult').eval()
    air = FairResidualModel('mult_aircraft').eval(); air.load_state_dict(dual.state_dict())
    assert sum(x.numel() for x in dual.parameters()) == sum(x.numel() for x in air.parameters()) == 1016577
    def query(n):
        x=torch.randn(2,96,n); m=torch.ones_like(x,dtype=torch.bool)
        return CausalQueryStream(x,m,m.any(-1),torch.ones_like(x)*.01,
                                 torch.arange(1,97).expand(2,-1)*30/96,torch.zeros_like(x,dtype=torch.long))
    p,v=query(16),query(12)
    modified,_=future_query_variants(p,cutoff=-1)
    torch.testing.assert_close(air(p,v)[0],air(modified,v)[0],atol=0,rtol=0)
    for model in (dual,air):
        old,head=model(p,v); assert head.abs().max()==0
        for a,b in zip(future_query_variants(p),future_query_variants(v)):
            torch.testing.assert_close(model(a,b)[0][:,:48],old[:,:48],atol=1e-5,rtol=1e-5)


def test_stable_gate_requires_both_folds_and_pairing():
    from chronaris.evaluation.application_tasks.shape_mult_results import metric_tables,shape_gates
    rows=[];pair=[]
    for fold,subjects in [('fold01',['a','b','c']),('fold02',['d','e'])]:
        for s in subjects:
            for method,value in [('aircraft_ridge',1.),('candidate_ridge',.9 if fold=='fold01' else .99),('candidate_head',.8)]:
                rows.append(dict(fold=fold,subject=s,sample_id=s,method=method,truth=0.,prediction=value))
            for method,value in [('different_record_1',1.),('different_record_2',1.),('different_record_3',1.),('zero_values',.7)]:
                pair.append(dict(fold=fold,subject=s,sample_id=s,method=method,truth=0.,prediction=value))
    people,summary=metric_tables(rows);_,paired=metric_tables(pair)
    gates,pair_gates=shape_gates(people,summary,paired)
    assert next(r for r in gates if r['fold']=='fold01' and r['route']=='ridge')['passed']
    assert not next(r for r in gates if r['fold']=='fold02' and r['route']=='ridge')['passed']
    assert not any(r['passed'] for r in pair_gates)


def test_exact_state_bytes_include_nan_dtype_markers():
    from chronaris.evaluation.application_tasks.shape_mult_verify import same_tensor_bytes
    a=torch.tensor([float('nan'),1.])
    assert same_tensor_bytes(a,a.clone())
    b=a.clone();b[1]=1.001
    assert not same_tensor_bytes(a,b)
    assert not same_tensor_bytes(a,a.double())
