import pytest

from chronaris.evaluation.application_tasks import v4_adoption as module


def _inputs():
    rows=[];tails=[]
    for candidate in ('reference','quality_gate'):
        for seed in module.SEEDS:
            tails.append(dict(candidate=candidate,seed=seed,missingness_p95=1.))
            for domain,task,metric,direction in module.TASKS:
                value=.5 if direction=='higher' else 1.
                if candidate=='quality_gate' and domain=='cogpilot' and metric=='macro_f1':value+=.01
                rows.append(dict(candidate=candidate,seed=seed,domain=domain,task=task,metric=metric,role='validation',value=value))
    return rows,{name:{'encoder_parameters':100,'training_elapsed_s':1.} for name in ('reference','quality_gate')},tails


def test_adoption_requires_replicated_gain_clean_guards_and_each_seed_tail():
    rows,meta,tails=_inputs()
    result=module.assess_candidate_adoption(rows,meta,tails)
    assert result['recommended_candidate']=='quality_gate' and not result['configuration_frozen']
    tails[-1]['missingness_p95']=1.21
    result=module.assess_candidate_adoption(rows,meta,tails)
    assert result['recommended_candidate']=='reference'
    assert [r['passed'] for r in result['assessments']['quality_gate']['tail_checks']]==[True,True,False]
    tails[-1]['missingness_p95']=1.2
    for row in rows:
        if row['candidate']=='quality_gate' and row['domain']=='clare' and row['metric']=='rmse':row['value']=1.051
    assert module.assess_candidate_adoption(rows,meta,tails)['recommended_candidate']=='reference'
    rows,meta,tails=_inputs()
    for row in rows:
        if row['candidate']=='quality_gate' and row['domain']=='clare' and row['metric']=='macro_f1':row['value']=.48
    assert module.assess_candidate_adoption(rows,meta,tails)['assessments']['quality_gate']['eligible']
    for row in rows:
        if row['candidate']=='quality_gate' and row['domain']=='clare' and row['metric']=='macro_f1':row['value']=.479
    assert module.assess_candidate_adoption(rows,meta,tails)['recommended_candidate']=='reference'
    rows,meta,tails=_inputs()
    for row in rows:
        if row['candidate']=='quality_gate' and row['domain']=='cogpilot' and row['metric']=='macro_f1':
            row['value']=.59 if row['seed']==17 else .5
    # A large improvement in only one seed is insufficient even when the mean clears 0.01.
    assert module.assess_candidate_adoption(rows,meta,tails)['recommended_candidate']=='reference'


def test_adoption_regression_threshold_zero_reference_and_complete_evidence():
    rows,meta,tails=_inputs()
    for row in rows:
        row['value']=1. if row['metric']=='rmse' else .5
        if row['candidate']=='quality_gate' and row['metric']=='rmse':row['value']=.98
    assert module.assess_candidate_adoption(rows,meta,tails)['recommended_candidate']=='quality_gate'
    for row in rows:
        if row['metric']=='rmse':row['value']=0.
    assert module.assess_candidate_adoption(rows,meta,tails)['recommended_candidate']=='reference'
    for row in rows:
        if row['candidate']=='quality_gate' and row['metric']=='rmse':row['value']=1e-15
    assert not module.assess_candidate_adoption(rows,meta,tails)['assessments']['quality_gate']['eligible']
    with pytest.raises(ValueError,match='three pressure seeds'):module.assess_candidate_adoption(rows,meta,tails[:-1])
    with pytest.raises(ValueError,match='every fixed task'):module.assess_candidate_adoption(rows[:-1],meta,tails)
    with pytest.raises(ValueError,match='invalid F1'):module.assess_candidate_adoption([rows[0] | {'value':2.}]+rows[1:],meta,tails)


def test_combined_decisions_wait_without_partial_recommendations(monkeypatch):
    monkeypatch.setattr(module,'collect_public_review_results',lambda **kwargs:{'status':'waiting_for_three_seed_review_plan'})
    monkeypatch.setattr(module,'collect_simulation_review_results',lambda **kwargs:{'status':'waiting_for_simulation_review_pressure'})
    result=module.collect_adoption_decisions(output_root='unused',pressure_root='unused')
    assert result['status']=='waiting_for_complete_three_seed_review' and result['decisions']=={}
    monkeypatch.setattr(module,'collect_simulation_review_results',lambda **kwargs:{'status':'blocked_by_simulation_review_execution_failure'})
    assert module.collect_adoption_decisions(output_root='unused',pressure_root='unused')['status']=='blocked_by_review_execution_failure'


def test_combined_decisions_cover_ten_method_routes_with_same_selection(monkeypatch):
    public=dict(status='public_review_verified_not_final_selection',selection_plan_sha256='a'*64,
                confirmation_feedback_used=False,task_rows=[],units=[])
    simulation=dict(status='simulation_review_verified_not_final_selection',selection_plan_sha256='a'*64,
                    confirmation_feedback_used=False,task_rows=[],completed=[])
    for method in module.METHODS:
        for route in module.ROUTES:
            rows,_,tails=_inputs()
            alternative='quality_gate' if method=='chronaris' else 'capacity64'
            for row in rows+tails:
                if row['candidate']=='quality_gate':row['candidate']=alternative
            for row in rows:
                row.update(method=method,route=route)
                (simulation if row['domain']=='simulation' else public)['task_rows'].append(row)
            for candidate in ('reference',alternative):
                for seed in module.SEEDS:
                    for domain in ('cogpilot','clare'):
                        for fold in range(3):
                            public['units'].append(dict(unit=f'{domain}/{method}/{candidate}/fold{fold+1:02d}/seed{seed}',route=route,
                                metadata={'encoder_parameters':100,'training_elapsed_s':1.}))
            simulation['completed'].extend(row | dict(method=method,route=route,encoder_parameters=100,training_elapsed_s=1.) for row in tails)
    monkeypatch.setattr(module,'collect_public_review_results',lambda **kwargs:public)
    monkeypatch.setattr(module,'collect_simulation_review_results',lambda **kwargs:simulation)
    result=module.collect_adoption_decisions(output_root='unused',pressure_root='unused')
    assert len(result['decisions'])==10 and result['status']=='single_factor_decisions_ready_not_frozen'
    assert all(v['recommended_candidate']==('quality_gate' if k.startswith('chronaris/') else 'capacity64')
               for k,v in result['decisions'].items())
    assert result['decisions']['chronaris/self_supervised']['rankings'][0]['encoder_parameters']==300
    simulation['selection_plan_sha256']='b'*64
    with pytest.raises(ValueError,match='different selections'):module.collect_adoption_decisions(output_root='unused',pressure_root='unused')
