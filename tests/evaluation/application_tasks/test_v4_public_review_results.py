import json

import pytest

from chronaris.evaluation.application_tasks import v4_public_review_results as module


def _rows():
    rows=[]
    for seed in (17,29,43):
        for domain,task,metric,_ in module.TASKS:
            if domain=='simulation':continue
            for index,subject in enumerate(('a','b','c','d')):
                rows.append(dict(method='chronaris',route='self_supervised',candidate='reference',seed=seed,
                    domain=domain,task=task,metric=metric,role='validation',subject=subject,
                    fold_index=min(index,2),value=float(index)))
    return rows


def test_public_review_averages_subjects_not_folds_and_requires_complete_pairs():
    expected={domain:list('abcd') for domain in ('cogpilot','clare')}
    rows=_rows()
    result=module.aggregate_review_subject_rows(rows,expected)
    assert len(result)==12 and all(row['value']==1.5 and row['subject_count']==4 for row in result)
    for changed,match in ((rows[:-1],'every fixed'),(rows+[rows[0]],'more than one'),
                          ([r for r in rows if r['seed']!=43],'three seeds'),
                          ([rows[0] | {'role':'held_out'}]+rows[1:],'unapproved')):
        with pytest.raises(ValueError,match=match):module.aggregate_review_subject_rows(changed,expected)


def test_public_review_reader_preserves_fold_seed_routes_and_shared_update_counts(tmp_path,monkeypatch):
    plan={'plan_sha256':'a'*64,'source_code_sha256':'b'*64,'screen_root':str(tmp_path/'screen'),'units':[]}
    calls=[]
    for fold in range(3):
        for seed in (17,29,43):
            plan['units'].append(dict(domain='clare',method='chronaris',candidate_name='reference',
                                     phase='review',fold_index=fold,seed=seed,routes=['self_supervised','task_guided']))
            root=tmp_path/'clare/chronaris/reference/review'/f'seed{seed}'/f'fold{fold+1:02d}'
            root.mkdir(parents=True)
            (root/'run_state.json').write_text(json.dumps({'completed_consumers':['self_supervised:1500','task_guided:500']}))
    # One pending CogPilot unit keeps this fixture from claiming complete public coverage.
    plan['units'].append(dict(domain='cogpilot',method='chronaris',candidate_name='reference',
                             phase='review',fold_index=0,seed=17,routes=['self_supervised']))
    (tmp_path/'selection_plan.json').write_text(json.dumps(plan))
    state=dict(plan_sha256='a'*64,completed_units=[],failed_units=[])
    (tmp_path/'run_state.json').write_text(json.dumps(state))
    monkeypatch.setattr(module,'load_verified_review_plan',lambda *args,**kwargs:plan)
    monkeypatch.setattr(module,'load_development_inputs',lambda domain,*args,**kwargs:calls.append((domain,kwargs['fold_index'])) or ())
    def read(**kwargs):
        unit,route=kwargs['unit'],kwargs['route']
        assert kwargs['unit_root'].parts[-2:]==(f"seed{unit['seed']}",f"fold{unit['fold_index']+1:02d}")
        return dict(subject_rows=[],files={},metadata={},training_updates={'pretraining':700,'supervised':350 if route=='task_guided' else 0})
    monkeypatch.setattr(module,'read_public_candidate_unit',read)
    result=module.collect_public_review_results(output_root=tmp_path)
    assert calls==[('clare',0),('clare',1),('clare',2)]  # Inputs are reused across routes and seeds.
    assert len(result['units'])==18 and result['task_rows']==[]
    assert result['audited_pretraining_updates']==9*700 and result['audited_supervised_updates']==9*350
    assert result['pending']==['cogpilot/chronaris/reference/fold01/seed17/self_supervised']
    state['failed_units']=['cogpilot/chronaris/reference/fold01/seed17']
    (tmp_path/'run_state.json').write_text(json.dumps(state))
    assert module.collect_public_review_results(output_root=tmp_path)['status']=='blocked_by_public_review_execution_failure'
    state['completed_units']=state['failed_units'];state['failed_units']=[]
    (tmp_path/'run_state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='lacks a selected route'):module.collect_public_review_results(output_root=tmp_path)
