import json
from pathlib import Path

import pytest

from chronaris.evaluation.application_tasks.chapter5_assets import bind, sha, verify, write
from chronaris.evaluation.application_tasks.chapter5_core import core_units, groups
from chronaris.evaluation.application_tasks.chapter5_native import compare_predictions, evaluate_main_unit, require_outer_freeze


def test_core_matrix_only_removes_enabled_components_and_preserves_single_factor():
    units=core_units()
    assert len(units)==36 and sum(len(u['routes']) for u in units)==69
    assert {u['seed'] for u in units}=={17,29,43}
    base=units[0]['options']
    for unit in units:
        if unit['purpose']=='main_reference':continue
        changed={k:v for k,v in unit['options'].items() if k!='name' and base.get(k)!=v}
        assert len(changed)==1
        assert unit['method']=='chronaris' and unit['options']['training']=={}
    assert not any('independent_pair' in u['options']['name'] for u in units)
    assert {u['options'].get('physics_weight',.05) for u in units if u['method']=='chronaris'}=={0.,.025,.05,.1}
    stages=[name for group in groups() for name,_ in group]
    assert stages[-1]=='chapter5_core_models'
    assert stages.index('chapter5_smoke__27')<stages.index('chapter5_core__0')
    assert not any('outer' in s or 'evaluate' in s for s in stages)


def prediction(**changes):
    return dict(sample_id='v1',task='regression',field=0,target_valid=True,target=1.,
        group_id='subject1',no_observation=False,sample_weight=1.,prediction=1.5,probability=None)|changes


def test_replay_checks_all_rows_targets_and_probabilities():
    original=[prediction()]
    assert compare_predictions(original,[prediction(prediction=1.500001)])['passed']
    for modified in ([prediction(prediction=1.6)],[prediction(target=2.)],[],original*2,
                     [prediction(probability=[.5,.5])],[prediction(no_observation=True)]):
        with pytest.raises(ValueError):compare_predictions(original,modified)
    with pytest.raises(ValueError):compare_predictions([prediction(probability=[.1,.9])],[prediction(probability=[.2,.8])])


def test_outer_gate_is_checked_before_loading_any_native_data(tmp_path):
    with pytest.raises(ValueError,match='core-model freeze'):
        evaluate_main_unit(main_root=tmp_path,index=0,output_root=tmp_path/'output',role='held_out',sample_limit=0)
    assert not (tmp_path/'output').exists()


def test_outer_requires_committed_receipt_and_unchanged_core_files(tmp_path,monkeypatch):
    from chronaris.evaluation.application_tasks import chapter5_native as module
    monkeypatch.setattr(module,'v4_workflow_source_sha256',lambda:'source')
    write(tmp_path/'main_models.json',{'main':True})
    model=tmp_path/'model';model.write_bytes(b'frozen')
    gate=dict(status='completed',core_models_frozen=True,source_code_sha256='source',
        main_models_sha256=sha(tmp_path/'main_models.json'),files={str(model):sha(model)})
    write(tmp_path/'core_models.json',gate);write(tmp_path/'receipt.json',gate)
    write(tmp_path/'pipeline_state.json',{'completed':{}})
    with pytest.raises(ValueError,match='committed'):require_outer_freeze(tmp_path/'core_models.json',tmp_path)
    write(tmp_path/'pipeline_state.json',{'completed':{'chapter5_core_models':{'path':str(tmp_path/'receipt.json'),'sha256':sha(tmp_path/'receipt.json')}}})
    assert require_outer_freeze(tmp_path/'core_models.json',tmp_path)==gate
    model.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):require_outer_freeze(tmp_path/'core_models.json',tmp_path)


def test_asset_binding_rejects_conflicting_and_modified_sources(tmp_path):
    p=tmp_path/'source';p.write_bytes(b'original');files={}
    bind(files,p,sha(p));verify(files)
    with pytest.raises(ValueError,match='conflicting'):bind(files,p,'0'*64)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):verify(files)


def test_legacy_naive_source_exception_retains_all_other_checks(tmp_path,monkeypatch):
    from chronaris.evaluation.application_tasks import v4_naive_baseline as module
    from chronaris.representation import FoldLineage
    fold=FoldLineage('f',('t',),('v',),('h',))
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'frozen')
    payload=dict(format='chronaris.v4_naive_fit.v1',training_status='completed',checkpoint_sha256=sha(checkpoint),
        fold=fold.to_dict(),config={'data_manifest_sha256':'data'},source_code_sha256='old',
        label_used_for_encoder_training=False,optimizer_updates=0,seed=17)
    write(tmp_path/'fit_manifest.json',payload)
    monkeypatch.setattr(module,'v4_workflow_source_sha256',lambda:'new')
    from types import SimpleNamespace
    encoder=SimpleNamespace(normalizer=SimpleNamespace(fit_sample_ids=('t',)),
        projector=SimpleNamespace(fit_sample_ids=('t',),random_state=17),
        config=SimpleNamespace(validity_policy='query_history',random_state=17))
    monkeypatch.setattr(module,'load_naive_time_sync_checkpoint',lambda path:encoder)
    with pytest.raises(ValueError):module.load_v4_naive_encoder(checkpoint,fold=fold,data_manifest_sha256='data')
    assert module.load_v4_naive_encoder(checkpoint,fold=fold,data_manifest_sha256='data',expected_training_source_sha256='old')[0] is encoder
    with pytest.raises(ValueError):module.load_v4_naive_encoder(checkpoint,fold=fold,data_manifest_sha256='other',expected_training_source_sha256='old')
