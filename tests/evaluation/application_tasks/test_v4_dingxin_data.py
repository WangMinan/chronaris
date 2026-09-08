import pytest

from chronaris.evaluation.application_tasks.v4_dingxin_data import dingxin_v4_inner_folds
from tests.evaluation.dingxin.test_simple_downstream_protocol import _synthetic_raw_bundle


def test_dingxin_inner_roles_keep_shared_views_and_entire_target_support_together():
    contexts = _synthetic_raw_bundle().contexts
    folds, embargo = dingxin_v4_inner_folds(contexts)
    for fold in folds:
        roles = (fold.train_sample_ids, fold.validation_sample_ids, embargo[fold.fold_id], fold.held_out_sample_ids)
        assert set().union(*map(set, roles)) == set(contexts.context_id)
        assert sum(map(len, roles)) == len(contexts)
        train, validation, excluded, held = (contexts[contexts.context_id.isin(ids)] for ids in roles)
        assert tuple(frame.vehicle_context_id.nunique() for frame in (train, validation, excluded)) == (18, 6, 6)
        assert train.target_end_exclusive_ms.max() <= validation.input_start_offset_ms.min()
        for frame in (train, validation, excluded, held):
            assert set(contexts[contexts.vehicle_context_id.isin(frame.vehicle_context_id)].context_id) == set(frame.context_id)
    changed = contexts.copy()
    changed.loc[changed.target_start_offset_ms == 115000, "target_end_exclusive_ms"] += 1
    with pytest.raises(ValueError, match="18/6/6"):
        dingxin_v4_inner_folds(changed)


def test_outer_consumers_refit_complete_sortie_without_reusing_encoder_targets():
    from chronaris.evaluation.application_tasks.v4_dingxin_data import (
        V4DingxinData, build_dingxin_outer_consumer_inputs)
    from chronaris.evaluation.application_tasks.application_task_heads import fit_application_task_parameters
    from chronaris.evaluation.application_tasks.v4_grouped_consumers import fit_native_consumers, evaluate_native_consumers
    from chronaris.representation.contracts import RepresentationContractError
    from tests.evaluation.application_tasks.test_v4_grouped_consumers import _output
    import numpy as np

    raw = _synthetic_raw_bundle()
    folds, embargo = dingxin_v4_inner_folds(raw.contexts)
    data = V4DingxinData(None, folds, embargo, raw, None, {}, {}, {}, {}, (), "a" * 64)
    for inner in folds:
        result = build_dingxin_outer_consumer_inputs(data, inner.fold_id)
        outer, targets = result["fold"], result["targets"]
        assert not outer.validation_sample_ids
        assert set(outer.train_sample_ids) == set(inner.train_sample_ids + inner.validation_sample_ids + embargo[inner.fold_id])
        assert result["encoder_fold"] == inner and data.fitted_targets is None
        assert targets.manifest["fit_scope"] == "outer_training_after_encoder_freeze"
        with pytest.raises(RepresentationContractError, match="internal training"):
            fit_application_task_parameters(targets, result["definitions"], inner.train_sample_ids)
        rng = np.random.default_rng(17)
        outputs = {role: _output(ids, rng.normal(size=(len(ids), 64))) for role, ids in
                   (("train", outer.train_sample_ids), ("held_out", outer.held_out_sample_ids))}
        bundle = fit_native_consumers(outputs=outputs, targets=targets, definitions=result["definitions"], context=result["context"])
        assert all(row["selected_parameter"] == 1 for row in bundle["fit_rows"])
        evaluated = evaluate_native_consumers(bundle, output=outputs["held_out"], targets=targets)
        assert evaluated["independent_unit"] == "sortie_descriptive_only"
        assert evaluated["sample_count"] == len(outer.held_out_sample_ids)


def test_vehicle_content_audit_catches_cross_sortie_duplicates_and_blocks_formal_evaluation(tmp_path,monkeypatch):
    import pandas as pd
    from types import SimpleNamespace
    from chronaris.evaluation.application_tasks.v4_dingxin_data import audit_dingxin_vehicle_reuse
    from chronaris.evaluation.application_tasks import v4_native_frozen_evaluation as evaluation
    from chronaris.representation import FoldLineage,collate_observation_samples
    from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
    from tests.representation.test_contracts import _sample
    contexts=pd.DataFrame([dict(context_id=s,vehicle_context_id='vehicle_'+s,sortie_id='held' if s=='c' else 'train',target_start_offset_ms=i*35000) for i,s in enumerate(('a','b','c'))])
    fold=FoldLineage('fold',('a',),('b',),('c',))
    def load(ids):
        result=collate_observation_samples([_sample(s,shift=1. if s=='b' else 0.) for s in ids])
        if 'c' in ids:result.physiology_values[ids.index('c')]+=9.
        return result
    data=SimpleNamespace(raw_targets=SimpleNamespace(contexts=contexts),index=SimpleNamespace(load_batch=load),folds=(fold,),embargo={'fold':()},data_manifest_sha256='a'*64)
    audit=audit_dingxin_vehicle_reuse(data)
    assert audit['declared_vehicle_contexts']==3 and audit['unique_vehicle_contents']==2
    assert audit['folds'][0]['inner_training_held_out_shared_contents']==1
    assert not audit['outer_roles_disjoint']
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'not loaded before data isolation')
    monkeypatch.setattr(evaluation,'load_development_inputs',lambda *args,**kwargs:(None,None,fold,None,'a'*64,None,None,data))
    with pytest.raises(ValueError,match='share identical vehicle'):
        evaluation.run_native_frozen_evaluation(domain='dingxin',fold_index=0,checkpoint=checkpoint,checkpoint_sha256=sha256_file(checkpoint),route='self_supervised',output_root=tmp_path/'formal',device='cpu',method='naive_time_sync')
    assert (tmp_path/'formal/vehicle_content_audit.json').exists()
