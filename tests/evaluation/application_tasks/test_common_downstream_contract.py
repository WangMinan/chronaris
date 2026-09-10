from dataclasses import replace
import json

import joblib
import pytest
import torch

from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskTargets
from chronaris.evaluation.application_tasks.common_downstream_contract import (
    build_common_contract, run_common_downstream, validate_representation_declaration)
from chronaris.evaluation.application_tasks.v4_public_data import PUBLIC_TASKS
from chronaris.representation import FoldLineage, collate_observation_samples
from chronaris.representation.window_features import WindowFeatureBatch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
from tests.evaluation.application_tasks.test_v4_grouped_consumers import _public_inputs
from tests.representation.test_contracts import _sample


def _case(tmp_path):
    outputs, _, _, context = _public_inputs()
    observations = {r: collate_observation_samples([_sample(s) for s in o.sample_ids]) for r, o in outputs.items()}
    outputs = {r: replace(o, timestamps_s=observations[r].query_timestamps_s,
        source_sample_hashes=observations[r].source_sample_hashes) for r, o in outputs.items()}
    fold = FoldLineage('test', outputs['train'].sample_ids, outputs['validation'].sample_ids, (), development_only=True)
    ids = fold.train_sample_ids + fold.validation_sample_ids
    values = {'workload_classification': torch.arange(len(ids)) % 2, 'workload_regression': torch.arange(float(len(ids)))}
    targets = ApplicationTaskTargets(ids, values, {k: torch.ones_like(v, dtype=torch.bool) for k, v in values.items()},
        {'domain': 'clare', 'source_role': 'fixed_development_subjects'})
    checkpoint = tmp_path/'encoder.pt'
    torch.save({'label_used_for_encoder_training': False, 'fold': fold.to_dict(),
        'normalizer': {'fit_sample_ids': list(fold.train_sample_ids)}}, checkpoint)
    outputs = {r: replace(o, checkpoint_sha256=sha256_file(checkpoint)) for r, o in outputs.items()}
    kwargs = dict(fold=fold, observations=observations, targets=targets, definitions=PUBLIC_TASKS['clare'],
        context=context, data_manifest_sha256='d'*64)
    contract = build_common_contract(domain='clare', **kwargs)
    declaration = dict(method='chronaris', checkpoint_path=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
        route='self_supervised', target_supervision='none', training_tasks=[], external_pretraining={'source': 'none'},
        kind='causal_sequence', feature_origin='hidden_state', extraction_location='encoder.hidden', feature_dim=64,
        encoder_frozen=True, task_heads_removed=True, encoder_fit_sample_ids=list(fold.train_sample_ids),
        preprocessing_fit_sample_ids=list(fold.train_sample_ids), evidence_files={str(checkpoint): sha256_file(checkpoint)})
    return kwargs, contract, outputs, declaration


def test_shared_contract_fits_independent_models_and_window_only_features(tmp_path):
    kwargs, contract, outputs, declaration = _case(tmp_path)
    # Round-trip the actual persisted contract, not only the in-memory builder.
    contract = json.loads(json.dumps(contract))
    root = tmp_path/'first'
    first = run_common_downstream(**kwargs, contract=contract, outputs=outputs, declaration=declaration, output_root=root)
    assert first == run_common_downstream(**kwargs, contract=contract, outputs=outputs, declaration=declaration, output_root=root)
    second_outputs = {r: replace(o, method_name='other', sequence_embedding=o.sequence_embedding*2,
        pooled_embedding=o.pooled_embedding*2) for r, o in outputs.items()}
    second = run_common_downstream(**kwargs, contract=contract, outputs=second_outputs,
        declaration=declaration | {'method': 'other'}, output_root=tmp_path/'second')
    assert first['comparison_key'] == second['comparison_key']
    bundles = [joblib.load(x['consumers']['components']['linear']['model_path'])['consumer'] for x in (first, second)]
    key = ('workload_regression', 0)
    assert not torch.equal(torch.from_numpy(bundles[0]['models'][key][0].mean_), torch.from_numpy(bundles[1]['models'][key][0].mean_))
    for b, o in zip(bundles, (outputs, second_outputs)):
        torch.testing.assert_close(torch.from_numpy(b['models'][key][0].mean_).float(), o['train'].pooled_embedding.mean(0))
    windows = {r: WindowFeatureBatch(o.sample_ids, kwargs['observations'][r].context_durations_s,
        o.pooled_embedding, o.valid_mask.any(1), o.method_name, o.fold_id, o.checkpoint_sha256, o.source_sample_hashes)
        for r, o in outputs.items()}
    result = run_common_downstream(**kwargs, contract=contract, outputs=windows,
        declaration=declaration | {'kind': 'window_end'}, output_root=tmp_path/'window')
    a = json.loads(open(first['consumers']['components']['linear']['result_path']).read())
    b = json.loads(open(result['consumers']['components']['linear']['result_path']).read())
    assert a['evaluations'] == b['evaluations']
    assert all(o.sequence_embedding is None for o in windows.values())
    with pytest.raises(ValueError, match='scalar linear'):
        run_common_downstream(**kwargs, contract=contract, outputs=windows,
            declaration=declaration | {'kind': 'window_end'}, output_root=tmp_path/'bad_sequence', families=('minirocket',))


@pytest.mark.parametrize('mutation', ['target', 'observation', 'order', 'cutoff', 'source', 'fit', 'prediction', 'supervision', 'checkpoint', 'checkpoint_pca', 'checkpoint_data', 'policy'])
def test_contract_rejects_changes_before_fitting(tmp_path, mutation):
    kwargs, contract, outputs, declaration = _case(tmp_path)
    if mutation == 'target':
        kwargs['targets'].values['workload_regression'][0] += 1
    elif mutation == 'observation':
        kwargs['observations']['train'].physiology_values[0, 0, 0] += 1
    elif mutation == 'order':
        outputs['validation'] = replace(outputs['validation'], sample_ids=tuple(reversed(outputs['validation'].sample_ids)))
    elif mutation == 'cutoff':
        outputs['validation'] = replace(outputs['validation'], timestamps_s=outputs['validation'].timestamps_s+1)
    elif mutation == 'source':
        outputs['validation'] = replace(outputs['validation'], source_sample_hashes=('f'*64,)*6)
    elif mutation == 'fit':
        declaration['preprocessing_fit_sample_ids'].append(outputs['validation'].sample_ids[0])
    elif mutation == 'prediction':
        declaration['feature_origin'] = 'final_predictions'
    elif mutation == 'supervision':
        declaration.update(route='task_guided', target_supervision='summary_labels', training_tasks=list(kwargs['targets'].values))
    elif mutation == 'checkpoint':
        open(declaration['checkpoint_path'], 'ab').write(b'changed')
    elif mutation in {'checkpoint_pca', 'checkpoint_data'}:
        payload = torch.load(declaration['checkpoint_path'], weights_only=True)
        if mutation == 'checkpoint_pca':
            payload['pca_state'] = {'fit_sample_ids': list(outputs['validation'].sample_ids)}
        else:
            payload['config'] = {'data_manifest_sha256': 'f'*64}
        torch.save(payload, declaration['checkpoint_path'])
        digest = sha256_file(declaration['checkpoint_path'])
        outputs = {r: replace(o, checkpoint_sha256=digest) for r, o in outputs.items()}
        declaration.update(checkpoint_sha256=digest, evidence_files={declaration['checkpoint_path']: digest})
    else:
        contract['policy']['public_parameters']['classification_c'] = [100.]
    with pytest.raises(ValueError):
        run_common_downstream(**kwargs, contract=contract, outputs=outputs, declaration=declaration, output_root=tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


def test_future_trajectory_supervision_has_a_separate_stratum(tmp_path):
    kwargs, contract, outputs, declaration = _case(tmp_path)
    torch.save({'label_used_for_encoder_training': True, 'target_supervision': 'future_trajectory'}, declaration['checkpoint_path'])
    digest = sha256_file(declaration['checkpoint_path'])
    outputs = {r: replace(o, checkpoint_sha256=digest) for r, o in outputs.items()}
    declaration.update(route='task_guided', target_supervision='future_trajectory', checkpoint_sha256=digest,
        evidence_files={declaration['checkpoint_path']: digest})
    assert validate_representation_declaration(contract, outputs, declaration, families=('linear',)) == 'task_guided:future_trajectory'
    with pytest.raises(ValueError, match='checkpoint|mixed'):
        validate_representation_declaration(contract, outputs, declaration | {'target_supervision': 'summary_labels',
            'training_tasks': list(kwargs['targets'].values)}, families=('linear',))


def test_different_summary_training_tasks_cannot_share_a_comparison_key(tmp_path):
    kwargs, contract, outputs, declaration = _case(tmp_path)
    keys = []
    for names in (['workload_classification'], ['workload_classification', 'workload_regression']):
        checkpoint = tmp_path/f'guided{len(names)}.pt'
        torch.save({'label_used_for_encoder_training': True, 'task_definitions': [{'name': n} for n in names]}, checkpoint)
        digest = sha256_file(checkpoint)
        result = run_common_downstream(**kwargs, contract=contract,
            outputs={r: replace(o, checkpoint_sha256=digest) for r, o in outputs.items()},
            declaration=declaration | {'route': 'task_guided', 'target_supervision': 'summary_labels', 'training_tasks': names,
                'checkpoint_path': str(checkpoint), 'checkpoint_sha256': digest, 'evidence_files': {str(checkpoint): digest}},
            output_root=tmp_path/f'result{len(names)}')
        keys.append(result['comparison_key'])
    assert keys[0] != keys[1]
