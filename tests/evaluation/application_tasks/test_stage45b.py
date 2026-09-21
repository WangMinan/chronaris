from dataclasses import replace
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch
import json
import os
import subprocess
import sys

import pytest
import torch

from chronaris.evaluation.application_tasks.checkpoint_selection import selection_split, GroupedCheckpointSelector
from chronaris.evaluation.application_tasks.stage45b_diagnostics import isolated_gradients
from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskTargets
from chronaris.evaluation.application_tasks.v4_public_data import PUBLIC_TASKS
from chronaris.representation import FoldLineage, collate_observation_samples, select_observation_batch
from tests.modeling.training.test_candidate_screen import _sample


def short_inputs(*args, **kwargs):
    ids = tuple(f's{i}' for i in range(16))
    raw = collate_observation_samples([_sample(s, float(i % 4)) for i, s in enumerate(ids)])
    fold = FoldLineage('selection_test', ids[:8], ids[8:], (), development_only=True)
    values = {'workload_classification': torch.arange(16) % 2, 'workload_regression': torch.arange(16).float() % 4}
    targets = ApplicationTaskTargets(ids, values, {k: torch.ones_like(v, dtype=torch.bool) for k,v in values.items()}, {'domain':'clare'})
    context = dict(domain='clare', groups={s:str(i//4) for i,s in enumerate(ids)}, regression={}, vehicle_groups={})
    provider = lambda selected: select_observation_batch(raw, selected)
    return provider, _sample("schema", 0.).schema, fold, 'd'*64, targets, PUBLIC_TASKS['clare'], context, {
        r:provider(getattr(fold,r+'_sample_ids')) for r in ('train','validation')}


def test_group_selection_isolation_and_gradient_missingness():
    provider, _, fold, _, targets, definitions, context, _ = short_inputs()
    a, evaluation, support = selection_split(fold, targets, definitions, context, inner_index=0)
    b, _, _ = selection_split(fold, targets, definitions, context, inner_index=1)
    assert a.train_sample_ids == b.validation_sample_ids
    assert evaluation.validation_sample_ids == fold.validation_sample_ids
    assert not set(a.train_sample_ids+a.validation_sample_ids) & set(fold.validation_sample_ids)
    selector = GroupedCheckpointSelector(provider=provider, fold=a, targets=targets, definitions=definitions,
                                         context=context, output_root=Path('/unused'))
    corrupted = replace(targets, values={k:v.clone() for k,v in targets.values.items()})
    corrupted.values['workload_regression'][8:] += 1000
    other = GroupedCheckpointSelector(provider=provider, fold=a, targets=corrupted, definitions=definitions,
                                      context=context, output_root=Path('/unused'))
    assert selector.manifest == other.manifest
    with pytest.raises(ValueError, match='overlap'):
        GroupedCheckpointSelector(provider=provider, fold=a, targets=targets, definitions=definitions,
            context=context | {'groups':{s:'same' for s in targets.sample_ids}}, output_root=Path('/unused'))
    p = torch.nn.Parameter(torch.ones(2))
    result = isolated_gradients({'a':p.sum(), 'opposite':-p.sum(), 'zero':p.sum()*0, 'missing':None}, [p])
    assert result['cosines']['a / opposite'] == pytest.approx(-1)
    assert result['terms']['zero']['norm'] == 0
    assert result['terms']['missing']['norm'] is None


def short_run(root, interrupt=False):
    from chronaris.evaluation.application_tasks import common_downstream_smoke as entry
    from chronaris.modeling.training import candidate_screen
    from chronaris.evaluation.application_tasks import application_finetuning
    from chronaris.evaluation.application_tasks import stage45_recipe
    recipe_builder = stage45_recipe.training_recipe
    original_select = GroupedCheckpointSelector.__call__
    def recipe(*args, **kwargs):
        candidate, pre, guided, arguments, record = recipe_builder(*args, **kwargs)
        return candidate, replace(pre, validation_interval=1), replace(guided, validation_interval=1), arguments, record
    def tied_selection(self, *args):
        return original_select(self, *args) | {'score': 0.}
    original = candidate_screen.atomic_save_candidate
    guided_save = application_finetuning._atomic_save
    def save(path, payload):
        original(path, payload)
        if interrupt == 'pretraining' and Path(path).name == 'last.pt' and payload.get('pending_validation'):
            raise RuntimeError('test interruption before selection')
    def save_guided(path, payload):
        guided_save(path, payload)
        if interrupt == 'guided' and payload.get('pending_validation'):
            raise RuntimeError('test interruption before selection')
    with patch.object(stage45_recipe, 'training_recipe', recipe), patch.object(GroupedCheckpointSelector, '__call__', tied_selection), patch.object(application_finetuning, '_atomic_save', save_guided), patch.object(entry, 'contract_development_inputs', short_inputs), patch.object(entry, 'development_gpu_lock', lambda: nullcontext(True)), patch.object(candidate_screen, 'atomic_save_candidate', save):
        return entry.run_common_contract_smoke(domain='clare', output_root=root, methods=('chronaris',), recipe='stage4_reference',
            cuda_graph_recurrence=True, selection_inner_index=0, private_projection_kind='linear')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA required')
@pytest.mark.parametrize("interruption_stage", ["pretraining", "guided"])
def test_grouped_entry_and_independent_process_resume(tmp_path, interruption_stage):
    # Exercises training, selection, new projection, original-selector controls, exports and consumers.
    reference = short_run(tmp_path/'reference')
    code = 'from tests.evaluation.application_tasks.test_stage45b import short_run; short_run(__import__("sys").argv[1], __import__("sys").argv[2])'
    environment = os.environ | {'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    interrupted = subprocess.run([sys.executable, '-c', code, str(tmp_path/'resumed'), interruption_stage], env=environment, capture_output=True, text=True)
    assert interrupted.returncode != 0 and 'test interruption before selection' in interrupted.stderr
    subprocess.run([sys.executable, '-c', code, str(tmp_path/'resumed'), 'resume'], env=environment, check=True)
    resumed = json.loads((tmp_path/'resumed/clare/summary.json').read_text())
    for route in ('self_supervised','task_guided'):
        records = [next(r for r in x['results'] if r.get('route') == route and 'training' in r) for x in (reference,resumed)]
        states = [torch.load(r['training']['best_checkpoint_path'],weights_only=True,map_location='cpu') for r in records]
        key = 'encoder_state_dict' if route == 'self_supervised' else 'model_state_dict'
        for name, tensor in states[0][key].items():
            assert torch.equal(tensor, states[1][key][name]), name
        assert states[0]['config']['checkpoint_selection']['selection_supervision'] == 'summary_labels'
        assert states[0]['best_update'] == (1 if route == 'self_supervised' else 3)
        assert states[1]['best_update'] == states[0]['best_update']
        last = [torch.load(r['training']['last_checkpoint_path'], weights_only=True, map_location='cpu') for r in records]
        for name, tensor in last[0][key].items():
            assert torch.equal(tensor, last[1][key][name]), name
        from chronaris.evaluation.application_tasks.checkpoint_performance import compare_values
        assert compare_values(last[0]['optimizer_state_dict'], last[1]['optimizer_state_dict'], atol=0., rtol=0.)['bitwise_equal']
        assert records[0]['consumers']['components']['linear']['task_summary'] == records[1]['consumers']['components']['linear']['task_summary']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA required')
def test_linear_projection_cuda_equivalence():
    from chronaris.modeling.training import build_trainable_fusion_encoder, EncoderCandidateConfig
    from chronaris.representation import TrainOnlyRobustNormalizer
    from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
    from chronaris.modeling.training.rng import isolated_training_rng
    from chronaris.evaluation.application_tasks.execution_equivalence import compare_runtime
    provider, schema, fold, _, _, _, _, _ = short_inputs()
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(provider, train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids, batch_size=4)
    batch = move_observation_batch(normalizer.transform(provider(fold.train_sample_ids[:4])), device='cuda')
    traces = []
    for graph in (False, True):
        with isolated_training_rng(17):
            encoder = build_trainable_fusion_encoder('chronaris', physiology_feature_names=schema.physiology_feature_names,
                vehicle_feature_names=schema.vehicle_feature_names, candidate_config=EncoderCandidateConfig('C', hidden_dim=32),
                chronaris_fusion_kind='safe_lag', chronaris_private_projection_kind='linear', chronaris_cuda_graph_recurrence=graph).cuda()
            optimizer = torch.optim.AdamW(encoder.parameters(), lr=.001)
            trace = []
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                out = encoder(batch)
                assert out.sequence_embedding.shape[-1] == 64
                out.sequence_embedding.square().mean().backward()
                gradients = {n:p.grad.detach().cpu().clone() for n,p in encoder.named_parameters() if p.grad is not None}
                optimizer.step()
                trace.append(dict(output=out.sequence_embedding.detach().cpu(), mask=out.modality_available_mask.cpu(),
                    gradients=gradients, state={n:p.detach().cpu().clone() for n,p in encoder.state_dict().items()}))
            traces.append(trace)
    for expected, actual in zip(*traces):
        assert compare_runtime(expected['output'], actual['output'], representation=True)['close']
        assert compare_runtime(expected, actual)['close']
