import os
import shutil
import json
from pathlib import Path
import time

import pytest
import torch

from chronaris.modeling.fusion_encoders.recent_models import (
    Chronos2WindowEncoder, TimeCMAWindowEncoder, historical_channels, history_prompts, verify_assets)
from chronaris.representation import collate_observation_samples
from tests.representation.test_contracts import _sample
from chronaris.modeling.training.rng import canonical_training_state_sha256
from chronaris.evaluation.application_tasks.recent_model_training import task_covered_positions
from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskTargets
from chronaris.evaluation.application_tasks.v4_public_data import PUBLIC_TASKS


def test_short_batches_cover_disjoint_tasks_without_using_validation_or_label_values():
    ids = tuple(f's{i}' for i in range(12))
    values = {'workload_classification': torch.zeros(12, dtype=torch.long), 'workload_regression': torch.arange(12.)}
    masks = {'workload_classification': torch.tensor([True]*4+[False]*4+[True]*4),
        'workload_regression': torch.tensor([False]*4+[True]*8)}
    targets = ApplicationTaskTargets(ids, values, masks, {})
    before = []
    for update in range(2):
        chosen = task_covered_positions(targets, PUBLIC_TASKS['clare'], ids[:8], update, 4)
        assert len(chosen)==len(set(chosen))==4 and all(0 <= i < 8 for i in chosen)
        assert all(mask[chosen].any() for mask in masks.values())
        before.append(chosen)
    values['workload_classification'].fill_(1)
    values['workload_regression'].mul_(-3)
    assert before == [task_covered_positions(targets, PUBLIC_TASKS['clare'], ids[:8], u, 4) for u in range(2)]
    with pytest.raises(ValueError, match='cannot cover'):
        task_covered_positions(targets, PUBLIC_TASKS['clare'], ids[:8], 0, 1)
    for mask in masks.values():
        mask.fill_(True)
    assert task_covered_positions(targets, PUBLIC_TASKS['clare'], ids[:8], 1, 4)==[4,5,6,7]


def test_checkpoint_hash_supports_bfloat16_and_scalar_optimizer_state():
    a = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16).T
    assert canonical_training_state_sha256(a) == canonical_training_state_sha256(a.contiguous())
    assert canonical_training_state_sha256(a) != canonical_training_state_sha256(a+1)
    assert len(canonical_training_state_sha256(torch.tensor(1.))) == 64


def test_assets_reject_modified_source_before_model_loading(tmp_path):
    source = tmp_path/'official.py'
    source.write_text('modified')
    manifest = tmp_path/'assets.json'
    manifest.write_text(json.dumps({'files': {str(source): '0'*64}}))
    with pytest.raises(ValueError, match='source/weight changed'):
        verify_assets(manifest)


def test_history_preparation_excludes_future_and_masked_values():
    raw = collate_observation_samples([_sample('sample')])
    values, mask = historical_channels(raw)
    names = tuple(f'channel_{i}' for i in range(values.shape[-1]))
    first = history_prompts(values, mask, raw.query_timestamps_s, names)
    changed = values.clone().masked_fill(~mask, 1e20)
    assert history_prompts(changed, mask, raw.query_timestamps_s, names) == first
    for stream in ('physiology', 'vehicle'):
        times = getattr(raw, stream+'_timestamps_s')
        x = getattr(raw, stream+'_values')
        x[times > raw.query_timestamps_s[0, 20]] += 1000
    altered, _ = historical_channels(raw)
    assert torch.equal(values[:, :21], altered[:, :21])
    assert 'sample' not in ''.join(first)
    assert 'seconds in the observed window' in first[0]


@pytest.mark.skipif(not os.environ.get('CHRONARIS_RECENT_ASSETS') or not torch.cuda.is_available(),
    reason='explicit pinned official assets and CUDA required')
def test_official_hidden_extraction_and_window_isolation():
    assets = verify_assets(os.environ['CHRONARIS_RECENT_ASSETS'])
    torch.manual_seed(17)
    values = torch.randn(2, 96, 3)
    mask = torch.ones_like(values, dtype=torch.bool)
    prompts = torch.randn(2, 768, 3)
    encoder = TimeCMAWindowEncoder(assets, 3).eval()
    captured = []
    hook = encoder.core.decoder.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
    with torch.inference_mode():
        encoder.core(values.cuda(), torch.zeros(2,96,1,device='cuda'), prompts.cuda())
        expected = captured.pop().flatten(1).cpu()
        actual = encoder(values, mask, prompts).cpu()
    hook.remove()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    del encoder
    encoder = Chronos2WindowEncoder(assets, 3).eval()
    with torch.inference_mode():
        original = encoder(values, mask).cpu()
        changed = values.clone()
        changed[1] *= 1000
        assert torch.equal(original[0], encoder(changed, mask).cpu()[0])
        embeds, _ = encoder.pipeline.embed(values.transpose(1,2), batch_size=3, context_length=96)
        expected = torch.stack([e[:,-2,:] for e in embeds]).flatten(1)
        torch.testing.assert_close(original, expected, atol=0, rtol=0)
        empty = encoder(torch.zeros_like(values), torch.zeros_like(mask))
        assert torch.equal(empty, torch.zeros_like(empty))


@pytest.mark.skipif(not os.environ.get('CHRONARIS_SENSOR_ASSETS') or not os.environ.get('CHRONARIS_SENSOR_RUN')
    or not torch.cuda.is_available(), reason='explicit real SensorLLM run, pinned weights and CUDA required')
def test_sensorllm_cold_load_restores_real_features_and_zero_observation(record_property, tmp_path):
    import numpy as np
    from chronaris.evaluation.application_tasks.common_downstream_smoke import contract_development_inputs
    from chronaris.modeling.fusion_encoders.sensorllm_adapter import SensorLLMWindowEncoder
    from chronaris.modeling.training.rng import isolated_training_rng
    from chronaris.representation import TrainOnlyRobustNormalizer
    assets = verify_assets(os.environ['CHRONARIS_SENSOR_ASSETS'])
    unit = Path(os.environ['CHRONARIS_SENSOR_RUN'])/'summary_adapted'
    saved = torch.load(unit/'encoder.pt', weights_only=True)
    _, _, _, digest, _, _, _, raw = contract_development_inputs('clare',
        data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
        registry_path='docs/requirements/thesis-v4-public-subjects.json')
    assert digest == saved['config']['data_manifest_sha256']
    normalizer = TrainOnlyRobustNormalizer.from_manifest(saved['normalizer'])
    values, mask = historical_channels(normalizer.transform(raw['validation']))
    fraction = torch.cuda.get_per_process_memory_fraction()
    encoder = None
    try:
        torch.cuda.set_per_process_memory_fraction(.95)
        with isolated_training_rng(17):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            encoder = SensorLLMWindowEncoder(assets, values.shape[-1])
            torch.cuda.synchronize()
            record_property('loader_seconds', time.perf_counter()-started)
            record_property('loader_peak_cuda_bytes', torch.cuda.max_memory_allocated())
            encoder.channel_names = saved['channel_names']
            encoder.load_checkpoint_state(saved['encoder'])
            encoder.eval().requires_grad_(False)
            with torch.inference_mode(), np.load(unit/'validation_features.npz') as exported:
                actual = encoder(values[:1], mask[:1]).cpu()
                assert torch.equal(actual, torch.from_numpy(exported['pooled_embedding'][:1]))
                assert torch.equal(actual, encoder(values[:1].masked_fill(~mask[:1], 1e6), mask[:1]).cpu())
                empty = encoder(values[:1], torch.zeros_like(mask[:1]))
                assert torch.equal(empty, torch.zeros_like(empty))
            encoder.configure_training()
            train_values, train_mask = historical_channels(normalizer.transform(raw['train']))
            alignment = dict(values=train_values[:2], mask=train_mask[:2], names=saved['channel_names'],
                train_ids=raw['train'].sample_ids[:2], binding={'scope':'test_only_no_research_evidence'})
            encoder.align_history(**alignment, root=tmp_path/'alignment', stop_after=1)
            (tmp_path/'replay').mkdir()
            shutil.copyfile(tmp_path/'alignment/alignment.pt', tmp_path/'replay/alignment.pt')
            primary = encoder.align_history(**alignment, root=tmp_path/'alignment', stop_after=2)
            expected = canonical_training_state_sha256(encoder.checkpoint_state())
            replay = encoder.align_history(**alignment, root=tmp_path/'replay', stop_after=2)
            assert primary['updates'] == replay['updates']
            assert expected == canonical_training_state_sha256(encoder.checkpoint_state())
    finally:
        del encoder
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(fraction)
