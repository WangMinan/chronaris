import time

import pytest
import torch

from chronaris.modeling.training.candidate_mechanisms import evaluate_candidate_mechanisms
from chronaris.modeling.training.candidate_validation import _evaluate_public_losses
from chronaris.modeling.training.pretext import CommonPretextHeadBundle, ExplicitTimeShiftHead
from chronaris.modeling.training.pretraining_encoders import EncoderCandidateConfig, build_trainable_fusion_encoder
from chronaris.representation import AugmentationPolicy, TrainOnlyRobustNormalizer, collate_observation_samples
from tests.modeling.training.test_candidate_screen import _sample


def test_validation_reuses_forward_without_changing_public_or_mechanism_values():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    batch = collate_observation_samples([_sample(f"sample_{i}", i) for i in range(4)])
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=batch.sample_ids, held_out_sample_ids=())
    encoder = build_trainable_fusion_encoder("chronaris", physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",), vehicle_field_labels=(),
        candidate_config=EncoderCandidateConfig(hidden_dim=8), chronaris_fusion_kind="safe_lag",
        chronaris_semantic_event_enabled=True)
    heads = CommonPretextHeadBundle(representation_dim=64, target_feature_count=2, modality_feature_counts=(1, 1))
    shift = ExplicitTimeShiftHead(64)
    common = dict(encoder=encoder, batch=batch, batch_provider=None, sample_ids=batch.sample_ids,
        batch_size=2, normalizer=normalizer, policy=AugmentationPolicy(), seed=17, device="cpu")
    mechanism = dict(shift_head=shift, mechanism_enabled=True, lag_aware_weight=0., explicit_shift_weight=.1, event_pair_weight=0.)
    calls = []
    hook = encoder.register_forward_hook(lambda *args: calls.append(1))
    started = time.perf_counter()
    original_public = _evaluate_public_losses(**common, heads=heads)
    original_mechanism = evaluate_candidate_mechanisms(**common, **mechanism)
    separate_s, separate_calls = time.perf_counter() - started, len(calls)
    calls.clear()
    started = time.perf_counter()
    combined = evaluate_candidate_mechanisms(**common, **mechanism, public_heads=heads)
    combined_s = time.perf_counter() - started
    assert combined.pop("public_losses") == pytest.approx(original_public, abs=1e-7)
    assert combined == original_mechanism
    assert separate_calls == 10 and len(calls) == 6
    print({"separate_forward_calls": separate_calls, "combined_forward_calls": len(calls),
           "separate_cpu_s": separate_s, "combined_cpu_s": combined_s})
    hook.remove()
