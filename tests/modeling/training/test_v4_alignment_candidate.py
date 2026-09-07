import pytest
import torch

from chronaris.modeling.training.candidate_mechanisms import build_candidate_mechanism_step, evaluate_candidate_mechanisms
from chronaris.modeling.training.pretraining_encoders import build_trainable_fusion_encoder, EncoderCandidateConfig
from chronaris.modeling.training.pretext import chronaris_auxiliary_weight_schedule
from chronaris.models.alignment.calibrated_physics import fit_physics_calibration
from chronaris.representation import AugmentationPolicy, TrainOnlyRobustNormalizer, collate_observation_samples, select_observation_batch
from tests.modeling.training.test_candidate_screen import _sample


def test_no_same_time_alignment_keeps_observation_anchor_and_other_mechanisms():
    torch.set_num_threads(1)
    samples = [_sample(f"sample_{i}", i) for i in range(3)]
    batch = collate_observation_samples(samples)
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=batch.sample_ids[:2], held_out_sample_ids=batch.sample_ids[2:])
    calibration = fit_physics_calibration(normalizer, lambda ids: select_observation_batch(batch, ids),
        train_sample_ids=batch.sample_ids[:2], vehicle_feature_names=samples[0].schema.vehicle_feature_names, relations=())
    encoder = build_trainable_fusion_encoder("chronaris", physiology_feature_names=samples[0].schema.physiology_feature_names,
        vehicle_feature_names=samples[0].schema.vehicle_feature_names, candidate_config=EncoderCandidateConfig(hidden_dim=8),
        chronaris_fusion_kind="safe_lag", chronaris_physics_calibration=calibration)
    encoded = encoder(normalizer.transform(batch), compute_chronaris_diagnostics=True)
    args = dict(encoder=encoder, shift_head=None, positive=encoded, negative=encoded, augmented=None,
        group_ids=batch.group_ids, epoch=5, device="cpu", mechanism_enabled=True, lag_aware_weight=0.,
        explicit_shift_weight=0., event_pair_weight=0., optimizer_updates=200)
    reference = build_candidate_mechanism_step(**args)
    disabled = build_candidate_mechanism_step(**args, continuous_alignment_weight=0.)
    before = {row["term_name"]: row for row in reference.rows}
    after = {row["term_name"]: row for row in disabled.rows}
    assert after["chronaris_continuous_alignment"]["weight"] == 0.
    for name in ("observation_anchor", "chronaris_physical_consistency", "chronaris_causal_direction"):
        assert after[name] == before[name]
    assert after["observation_anchor"]["weight"] == 1. and after["observation_anchor"]["count"] > 0
    assert float((reference.additional_loss - disabled.additional_loss).detach()) == pytest.approx(
        before["chronaris_continuous_alignment"]["weighted_loss"], abs=1e-6)
    validation = evaluate_candidate_mechanisms(encoder=encoder, shift_head=None, batch=batch, batch_provider=None,
        sample_ids=batch.sample_ids[2:], batch_size=1, normalizer=normalizer, policy=AugmentationPolicy(), seed=17,
        device="cpu", mechanism_enabled=True, lag_aware_weight=0., explicit_shift_weight=0., event_pair_weight=0.,
        continuous_alignment_weight=0.)
    terms = {row["term_name"]: row for row in validation["terms"]}
    assert terms["chronaris_continuous_alignment"]["weight"] == 0.
    assert terms["observation_anchor"]["weight"] == 1. and terms["observation_anchor"]["count"] > 0
    for update in (0, 50, 125, 200):
        original = chronaris_auxiliary_weight_schedule(1, optimizer_updates=update)
        changed = chronaris_auxiliary_weight_schedule(1, optimizer_updates=update, continuous_alignment_weight=0.)
        assert changed.continuous_alignment == 0.
        assert original.physical_consistency == changed.physical_consistency and original.causal_direction == changed.causal_direction
