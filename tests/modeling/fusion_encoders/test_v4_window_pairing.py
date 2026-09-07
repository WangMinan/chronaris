from dataclasses import replace
from types import SimpleNamespace
import math

import pytest
import torch

from chronaris.models.fusion.window_pairing import IndependentWindowPairing, WindowPairingOutput
from chronaris.modeling.training.pretext import independent_window_pair_loss_term
from chronaris.modeling.training.pretraining_encoders import EncoderCandidateConfig, build_trainable_fusion_encoder
from chronaris.representation import collate_observation_samples
from tests.modeling.fusion_encoders.test_safe_lag_fusion import _sample


def test_native_time_bins_exclude_future_and_empty_windows_clear_projection_bias():
    model = IndependentWindowPairing(1)
    with torch.no_grad():
        model.physiology_projection.weight.zero_()
        model.physiology_projection.weight[:4] = torch.eye(4)
        model.physiology_projection.bias.fill_(3.)
    states = torch.tensor([1., 3., 5., 7., 9., 11., 999.]).reshape(1, 7, 1).repeat(2, 1, 1).requires_grad_()
    mask = torch.ones(2, 7, dtype=torch.bool); mask[1] = False
    stream = SimpleNamespace(updated_hidden_states=states, mask=mask,
        offsets_s=torch.tensor([0., 7.49, 7.5, 15., 22.5, 29.9, 30.1], dtype=torch.float64).repeat(2, 1))
    output = model(SimpleNamespace(physiology=stream, vehicle=stream), torch.tensor([30., 30.]))
    torch.testing.assert_close(output.physiology[0, :4], torch.tensor([5., 8., 10., 13.]))
    assert not output.physiology[1].any() and not output.vehicle[1].any()
    output.physiology.sum().backward()
    assert not states.grad[:, -1].any() and not states.grad[1].any()


def test_pairing_ends_remain_independent_even_when_event_fusion_is_enabled():
    torch.manual_seed(17)
    samples = [_sample(f"sample_{i}", future_scale=i + 1.) for i in range(3)]
    batch = collate_observation_samples(samples)
    encoder = build_trainable_fusion_encoder("chronaris", physiology_feature_names=samples[0].schema.physiology_feature_names,
        vehicle_feature_names=samples[0].schema.vehicle_feature_names, candidate_config=EncoderCandidateConfig(hidden_dim=8),
        chronaris_fusion_kind="safe_lag", chronaris_semantic_event_enabled=True, chronaris_learnable_semantic_queries=True,
        chronaris_independent_pairing_enabled=True).eval()
    original = encoder(batch).auxiliary["independent_pairing"]
    for stream, other in (("vehicle", "physiology"), ("physiology", "vehicle")):
        changed = replace(batch, **{stream + "_" + name: getattr(batch, stream + "_" + name).flip(0)
            for name in ("values", "timestamps_s", "point_mask", "feature_mask", "observation_age_s")})
        permuted = encoder(changed).auxiliary["independent_pairing"]
        torch.testing.assert_close(getattr(original, other), getattr(permuted, other), atol=1e-6, rtol=0)
        torch.testing.assert_close(getattr(original, stream).flip(0), getattr(permuted, stream), atol=1e-6, rtol=0)


def test_independent_pair_loss_uses_all_actual_batch_negatives_and_marks_unavailable():
    physiology = torch.eye(4, 32).requires_grad_()
    vehicle = torch.eye(4, 32).requires_grad_()
    valid = torch.ones(4, dtype=torch.bool)
    pairing = WindowPairingOutput(physiology, vehicle, valid, valid)
    term, metrics = independent_window_pair_loss_term(pairing, ("a", "b", "b", "c"), weight=.1)
    expected = sum(math.log1p(n * math.exp(-10.)) for n in (3, 2, 2, 3)) / 4
    assert float(term.raw_loss.detach()) == pytest.approx(expected, abs=1e-7)
    assert term.count == metrics["valid_pair_count"] == 4 and metrics["negative_pair_count"] == 10
    assert metrics["actual_batch_size"] == 4 and metrics["recall_at_1"] == 1.
    term.weighted_loss.backward()
    assert physiology.grad.abs().sum() > 0 and vehicle.grad.abs().sum() > 0
    for groups, data in ((("a",) * 4, pairing), (("a", "b", "b", "c"),
        replace(pairing, physiology_valid=torch.tensor([True, False, False, False])))):
        unavailable, metrics = independent_window_pair_loss_term(data, groups, weight=.1)
        assert unavailable.status == "unavailable" and unavailable.raw_loss is None and unavailable.weighted_loss is None
        assert metrics["valid_pair_count"] == metrics["negative_pair_count"] == 0
