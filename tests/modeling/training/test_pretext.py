from __future__ import annotations

import torch
import pytest
from types import SimpleNamespace

from chronaris.modeling.training import (
    CommonPretextHeadBundle,
    CommonPretextWeights,
    ExplicitTimeShiftHead,
    chronaris_auxiliary_weight_schedule,
    event_pair_contrastive_loss_term,
    explicit_time_shift_loss_term,
)
from chronaris.representation import CommonPretextTargets


def _targets(*, reconstruction_active: bool = True):
    target = torch.randn(2, 96, 5)
    reconstruction_mask = torch.zeros_like(target, dtype=torch.bool)
    if reconstruction_active:
        reconstruction_mask[:, 10:20] = True
    next_mask = torch.ones_like(target, dtype=torch.bool)
    next_mask[:, -1] = False
    return CommonPretextTargets(
        reconstruction_target=target,
        reconstruction_mask=reconstruction_mask,
        next_query_target=torch.roll(target, shifts=-1, dims=1),
        next_query_mask=next_mask,
        target_feature_count=5,
        augmentation_ids=("a" * 64, "b" * 64),
    )


def test_common_pretext_heads_produce_finite_shared_losses_and_gradients() -> None:
    torch.manual_seed(17)
    head = CommonPretextHeadBundle(
        representation_dim=64,
        target_feature_count=5,
    )
    positive = torch.randn(2, 96, 64, requires_grad=True)
    negative = torch.randn(2, 96, 64, requires_grad=True)
    output = head(positive, negative, _targets())

    assert [term.term_name for term in output.terms] == [
        "masked_reconstruction",
        "short_horizon_prediction",
        "lag_discrimination",
    ]
    assert all(term.status == "active" and term.count > 0 for term in output.terms)
    assert torch.isfinite(output.total_loss)
    output.total_loss.backward()
    assert positive.grad is not None and torch.isfinite(positive.grad).all()
    assert negative.grad is not None and torch.isfinite(negative.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )


def test_pretext_unavailable_term_is_not_fabricated_as_zero() -> None:
    head = CommonPretextHeadBundle(
        representation_dim=64,
        target_feature_count=5,
    )
    positive = torch.randn(2, 96, 64)
    negative = torch.randn(2, 96, 64)
    output = head(
        positive,
        negative,
        _targets(reconstruction_active=False),
        weights=CommonPretextWeights(),
    )
    reconstruction = output.terms[0]

    assert reconstruction.status == "unavailable"
    assert reconstruction.count == 0
    assert reconstruction.raw_loss is None
    assert reconstruction.weighted_loss is None
    assert torch.isfinite(output.total_loss)


def test_chronaris_auxiliary_schedule_matches_frozen_ramp() -> None:
    epoch_1 = chronaris_auxiliary_weight_schedule(1)
    assert epoch_1.continuous_alignment == pytest.approx(0.04)
    assert epoch_1.physical_consistency == pytest.approx(0.02)
    assert epoch_1.causal_direction == pytest.approx(0.02)
    epoch_5 = chronaris_auxiliary_weight_schedule(5)
    assert epoch_5.continuous_alignment == 0.2
    assert epoch_5.physical_consistency == 0.1
    assert epoch_5.causal_direction == 0.1
    assert chronaris_auxiliary_weight_schedule(25) == epoch_5


def test_explicit_shift_weight_changes_head_gradient() -> None:
    torch.manual_seed(29)
    sequence = torch.randn(5, 6, 64)
    valid = torch.ones(5, 6, dtype=torch.bool)
    labels = torch.arange(5)
    zero_head = ExplicitTimeShiftHead(64)
    active_head = ExplicitTimeShiftHead(64)
    active_head.load_state_dict(zero_head.state_dict())

    zero = explicit_time_shift_loss_term(zero_head(sequence, valid), labels, weight=0.0)
    active = explicit_time_shift_loss_term(
        active_head(sequence, valid), labels, weight=0.1
    )
    zero.weighted_loss.backward()
    active.weighted_loss.backward()

    assert torch.isfinite(active.raw_loss)
    assert sum(parameter.grad.abs().sum() for parameter in zero_head.parameters()) == 0
    assert sum(parameter.grad.abs().sum() for parameter in active_head.parameters()) > 0


def test_event_pairing_uses_only_cross_group_negatives_and_reports_unavailable() -> None:
    query_context = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        ]
    )
    semantic = SimpleNamespace(
        query_names=(
            "flight_event",
            "physiology_response",
            "human_aircraft_coordination",
        ),
        query_context_states=query_context,
    )

    active, metrics = event_pair_contrastive_loss_term(
        semantic,
        ("group_a", "group_b"),
        temperature=0.1,
        weight=0.1,
    )
    unavailable, _ = event_pair_contrastive_loss_term(
        semantic,
        ("group_a", "group_a"),
        temperature=0.1,
        weight=0.1,
    )

    assert active.status == "active" and active.count == 2
    assert metrics["positive_similarity"] > metrics["negative_similarity"]
    assert metrics["recall_at_1"] == 1.0
    assert unavailable.status == "unavailable"
    assert unavailable.reason == "no_cross_group_negative"
