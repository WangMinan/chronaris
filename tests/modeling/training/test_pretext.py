from __future__ import annotations

import torch
import pytest

from chronaris.modeling.training import (
    CommonPretextHeadBundle,
    CommonPretextWeights,
    chronaris_auxiliary_weight_schedule,
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
    assert chronaris_auxiliary_weight_schedule(1).physical_consistency == 0.0
    assert chronaris_auxiliary_weight_schedule(10).causal_direction == 0.0
    epoch_11 = chronaris_auxiliary_weight_schedule(11)
    assert epoch_11.continuous_alignment == pytest.approx(0.02)
    assert epoch_11.physical_consistency == pytest.approx(0.01)
    epoch_20 = chronaris_auxiliary_weight_schedule(20)
    assert epoch_20.continuous_alignment == 0.2
    assert epoch_20.physical_consistency == 0.1
    assert epoch_20.causal_direction == 0.1
    assert chronaris_auxiliary_weight_schedule(25) == epoch_20
