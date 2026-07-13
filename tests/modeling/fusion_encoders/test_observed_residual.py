from __future__ import annotations

import torch

from chronaris.evaluation.application_tasks.core_recovery_tasks import (
    ChronarisCoreTaskHeads,
    CoreTaskLossConfig,
    TaskAwareTargetBundle,
    core_task_losses,
)
from chronaris.modeling.fusion_encoders import (
    ChronarisContinuousEncoderConfig,
    ChronarisContinuousFusionEncoder,
    ObservedStateResidual,
    build_causal_observed_features,
    fit_observed_state_projector,
)
from chronaris.representation import TrainOnlyRobustNormalizer, collate_observation_samples
from tests.representation.test_contracts import _sample


def _model_and_batch():
    raw = collate_observation_samples(
        [_sample("train"), _sample("test", shift=2.0)]
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        raw,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    normalized = normalizer.transform(raw)
    projector = fit_observed_state_projector(
        normalized,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    continuous = ChronarisContinuousFusionEncoder(
        ChronarisContinuousEncoderConfig(
            physiology_feature_names=("physiology.a", "physiology.b"),
            vehicle_feature_names=("vehicle.a",),
            dropout=0.0,
        )
    )
    return ObservedStateResidual(
        continuous_encoder=continuous,
        projector=projector,
    ), normalized


def test_zero_initialized_residual_is_exact_observed_state_anchor() -> None:
    torch.manual_seed(17)
    model, batch = _model_and_batch()
    output = model(batch)

    assert output.observed_sequence.shape == (2, 96, 64)
    assert torch.equal(output.delta_sequence, torch.zeros_like(output.delta_sequence))
    assert torch.equal(output.sequence_for("maneuver"), output.observed_sequence)
    assert torch.equal(
        output.sequence_for("physiology_response"),
        output.observed_sequence,
    )
    _features, expected_valid = build_causal_observed_features(batch)
    assert torch.equal(output.valid_mask, expected_valid)
    assert all(
        torch.allclose(value, torch.full_like(value, 0.05), atol=1e-6)
        for value in output.task_gate_weights.values()
    )


def test_task_residual_and_heads_produce_finite_joint_losses() -> None:
    torch.manual_seed(29)
    model, batch = _model_and_batch()
    with torch.no_grad():
        torch.nn.init.normal_(model.delta_projection[-1].weight, std=0.01)
    representation = model(batch)
    heads = ChronarisCoreTaskHeads(physiology_target_count=2)
    output = heads(representation)
    targets = TaskAwareTargetBundle(
        sample_ids=batch.sample_ids,
        maneuver_class=torch.tensor([0, 2], dtype=torch.long),
        maneuver_score=torch.tensor([0.2, 1.8]),
        response_value=torch.tensor([0.1, 0.8]),
        high_response=torch.tensor([0.0, 1.0]),
        response_available=torch.tensor([True, True]),
        field_deltas=torch.tensor([[0.1, 0.2], [0.3, 0.5]]),
        field_delta_mask=torch.tensor([[True, True], [True, False]]),
        maneuver_soft_targets=torch.tensor([[0.8, 0.2, 0.0], [0.0, 0.1, 0.9]]),
    )
    losses = core_task_losses(
        output,
        targets,
        config=CoreTaskLossConfig(high_response_weight=0.5),
    )
    losses["total"].backward()

    assert not torch.equal(
        representation.sequence_for("maneuver"),
        representation.observed_sequence,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    assert model.delta_projection[-1].weight.grad is not None
