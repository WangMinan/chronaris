from __future__ import annotations

import torch

from chronaris.models.fusion.semantic_event import (
    CausalEventFusion,
    CausalEventFusionConfig,
    SemanticEventTensorInput,
    SemanticQuerySpec,
)


def _inputs() -> SemanticEventTensorInput:
    torch.manual_seed(17)
    return SemanticEventTensorInput(
        physiology_states=torch.randn(2, 4, 4),
        vehicle_states=torch.randn(2, 4, 4),
        attention_weights=torch.eye(4).expand(2, -1, -1).clone(),
        vehicle_event_scores=torch.tensor([[0.0, 1.0, 0.1, 0.8], [0.0, 0.7, 0.1, 1.0]]),
        vehicle_offsets_s=torch.arange(4, dtype=torch.float32).expand(2, -1),
    )


def test_deterministic_semantic_queries_remain_parameter_free() -> None:
    model = CausalEventFusion(CausalEventFusionConfig())
    output = model(_inputs())

    assert model.query_bank.query_residual is None
    assert output.query_states.shape == (2, 3, 4)


def test_learnable_semantic_query_residual_receives_nonzero_gradient() -> None:
    model = CausalEventFusion(
        CausalEventFusionConfig(
            state_dim=4,
            learnable_queries=True,
            event_top_k=2,
            event_score_quantile=0.5,
            query_specs=(
                SemanticQuerySpec("flight_event", "vehicle_plus_event"),
                SemanticQuerySpec("physiology_response", "physiology_plus_gap"),
                SemanticQuerySpec("human_aircraft_coordination", "coordination_gap"),
            ),
        )
    )

    output = model(_inputs())
    output.query_context_states.square().sum().backward()

    gradient = model.query_bank.query_residual.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
