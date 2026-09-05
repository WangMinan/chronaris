from dataclasses import replace

import torch
from torch.nn import functional as F

from chronaris.models.fusion.causal import (
    CausalFusionTensorInput, CausalMaskedCrossModalFusion,
    compute_vehicle_event_strengths, normalize_visible_event_scores,
)
from chronaris.models.fusion.semantic_event import CausalEventFusion, CausalEventFusionConfig, SemanticEventTensorInput
from chronaris.modeling.fusion_encoders.multiscale_causal import MultiScaleCausalFusionInput
from chronaris.modeling.fusion_encoders.safe_lag_fusion import SafeLagAwareFusion, SafeLagAwareFusionConfig


def test_causal_event_bias_and_empty_history_ignore_future_values():
    states = torch.tensor([[[0., 0.], [1., 0.], [2., 1.], [3., 2.]]], requires_grad=True)
    raw = compute_vehicle_event_strengths(states)
    visible = torch.tensor([[[True, True, False, False], [True, True, True, False]]])
    scores = normalize_visible_event_scores(raw, visible)
    torch.testing.assert_close(scores[0, 0], torch.tensor([0., 1., 0., 0.]))
    scores.sum().backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    model = CausalMaskedCrossModalFusion()
    inputs = CausalFusionTensorInput(
        physiology_states=torch.ones(1, 3, 2), vehicle_states=states.detach(),
        physiology_offsets_s=torch.tensor([[-1., 0., 1.]]),
        vehicle_offsets_s=torch.arange(4)[None].float(),
    )
    baseline = model(inputs)
    changed = states.detach().clone()
    changed[:, 2:] += 10000
    altered = model(replace(inputs, vehicle_states=changed))
    torch.testing.assert_close(baseline.fused_states, altered.fused_states, atol=1e-6, rtol=0)
    assert not baseline.causal_mask[0, 0].any()
    assert not baseline.attended_vehicle_states[0, 0].any()


def test_semantic_pool_ignores_padding_and_does_not_invent_empty_events():
    torch.manual_seed(17)
    model = CausalEventFusion(CausalEventFusionConfig(state_dim=4, learnable_queries=True))
    with torch.no_grad():
        model.query_bank.query_residual.normal_()
    inputs = SemanticEventTensorInput(
        physiology_states=torch.randn(1, 4, 4), vehicle_states=torch.randn(1, 4, 4),
        attention_weights=torch.ones(1, 1, 4), vehicle_event_scores=torch.tensor([[0., 1., .3, .8]]),
        vehicle_offsets_s=torch.arange(4)[None].float(),
        physiology_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        vehicle_valid_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    padded = replace(inputs,
        physiology_states=F.pad(inputs.physiology_states, (0, 0, 0, 5), value=100),
        vehicle_states=F.pad(inputs.vehicle_states, (0, 0, 0, 5), value=100),
        attention_weights=F.pad(inputs.attention_weights, (0, 5)),
        vehicle_event_scores=F.pad(inputs.vehicle_event_scores, (0, 5), value=100),
        vehicle_offsets_s=torch.arange(9)[None].float(),
        physiology_valid_mask=F.pad(inputs.physiology_valid_mask, (0, 5)),
        vehicle_valid_mask=F.pad(inputs.vehicle_valid_mask, (0, 5)),
    )
    torch.testing.assert_close(model(inputs).query_context_states, model(padded).query_context_states, atol=1e-6, rtol=0)
    empty = model(replace(inputs, vehicle_valid_mask=torch.zeros_like(inputs.vehicle_valid_mask)))
    assert not empty.event_token_mask.any() and not empty.query_context_states.any()


def test_each_missing_fusion_branch_is_zero():
    torch.manual_seed(17)
    model = SafeLagAwareFusion(SafeLagAwareFusionConfig(hidden_dim=4))
    for missing in ("physiology", "vehicle", "both"):
        phys = torch.full((1, 4), missing not in ("physiology", "both"), dtype=torch.bool)
        vehicle = torch.full((1, 4), missing not in ("vehicle", "both"), dtype=torch.bool)
        output = model(MultiScaleCausalFusionInput(
            physiology_states=torch.zeros(1, 4, 4), vehicle_states=torch.ones(1, 4, 4),
            physiology_valid_mask=phys, vehicle_valid_mask=vehicle,
            query_timestamps_s=torch.arange(4)[None].float(),
        ))
        if not phys.any():
            assert not output.sequence_embedding[..., :24].any()
        if not vehicle.any():
            assert not output.sequence_embedding[..., 24:48].any()
        assert not output.sequence_embedding[..., 48:].any()
