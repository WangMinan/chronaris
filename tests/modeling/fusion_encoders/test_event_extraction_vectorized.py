from dataclasses import replace

import pytest
import torch

from chronaris.models.fusion.semantic_event import CausalEventFusionConfig, EventTokenExtractor, SemanticEventTensorInput
from tests.modeling.fusion_encoders.event_extraction_reference import reference_extract


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_vectorized_events_match_scalar_outputs_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(17)
    for points, top_k in ((2, 4), (17, 4), (96, 4)):
        for with_mask in (False, True):
            states = torch.randn(4, points, 8, device=device, requires_grad=True)
            scores = torch.rand(4, points, device=device, requires_grad=True)
            attention = torch.rand(4, 2, points, device=device, requires_grad=True)
            valid = torch.rand(4, points, device=device) > .3
            valid[0] = False
            valid[-1] = True
            inputs = SemanticEventTensorInput(
                physiology_states=states, vehicle_states=states,
                attention_weights=attention,
                vehicle_event_scores=scores * torch.tensor([0., 1., 1., 1.], device=device)[:, None],
                vehicle_offsets_s=torch.arange(points, dtype=torch.float64, device=device)[None].expand(4, -1),
                physiology_valid_mask=valid if with_mask else None,
                vehicle_valid_mask=valid if with_mask else None,
            )
            config = CausalEventFusionConfig(state_dim=8, event_top_k=top_k)
            expected = reference_extract(inputs, config)
            actual = EventTokenExtractor(config).extract(inputs)
            for left, right in zip(expected, actual, strict=True):
                torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
            ref_loss = sum(value.square().sum() for value in expected[:3])
            new_loss = sum(value.square().sum() for value in actual[:3])
            reference_gradients = torch.autograd.grad(ref_loss, (states, scores, attention), retain_graph=True)
            gradients = torch.autograd.grad(new_loss, (states, scores, attention))
            for left, right in zip(reference_gradients, gradients, strict=True):
                torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-5)

    # Equal salience must retain the scalar reference's chronological tie order.
    tied = replace(inputs, vehicle_event_scores=torch.ones_like(scores),
                   attention_weights=torch.ones_like(attention))
    expected, actual = reference_extract(tied, config), EventTokenExtractor(config).extract(tied)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
