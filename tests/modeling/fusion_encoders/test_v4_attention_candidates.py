from dataclasses import replace
import math

import pytest
import torch
from torch.nn import functional as F

from chronaris.modeling.fusion_encoders.safe_lag_fusion import SafeLagAwareFusion, SafeLagAwareFusionConfig
from tests.modeling.fusion_encoders.test_safe_lag_fusion import _fusion_input


@pytest.mark.parametrize("kind", ["legacy_cosine", "cosine_temperature", "projected_dot_product"])
def test_attention_formula_gradients_and_missing_history_contract(kind):
    inputs = _fusion_input(batch=2, timepoints=8, hidden=8)
    model = SafeLagAwareFusion(SafeLagAwareFusionConfig(hidden_dim=8, attention_kind=kind,
        lag_ranges_s=((0., 30.),), use_scale_gate=False))
    output = model(inputs)
    if kind == "projected_dot_product":
        q, k, v = (model.query_projection(inputs.physiology_states), model.key_projection(inputs.vehicle_states),
                   model.value_projection(inputs.vehicle_states))
        scores = q @ k.transpose(-1, -2) / math.sqrt(8)
    else:
        v = inputs.vehicle_states
        scores = F.normalize(inputs.physiology_states, dim=-1) @ F.normalize(v, dim=-1).transpose(-1, -2)
        if kind == "legacy_cosine":
            scores /= math.sqrt(8)
        else:
            assert float(model.effective_attention_temperature.detach()) == pytest.approx(1.)
            scores /= model.effective_attention_temperature
    expected = scores.masked_fill(~output.lag_masks[0], -torch.inf).softmax(-1)
    torch.testing.assert_close(output.attention_weights[0], expected)
    torch.testing.assert_close(output.attended_vehicle_states, expected @ v)
    output.sequence_embedding.square().sum().backward()
    parameters = ([model.temperature_logit] if kind == "cosine_temperature" else
                  [model.query_projection.weight, model.key_projection.weight, model.value_projection.weight]
                  if kind == "projected_dot_product" else [])
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for parameter in parameters)
    changed = replace(inputs, vehicle_states=inputs.vehicle_states.clone(), physiology_states=inputs.physiology_states.clone())
    changed.vehicle_states[:, 4:] += 1000
    changed.physiology_states[:, 4:] -= 1000
    torch.testing.assert_close(model(changed).sequence_embedding[:, :4], output.sequence_embedding[:, :4], atol=1e-6, rtol=0)
    for absent in (("physiology",), ("vehicle",), ("physiology", "vehicle")):
        updates = {stream + "_valid_mask": torch.zeros_like(getattr(inputs, stream + "_valid_mask")) for stream in absent}
        updates.update({stream + "_states": torch.zeros_like(getattr(inputs, stream + "_states")) for stream in absent})
        missing = model(replace(inputs, **updates))
        assert not missing.cross_features.any() and not missing.cross_gate.any()
        for stream in absent:
            assert not getattr(missing, stream + "_private").any()
    if kind == "cosine_temperature":
        with torch.no_grad():
            for value in (-1000., 1000.):
                model.temperature_logit.fill_(value)
                assert .05 - 1e-8 <= float(model.effective_attention_temperature) <= 2.


def test_reference_attention_has_no_new_parameters_and_keeps_existing_weights():
    model = SafeLagAwareFusion(SafeLagAwareFusionConfig(hidden_dim=8))
    assert not any(name.startswith(("temperature", "query_projection", "key_projection", "value_projection")) for name in model.state_dict())
    clone = SafeLagAwareFusion(SafeLagAwareFusionConfig(hidden_dim=8, attention_kind="legacy_cosine"))
    clone.load_state_dict(model.state_dict(), strict=True)
    inputs = _fusion_input(hidden=8)
    assert torch.equal(model(inputs).sequence_embedding, clone(inputs).sequence_embedding)
