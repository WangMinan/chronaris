import pytest
import torch

from chronaris.modeling.training.pretext import CommonPretextHeadBundle
from chronaris.representation.pretext_targets import CommonPretextTargets


def test_private_reconstruction_has_separate_gradients_masks_and_equal_modal_weights():
    torch.set_num_threads(1)
    heads = CommonPretextHeadBundle(representation_dim=64, target_feature_count=4,
        modality_feature_counts=(1, 3), single_stream_fidelity_weight=.2)
    with torch.no_grad():
        for parameter in heads.single_stream_heads.parameters():
            parameter.zero_()
        for head in heads.single_stream_heads.values():
            head.weight[:, 0] = 1
    sequence = torch.zeros(2, 3, 64, requires_grad=True)
    phys, vehicle = (torch.zeros(2, 3, 24, requires_grad=True) for _ in range(2))
    values = torch.tensor([1., 3., 3., 3.]).expand(2, 3, -1).clone()
    observed = torch.ones_like(values, dtype=torch.bool)
    targets = CommonPretextTargets(values, torch.zeros_like(observed), values, observed, 4, ("a", "b"), observation_mask=observed)
    valid = torch.ones((2, 3), dtype=torch.bool)
    def forward(vehicle_state, vehicle_valid=valid):
        return heads(sequence, sequence, targets, positive_valid_mask=valid, negative_valid_mask=valid,
            single_stream_inputs={"physiology": (phys, valid), "vehicle": (vehicle_state, vehicle_valid)})
    first, changed = forward(vehicle), forward(vehicle + 10)
    a, b = first.terms[-1], changed.terms[-1]
    assert a.raw_loss.item() == pytest.approx(1.5)
    assert a.weighted_loss.item() == pytest.approx(.3)
    torch.testing.assert_close(a.components[0].raw_loss, b.components[0].raw_loss, atol=0, rtol=0)
    a.components[0].weighted_loss.backward(retain_graph=True)
    assert phys.grad.abs().sum() > 0 and vehicle.grad is None and sequence.grad is None
    missing = forward(vehicle, torch.zeros_like(valid)).terms[-1]
    assert missing.components[1].status == "unavailable" and missing.components[1].count == 0
    assert missing.raw_loss.item() == pytest.approx(.5)
    restored = CommonPretextHeadBundle(representation_dim=64, target_feature_count=4, **heads.objective_config())
    restored.load_state_dict(heads.state_dict(), strict=True)
