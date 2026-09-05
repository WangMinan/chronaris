from types import SimpleNamespace

import pytest
import torch

from chronaris.models.alignment.calibrated_physics import (
    KinematicRelation, calibrated_kinematic_losses, fit_physics_calibration,
)
from chronaris.modeling.fusion_encoders.chronaris_physics import build_chronaris_physics_audit


def test_physics_restores_units_and_rejects_zero_information_via_same_output_anchor():
    names = ("vehicle.speed_mps", "vehicle.longitudinal_acc_mps2")
    times = torch.arange(4).float()[None]
    raw = torch.stack((2 * times + 10, torch.full_like(times, 2)), dim=-1)
    center, scale = torch.tensor([10., 2.]), torch.tensor([2., .5])
    normalized = (raw - center) / scale
    mask = torch.ones_like(raw, dtype=torch.bool)
    stream = SimpleNamespace(values=normalized, offsets_s=times, feature_names=names,
                             feature_valid_mask=mask, mask=mask.any(-1))
    normalizer = SimpleNamespace(fit_sample_ids=("train",), fit_sample_hash="a" * 64,
                                 vehicle=SimpleNamespace(center=center, scale=scale))
    calls = []
    def provider(ids):
        calls.extend(ids)
        assert tuple(ids) == ("train",)
        return SimpleNamespace(sample_ids=tuple(ids), vehicle_values=raw,
                               vehicle_timestamps_s=times, vehicle_feature_mask=mask,
                               vehicle_point_mask=mask.any(-1))
    relation = KinematicRelation(*names, "vehicle_rigid_body_translation", "m/s^2", "analytic test")
    calibration = fit_physics_calibration(normalizer, provider, train_sample_ids=("train",),
                                          vehicle_feature_names=names, relations=(relation,))
    assert calls == ["train"]
    assert calibration["relations"][0]["residual_scale"] == .001
    loss = calibrated_kinematic_losses(normalized, stream, calibration)[0]
    assert loss[1] == 0 and loss[2] == 3
    empty = SimpleNamespace(values=torch.zeros(1, 4, 1), feature_valid_mask=torch.zeros(1, 4, 1, dtype=torch.bool),
                            mask=torch.zeros(1, 4, dtype=torch.bool))
    zero_physical = (-center / scale).expand_as(normalized).clone().requires_grad_()
    output = SimpleNamespace(vehicle=SimpleNamespace(reconstructions=zero_physical),
                             physiology=SimpleNamespace(reconstructions=empty.values))
    batch = SimpleNamespace(vehicle=stream, physiology=empty)
    audit = build_chronaris_physics_audit(output, batch, enabled=True, weight=.05, calibration=calibration)
    assert audit.total_weighted_value == 0
    assert audit.observation_anchor > 0
    audit.observation_anchor.backward()
    assert torch.count_nonzero(zero_physical.grad) > 0
    disabled = build_chronaris_physics_audit(output, batch, enabled=False, weight=.05, calibration=calibration)
    torch.testing.assert_close(disabled.observation_anchor, audit.observation_anchor)
    with pytest.raises(ValueError, match="training samples"):
        fit_physics_calibration(normalizer, provider, train_sample_ids=("held_out",),
                                vehicle_feature_names=names, relations=(relation,))
