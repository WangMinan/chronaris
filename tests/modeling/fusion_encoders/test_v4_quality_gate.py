from dataclasses import replace

import torch

from chronaris.modeling.fusion_encoders.observation_quality import causal_observation_quality
from chronaris.modeling.fusion_encoders.chronaris_continuous import ChronarisContinuousEncoderConfig, ChronarisContinuousFusionEncoder
from chronaris.representation import collate_observation_samples
from tests.representation.test_augmentation_apply import _sample


def test_native_quality_matches_visible_field_seconds_and_ignores_future_support():
    torch.set_num_threads(1)
    sample = _sample("a")
    batch = collate_observation_samples([sample])
    quality = causal_observation_quality(batch)
    for q, query in enumerate(batch.query_timestamps_s[0]):
        for s, stream in enumerate(("physiology", "vehicle")):
            times = getattr(batch, f"{stream}_timestamps_s")[0]
            masks = getattr(batch, f"{stream}_feature_mask")[0]
            observed_bins, ages = set(), []
            for feature in range(masks.shape[-1]):
                visible = times[masks[:, feature] & (times <= query)]
                observed_bins.update((int(t), feature) for t in visible)
                ages.append(min(float(query - visible[-1]), 30.) / 30. if len(visible) else 1.)
            expected = torch.tensor([sum(ages) / len(ages), len(observed_bins) / ((int(query) + 1) * len(ages)), min(ages)])
            torch.testing.assert_close(quality[0, q, s*3:s*3+3], expected)
    changes = {}
    for stream in ("physiology", "vehicle"):
        times = getattr(batch, f"{stream}_timestamps_s")
        masks = getattr(batch, f"{stream}_feature_mask").clone()
        masks[times > 10] = False
        changes[f"{stream}_feature_mask"] = masks
        changes[f"{stream}_point_mask"] = masks.any(dim=-1)
    changed = replace(batch, **changes)
    history = batch.query_timestamps_s <= 10
    torch.testing.assert_close(quality[history], causal_observation_quality(changed)[history], atol=0, rtol=0)
    future_times = replace(batch, **{f"{stream}_timestamps_s": torch.where(
        getattr(batch, f"{stream}_timestamps_s") > 10, getattr(batch, f"{stream}_timestamps_s") + 100,
        getattr(batch, f"{stream}_timestamps_s")) for stream in ("physiology", "vehicle")})
    torch.testing.assert_close(quality[history], causal_observation_quality(future_times)[history], atol=0, rtol=0)
    model = ChronarisContinuousFusionEncoder(ChronarisContinuousEncoderConfig(
        physiology_feature_names=sample.schema.physiology_feature_names,
        vehicle_feature_names=sample.schema.vehicle_feature_names, hidden_dim=8,
        fusion_kind="safe_lag", semantic_event_enabled=True, quality_gate_enabled=True, dropout=0.)).eval()
    original, truncated = model(batch), model(changed)
    torch.testing.assert_close(original.sequence_embedding[history], truncated.sequence_embedding[history], atol=1e-6, rtol=0)
    original.sequence_embedding.sum().backward()
    assert model.causal_fusion.cross_gate.weight.grad[:, -6:].abs().sum() > 0
    for stream in ("physiology", "vehicle"):
        missing = replace(batch, **{f"{stream}_{field}": torch.zeros_like(getattr(batch, f"{stream}_{field}"))
                                   for field in ("feature_mask", "point_mask")})
        fusion = model(missing).fusion_output
        assert torch.count_nonzero(getattr(fusion, f"{stream}_private")) == 0
        assert torch.count_nonzero(fusion.cross_gate) == 0
    empty = replace(batch, **{f"{stream}_{field}": torch.zeros_like(getattr(batch, f"{stream}_{field}"))
        for stream in ("physiology", "vehicle") for field in ("feature_mask", "point_mask")})
    assert torch.isfinite(causal_observation_quality(empty)).all()
    assert torch.count_nonzero(model(empty).sequence_embedding) == 0
