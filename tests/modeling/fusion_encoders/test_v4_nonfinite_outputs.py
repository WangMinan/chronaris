from dataclasses import replace

import pytest
import torch

from chronaris.modeling.training.pretraining_encoders import build_trainable_fusion_encoder, EncoderCandidateConfig
from chronaris.representation import collate_observation_samples
from tests.modeling.fusion_encoders.test_deep_baselines import _dual_sample


@pytest.mark.parametrize("method", ["physiology_only", "vehicle_only", "mult", "contiformer", "chronaris"])
def test_valid_output_nonfinite_raises_but_unobserved_padding_is_zero(method):
    torch.set_num_threads(1)
    batch = collate_observation_samples([_dual_sample("sample")])
    encoder = build_trainable_fusion_encoder(method,
        physiology_feature_names=("physiology.a",), vehicle_feature_names=("vehicle.a",),
        vehicle_field_labels=(), candidate_config=EncoderCandidateConfig(hidden_dim=8, dropout=0.))
    backbone = encoder.backbone
    module = backbone.causal_fusion if method == "chronaris" else backbone
    projection = getattr(module, "output_projection", getattr(module, "contract_projection", None))
    assert projection is not None
    handle = projection.register_forward_hook(lambda _module, _inputs, output: torch.full_like(output, torch.nan))
    with pytest.raises(ValueError, match="non-finite valid"):
        encoder(batch)
    empty = replace(batch, **{f"{stream}_{field}": torch.zeros_like(getattr(batch, f"{stream}_{field}"))
        for stream in ("physiology", "vehicle") for field in ("feature_mask", "point_mask")})
    assert torch.count_nonzero(encoder(empty).sequence_embedding) == 0
    handle.remove()
