from __future__ import annotations

import torch

from chronaris.modeling.training import (
    EncoderCandidateConfig,
    build_trainable_fusion_encoder,
    initialize_encoder_from_transfer_source,
)


def test_chronaris_transfer_copies_shared_mechanisms_and_skips_schema_layers(
    tmp_path,
) -> None:
    candidate = EncoderCandidateConfig(candidate_id="C", hidden_dim=32)
    source = build_trainable_fusion_encoder(
        "chronaris",
        physiology_feature_names=("p.a",),
        vehicle_feature_names=("v.a",),
        candidate_config=candidate,
    )
    target = build_trainable_fusion_encoder(
        "chronaris",
        physiology_feature_names=("p.a", "p.b"),
        vehicle_feature_names=("v.a", "v.b", "v.c"),
        candidate_config=candidate,
    )
    shared_name = next(
        name
        for name, value in source.state_dict().items()
        if name in target.state_dict()
        and value.shape == target.state_dict()[name].shape
        and value.is_floating_point()
    )
    source_state = source.state_dict()
    source_state[shared_name] = torch.full_like(source_state[shared_name], 0.125)
    source.load_state_dict(source_state)
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "format": "chronaris.common_pretraining_checkpoint.v1",
            "training_status": "completed",
            "method_name": "chronaris",
            "seed": 17,
            "label_used_for_encoder_training": False,
            "physiology_feature_names": ["p.a"],
            "vehicle_feature_names": ["v.a"],
            "encoder_state_dict": source.state_dict(),
        },
        checkpoint,
    )

    manifest = initialize_encoder_from_transfer_source(
        target,
        checkpoint,
        expected_method="chronaris",
        expected_seed=17,
    )

    assert manifest.copied_tensor_count > 0
    assert manifest.skipped_shape_tensor_count > 0
    assert 0 < manifest.copied_element_fraction < 1
    assert manifest.schema_specific_layers_reinitialized is True
    assert torch.equal(
        target.state_dict()[shared_name],
        torch.full_like(target.state_dict()[shared_name], 0.125),
    )
