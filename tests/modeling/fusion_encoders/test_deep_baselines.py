from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from chronaris.modeling.fusion_encoders import (
    CausalContiFormerFusionEncoder,
    CausalMulTFusionEncoder,
    DeepBaselineEncoderConfig,
    DeepBaselineFusionAdapter,
    load_deep_baseline_checkpoint,
    save_deep_baseline_checkpoint,
)
from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
from third_party.mult.modules.transformer import TransformerEncoder


def _dual_sample(
    sample_id: str,
    *,
    physiology_future: float = 4.0,
    vehicle_future: float = 14.0,
) -> ObservedDualStreamSample:
    schema = ObservationSchema(
        schema_id="deep_baseline_test.v1",
        source_kind="unit_test",
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        physiology_feature_roles=("observed",),
        vehicle_feature_roles=("observed",),
    )
    return ObservedDualStreamSample(
        sample_id=sample_id,
        group_id=sample_id,
        schema=schema,
        physiology_values=np.asarray([[1.0], [2.0], [physiology_future]], dtype=np.float32),
        physiology_timestamps_s=np.asarray([0.0, 5.0, 20.0], dtype=np.float64),
        physiology_feature_mask=np.ones((3, 1), dtype=bool),
        vehicle_values=np.asarray([[10.0], [12.0], [vehicle_future]], dtype=np.float32),
        vehicle_timestamps_s=np.asarray([0.0, 7.0, 20.0], dtype=np.float64),
        vehicle_feature_mask=np.ones((3, 1), dtype=bool),
        source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest(),
    )


def _adapter(method_name: str, batch):
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=("train",),
        held_out_sample_ids=("test",),
    )
    config = DeepBaselineEncoderConfig(
        method_name=method_name,
        physiology_feature_dim=1,
        vehicle_feature_dim=1,
        dropout=0.0,
    )
    backbone = (
        CausalMulTFusionEncoder(config)
        if method_name == "mult"
        else CausalContiFormerFusionEncoder(config)
    ).eval()
    return DeepBaselineFusionAdapter(
        backbone=backbone,
        normalizer=normalizer,
        fold_id="fold_a",
        checkpoint_sha256=("5" if method_name == "mult" else "6") * 64,
    )


@pytest.mark.parametrize("method_name", ["mult", "contiformer"])
def test_deep_baselines_are_task_head_free_fixed_shape_and_future_invariant(method_name):
    torch.manual_seed(19)
    train = _dual_sample("train")
    base = _dual_sample("test")
    changed = _dual_sample(
        "test",
        physiology_future=9999.0,
        vehicle_future=-9999.0,
    )
    adapter = _adapter(method_name, collate_observation_samples([train, base]))

    first = adapter(collate_observation_samples([base]))
    second = adapter(collate_observation_samples([changed]))
    past = first.timestamps_s[0] < 20.0

    assert first.sequence_embedding.shape == (1, 96, 64)
    assert first.valid_mask.all()
    assert torch.allclose(
        first.sequence_embedding[0, past],
        second.sequence_embedding[0, past],
        atol=1e-5,
        rtol=1e-5,
    )
    assert adapter.to_manifest()["sequence_source"] == "task_head_free"
    assert adapter.to_manifest()["causal_attention"] is True


@pytest.mark.parametrize("method_name", ["mult", "contiformer"])
def test_deep_baselines_respond_to_each_historical_stream(method_name):
    torch.manual_seed(23)
    train = _dual_sample("train")
    base = _dual_sample("test")
    adapter = _adapter(method_name, collate_observation_samples([train, base]))
    batch = collate_observation_samples([base])
    physiology_changed = replace(
        batch,
        physiology_values=batch.physiology_values + batch.physiology_feature_mask * 2.0,
    )
    vehicle_changed = replace(
        batch,
        vehicle_values=batch.vehicle_values + batch.vehicle_feature_mask * 2.0,
    )

    output = adapter(batch)
    phys_output = adapter(physiology_changed)
    vehicle_output = adapter(vehicle_changed)

    assert not torch.equal(output.sequence_embedding, phys_output.sequence_embedding)
    assert not torch.equal(output.sequence_embedding, vehicle_output.sequence_embedding)
    assert torch.isfinite(phys_output.sequence_embedding).all()
    assert torch.isfinite(vehicle_output.sequence_embedding).all()


@pytest.mark.parametrize("method_name", ["mult", "contiformer"])
def test_deep_baseline_checkpoint_round_trip(method_name, tmp_path):
    torch.manual_seed(29)
    batch = collate_observation_samples(
        [_dual_sample("train"), _dual_sample("test")]
    )
    adapter = _adapter(method_name, batch)
    path = save_deep_baseline_checkpoint(
        tmp_path / f"{method_name}.pt",
        backbone=adapter.backbone,
        normalizer=adapter.normalizer,
        seed=29,
    )
    backbone, normalizer, metadata = load_deep_baseline_checkpoint(path)
    loaded = DeepBaselineFusionAdapter(
        backbone=backbone,
        normalizer=normalizer,
        fold_id="fold_a",
        checkpoint_sha256=sha256_file(path),
    )
    original = DeepBaselineFusionAdapter(
        backbone=adapter.backbone,
        normalizer=adapter.normalizer,
        fold_id="fold_a",
        checkpoint_sha256=sha256_file(path),
    )

    assert metadata["sequence_source"] == "task_head_free"
    assert torch.equal(original(batch).sequence_embedding, loaded(batch).sequence_embedding)


def test_vendored_mult_padding_mask_blocks_masked_key_values():
    torch.manual_seed(31)
    encoder = TransformerEncoder(
        embed_dim=8,
        num_heads=2,
        layers=1,
        attn_dropout=0.0,
        relu_dropout=0.0,
        res_dropout=0.0,
        embed_dropout=0.0,
        attn_mask=True,
    ).eval()
    query = torch.randn(3, 1, 8)
    key = torch.randn(3, 1, 8)
    changed = key.clone()
    changed[2] = 1_000_000.0
    padding = torch.tensor([[False, False, True]])

    first = encoder(query, key, key, key_padding_mask=padding)
    second = encoder(query, changed, changed, key_padding_mask=padding)

    assert torch.allclose(first, second, atol=1e-5, rtol=1e-5)
