from dataclasses import replace

import pytest
import torch

from chronaris.modeling.fusion_encoders import NaiveTimeSyncEncoder, NaiveTimeSyncFusionAdapter
from chronaris.modeling.fusion_encoders.single_stream import SingleStreamFusionAdapter
from chronaris.modeling.fusion_encoders.deep_baselines import DeepBaselineFusionAdapter
from chronaris.modeling.fusion_encoders.chronaris_continuous import ChronarisContinuousFusionAdapter
from chronaris.modeling.training import TrainedFusionAdapter
from chronaris.modeling.training.pretraining_encoders import build_trainable_fusion_encoder
from chronaris.representation import TrainOnlyRobustNormalizer, collate_observation_samples
from chronaris.evaluation.application_tasks.simulation_stress_representation_run import _export_one
from tests.representation.test_contracts import _sample


@pytest.mark.parametrize("method,legacy", [
    (method, legacy)
    for method in ("physiology_only", "vehicle_only", "mult", "contiformer", "chronaris", "naive_time_sync")
    for legacy in ([False] if method == "naive_time_sync" else [False, True])
])
def test_frozen_adapters_preserve_observed_outputs_and_export_empty_samples(method, legacy, tmp_path):
    samples = [_sample("train"), _sample("test", shift=2)]
    batch = collate_observation_samples(samples)
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch, train_sample_ids=("train",), held_out_sample_ids=("test",)
    )
    if method == "naive_time_sync":
        encoder = NaiveTimeSyncEncoder().fit(batch, train_sample_ids=("train",), held_out_sample_ids=("test",))
        adapter = NaiveTimeSyncFusionAdapter(encoder=encoder, fold_id="test", checkpoint_sha256="0" * 64)
        sequence, _ = encoder.encode(batch)
        pooled = sequence.mean(dim=1)
    else:
        encoder = build_trainable_fusion_encoder(
            method, physiology_feature_names=samples[0].schema.physiology_feature_names,
            vehicle_feature_names=samples[0].schema.vehicle_feature_names,
            vehicle_field_labels=(),
        ).eval()
        adapter = TrainedFusionAdapter(encoder=encoder, normalizer=normalizer, fold_id="test", checkpoint_sha256="0" * 64)
        with torch.inference_mode():
            reference = encoder(normalizer.transform(batch))
        sequence = reference.sequence_embedding
        valid = reference.modality_available_mask
        pooled = (sequence * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        if legacy:
            adapter_class = (
                SingleStreamFusionAdapter if method in ("physiology_only", "vehicle_only")
                else ChronarisContinuousFusionAdapter if method == "chronaris"
                else DeepBaselineFusionAdapter
            )
            adapter = adapter_class(backbone=encoder.backbone, normalizer=normalizer, fold_id="test", checkpoint_sha256="0" * 64)
            if method != "chronaris":
                pooled = sequence.mean(dim=1)
    observed = adapter(batch)
    assert torch.equal(observed.sequence_embedding, sequence)
    assert torch.equal(observed.pooled_embedding, pooled)

    empty = replace(samples[1],
        physiology_values=samples[1].physiology_values[:0],
        physiology_timestamps_s=samples[1].physiology_timestamps_s[:0],
        physiology_feature_mask=samples[1].physiology_feature_mask[:0],
        vehicle_values=samples[1].vehicle_values[:0],
        vehicle_timestamps_s=samples[1].vehicle_timestamps_s[:0],
        vehicle_feature_mask=samples[1].vehicle_feature_mask[:0],
    )
    mixed = collate_observation_samples([samples[0], empty])
    output, status = _export_one(adapter=adapter, batch=mixed, destination=tmp_path, batch_size=1, resume=True)
    assert status == "completed"
    assert output.sample_ids == mixed.sample_ids
    assert output.valid_mask[0].any() and not output.valid_mask[1].any()
    assert not output.sequence_embedding[1].any() and not output.pooled_embedding[1].any()
    resumed, status = _export_one(adapter=adapter, batch=mixed, destination=tmp_path, batch_size=1, resume=True)
    assert status == "resumed"
    assert torch.equal(output.sequence_embedding, resumed.sequence_embedding)
    drifted = replace(mixed, source_sample_hashes=("1" * 64, "2" * 64))
    with pytest.raises(ValueError, match="lineage changed"):
        _export_one(adapter=adapter, batch=drifted, destination=tmp_path, batch_size=1, resume=True)
