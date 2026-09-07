import torch
import pytest

from chronaris.evaluation.application_tasks.v4_correctness import remove_future_observations
from chronaris.evaluation.application_tasks.v4_encoding_diagnostics import collect_encoding_diagnostics
from chronaris.modeling.training.pretraining_encoders import EncoderCandidateConfig, build_trainable_fusion_encoder
from chronaris.models.alignment.calibrated_physics import fit_physics_calibration
from chronaris.representation import TrainOnlyRobustNormalizer, collate_observation_samples, select_observation_batch
from tests.representation.test_contracts import _sample


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_validation_diagnostics_keep_empty_contexts_unavailable_and_anchor_predictions(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA encoding diagnostics")
    torch.set_num_threads(1)
    samples = [_sample(f"sample_{i}", shift=i / 10) for i in range(3)]
    batch = collate_observation_samples(samples)
    normalizer = TrainOnlyRobustNormalizer().fit(batch, train_sample_ids=batch.sample_ids[:2], held_out_sample_ids=batch.sample_ids[2:])
    calibration = fit_physics_calibration(normalizer, lambda ids: select_observation_batch(batch, ids),
        train_sample_ids=batch.sample_ids[:2], vehicle_feature_names=samples[0].schema.vehicle_feature_names, relations=())
    encoder = build_trainable_fusion_encoder("chronaris", physiology_feature_names=samples[0].schema.physiology_feature_names,
        vehicle_feature_names=samples[0].schema.vehicle_feature_names, candidate_config=EncoderCandidateConfig(hidden_dim=8),
        chronaris_fusion_kind="safe_lag", chronaris_physics_calibration=calibration).to(device)
    original = {name: value.clone() for name, value in encoder.state_dict().items()}
    clean = collect_encoding_diagnostics(encoder=encoder, normalizer=normalizer, batch=batch, batch_size=2)
    assert clean["sample_count"] == 3 and clean["label_used"] is False
    assert clean["distributions"]["physiology_query_state_norm"]["count"] > 0
    feature_count = len(samples[0].schema.physiology_feature_names) + len(samples[0].schema.vehicle_feature_names)
    assert len(clean["observation_fit"]) == feature_count and all(row["count"] > 0 for row in clean["observation_fit"])
    assert any(row["standardized_rmse"] > 0 for row in clean["observation_fit"])
    missing = collect_encoding_diagnostics(encoder=encoder, normalizer=normalizer,
        batch=remove_future_observations(batch, cutoff_s=-1.), batch_size=2)
    assert missing["sample_count"] == 3 and missing["physical_components"] == []
    for stream in ("physiology", "vehicle"):
        assert missing["distributions"][stream + "_query_state_norm"] == {"status": "unavailable_no_valid_values", "count": 0}
        assert missing["validity_counts"][stream + "_unobserved_windows"] == 3
    assert all(row["standardized_rmse"] is None for row in missing["observation_fit"])
    assert all(torch.equal(value, encoder.state_dict()[name]) for name, value in original.items())
