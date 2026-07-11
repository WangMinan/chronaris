from __future__ import annotations

import hashlib

import numpy as np
import torch

from chronaris.modeling.training import (
    CandidateScreenConfig,
    EncoderCandidateConfig,
    CandidateScreenResult,
    load_common_pretraining_checkpoint,
    rank_encoder_candidates,
    train_pretext_candidate,
)
from chronaris.representation import (
    FoldLineage,
    ObservationSchema,
    ObservedDualStreamSample,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
)


def _sample(sample_id: str, offset: float):
    schema = ObservationSchema(
        schema_id="candidate_screen_test.v1",
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
        physiology_values=np.asarray(
            [[1 + offset], [2 + offset], [3 + offset], [4 + offset]], dtype=np.float32
        ),
        physiology_timestamps_s=np.asarray([0.0, 5.0, 10.0, 20.0]),
        physiology_feature_mask=np.ones((4, 1), dtype=bool),
        vehicle_values=np.asarray(
            [[5 + offset], [6 + offset], [7 + offset], [8 + offset]], dtype=np.float32
        ),
        vehicle_timestamps_s=np.asarray([0.0, 4.0, 12.0, 20.0]),
        vehicle_feature_mask=np.ones((4, 1), dtype=bool),
        source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest(),
    )


def test_candidate_screen_early_stopping_checkpoint_is_loadable(tmp_path) -> None:
    batch = collate_observation_samples(
        [
            _sample("train_a", 0),
            _sample("train_b", 1),
            _sample("validation", 2),
            _sample("held_out", 3),
        ]
    )
    fold = FoldLineage(
        fold_id="candidate_screen_fold",
        train_sample_ids=("train_a", "train_b"),
        validation_sample_ids=("validation",),
        held_out_sample_ids=("held_out",),
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
    )
    candidate = EncoderCandidateConfig(candidate_id="C", hidden_dim=32)
    result = train_pretext_candidate(
        "physiology_only",
        candidate=candidate,
        batch=batch,
        fold=fold,
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        vehicle_field_labels=(),
        normalizer=normalizer,
        output_root=tmp_path,
        config=CandidateScreenConfig(max_epochs=2, batch_size=2, patience=1),
    )
    resumed = train_pretext_candidate(
        "physiology_only",
        candidate=candidate,
        batch=batch,
        fold=fold,
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        vehicle_field_labels=(),
        normalizer=normalizer,
        output_root=tmp_path,
        config=CandidateScreenConfig(max_epochs=2, batch_size=2, patience=1),
    )
    encoder, _heads, _normalizer, payload = load_common_pretraining_checkpoint(
        result.best_checkpoint_path
    )

    assert result.status == "completed"
    assert resumed.status == "resumed"
    assert result.best_epoch >= 1
    assert result.completed_epochs in {1, 2}
    assert set(result.best_validation_losses) == {
        "masked_reconstruction",
        "short_horizon_prediction",
        "lag_discrimination",
    }
    assert payload["selection_uses_public_pretext_only"] is True
    assert payload["label_used_for_encoder_training"] is False
    assert payload["simulation_oracle_opened"] is False
    assert encoder.config_manifest()["backbone_config"]["hidden_dim"] == 32


def test_candidate_ranking_uses_within_method_min_max_and_parameter_tie_break() -> None:
    results = []
    losses = {
        "A": (2.0, 2.0, 2.0),
        "B": (1.0, 3.0, 3.0),
        "C": (1.0, 1.0, 1.0),
        "D": (3.0, 1.0, 1.0),
    }
    for candidate_id, values in losses.items():
        results.append(
            CandidateScreenResult(
                method_name="chronaris",
                candidate_id=candidate_id,
                status="completed",
                best_checkpoint_path=f"/{candidate_id}.pt",
                last_checkpoint_path=f"/{candidate_id}-last.pt",
                protocol_sha256="0" * 64,
                best_epoch=1,
                completed_epochs=2,
                stopped_early=False,
                best_validation_losses=dict(
                    zip(
                        (
                            "masked_reconstruction",
                            "short_horizon_prediction",
                            "lag_discrimination",
                        ),
                        values,
                        strict=True,
                    )
                ),
                best_public_selection_loss=1.0,
                training_elapsed_s=1.0,
                parameter_count=50 if candidate_id == "C" else 100,
                epoch_rows=(),
            )
        )

    rows = rank_encoder_candidates(results)

    assert [row["candidate_id"] for row in rows] == ["C", "A", "B", "D"]
    assert rows[0]["selected"] is True
    assert rows[0]["normalized_validation_losses"] == {
        "masked_reconstruction": 0.0,
        "short_horizon_prediction": 0.0,
        "lag_discrimination": 0.0,
    }


def test_candidate_screen_resumes_running_last_checkpoint_from_next_epoch(tmp_path) -> None:
    batch = collate_observation_samples(
        [
            _sample("train_a", 0),
            _sample("train_b", 1),
            _sample("validation", 2),
            _sample("held_out", 3),
        ]
    )
    fold = FoldLineage(
        fold_id="candidate_resume_fold",
        train_sample_ids=("train_a", "train_b"),
        validation_sample_ids=("validation",),
        held_out_sample_ids=("held_out",),
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
    )
    candidate = EncoderCandidateConfig(candidate_id="C", hidden_dim=32)
    common = dict(
        method_name="physiology_only",
        candidate=candidate,
        batch=batch,
        fold=fold,
        physiology_feature_names=("physiology.a",),
        vehicle_feature_names=("vehicle.a",),
        vehicle_field_labels=(),
        normalizer=normalizer,
        output_root=tmp_path,
        config=CandidateScreenConfig(max_epochs=3, batch_size=2, patience=3),
    )
    first = train_pretext_candidate(**common)
    last_path = tmp_path / "physiology_only" / "C" / "last.pt"
    payload = torch.load(last_path, map_location="cpu", weights_only=True)
    payload["training_status"] = "running"
    payload["completed_epochs"] = 2
    payload["epoch"] = 2
    payload["epoch_rows"] = payload["epoch_rows"][:2]
    payload["step_count"] = 2
    torch.save(payload, last_path)

    resumed = train_pretext_candidate(**common)

    assert first.completed_epochs == 3
    assert resumed.completed_epochs == 3
    assert [row["epoch"] for row in resumed.epoch_rows] == [1, 2, 3]
