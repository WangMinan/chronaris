from __future__ import annotations

import torch

from chronaris.evaluation.application_tasks.core_recovery_tasks import (
    TaskAwareTargetBundle,
)
from chronaris.evaluation.application_tasks.core_recovery_training import (
    CoreRecoveryMethodModel,
    CoreRecoveryTrainingConfig,
    train_core_recovery_method,
)
from chronaris.modeling.fusion_encoders import (
    ChronarisContinuousEncoderConfig,
    ChronarisContinuousFusionEncoder,
    ObservedStateResidual,
    fit_observed_state_projector,
)
from chronaris.representation import TrainOnlyRobustNormalizer, collate_observation_samples
from tests.representation.test_contracts import _sample


def test_core_recovery_training_uses_steps_labels_and_no_outer_test(tmp_path) -> None:
    raw = collate_observation_samples(
        [
            _sample("train_a"),
            _sample("train_b", shift=1.0),
            _sample("validation", shift=2.0),
        ]
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        raw,
        train_sample_ids=("train_a", "train_b"),
        held_out_sample_ids=("validation",),
    )
    normalized = normalizer.transform(raw)
    projector = fit_observed_state_projector(
        normalized,
        train_sample_ids=("train_a", "train_b"),
        held_out_sample_ids=("validation",),
    )
    residual = ObservedStateResidual(
        continuous_encoder=ChronarisContinuousFusionEncoder(
            ChronarisContinuousEncoderConfig(
                physiology_feature_names=("physiology.a", "physiology.b"),
                vehicle_feature_names=("vehicle.a",),
                dropout=0.0,
            )
        ),
        projector=projector,
    )
    model = CoreRecoveryMethodModel(
        method_name="chronaris",
        encoder=None,
        chronaris_residual=residual,
        normalizer=normalizer,
        physiology_target_count=2,
    )
    targets = TaskAwareTargetBundle(
        sample_ids=raw.sample_ids,
        maneuver_class=torch.tensor([0, 2, 1], dtype=torch.long),
        maneuver_score=torch.tensor([0.1, 1.9, 1.0]),
        response_value=torch.tensor([0.2, 0.8, 0.5]),
        high_response=torch.tensor([0.0, 1.0, 1.0]),
        response_available=torch.tensor([True, True, True]),
        field_deltas=torch.tensor([[0.1, 0.2], [0.4, 0.6], [0.2, 0.3]]),
        field_delta_mask=torch.ones((3, 2), dtype=torch.bool),
    )
    source = tmp_path / "source.pt"
    torch.save({"source": True}, source)
    result = train_core_recovery_method(
        model=model,
        batch=raw,
        targets=targets,
        train_sample_ids=("train_a", "train_b"),
        validation_sample_ids=("validation",),
        source_checkpoint_path=source,
        output_root=tmp_path / "outputs",
        candidate_id="chronaris-c01",
        config=CoreRecoveryTrainingConfig(
            frozen_backbone_steps=1,
            partial_unfreeze_steps=1,
            validation_interval_steps=1,
            patience_evaluations=4,
            batch_size=2,
            device="cpu",
        ),
    )
    payload = torch.load(result.best_checkpoint_path, map_location="cpu", weights_only=True)

    assert result.status == "completed"
    assert result.completed_steps == 2
    assert payload["sample_exposure_count"] == 4
    assert payload["label_used_for_encoder_training"] is True
    assert payload["outer_test_accessed"] is False
    assert payload["representation_family"] == "task_aware_core_recovery_v1"
