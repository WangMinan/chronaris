from __future__ import annotations

import hashlib

import numpy as np
import torch

from chronaris.modeling.fusion_encoders.chronaris_v2 import (
    ChronarisV2EncoderConfig,
    ChronarisV2FusionEncoder,
)
from chronaris.modeling.training.chronaris_v2_curriculum import (
    chronaris_v2_augmentation_policy,
    chronaris_v2_curriculum_stage,
)
from chronaris.modeling.training.chronaris_v2_objectives import (
    ChronarisV2ObjectiveHeads,
    build_chronaris_v2_objective_targets,
    build_known_lag_batch,
    chronaris_v2_objective_weight_schedule,
)
from chronaris.modeling.training.pcgrad import (
    GradientConflictController,
    loss_gradient_cosines,
    pcgrad_backward,
)
from chronaris.modeling.training.chronaris_v2_training import (
    ChronarisV2CandidateConfig,
    ChronarisV2TrainingConfig,
    chronaris_v2_hyperparameter_grid,
    train_chronaris_v2_candidate,
    _early_stopping_allowed,
    _shared_pcgrad_backward,
)
from chronaris.modeling.training.common_pretraining import (
    load_common_pretraining_checkpoint,
)
from chronaris.representation import (
    FoldLineage,
    ObservationSchema,
    ObservedDualStreamSample,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
)


def _batch(sample_ids=("a", "b")):
    physiology_names = ("physiology.spo2", "physiology.eeg")
    vehicle_names = ("vehicle.speed_mps", "vehicle.pitch_rad")
    schema = ObservationSchema(
        schema_id="chronaris_v2_objective_test.v1",
        source_kind="unit_test",
        physiology_feature_names=physiology_names,
        vehicle_feature_names=vehicle_names,
        physiology_feature_roles=("observed", "observed"),
        vehicle_feature_roles=("observed", "observed"),
    )

    def sample(sample_id, offset):
        physiology = np.asarray(
            [[96.0, 0.1], [95.0, 0.2], [94.0, 0.4], [93.0, 0.8]],
            dtype=np.float32,
        ) + offset
        vehicle = np.asarray(
            [[10.0, 0.0], [12.0, 0.1], [14.0, 0.2], [16.0, 0.3]],
            dtype=np.float32,
        ) + offset
        return ObservedDualStreamSample(
            sample_id=sample_id,
            group_id=sample_id,
            schema=schema,
            physiology_values=physiology,
            physiology_timestamps_s=np.asarray([0.0, 5.0, 10.0, 20.0]),
            physiology_feature_mask=np.ones_like(physiology, dtype=bool),
            vehicle_values=vehicle,
            vehicle_timestamps_s=np.asarray([0.0, 4.0, 12.0, 20.0]),
            vehicle_feature_mask=np.ones_like(vehicle, dtype=bool),
            source_sample_hash=hashlib.sha256(sample_id.encode()).hexdigest(),
        )

    return collate_observation_samples(
        tuple(sample(sample_id, float(index)) for index, sample_id in enumerate(sample_ids))
    )


def test_v2_targets_and_losses_are_lag_conditioned_without_same_time_alignment() -> None:
    torch.manual_seed(17)
    batch = _batch()
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=("a",),
        held_out_sample_ids=("b",),
    )
    normalized = normalizer.transform(batch)
    config = ChronarisV2EncoderConfig(
        physiology_feature_names=("physiology.spo2", "physiology.eeg"),
        vehicle_feature_names=("vehicle.speed_mps", "vehicle.pitch_rad"),
        internal_hidden_dim=32,
        physiology_hidden_dim=16,
        vehicle_hidden_dim=32,
        num_heads=4,
        dropout=0.0,
    )
    encoder = ChronarisV2FusionEncoder(config)
    encoder.attach_normalizer(normalizer)
    clean = encoder(normalized, compute_diagnostics=True)
    lag_labels = torch.tensor([0, 3], dtype=torch.long)
    lag = encoder(build_known_lag_batch(normalized, lag_labels))
    targets = build_chronaris_v2_objective_targets(normalized)
    heads = ChronarisV2ObjectiveHeads(
        physiology_feature_count=2,
        vehicle_feature_count=2,
    )
    output = heads(
        clean,
        targets,
        weights=chronaris_v2_objective_weight_schedule(20),
        corrupted_encoding=clean,
        lag_encoding=lag,
        lag_labels=lag_labels,
    )
    output.total_loss.backward()

    names = {term.name for term in output.terms}
    assert "future_physiology_delta" in names
    assert "lag_bin_classification" in names
    assert "vehicle_private_retention" in names
    assert "physiology_private_retention" in names
    assert all("alignment" not in name for name in names)
    assert targets.future_physiology_delta.shape == (2, 96, 4, 2)
    assert targets.future_physiology_mask.any()
    assert torch.isfinite(output.total_loss)


def test_curriculum_matches_predeclared_missingness_levels() -> None:
    assert chronaris_v2_curriculum_stage(1).point_dropout_probability == 0.10
    assert chronaris_v2_curriculum_stage(6).point_dropout_probability == 0.20
    assert chronaris_v2_curriculum_stage(11).block_duration_s == 10.0
    assert chronaris_v2_curriculum_stage(20).clock_offset_limit_s == 1.0
    assert chronaris_v2_augmentation_policy(20).point_dropout_probability == 0.40
    assert chronaris_v2_objective_weight_schedule(10).future_physiology_delta == 0
    assert chronaris_v2_objective_weight_schedule(20).future_physiology_delta == 0.5


def test_v2_early_stopping_cannot_preempt_joint_unfreeze() -> None:
    config = ChronarisV2TrainingConfig(
        max_epochs=50,
        patience=1,
        minimum_epochs_before_early_stopping=30,
    )

    assert not _early_stopping_allowed(29, config)
    assert _early_stopping_allowed(30, config)


def test_pcgrad_switches_after_observed_conflict_rate_exceeds_twenty_percent() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    losses = {
        "left": parameter.sum(),
        "right": -parameter.sum(),
    }
    cosines = loss_gradient_cosines(losses, (parameter,))
    controller = GradientConflictController(minimum_pair_count=10)
    for _ in range(10):
        controller.observe(tuple(cosines.values()))

    pairwise = pcgrad_backward(losses, (parameter,))

    assert controller.conflict_rate == 1.0
    assert controller.use_pcgrad is True
    assert next(iter(pairwise.values())) < -0.999
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


def test_pcgrad_controller_counts_conflicted_steps_not_individual_pairs() -> None:
    controller = GradientConflictController(minimum_pair_count=2)
    controller.observe_step((0.2, -0.2, 0.4))
    controller.observe_step((0.1, 0.2, 0.3))

    assert controller.conflict_history == [True, False]
    assert controller.conflict_rate == 0.5
    assert controller.use_pcgrad


def test_shared_pcgrad_leaves_exclusive_head_gradient_unprojected() -> None:
    shared = torch.nn.Parameter(torch.tensor([1.0]))
    exclusive = torch.nn.Parameter(torch.tensor([2.0]))
    losses = {
        "left": (shared + exclusive).square().sum(),
        "right": (-shared + exclusive).square().sum(),
    }
    pairwise = _shared_pcgrad_backward(
        losses,
        total_loss=sum(losses.values()),
        shared_parameters=(shared,),
        exclusive_parameters=(exclusive,),
    )

    assert next(iter(pairwise.values())) < -0.999
    assert exclusive.grad is not None
    torch.testing.assert_close(exclusive.grad, torch.tensor([8.0]))
    assert shared.grad is not None
    assert torch.isfinite(shared.grad).all()


def test_v2_grid_and_one_epoch_training_round_trip(tmp_path) -> None:
    batch = _batch(("a", "b", "c"))
    fold = FoldLineage(
        fold_id="v2-training-test",
        train_sample_ids=("a",),
        validation_sample_ids=("b",),
        held_out_sample_ids=("c",),
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
    )
    candidate = chronaris_v2_hyperparameter_grid()[0]
    result = train_chronaris_v2_candidate(
        candidate=candidate,
        batch=batch,
        fold=fold,
        physiology_feature_names=("physiology.spo2", "physiology.eeg"),
        vehicle_feature_names=("vehicle.speed_mps", "vehicle.pitch_rad"),
        vehicle_field_labels=(),
        normalizer=normalizer,
        output_root=tmp_path,
        config=ChronarisV2TrainingConfig(
            max_epochs=1,
            batch_size=1,
            patience=1,
            seed=17,
        ),
    )
    encoder, _heads, _normalizer, payload = load_common_pretraining_checkpoint(
        result.best_checkpoint_path
    )

    assert len(chronaris_v2_hyperparameter_grid()) == 24
    assert result.status == "completed"
    assert result.completed_epochs == 1
    assert payload["format"] == "chronaris.common_pretraining_checkpoint.v2"
    assert payload["architecture_version"] == "v2"
    assert encoder.backbone.config.architecture_version == "v2"
    assert payload["label_used_for_encoder_training"] is False
    assert payload["simulation_oracle_opened"] is False
