"""Shared frozen encoder training for thesis outer folds."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from chronaris.modeling.fusion_encoders import (
    NaiveTimeSyncEncoder,
    NaiveTimeSyncFusionAdapter,
    load_naive_time_sync_checkpoint,
    save_naive_time_sync_checkpoint,
)
from chronaris.modeling.training import (
    ENCODER_SCREEN_CANDIDATES,
    TRAINABLE_FUSION_METHODS,
    CandidateScreenConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_pretext_candidate,
)
from chronaris.representation import AugmentationPolicy
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def train_frozen_outer_adapters(
    *,
    provider,
    fold,
    schema,
    normalizer,
    selected_models,
    output_root,
    seed,
    max_epochs,
    patience,
    batch_size,
    learning_rate,
    weight_decay,
    device,
    resume,
    vehicle_field_labels=None,
):
    candidates = {value.candidate_id: value for value in ENCODER_SCREEN_CANDIDATES}
    adapters = {}
    hashes = {}
    rows = []
    labels = vehicle_field_labels or tuple(
        (name, name) for name in schema.vehicle_feature_names
    )
    for method in TRAINABLE_FUSION_METHODS:
        candidate = replace(
            candidates[selected_models[method]["candidate_id"]],
            learning_rate=learning_rate,
        )
        chronaris = method == "chronaris"
        result = train_pretext_candidate(
            method,
            candidate=candidate,
            batch=None,
            batch_provider=provider,
            fold=fold,
            physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names,
            vehicle_field_labels=labels,
            normalizer=normalizer,
            output_root=Path(output_root) / "checkpoints",
            config=CandidateScreenConfig(
                max_epochs=max_epochs,
                batch_size=batch_size,
                patience=patience,
                weight_decay=weight_decay,
                seed=seed,
                device=device,
                deterministic=True,
                semantic_event_enabled=chronaris,
                learnable_semantic_queries=chronaris,
                heartbeat_interval_s=30.0,
            ),
            augmentation_policy=AugmentationPolicy(),
            chronaris_fusion_kind="safe_lag" if chronaris else "multiscale",
            chronaris_mechanism_enabled=chronaris,
            chronaris_explicit_shift_enabled=chronaris,
            chronaris_explicit_shift_weight=0.1 if chronaris else 0.0,
            chronaris_event_pair_weight=0.1 if chronaris else 0.0,
            resume=resume,
        )
        encoder, _heads, loaded_normalizer, payload = (
            load_common_pretraining_checkpoint(
                result.best_checkpoint_path,
                device=device,
            )
        )
        checkpoint_hash = sha256_file(result.best_checkpoint_path)
        adapters[method] = TrainedFusionAdapter(
            encoder=encoder,
            normalizer=loaded_normalizer,
            fold_id=fold.fold_id,
            checkpoint_sha256=checkpoint_hash,
        )
        hashes[method] = checkpoint_hash
        rows.append(
            {
                "method": method,
                "status": result.status,
                "best_epoch": result.best_epoch,
                "completed_epochs": result.completed_epochs,
                "validation_self_supervised_loss": (
                    result.best_public_selection_loss
                ),
                "parameter_count": result.parameter_count,
                "training_elapsed_s": result.training_elapsed_s,
                "checkpoint_path": result.best_checkpoint_path,
                "checkpoint_sha256": checkpoint_hash,
                "canonical_training_state_sha256": payload[
                    "canonical_training_state_sha256"
                ],
                "protocol_sha256": result.protocol_sha256,
                "training_device": payload["training_device_history"][-1],
            }
        )
    naive_path = Path(output_root) / "checkpoints/naive_time_sync/best.pt"
    if naive_path.is_file() and resume:
        naive = load_naive_time_sync_checkpoint(naive_path)
        naive_status = "resumed"
    else:
        naive = NaiveTimeSyncEncoder().fit_from_batch_provider(
            provider,
            train_sample_ids=fold.train_sample_ids,
            held_out_sample_ids=(
                fold.validation_sample_ids + fold.held_out_sample_ids
            ),
            normalizer=normalizer,
            batch_size=batch_size,
        )
        save_naive_time_sync_checkpoint(naive_path, encoder=naive)
        naive_status = "completed"
    checkpoint_hash = sha256_file(naive_path)
    adapters["naive_time_sync"] = NaiveTimeSyncFusionAdapter(
        encoder=naive,
        fold_id=fold.fold_id,
        checkpoint_sha256=checkpoint_hash,
    )
    hashes["naive_time_sync"] = checkpoint_hash
    rows.append(
        {
            "method": "naive_time_sync",
            "status": naive_status,
            "best_epoch": None,
            "completed_epochs": None,
            "validation_self_supervised_loss": None,
            "parameter_count": 0,
            "training_elapsed_s": 0.0,
            "checkpoint_path": str(naive_path),
            "checkpoint_sha256": checkpoint_hash,
            "canonical_training_state_sha256": None,
            "protocol_sha256": None,
            "training_device": "cpu_unsupervised_transform",
        }
    )
    return adapters, hashes, rows
