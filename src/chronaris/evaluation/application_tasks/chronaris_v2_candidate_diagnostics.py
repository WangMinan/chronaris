"""Task-independent gate diagnostics for trained Chronaris v2 candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import torch

from chronaris.evaluation.representation_diagnostics import (
    fit_feature_recovery_probe,
    fit_fidelity_probe,
    representation_health,
)
from chronaris.modeling.training import (
    TaskIndependentCandidateEvidence,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
)
from chronaris.modeling.training.pretraining_encoders import TrainableFusionEncoder
from chronaris.modeling.fusion_encoders.semantic_groups import (
    build_vehicle_semantic_group_map,
)
from chronaris.representation import (
    DualStreamObservationBatch,
    masked_mean_pool,
    select_observation_batch,
)


V1_CLOCK_OFFSET_MAE_S = 0.8883
V1_RESPONSE_LAG_MAE_S = 7.4859


@dataclass(frozen=True, slots=True)
class CandidateDiagnosticResult:
    evidence: TaskIndependentCandidateEvidence
    details: Mapping[str, object]


def diagnose_chronaris_candidate(
    *,
    candidate_checkpoint: str | Path,
    physiology_reference_checkpoint: str | Path,
    vehicle_reference_checkpoint: str | Path,
    batch: DualStreamObservationBatch | None = None,
    train_sample_ids: tuple[str, ...] = (),
    validation_sample_ids: tuple[str, ...] = (),
    train_batch: DualStreamObservationBatch | None = None,
    validation_batch: DualStreamObservationBatch | None = None,
    fold_id: str,
    device: str = "cuda",
    mechanism_sample_count: int = 5,
    candidate_id: str | None = None,
    v1_clock_offset_mae_s: float = V1_CLOCK_OFFSET_MAE_S,
    v1_response_lag_mae_s: float = V1_RESPONSE_LAG_MAE_S,
) -> CandidateDiagnosticResult:
    """Evaluate one candidate without task labels or locked-confirmation data."""

    candidate_path = Path(candidate_checkpoint)
    candidate_encoder, _heads, candidate_normalizer, payload = (
        load_common_pretraining_checkpoint(candidate_path, device=device)
    )
    if payload.get("format") not in {
        "chronaris.common_pretraining_checkpoint.v1",
        "chronaris.common_pretraining_checkpoint.v2",
    }:
        raise ValueError("candidate diagnostics require a Chronaris checkpoint")
    if any(
        bool(payload.get(key))
        for key in (
            "label_used_for_encoder_training",
            "simulation_oracle_opened",
            "locked_test_opened",
        )
    ):
        raise ValueError("candidate checkpoint contains forbidden selection evidence")
    physiology_adapter = _adapter(
        physiology_reference_checkpoint,
        fold_id=fold_id,
        device=device,
    )
    vehicle_adapter = _adapter(
        vehicle_reference_checkpoint,
        fold_id=fold_id,
        device=device,
    )
    candidate_adapter = TrainedFusionAdapter(
        encoder=candidate_encoder,
        normalizer=candidate_normalizer,
        fold_id=fold_id,
        checkpoint_sha256=_sha256_file(candidate_path),
    )
    if batch is not None and (train_batch is not None or validation_batch is not None):
        raise ValueError("provide either a combined batch or separate train/validation batches")
    if batch is not None:
        train = select_observation_batch(batch, train_sample_ids)
        validation = select_observation_batch(batch, validation_sample_ids)
    elif train_batch is not None and validation_batch is not None:
        train, validation = train_batch, validation_batch
    else:
        raise ValueError("candidate diagnostics require train and validation observations")
    candidate_train = candidate_adapter(train)
    candidate_validation = candidate_adapter(validation)
    physiology_train_representation = physiology_adapter(train)
    physiology_validation_representation = physiology_adapter(validation)
    vehicle_train_representation = vehicle_adapter(train)
    vehicle_validation_representation = vehicle_adapter(validation)
    physiology_probe = fit_fidelity_probe(
        candidate_train,
        physiology_train_representation,
        candidate_validation,
        physiology_validation_representation,
    )
    vehicle_probe = fit_fidelity_probe(
        candidate_train,
        vehicle_train_representation,
        candidate_validation,
        vehicle_validation_representation,
    )
    normalized_train = candidate_normalizer.transform(train)
    normalized_validation = candidate_normalizer.transform(validation)
    physiology_train_target = _stream_feature_targets(
        normalized_train,
        stream_name="physiology",
    )
    physiology_validation_target = _stream_feature_targets(
        normalized_validation,
        stream_name="physiology",
    )
    vehicle_groups = (
        candidate_encoder.backbone.semantic_group_map.groups
        if hasattr(candidate_encoder.backbone, "semantic_group_map")
        else build_vehicle_semantic_group_map(
            tuple(payload["vehicle_feature_names"]),
            field_labels=dict(payload.get("vehicle_field_labels", ())),
        ).groups
    )
    vehicle_train_target = _stream_feature_targets(
        normalized_train,
        stream_name="vehicle",
        groups=vehicle_groups,
    )
    vehicle_validation_target = _stream_feature_targets(
        normalized_validation,
        stream_name="vehicle",
        groups=vehicle_groups,
    )
    candidate_physiology_feature_probe = fit_feature_recovery_probe(
        candidate_train,
        physiology_train_target,
        candidate_validation,
        physiology_validation_target,
        target_name="physiology_observed_features",
    )
    reference_physiology_feature_probe = fit_feature_recovery_probe(
        physiology_train_representation,
        physiology_train_target,
        physiology_validation_representation,
        physiology_validation_target,
        target_name="physiology_observed_features",
    )
    candidate_vehicle_feature_probe = fit_feature_recovery_probe(
        candidate_train,
        vehicle_train_target,
        candidate_validation,
        vehicle_validation_target,
        target_name="vehicle_semantic_groups",
    )
    reference_vehicle_feature_probe = fit_feature_recovery_probe(
        vehicle_train_representation,
        vehicle_train_target,
        vehicle_validation_representation,
        vehicle_validation_target,
        target_name="vehicle_semantic_groups",
    )
    health = representation_health(candidate_validation)
    mechanism_ids = validation.sample_ids[:mechanism_sample_count]
    mechanism_batch = select_observation_batch(validation, mechanism_ids)
    clock_offset_mae = _clock_offset_recovery_mae(
        candidate_adapter,
        mechanism_batch,
    )
    response_lag_mae = _response_lag_recovery_mae(
        candidate_encoder,
        candidate_normalizer,
        mechanism_batch,
        device=device,
    )
    future_invariance = _future_invariance_passed(
        candidate_encoder,
        candidate_normalizer,
        mechanism_batch,
        device=device,
    )
    lag_mask_passed = _lag_mask_passed(
        candidate_encoder,
        candidate_normalizer,
        mechanism_batch,
        device=device,
    )
    expected_pool = masked_mean_pool(
        candidate_validation.sequence_embedding,
        candidate_validation.valid_mask,
    )
    pooling_passed = bool(
        torch.allclose(
            candidate_validation.pooled_embedding,
            expected_pool,
            atol=1e-7,
            rtol=1e-6,
        )
        and torch.count_nonzero(
            candidate_validation.sequence_embedding[
                ~candidate_validation.valid_mask
            ]
        )
        == 0
    )
    vehicle_fidelity = _recovery_ratio(
        candidate_vehicle_feature_probe.normalized_rmse,
        reference_vehicle_feature_probe.normalized_rmse,
    )
    physiology_fidelity = _recovery_ratio(
        candidate_physiology_feature_probe.normalized_rmse,
        reference_physiology_feature_probe.normalized_rmse,
    )
    evidence = TaskIndependentCandidateEvidence(
        candidate_id=(
            candidate_id
            if candidate_id is not None
            else str(payload["candidate_config"]["candidate_id"])
        ),
        public_self_supervised_validation_loss=float(
            payload["best_public_selection_loss"]
        ),
        vehicle_fidelity_ratio=vehicle_fidelity,
        physiology_fidelity_ratio=physiology_fidelity,
        worst_fold_fidelity_ratio=min(vehicle_fidelity, physiology_fidelity),
        effective_rank=health.effective_rank,
        near_zero_variance_fraction=health.near_zero_variance_fraction,
        clock_offset_mae_s=clock_offset_mae,
        response_lag_mae_s=response_lag_mae,
        v1_clock_offset_mae_s=v1_clock_offset_mae_s,
        v1_response_lag_mae_s=v1_response_lag_mae_s,
        causal_future_invariance_passed=future_invariance,
        invalid_query_pooling_passed=pooling_passed,
        lag_mask_passed=lag_mask_passed,
        parameter_count=int(payload["parameter_count"]),
    )
    return CandidateDiagnosticResult(
        evidence=evidence,
        details={
            "candidate_checkpoint": str(candidate_path),
            "candidate_checkpoint_sha256": _sha256_file(candidate_path),
            "physiology_fidelity_probe": physiology_probe.to_dict(),
            "vehicle_fidelity_probe": vehicle_probe.to_dict(),
            "candidate_physiology_feature_probe": (
                candidate_physiology_feature_probe.to_dict()
            ),
            "reference_physiology_feature_probe": (
                reference_physiology_feature_probe.to_dict()
            ),
            "candidate_vehicle_semantic_probe": (
                candidate_vehicle_feature_probe.to_dict()
            ),
            "reference_vehicle_semantic_probe": (
                reference_vehicle_feature_probe.to_dict()
            ),
            "representation_health": health.to_dict(),
            "mechanism_sample_ids": list(mechanism_ids),
            "clock_offset_recovery_mae_s": clock_offset_mae,
            "response_lag_recovery_mae_s": response_lag_mae,
            "causal_future_invariance_passed": future_invariance,
            "invalid_query_pooling_passed": pooling_passed,
            "lag_mask_passed": lag_mask_passed,
            "task_labels_opened": False,
            "simulation_oracle_opened": False,
            "locked_test_opened": False,
        },
    )


diagnose_v2_candidate = diagnose_chronaris_candidate


def _adapter(path, *, fold_id, device):
    resolved = Path(path)
    encoder, _heads, normalizer, payload = load_common_pretraining_checkpoint(
        resolved,
        device=device,
    )
    if bool(payload.get("label_used_for_encoder_training")):
        raise ValueError("single-stream fidelity reference used task labels")
    return TrainedFusionAdapter(
        encoder=encoder,
        normalizer=normalizer,
        fold_id=fold_id,
        checkpoint_sha256=_sha256_file(resolved),
    )


def _stream_feature_targets(batch, *, stream_name, groups=None):
    values = getattr(batch, f"{stream_name}_values")
    point_mask = getattr(batch, f"{stream_name}_point_mask")
    feature_mask = getattr(batch, f"{stream_name}_feature_mask")
    valid = feature_mask & point_mask.unsqueeze(-1)
    if groups is None:
        numerator = (values * valid.to(values.dtype)).sum(dim=1)
        denominator = valid.sum(dim=1).clamp_min(1).to(values.dtype)
        return numerator / denominator
    targets = []
    for indices in groups.values():
        index = torch.tensor(indices, dtype=torch.long, device=values.device)
        group_values = values.index_select(-1, index)
        group_valid = valid.index_select(-1, index)
        numerator = (group_values * group_valid.to(values.dtype)).sum(dim=(1, 2))
        denominator = group_valid.sum(dim=(1, 2)).clamp_min(1).to(values.dtype)
        targets.append(numerator / denominator)
    return torch.stack(targets, dim=-1)


def _recovery_ratio(candidate_normalized_rmse, reference_normalized_rmse):
    candidate = max(float(candidate_normalized_rmse), 1e-12)
    reference = max(float(reference_normalized_rmse), 1e-12)
    return min(reference / candidate, 1e6)


def _clock_offset_recovery_mae(adapter, batch):
    offsets = (-1.0, -0.5, 0.0, 0.5, 1.0)
    reference = adapter(batch).pooled_embedding
    errors = []
    for injected in offsets:
        shifted = replace(
            batch,
            vehicle_timestamps_s=batch.vehicle_timestamps_s + injected,
        )
        scores = []
        for correction in offsets:
            corrected = replace(
                shifted,
                vehicle_timestamps_s=shifted.vehicle_timestamps_s + correction,
            )
            representation = adapter(corrected).pooled_embedding
            scores.append(float((representation - reference).square().mean()))
        best_correction = offsets[min(range(len(scores)), key=scores.__getitem__)]
        estimated_injected = -best_correction
        errors.append(abs(estimated_injected - injected))
    return sum(errors) / len(errors)


def _response_lag_recovery_mae(encoder, normalizer, batch, *, device):
    normalized = normalizer.transform(batch)
    probe = _encode(encoder, normalized, device=device)
    scale_count = probe.fusion_output.scale_gate_weights.shape[-1]
    centers = (
        (2.5, 10.0, 22.5)
        if scale_count == 3
        else (1.0, 3.5, 7.5, 15.0, 25.0)
    )
    labels = torch.tensor(
        [centers[index % len(centers)] for index in range(len(batch.sample_ids))],
        dtype=batch.vehicle_timestamps_s.dtype,
    )
    shifted = replace(
        batch,
        vehicle_timestamps_s=batch.vehicle_timestamps_s + labels.unsqueeze(-1),
    )
    encoded = _encode(encoder, normalizer.transform(shifted), device=device)
    gates = encoded.fusion_output.scale_gate_weights
    valid = encoded.fusion_output.scale_available_mask
    weights = gates * valid.to(gates.dtype)
    pooled = weights.sum(dim=1)
    center_tensor = gates.new_tensor(centers)
    estimate = (pooled * center_tensor).sum(dim=-1) / pooled.sum(dim=-1).clamp_min(
        1e-8
    )
    return float((estimate.cpu() - labels).abs().mean())


def _future_invariance_passed(encoder, normalizer, batch, *, device):
    cutoff = float(batch.query_timestamps_s[0, batch.query_timestamps_s.shape[1] // 2])
    changed_values = batch.vehicle_values.clone()
    future = batch.vehicle_timestamps_s > cutoff
    changed_values[future] = changed_values[future] * 1.5 + 7.0
    changed = replace(batch, vehicle_values=changed_values)
    original_encoding = _encode(encoder, normalizer.transform(batch), device=device)
    changed_encoding = _encode(encoder, normalizer.transform(changed), device=device)
    historical = batch.query_timestamps_s <= cutoff
    return bool(
        torch.allclose(
            original_encoding.sequence_embedding.cpu()[historical],
            changed_encoding.sequence_embedding.cpu()[historical],
            atol=1e-5,
            rtol=1e-5,
        )
    )


def _lag_mask_passed(encoder, normalizer, batch, *, device):
    encoding = _encode(encoder, normalizer.transform(batch), device=device)
    return all(
        bool(torch.count_nonzero(weights.masked_select(~mask)) == 0)
        for weights, mask in zip(
            encoding.fusion_output.attention_weights,
            encoding.fusion_output.lag_masks,
            strict=True,
        )
    )


def _encode(encoder: TrainableFusionEncoder, batch, *, device):
    from chronaris.modeling.fusion_encoders.single_stream import (
        move_observation_batch,
    )

    encoder.eval()
    with torch.inference_mode():
        output = encoder(
            move_observation_batch(batch, device=device)
        ).auxiliary["chronaris_encoding"]
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
