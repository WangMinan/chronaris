"""Early-stopped, public-pretext-only encoder candidate screening."""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import nn
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.chronaris_auxiliary import lag_aware_alignment_loss
from chronaris.modeling.training.pretext import (
    CommonPretextHeadBundle,
    CommonPretextWeights,
    pretext_loss_terms_to_rows,
)
from chronaris.modeling.training.pretraining_encoders import (
    TRAINABLE_FUSION_METHODS,
    EncoderCandidateConfig,
    build_trainable_fusion_encoder,
)
from chronaris.modeling.training.rng import (
    canonical_training_state_sha256,
    capture_rng_state,
    isolated_training_rng,
    restore_rng_state,
)
from chronaris.modeling.training.transfer_initialization import (
    describe_transfer_source,
    initialize_encoder_from_transfer_source,
)
from chronaris.representation import (
    AugmentationPolicy,
    DualStreamObservationBatch,
    FoldLineage,
    TrainOnlyRobustNormalizer,
    apply_augmentation_realizations,
    build_batch_augmentation_realizations,
    build_common_pretext_targets,
    build_lag_discrimination_inputs,
    move_common_pretext_targets,
    select_observation_batch,
)
from chronaris.representation.contracts import FUSION_OUTPUT_DIM, RepresentationContractError


PUBLIC_SELECTION_WEIGHTS = {
    "masked_reconstruction": 0.50,
    "short_horizon_prediction": 0.25,
    "lag_discrimination": 0.25,
}

@dataclass(frozen=True, slots=True)
class CandidateScreenConfig:
    max_epochs: int = 50
    batch_size: int = 128
    patience: int = 8
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    seed: int = 17
    minimum_delta: float = 0.0
    device: str = "cpu"
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.max_epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("candidate screen epoch/batch/patience must be positive")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("candidate screen optimizer configuration is invalid")
        if self.minimum_delta < 0:
            raise ValueError("candidate screen minimum delta must be non-negative")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("candidate screen device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("candidate screen requested unavailable CUDA device")


@dataclass(frozen=True, slots=True)
class CandidateScreenResult:
    method_name: str
    candidate_id: str
    status: str
    best_checkpoint_path: str
    last_checkpoint_path: str
    protocol_sha256: str
    best_epoch: int
    completed_epochs: int
    stopped_early: bool
    best_validation_losses: Mapping[str, float]
    best_public_selection_loss: float
    training_elapsed_s: float
    parameter_count: int
    epoch_rows: tuple[Mapping[str, object], ...]


def train_pretext_candidate(
    method_name: str,
    *,
    candidate: EncoderCandidateConfig,
    batch: DualStreamObservationBatch | None,
    fold: FoldLineage,
    physiology_feature_names: tuple[str, ...],
    vehicle_feature_names: tuple[str, ...],
    vehicle_field_labels: tuple[tuple[str, str], ...],
    normalizer: TrainOnlyRobustNormalizer,
    output_root: str | Path,
    config: CandidateScreenConfig | None = None,
    augmentation_policy: AugmentationPolicy | None = None,
    batch_provider: Callable[[Sequence[str]], DualStreamObservationBatch] | None = None,
    initialization_checkpoint: str | Path | None = None,
    chronaris_fusion_kind: str = "multiscale",
    chronaris_lag_aware_weight: float = 0.0,
    include_candidate_subdirectory: bool = True,
    resume: bool = True,
) -> CandidateScreenResult:
    resolved = config or CandidateScreenConfig()
    with isolated_training_rng(
        resolved.seed,
        deterministic=resolved.deterministic,
    ):
        return _train_pretext_candidate(
            method_name,
            candidate=candidate,
            batch=batch,
            fold=fold,
            physiology_feature_names=physiology_feature_names,
            vehicle_feature_names=vehicle_feature_names,
            vehicle_field_labels=vehicle_field_labels,
            normalizer=normalizer,
            output_root=output_root,
            config=resolved,
            augmentation_policy=augmentation_policy,
            batch_provider=batch_provider,
            initialization_checkpoint=initialization_checkpoint,
            chronaris_fusion_kind=chronaris_fusion_kind,
            chronaris_lag_aware_weight=chronaris_lag_aware_weight,
            include_candidate_subdirectory=include_candidate_subdirectory,
            resume=resume,
        )


def _train_pretext_candidate(
    method_name: str,
    *,
    candidate: EncoderCandidateConfig,
    batch: DualStreamObservationBatch | None,
    fold: FoldLineage,
    physiology_feature_names: tuple[str, ...],
    vehicle_feature_names: tuple[str, ...],
    vehicle_field_labels: tuple[tuple[str, str], ...],
    normalizer: TrainOnlyRobustNormalizer,
    output_root: str | Path,
    config: CandidateScreenConfig | None = None,
    augmentation_policy: AugmentationPolicy | None = None,
    batch_provider: Callable[[Sequence[str]], DualStreamObservationBatch] | None = None,
    initialization_checkpoint: str | Path | None = None,
    chronaris_fusion_kind: str = "multiscale",
    chronaris_lag_aware_weight: float = 0.0,
    include_candidate_subdirectory: bool = True,
    resume: bool = True,
) -> CandidateScreenResult:
    """Train one frozen candidate without opening task labels or simulation truth."""

    if method_name not in TRAINABLE_FUSION_METHODS:
        raise ValueError(f"unsupported trainable method: {method_name}")
    if chronaris_lag_aware_weight < 0:
        raise ValueError("Chronaris lag-aware weight must be non-negative")
    if method_name != "chronaris" and (
        chronaris_fusion_kind != "multiscale" or chronaris_lag_aware_weight > 0
    ):
        raise ValueError("Chronaris-specific options require method_name='chronaris'")
    if not fold.train_sample_ids or not fold.validation_sample_ids:
        raise ValueError("candidate screen requires non-empty train and validation roles")
    if (batch is None) == (batch_provider is None):
        raise ValueError("provide exactly one of batch or batch_provider")
    resolved = config or CandidateScreenConfig()
    policy = augmentation_policy or AugmentationPolicy()
    root = Path(output_root) / method_name
    if include_candidate_subdirectory:
        root /= candidate.candidate_id
    best_path = root / "best.pt"
    last_path = root / "last.pt"
    transfer_source = (
        describe_transfer_source(
            initialization_checkpoint,
            expected_method=method_name,
            expected_seed=resolved.seed,
        ).to_dict()
        if initialization_checkpoint is not None
        else None
    )
    protocol_hash = _protocol_hash(
        method_name=method_name,
        candidate=asdict(candidate),
        config=asdict(resolved),
        augmentation_policy=asdict(policy),
        fold=fold.to_dict(),
        normalizer=normalizer.to_manifest(),
        physiology_feature_names=physiology_feature_names,
        vehicle_feature_names=vehicle_feature_names,
        vehicle_field_labels=vehicle_field_labels,
        data_access_mode="lazy_batch_provider" if batch_provider else "materialized_batch",
        transfer_source=transfer_source,
        chronaris_fusion_kind=chronaris_fusion_kind,
        chronaris_lag_aware_weight=chronaris_lag_aware_weight,
    )
    resume_payload = None
    if resume and last_path.exists():
        last_payload = _load_payload(last_path)
        if last_payload.get("format") != "chronaris.common_pretraining_checkpoint.v2":
            raise RepresentationContractError(
                "v1 pretraining checkpoints are inference-only and cannot resume"
            )
        if (
            last_payload.get("protocol_sha256") != protocol_hash
            and not _checkpoint_is_semantically_compatible(
                last_payload,
                candidate=candidate,
                config=resolved,
                policy=policy,
                fold=fold,
                normalizer=normalizer,
                physiology_feature_names=physiology_feature_names,
                vehicle_feature_names=vehicle_feature_names,
                vehicle_field_labels=vehicle_field_labels,
                transfer_source=transfer_source,
                chronaris_fusion_kind=chronaris_fusion_kind,
                chronaris_lag_aware_weight=chronaris_lag_aware_weight,
            )
        ):
            raise RepresentationContractError(
                f"candidate screen checkpoint protocol changed for {method_name}/{candidate.candidate_id}"
            )
        if last_payload.get("training_status") == "completed":
            payload = _load_payload(best_path)
            return _result(payload, best_path, last_path, status="resumed")
        resume_payload = last_payload

    encoder = build_trainable_fusion_encoder(
        method_name,
        physiology_feature_names=physiology_feature_names,
        vehicle_feature_names=vehicle_feature_names,
        vehicle_field_labels=vehicle_field_labels,
        candidate_config=candidate,
        chronaris_fusion_kind=chronaris_fusion_kind,
    ).to(resolved.device)
    heads = CommonPretextHeadBundle(
        representation_dim=FUSION_OUTPUT_DIM,
        target_feature_count=len(physiology_feature_names) + len(vehicle_feature_names),
    ).to(resolved.device)
    optimizer = torch.optim.AdamW(
        (*encoder.parameters(), *heads.parameters()),
        lr=candidate.learning_rate,
        weight_decay=resolved.weight_decay,
    )
    transfer_initialization = None
    if resume_payload is not None:
        encoder.load_state_dict(resume_payload["encoder_state_dict"], strict=True)
        heads.load_state_dict(resume_payload["head_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        restore_rng_state(resume_payload["rng_state"])
        transfer_initialization = resume_payload.get("transfer_initialization")
    elif initialization_checkpoint is not None:
        transfer_initialization = initialize_encoder_from_transfer_source(
            encoder,
            initialization_checkpoint,
            expected_method=method_name,
            expected_seed=resolved.seed,
        ).to_dict()
    best_score = (
        float(resume_payload["best_public_selection_loss"])
        if resume_payload is not None
        else float("inf")
    )
    best_epoch = int(resume_payload["best_epoch"]) if resume_payload is not None else 0
    best_losses: dict[str, float] = (
        dict(resume_payload["best_validation_losses"])
        if resume_payload is not None
        else {}
    )
    epoch_rows: list[Mapping[str, object]] = (
        list(resume_payload["epoch_rows"]) if resume_payload is not None else []
    )
    training_rows = list(resume_payload.get("training_rows", [])) if resume_payload else []
    augmentation_rows = (
        list(resume_payload.get("augmentation_rows", [])) if resume_payload else []
    )
    epochs_without_improvement = (
        int(epoch_rows[-1]["epochs_without_improvement"]) if epoch_rows else 0
    )
    step_count = int(resume_payload["step_count"]) if resume_payload is not None else 0
    elapsed_offset = (
        float(resume_payload["training_elapsed_s"]) if resume_payload is not None else 0.0
    )
    device_history = list(
        resume_payload.get(
            "training_device_history",
            [resume_payload.get("config", {}).get("device", "cpu")],
        )
        if resume_payload is not None
        else [resolved.device]
    )
    if device_history[-1] != resolved.device:
        device_history.append(resolved.device)
    start_epoch = (
        int(resume_payload["completed_epochs"]) + 1 if resume_payload is not None else 1
    )
    started = time.perf_counter()
    for epoch in range(start_epoch, resolved.max_epochs + 1):
        encoder.train()
        heads.train()
        train_totals = _empty_loss_totals()
        gradient_norms = []
        for sample_ids in _batch_ids(fold.train_sample_ids, resolved.batch_size):
            raw = _load_batch(batch, batch_provider, sample_ids)
            normalized = normalizer.transform(raw)
            plans = build_batch_augmentation_realizations(
                sample_ids,
                epoch=epoch,
                global_seed=resolved.seed,
                policy=policy,
            )
            augmented = apply_augmentation_realizations(normalized, plans, policy=policy)
            targets = build_common_pretext_targets(normalized, augmented)
            lag_inputs = build_lag_discrimination_inputs(
                augmented.batch,
                augmented.augmentation_ids,
            )
            positive_batch = move_observation_batch(
                augmented.batch,
                device=resolved.device,
            )
            negative_batch = move_observation_batch(
                lag_inputs.negative_batch,
                device=resolved.device,
            )
            targets = move_common_pretext_targets(
                targets,
                device=resolved.device,
            )
            optimizer.zero_grad(set_to_none=True)
            positive = encoder(
                positive_batch,
                compute_chronaris_diagnostics=(
                    method_name == "chronaris" and chronaris_lag_aware_weight > 0
                ),
            )
            output = heads(
                positive.sequence_embedding,
                encoder(negative_batch).sequence_embedding,
                targets,
                weights=CommonPretextWeights(),
            )
            total_loss = output.total_loss
            if chronaris_lag_aware_weight > 0:
                alignment = positive.auxiliary.get("alignment_output")
                if alignment is None:
                    raise RepresentationContractError(
                        "enabled lag-aware loss requires Chronaris alignment output"
                    )
                total_loss = total_loss + chronaris_lag_aware_weight * (
                    lag_aware_alignment_loss(
                        alignment,
                        min_lag_s=0.0,
                        max_lag_s=15.0,
                    ).loss
                )
            total_loss.backward()
            parameters = tuple((*encoder.parameters(), *heads.parameters()))
            gradient_norms.append(
                float(nn.utils.clip_grad_norm_(parameters, resolved.gradient_clip_norm))
            )
            optimizer.step()
            global_step = step_count + 1
            training_rows.extend(
                {
                    "method_name": method_name,
                    "candidate_id": candidate.candidate_id,
                    "epoch": epoch,
                    "step": global_step,
                    "batch_sample_ids": list(sample_ids),
                    "augmentation_ids": list(augmented.augmentation_ids),
                    "gradient_norm_before_clip": gradient_norms[-1],
                    **row,
                }
                for row in pretext_loss_terms_to_rows(output.terms)
            )
            augmentation_rows.extend(
                {
                    "method_name": method_name,
                    "candidate_id": candidate.candidate_id,
                    "epoch": epoch,
                    "step": global_step,
                    **row.to_dict(),
                }
                for row in augmented.audit_rows
            )
            _accumulate_loss_terms(train_totals, output.terms)
            step_count += 1
        train_losses = _finalize_loss_totals(train_totals)
        validation_losses = _evaluate_public_losses(
            encoder=encoder,
            heads=heads,
            batch=batch,
            batch_provider=batch_provider,
            sample_ids=fold.validation_sample_ids,
            batch_size=resolved.batch_size,
            normalizer=normalizer,
            policy=policy,
            seed=resolved.seed,
            device=resolved.device,
        )
        score = _public_selection_loss(validation_losses)
        improved = score < best_score - resolved.minimum_delta
        if improved:
            best_score = score
            best_epoch = epoch
            best_losses = dict(validation_losses)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "method_name": method_name,
            "candidate_id": candidate.candidate_id,
            "epoch": epoch,
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "public_selection_loss": score,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "mean_gradient_norm_before_clip": sum(gradient_norms) / len(gradient_norms),
        }
        epoch_rows.append(row)
        payload = _checkpoint_payload(
            method_name=method_name,
            candidate=candidate,
            config=resolved,
            policy=policy,
            fold=fold,
            normalizer=normalizer,
            encoder=encoder,
            heads=heads,
            optimizer=optimizer,
            protocol_hash=protocol_hash,
            physiology_feature_names=physiology_feature_names,
            vehicle_feature_names=vehicle_feature_names,
            vehicle_field_labels=vehicle_field_labels,
            best_epoch=best_epoch,
            completed_epochs=epoch,
            best_losses=best_losses,
            best_score=best_score,
            stopped_early=False,
            step_count=step_count,
            epoch_rows=epoch_rows,
            elapsed=elapsed_offset + time.perf_counter() - started,
            transfer_source=transfer_source,
            transfer_initialization=transfer_initialization,
            device_history=device_history,
            chronaris_fusion_kind=chronaris_fusion_kind,
            chronaris_lag_aware_weight=chronaris_lag_aware_weight,
            training_rows=training_rows,
            augmentation_rows=augmentation_rows,
        )
        _atomic_save(last_path, payload)
        if improved:
            _atomic_save(best_path, payload)
        if epochs_without_improvement >= resolved.patience:
            break
    elapsed = elapsed_offset + time.perf_counter() - started
    stopped_early = len(epoch_rows) < resolved.max_epochs
    final_payload = _load_payload(best_path)
    final_payload.update(
        {
            "training_status": "completed",
            "training_elapsed_s": elapsed,
            "completed_epochs": len(epoch_rows),
            "stopped_early": stopped_early,
            "epoch_rows": epoch_rows,
            "step_count": step_count,
            "config": asdict(resolved),
            "training_device_history": device_history,
            "training_rows": training_rows,
            "augmentation_rows": augmentation_rows,
        }
    )
    _atomic_save(best_path, final_payload)
    last_payload = _load_payload(last_path)
    last_payload.update(final_payload | {
        "encoder_state_dict": last_payload["encoder_state_dict"],
        "head_state_dict": last_payload["head_state_dict"],
        "optimizer_state_dict": last_payload["optimizer_state_dict"],
    })
    _atomic_save(last_path, last_payload)
    return _result(final_payload, best_path, last_path, status="completed")


def _evaluate_public_losses(
    *, encoder, heads, batch, batch_provider, sample_ids, batch_size, normalizer, policy, seed, device
) -> Mapping[str, float]:
    encoder.eval()
    heads.eval()
    totals = _empty_loss_totals()
    with torch.inference_mode():
        for ids in _batch_ids(sample_ids, batch_size):
            raw = _load_batch(batch, batch_provider, ids)
            normalized = normalizer.transform(raw)
            plans = build_batch_augmentation_realizations(
                ids, epoch=0, global_seed=seed, policy=policy
            )
            augmented = apply_augmentation_realizations(normalized, plans, policy=policy)
            targets = build_common_pretext_targets(normalized, augmented)
            lag_inputs = build_lag_discrimination_inputs(
                augmented.batch, augmented.augmentation_ids
            )
            positive_batch = move_observation_batch(augmented.batch, device=device)
            negative_batch = move_observation_batch(
                lag_inputs.negative_batch,
                device=device,
            )
            targets = move_common_pretext_targets(targets, device=device)
            output = heads(
                encoder(positive_batch).sequence_embedding,
                encoder(negative_batch).sequence_embedding,
                targets,
                weights=CommonPretextWeights(),
            )
            _accumulate_loss_terms(totals, output.terms)
    return _finalize_loss_totals(totals)


def _empty_loss_totals() -> dict[str, list[float]]:
    return {name: [0.0, 0.0] for name in PUBLIC_SELECTION_WEIGHTS}


def _accumulate_loss_terms(totals, terms) -> None:
    for term in terms:
        if term.raw_loss is not None and term.count > 0:
            totals[term.term_name][0] += float(term.raw_loss.detach()) * term.count
            totals[term.term_name][1] += term.count


def _finalize_loss_totals(totals) -> Mapping[str, float]:
    losses = {}
    for name, (loss_sum, count) in totals.items():
        if count <= 0:
            raise RepresentationContractError(f"candidate screen loss unavailable: {name}")
        losses[name] = loss_sum / count
    return losses


def _public_selection_loss(losses: Mapping[str, float]) -> float:
    return sum(PUBLIC_SELECTION_WEIGHTS[name] * losses[name] for name in PUBLIC_SELECTION_WEIGHTS)


def _load_batch(batch, provider, sample_ids):
    loaded = provider(sample_ids) if provider is not None else select_observation_batch(batch, sample_ids)
    if tuple(loaded.sample_ids) != tuple(sample_ids):
        raise RepresentationContractError("candidate screen batch provider changed sample order")
    return loaded


def _batch_ids(sample_ids, batch_size):
    values = tuple(sample_ids)
    return tuple(values[index : index + batch_size] for index in range(0, len(values), batch_size))


def _checkpoint_payload(**values):
    encoder = values.pop("encoder")
    heads = values.pop("heads")
    optimizer = values.pop("optimizer")
    rng_state = capture_rng_state()
    payload = {
        "format": "chronaris.common_pretraining_checkpoint.v2",
        "training_status": "running",
        "method_name": values["method_name"],
        "protocol_sha256": values["protocol_hash"],
        "encoder_state_dict": encoder.state_dict(),
        "head_state_dict": heads.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "normalizer": dict(values["normalizer"].to_manifest()),
        "fold": values["fold"].to_dict(),
        "config": asdict(values["config"]),
        "augmentation_policy": asdict(values["policy"]),
        "candidate_config": asdict(values["candidate"]),
        "physiology_feature_names": list(values["physiology_feature_names"]),
        "vehicle_feature_names": list(values["vehicle_feature_names"]),
        "vehicle_field_labels": [list(value) for value in values["vehicle_field_labels"]],
        "encoder_manifest": dict(encoder.config_manifest()),
        "seed": values["config"].seed,
        "epoch": values["completed_epochs"],
        "best_epoch": values["best_epoch"],
        "completed_epochs": values["completed_epochs"],
        "best_validation_losses": dict(values["best_losses"]),
        "best_public_selection_loss": values["best_score"],
        "stopped_early": values["stopped_early"],
        "step_count": values["step_count"],
        "training_elapsed_s": values["elapsed"],
        "parameter_count": encoder.parameter_count,
        "head_parameter_count": sum(p.numel() for p in heads.parameters()),
        "epoch_rows": list(values["epoch_rows"]),
        "training_rows": list(values["training_rows"]),
        "augmentation_rows": list(values["augmentation_rows"]),
        "label_used_for_encoder_training": False,
        "simulation_oracle_opened": False,
        "selection_uses_public_pretext_only": True,
        "selection_weights": dict(PUBLIC_SELECTION_WEIGHTS),
        "transfer_source": values.get("transfer_source"),
        "transfer_initialization": values.get("transfer_initialization"),
        "training_device_history": list(values["device_history"]),
        "augmentation_device": "cpu",
        "chronaris_fusion_kind": values["chronaris_fusion_kind"],
        "chronaris_lag_aware_weight": values["chronaris_lag_aware_weight"],
        "rng_state": rng_state,
    }
    payload["canonical_training_state_sha256"] = canonical_training_state_sha256(
        payload["encoder_state_dict"],
        payload["head_state_dict"],
        payload["optimizer_state_dict"],
        rng_state,
    )
    return payload


def _protocol_hash(**payload) -> str:
    payload["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_is_semantically_compatible(
    payload,
    *,
    candidate,
    config,
    policy,
    fold,
    normalizer,
    physiology_feature_names,
    vehicle_feature_names,
    vehicle_field_labels,
    transfer_source,
    chronaris_fusion_kind,
    chronaris_lag_aware_weight,
) -> bool:
    """Allow code-only changes when every persisted training input still matches."""
    config_matches = _training_configs_match_ignoring_device(
        payload.get("config", {}), asdict(config)
    )
    return all(
        (
            payload.get("candidate_config") == asdict(candidate),
            config_matches,
            payload.get("augmentation_policy") == asdict(policy),
            payload.get("fold") == fold.to_dict(),
            payload.get("normalizer", {}).get("transform_sha256")
            == normalizer.to_manifest().get("transform_sha256"),
            payload.get("physiology_feature_names") == list(physiology_feature_names),
            payload.get("vehicle_feature_names") == list(vehicle_feature_names),
            payload.get("vehicle_field_labels")
            == [list(value) for value in vehicle_field_labels],
            payload.get("label_used_for_encoder_training") is False,
            payload.get("simulation_oracle_opened") is False,
            payload.get("transfer_source") == transfer_source,
            payload.get("chronaris_fusion_kind", "multiscale")
            == chronaris_fusion_kind,
            float(payload.get("chronaris_lag_aware_weight", 0.0))
            == chronaris_lag_aware_weight,
        )
    )


def _training_configs_match_ignoring_device(stored, expected) -> bool:
    stored_values = dict(stored)
    expected_values = dict(expected)
    stored_values.pop("device", None)
    expected_values.pop("device", None)
    return stored_values == expected_values


def _load_payload(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") not in {
        "chronaris.common_pretraining_checkpoint.v1",
        "chronaris.common_pretraining_checkpoint.v2",
    }:
        raise RepresentationContractError("unsupported candidate screen checkpoint")
    return payload


def _atomic_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _result(payload, best_path, last_path, *, status) -> CandidateScreenResult:
    return CandidateScreenResult(
        method_name=str(payload["method_name"]),
        candidate_id=str(payload["candidate_config"]["candidate_id"]),
        status=status,
        best_checkpoint_path=str(best_path),
        last_checkpoint_path=str(last_path),
        protocol_sha256=str(payload["protocol_sha256"]),
        best_epoch=int(payload["best_epoch"]),
        completed_epochs=int(payload["completed_epochs"]),
        stopped_early=bool(payload["stopped_early"]),
        best_validation_losses=dict(payload["best_validation_losses"]),
        best_public_selection_loss=float(payload["best_public_selection_loss"]),
        training_elapsed_s=float(payload["training_elapsed_s"]),
        parameter_count=int(payload["parameter_count"]),
        epoch_rows=tuple(payload["epoch_rows"]),
    )
