"""Early-stopped Chronaris retraining with active method-specific regularizers."""

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
from chronaris.modeling.training.candidate_screen import PUBLIC_SELECTION_WEIGHTS
from chronaris.modeling.training.chronaris_auxiliary import (
    build_chronaris_auxiliary_losses,
    chronaris_auxiliary_losses_to_rows,
)
from chronaris.modeling.training.pretext import (
    CommonPretextHeadBundle,
    CommonPretextWeights,
    chronaris_auxiliary_weight_schedule,
)
from chronaris.modeling.training.pretraining_encoders import (
    ENCODER_SCREEN_CANDIDATES,
    EncoderCandidateConfig,
    build_trainable_fusion_encoder,
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
    select_observation_batch,
)
from chronaris.representation.contracts import FUSION_OUTPUT_DIM, RepresentationContractError


@dataclass(frozen=True, slots=True)
class LockedChronarisTrainingConfig:
    max_epochs: int = 50
    batch_size: int = 128
    patience: int = 8
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    seed: int = 17
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(self.max_epochs, self.batch_size, self.patience) <= 0:
            raise ValueError("locked Chronaris epoch/batch/patience must be positive")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("locked Chronaris optimizer configuration is invalid")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("locked Chronaris device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("locked Chronaris requested unavailable CUDA device")


@dataclass(frozen=True, slots=True)
class LockedChronarisTrainingResult:
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
    auxiliary_rows: tuple[Mapping[str, object], ...]


def train_locked_chronaris(
    *,
    batch: DualStreamObservationBatch | None,
    fold: FoldLineage,
    physiology_feature_names: tuple[str, ...],
    vehicle_feature_names: tuple[str, ...],
    vehicle_field_labels: tuple[tuple[str, str], ...],
    normalizer: TrainOnlyRobustNormalizer,
    output_root: str | Path,
    config: LockedChronarisTrainingConfig | None = None,
    augmentation_policy: AugmentationPolicy | None = None,
    batch_provider: Callable[[Sequence[str]], DualStreamObservationBatch] | None = None,
    candidate_config: EncoderCandidateConfig | None = None,
    variant: str = "full",
    fusion_kind: str = "multiscale",
    initialization_checkpoint: str | Path | None = None,
    resume: bool = True,
) -> LockedChronarisTrainingResult:
    resolved = config or LockedChronarisTrainingConfig()
    policy = augmentation_policy or AugmentationPolicy()
    if (batch is None) == (batch_provider is None):
        raise ValueError("provide exactly one of batch or batch_provider")
    candidate = candidate_config or ENCODER_SCREEN_CANDIDATES[0]
    root = Path(output_root) / "chronaris"
    best_path = root / "best.pt"
    last_path = root / "last.pt"
    transfer_source = (
        describe_transfer_source(
            initialization_checkpoint,
            expected_method="chronaris",
            expected_seed=resolved.seed,
        ).to_dict()
        if initialization_checkpoint is not None
        else None
    )
    protocol_hash = _protocol_hash(
        config=asdict(resolved),
        policy=asdict(policy),
        candidate=asdict(candidate),
        fold=fold.to_dict(),
        normalizer=normalizer.to_manifest(),
        physiology_feature_names=physiology_feature_names,
        vehicle_feature_names=vehicle_feature_names,
        vehicle_field_labels=vehicle_field_labels,
        variant=variant,
        fusion_kind=fusion_kind,
        transfer_source=transfer_source,
    )
    resume_payload = None
    if resume and last_path.exists():
        resume_payload = _load(last_path)
        if (
            resume_payload.get("protocol_sha256") != protocol_hash
            and not _checkpoint_is_semantically_compatible(
                resume_payload,
                config=resolved,
                policy=policy,
                candidate=candidate,
                fold=fold,
                normalizer=normalizer,
                physiology_feature_names=physiology_feature_names,
                vehicle_feature_names=vehicle_feature_names,
                vehicle_field_labels=vehicle_field_labels,
                variant=variant,
                fusion_kind=fusion_kind,
                transfer_source=transfer_source,
            )
        ):
            raise RepresentationContractError("locked Chronaris protocol changed")
        if resume_payload.get("training_status") == "completed":
            return _result(_load(best_path), best_path, last_path, status="resumed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(resolved.seed)
        encoder = build_trainable_fusion_encoder(
            "chronaris",
            physiology_feature_names=physiology_feature_names,
            vehicle_feature_names=vehicle_feature_names,
            vehicle_field_labels=vehicle_field_labels,
            candidate_config=candidate,
            chronaris_variant=variant,
            chronaris_fusion_kind=fusion_kind,
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
        transfer_initialization = resume_payload.get("transfer_initialization")
    elif initialization_checkpoint is not None:
        transfer_initialization = initialize_encoder_from_transfer_source(
            encoder,
            initialization_checkpoint,
            expected_method="chronaris",
            expected_seed=resolved.seed,
        ).to_dict()
    epoch_rows = list(resume_payload["epoch_rows"]) if resume_payload else []
    auxiliary_rows = list(resume_payload["auxiliary_rows"]) if resume_payload else []
    best_score = float(resume_payload["best_public_selection_loss"]) if resume_payload else float("inf")
    best_epoch = int(resume_payload["best_epoch"]) if resume_payload else 0
    best_losses = dict(resume_payload["best_validation_losses"]) if resume_payload else {}
    without_improvement = int(epoch_rows[-1]["epochs_without_improvement"]) if epoch_rows else 0
    start_epoch = int(resume_payload["completed_epochs"]) + 1 if resume_payload else 1
    elapsed_offset = float(resume_payload["training_elapsed_s"]) if resume_payload else 0.0
    started = time.perf_counter()
    for epoch in range(start_epoch, resolved.max_epochs + 1):
        encoder.train()
        heads.train()
        weights = chronaris_auxiliary_weight_schedule(epoch)
        train_common = 0.0
        train_auxiliary = 0.0
        step_count = 0
        for ids in _batch_ids(fold.train_sample_ids, resolved.batch_size):
            normalized = _normalized_batch(
                batch, batch_provider, ids, normalizer, resolved.device
            )
            plans = build_batch_augmentation_realizations(
                ids, epoch=epoch, global_seed=resolved.seed, policy=policy
            )
            augmented = apply_augmentation_realizations(normalized, plans, policy=policy)
            targets = build_common_pretext_targets(normalized, augmented)
            lag_inputs = build_lag_discrimination_inputs(
                augmented.batch, augmented.augmentation_ids
            )
            optimizer.zero_grad(set_to_none=True)
            positive = encoder(
                augmented.batch,
                compute_chronaris_diagnostics=True,
            )
            negative = encoder(lag_inputs.negative_batch)
            common = heads(
                positive.sequence_embedding,
                negative.sequence_embedding,
                targets,
                weights=CommonPretextWeights(),
            )
            auxiliary = build_chronaris_auxiliary_losses(
                positive,
                negative,
                weights=weights,
            )
            (common.total_loss + auxiliary.total_loss).backward()
            nn.utils.clip_grad_norm_(
                (*encoder.parameters(), *heads.parameters()),
                resolved.gradient_clip_norm,
            )
            optimizer.step()
            train_common += float(common.total_loss.detach())
            train_auxiliary += float(auxiliary.total_loss.detach())
            step_count += 1
            auxiliary_rows.extend(
                {
                    "epoch": epoch,
                    "step": step_count,
                    **row,
                }
                for row in chronaris_auxiliary_losses_to_rows(
                    auxiliary,
                    weights=weights,
                )
            )
        validation_losses = _validation_losses(
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
        score = sum(
            PUBLIC_SELECTION_WEIGHTS[name] * validation_losses[name]
            for name in PUBLIC_SELECTION_WEIGHTS
        )
        improved = score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_losses = dict(validation_losses)
            without_improvement = 0
        else:
            without_improvement += 1
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_common_loss": train_common / step_count,
                "train_auxiliary_loss": train_auxiliary / step_count,
                "validation_losses": validation_losses,
                "public_selection_loss": score,
                "improved": improved,
                "epochs_without_improvement": without_improvement,
            }
        )
        payload = _payload(
            encoder=encoder,
            heads=heads,
            optimizer=optimizer,
            protocol_hash=protocol_hash,
            config=resolved,
            policy=policy,
            candidate=candidate,
            fold=fold,
            normalizer=normalizer,
            physiology_feature_names=physiology_feature_names,
            vehicle_feature_names=vehicle_feature_names,
            vehicle_field_labels=vehicle_field_labels,
            best_epoch=best_epoch,
            completed_epochs=epoch,
            best_losses=best_losses,
            best_score=best_score,
            epoch_rows=epoch_rows,
            auxiliary_rows=auxiliary_rows,
            elapsed=elapsed_offset + time.perf_counter() - started,
            transfer_source=transfer_source,
            transfer_initialization=transfer_initialization,
        )
        _save(last_path, payload)
        if improved:
            _save(best_path, payload)
        if without_improvement >= resolved.patience:
            break
    elapsed = elapsed_offset + time.perf_counter() - started
    final = _load(best_path)
    final.update(
        training_status="completed",
        completed_epochs=len(epoch_rows),
        stopped_early=len(epoch_rows) < resolved.max_epochs,
        training_elapsed_s=elapsed,
        epoch_rows=epoch_rows,
        auxiliary_rows=auxiliary_rows,
    )
    _save(best_path, final)
    last = _load(last_path)
    last["training_status"] = "completed"
    last["stopped_early"] = final["stopped_early"]
    _save(last_path, last)
    return _result(final, best_path, last_path, status="completed")


def _validation_losses(**values):
    encoder, heads = values["encoder"], values["heads"]
    encoder.eval()
    heads.eval()
    totals = {name: [0.0, 0] for name in PUBLIC_SELECTION_WEIGHTS}
    with torch.inference_mode():
        for ids in _batch_ids(values["sample_ids"], values["batch_size"]):
            normalized = _normalized_batch(
                values["batch"], values["batch_provider"], ids,
                values["normalizer"], values["device"]
            )
            plans = build_batch_augmentation_realizations(
                ids, epoch=0, global_seed=values["seed"], policy=values["policy"]
            )
            augmented = apply_augmentation_realizations(normalized, plans, policy=values["policy"])
            targets = build_common_pretext_targets(normalized, augmented)
            lag_inputs = build_lag_discrimination_inputs(augmented.batch, augmented.augmentation_ids)
            output = heads(
                encoder(augmented.batch).sequence_embedding,
                encoder(lag_inputs.negative_batch).sequence_embedding,
                targets,
                weights=CommonPretextWeights(),
            )
            for term in output.terms:
                if term.raw_loss is not None:
                    totals[term.term_name][0] += float(term.raw_loss) * term.count
                    totals[term.term_name][1] += term.count
    return {name: total / count for name, (total, count) in totals.items()}


def _normalized_batch(batch, provider, ids, normalizer, device):
    raw = provider(ids) if provider is not None else select_observation_batch(batch, ids)
    if tuple(raw.sample_ids) != tuple(ids):
        raise RepresentationContractError("locked Chronaris provider changed sample order")
    return move_observation_batch(normalizer.transform(raw), device=device)


def _batch_ids(sample_ids, batch_size):
    values = tuple(sample_ids)
    return tuple(values[index:index + batch_size] for index in range(0, len(values), batch_size))


def _payload(**values):
    encoder, heads, optimizer = values["encoder"], values["heads"], values["optimizer"]
    return {
        "format": "chronaris.common_pretraining_checkpoint.v1",
        "training_status": "running",
        "method_name": "chronaris",
        "protocol_sha256": values["protocol_hash"],
        "encoder_state_dict": encoder.state_dict(),
        "head_state_dict": heads.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "normalizer": dict(values["normalizer"].to_manifest()),
        "fold": values["fold"].to_dict(),
        "config": asdict(values["config"]),
        "augmentation_policy": asdict(values["policy"]),
        "candidate_config": asdict(values["candidate"]),
        "encoder_manifest": dict(encoder.config_manifest()),
        "physiology_feature_names": list(values["physiology_feature_names"]),
        "vehicle_feature_names": list(values["vehicle_feature_names"]),
        "vehicle_field_labels": [list(value) for value in values["vehicle_field_labels"]],
        "seed": values["config"].seed,
        "epoch": values["completed_epochs"],
        "best_epoch": values["best_epoch"],
        "completed_epochs": values["completed_epochs"],
        "best_validation_losses": dict(values["best_losses"]),
        "best_public_selection_loss": values["best_score"],
        "stopped_early": False,
        "training_elapsed_s": values["elapsed"],
        "parameter_count": encoder.parameter_count,
        "head_parameter_count": sum(p.numel() for p in heads.parameters()),
        "epoch_rows": list(values["epoch_rows"]),
        "auxiliary_rows": list(values["auxiliary_rows"]),
        "training_rows": [],
        "augmentation_rows": [],
        "label_used_for_encoder_training": False,
        "simulation_oracle_opened": False,
        "chronaris_auxiliary_enabled": True,
        "selection_uses_public_pretext_only": True,
        "early_stopping_uses_public_pretext_only": True,
        "transfer_source": values.get("transfer_source"),
        "transfer_initialization": values.get("transfer_initialization"),
    }


def _checkpoint_is_semantically_compatible(
    payload,
    *,
    config,
    policy,
    candidate,
    fold,
    normalizer,
    physiology_feature_names,
    vehicle_feature_names,
    vehicle_field_labels,
    variant,
    fusion_kind,
    transfer_source,
):
    return all(
        (
            payload.get("method_name") == "chronaris",
            payload.get("config") == asdict(config),
            payload.get("augmentation_policy") == asdict(policy),
            payload.get("candidate_config") == asdict(candidate),
            payload.get("fold") == fold.to_dict(),
            payload.get("normalizer", {}).get("transform_sha256")
            == normalizer.to_manifest().get("transform_sha256"),
            payload.get("physiology_feature_names")
            == list(physiology_feature_names),
            payload.get("vehicle_feature_names") == list(vehicle_feature_names),
            payload.get("vehicle_field_labels")
            == [list(value) for value in vehicle_field_labels],
            payload.get("encoder_manifest", {})
            .get("backbone_config", {})
            .get("variant", "full")
            == variant,
            payload.get("encoder_manifest", {})
            .get("backbone_config", {})
            .get("fusion_kind", "multiscale")
            == fusion_kind,
            payload.get("transfer_source") == transfer_source,
            payload.get("label_used_for_encoder_training") is False,
            payload.get("simulation_oracle_opened") is False,
        )
    )


def _protocol_hash(**payload):
    payload["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _load(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "chronaris.common_pretraining_checkpoint.v1":
        raise RepresentationContractError("unsupported locked Chronaris checkpoint")
    return payload


def _save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _result(payload, best_path, last_path, *, status):
    return LockedChronarisTrainingResult(
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
        auxiliary_rows=tuple(payload["auxiliary_rows"]),
    )
