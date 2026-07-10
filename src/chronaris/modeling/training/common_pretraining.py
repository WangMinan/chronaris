"""Resumable common-pretext training for the five trainable fusion encoders."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.pretext import (
    CommonPretextHeadBundle,
    CommonPretextWeights,
    chronaris_auxiliary_weight_schedule,
    pretext_loss_terms_to_rows,
)
from chronaris.modeling.training.pretraining_encoders import (
    TRAINABLE_FUSION_METHODS,
    TrainableFusionEncoder,
    build_trainable_fusion_encoder,
)
from chronaris.representation import (
    AugmentationPolicy,
    DualStreamObservationBatch,
    FoldLineage,
    FusionStreamBatch,
    TrainOnlyRobustNormalizer,
    apply_augmentation_realizations,
    build_batch_augmentation_realizations,
    build_common_pretext_targets,
    build_lag_discrimination_inputs,
    select_observation_batch,
)
from chronaris.representation.contracts import FUSION_OUTPUT_DIM, RepresentationContractError


@dataclass(frozen=True, slots=True)
class CommonPretrainingConfig:
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    seed: int = 17

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("pretraining epoch/batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("pretraining optimizer configuration is invalid")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient clip norm must be positive")


@dataclass(frozen=True, slots=True)
class CommonPretrainingResult:
    method_name: str
    status: str
    best_checkpoint_path: str
    last_checkpoint_path: str
    protocol_sha256: str
    training_elapsed_s: float
    parameter_count: int
    head_parameter_count: int
    step_count: int
    training_rows: tuple[Mapping[str, object], ...]
    augmentation_rows: tuple[Mapping[str, object], ...]


class TrainedFusionAdapter:
    output_dim = FUSION_OUTPUT_DIM

    def __init__(
        self,
        *,
        encoder: TrainableFusionEncoder,
        normalizer: TrainOnlyRobustNormalizer,
        fold_id: str,
        checkpoint_sha256: str,
    ) -> None:
        self.encoder = encoder
        self.normalizer = normalizer
        self.fold_id = fold_id
        self.checkpoint_sha256 = checkpoint_sha256
        self.method_name = encoder.method_name

    def __call__(self, batch: DualStreamObservationBatch) -> FusionStreamBatch:
        device = next(self.encoder.parameters()).device
        normalized = self.normalizer.transform(
            move_observation_batch(batch, device=device)
        )
        self.encoder.eval()
        with torch.inference_mode():
            encoded = self.encoder(normalized)
        sequence = encoded.sequence_embedding
        valid = torch.ones(
            sequence.shape[:2],
            dtype=torch.bool,
            device=sequence.device,
        )
        return FusionStreamBatch(
            sample_ids=batch.sample_ids,
            timestamps_s=batch.query_timestamps_s.to(device),
            sequence_embedding=sequence,
            valid_mask=valid,
            pooled_embedding=sequence.mean(dim=1),
            method_name=self.method_name,
            fold_id=self.fold_id,
            checkpoint_sha256=self.checkpoint_sha256,
            source_sample_hashes=batch.source_sample_hashes,
        )


def train_common_pretext_method(
    method_name: str,
    *,
    batch: DualStreamObservationBatch,
    fold: FoldLineage,
    physiology_feature_names: tuple[str, ...],
    vehicle_feature_names: tuple[str, ...],
    vehicle_field_labels: tuple[tuple[str, str], ...],
    normalizer: TrainOnlyRobustNormalizer,
    output_root: str | Path,
    config: CommonPretrainingConfig | None = None,
    augmentation_policy: AugmentationPolicy | None = None,
    resume: bool = True,
) -> CommonPretrainingResult:
    if method_name not in TRAINABLE_FUSION_METHODS:
        raise ValueError(f"unsupported trainable method: {method_name}")
    resolved_config = config or CommonPretrainingConfig()
    resolved_policy = augmentation_policy or AugmentationPolicy()
    root = Path(output_root) / method_name
    best_path = root / "best.pt"
    last_path = root / "last.pt"
    protocol_hash = _training_protocol_hash(
        method_name=method_name,
        fold=fold,
        config=resolved_config,
        augmentation_policy=resolved_policy,
        normalizer=normalizer,
        physiology_feature_names=physiology_feature_names,
        vehicle_feature_names=vehicle_feature_names,
        vehicle_field_labels=vehicle_field_labels,
    )
    if resume and best_path.exists() and last_path.exists():
        payload = _load_checkpoint_payload(best_path)
        if payload.get("protocol_sha256") != protocol_hash:
            raise RepresentationContractError(
                f"pretraining checkpoint protocol changed for {method_name}"
            )
        if payload.get("training_status") == "completed":
            return _result_from_payload(payload, best_path, last_path, status="resumed")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(resolved_config.seed)
        encoder = build_trainable_fusion_encoder(
            method_name,
            physiology_feature_names=physiology_feature_names,
            vehicle_feature_names=vehicle_feature_names,
            vehicle_field_labels=vehicle_field_labels,
        )
        heads = CommonPretextHeadBundle(
            representation_dim=FUSION_OUTPUT_DIM,
            target_feature_count=(
                len(physiology_feature_names) + len(vehicle_feature_names)
            ),
        )
    optimizer = torch.optim.AdamW(
        (*encoder.parameters(), *heads.parameters()),
        lr=resolved_config.learning_rate,
        weight_decay=resolved_config.weight_decay,
    )
    encoder.train()
    heads.train()
    training_rows = []
    augmentation_rows = []
    step_count = 0
    started = time.perf_counter()
    for epoch in range(1, resolved_config.epochs + 1):
        auxiliary_weights = chronaris_auxiliary_weight_schedule(epoch)
        for step_index, sample_ids in enumerate(
            _batch_ids(fold.train_sample_ids, resolved_config.batch_size),
            start=1,
        ):
            raw = select_observation_batch(batch, sample_ids)
            normalized = normalizer.transform(raw)
            plans = build_batch_augmentation_realizations(
                sample_ids,
                epoch=epoch,
                global_seed=resolved_config.seed,
                policy=resolved_policy,
            )
            augmented = apply_augmentation_realizations(
                normalized,
                plans,
                policy=resolved_policy,
            )
            targets = build_common_pretext_targets(normalized, augmented)
            lag_inputs = build_lag_discrimination_inputs(
                augmented.batch,
                augmented.augmentation_ids,
            )
            optimizer.zero_grad(set_to_none=True)
            positive = encoder(augmented.batch)
            negative = encoder(lag_inputs.negative_batch)
            loss_output = heads(
                positive.sequence_embedding,
                negative.sequence_embedding,
                targets,
                weights=CommonPretextWeights(),
            )
            loss_output.total_loss.backward()
            parameters = tuple((*encoder.parameters(), *heads.parameters()))
            gradient_norm = float(
                nn.utils.clip_grad_norm_(
                    parameters,
                    resolved_config.gradient_clip_norm,
                ).detach()
            )
            optimizer.step()
            step_count += 1
            for row in pretext_loss_terms_to_rows(loss_output.terms):
                training_rows.append(
                    {
                        "method_name": method_name,
                        "epoch": epoch,
                        "step": step_index,
                        "batch_sample_ids": list(sample_ids),
                        "augmentation_ids": list(augmented.augmentation_ids),
                        "gradient_norm_before_clip": gradient_norm,
                        "continuous_alignment_weight": auxiliary_weights.continuous_alignment,
                        "physical_consistency_weight": auxiliary_weights.physical_consistency,
                        "causal_direction_weight": auxiliary_weights.causal_direction,
                        **row,
                    }
                )
            augmentation_rows.extend(
                {
                    "method_name": method_name,
                    "epoch": epoch,
                    "step": step_index,
                    **row.to_dict(),
                }
                for row in augmented.audit_rows
            )
    elapsed = time.perf_counter() - started
    payload = {
        "format": "chronaris.common_pretraining_checkpoint.v1",
        "training_status": "completed",
        "method_name": method_name,
        "protocol_sha256": protocol_hash,
        "encoder_state_dict": encoder.state_dict(),
        "head_state_dict": heads.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "normalizer": dict(normalizer.to_manifest()),
        "fold": fold.to_dict(),
        "config": asdict(resolved_config),
        "augmentation_policy": asdict(resolved_policy),
        "physiology_feature_names": list(physiology_feature_names),
        "vehicle_feature_names": list(vehicle_feature_names),
        "vehicle_field_labels": [list(value) for value in vehicle_field_labels],
        "encoder_manifest": dict(encoder.config_manifest()),
        "seed": resolved_config.seed,
        "epoch": resolved_config.epochs,
        "step_count": step_count,
        "training_elapsed_s": elapsed,
        "parameter_count": encoder.parameter_count,
        "head_parameter_count": sum(p.numel() for p in heads.parameters()),
        "training_rows": training_rows,
        "augmentation_rows": augmentation_rows,
        "label_used_for_encoder_training": False,
        "simulation_oracle_opened": False,
    }
    _atomic_torch_save(last_path, payload)
    _atomic_torch_save(best_path, payload)
    return _result_from_payload(payload, best_path, last_path, status="completed")


def load_common_pretraining_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
):
    payload = _load_checkpoint_payload(path, device=device)
    if payload.get("training_status") != "completed":
        raise RepresentationContractError("common pretraining checkpoint is incomplete")
    if bool(payload.get("label_used_for_encoder_training")):
        raise RepresentationContractError("pretraining checkpoint used downstream labels")
    method_name = str(payload["method_name"])
    physiology_names = tuple(payload["physiology_feature_names"])
    vehicle_names = tuple(payload["vehicle_feature_names"])
    field_labels = tuple(tuple(value) for value in payload["vehicle_field_labels"])
    encoder = build_trainable_fusion_encoder(
        method_name,
        physiology_feature_names=physiology_names,
        vehicle_feature_names=vehicle_names,
        vehicle_field_labels=field_labels,
    ).to(device)
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    heads = CommonPretextHeadBundle(
        representation_dim=FUSION_OUTPUT_DIM,
        target_feature_count=len(physiology_names) + len(vehicle_names),
    ).to(device)
    heads.load_state_dict(payload["head_state_dict"], strict=True)
    normalizer = TrainOnlyRobustNormalizer.from_manifest(payload["normalizer"])
    return encoder, heads, normalizer, payload


def _training_protocol_hash(**payload) -> str:
    normalized = {
        key: (
            value.to_dict()
            if isinstance(value, FoldLineage)
            else asdict(value)
            if hasattr(value, "__dataclass_fields__")
            else value.to_manifest()
            if isinstance(value, TrainOnlyRobustNormalizer)
            else value
        )
        for key, value in payload.items()
    }
    normalized["code_sha256"] = _code_sha256()
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("pretext.py"),
        Path(__file__).with_name("pretraining_encoders.py"),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_checkpoint_payload(path, *, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "chronaris.common_pretraining_checkpoint.v1":
        raise RepresentationContractError("unsupported common pretraining checkpoint")
    return payload


def _result_from_payload(payload, best_path, last_path, *, status):
    return CommonPretrainingResult(
        method_name=str(payload["method_name"]),
        status=status,
        best_checkpoint_path=str(best_path),
        last_checkpoint_path=str(last_path),
        protocol_sha256=str(payload["protocol_sha256"]),
        training_elapsed_s=float(payload["training_elapsed_s"]),
        parameter_count=int(payload["parameter_count"]),
        head_parameter_count=int(payload["head_parameter_count"]),
        step_count=int(payload["step_count"]),
        training_rows=tuple(payload["training_rows"]),
        augmentation_rows=tuple(payload["augmentation_rows"]),
    )


def _atomic_torch_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _batch_ids(sample_ids, batch_size):
    return tuple(
        tuple(sample_ids[index : index + batch_size])
        for index in range(0, len(sample_ids), batch_size)
    )
