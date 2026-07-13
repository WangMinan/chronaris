"""Equal-budget task-aware adaptation for the Chronaris core-task recovery."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error
from torch import nn

from chronaris.evaluation.application_tasks.core_recovery_tasks import (
    ChronarisCoreTaskHeads,
    CoreTaskHeadOutput,
    CoreTaskLossConfig,
    TaskAwareTargetBundle,
    core_task_losses,
    initialize_task_head_biases,
    select_task_aware_targets,
)
from chronaris.modeling.fusion_encoders.observed_residual import (
    ObservedStateResidual,
    fit_observed_state_projector,
)
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training import (
    TrainableFusionEncoder,
    load_common_pretraining_checkpoint,
)
from chronaris.representation import (
    DualStreamObservationBatch,
    TrainOnlyRobustNormalizer,
    select_observation_batch,
)
from chronaris.representation.contracts import RepresentationContractError
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


CORE_RECOVERY_CHECKPOINT_FORMAT = "chronaris.core_task_recovery_checkpoint.v1"


@dataclass(frozen=True, slots=True)
class CoreRecoveryTrainingConfig:
    adapter_learning_rate: float = 3e-4
    backbone_learning_rate_ratio: float = 0.05
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    batch_size: int = 32
    frozen_backbone_steps: int = 60
    partial_unfreeze_steps: int = 60
    validation_interval_steps: int = 10
    patience_evaluations: int = 4
    high_response_weight: float = 1.0
    seed: int = 17
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.adapter_learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("core-recovery optimizer configuration is invalid")
        if self.backbone_learning_rate_ratio not in {0.05, 0.1}:
            raise ValueError("backbone learning-rate ratio is outside the locked grid")
        if min(
            self.gradient_clip_norm,
            self.batch_size,
            self.frozen_backbone_steps,
            self.validation_interval_steps,
            self.patience_evaluations,
        ) <= 0:
            raise ValueError("core-recovery step budget must be positive")
        if self.partial_unfreeze_steps < 0:
            raise ValueError("partial-unfreeze step budget cannot be negative")
        if self.high_response_weight not in {0.5, 1.0}:
            raise ValueError("high-response weight is outside the locked grid")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("core-recovery device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("core-recovery requested unavailable CUDA")


@dataclass(frozen=True, slots=True)
class CoreRecoveryTrainingResult:
    method_name: str
    status: str
    best_checkpoint_path: str
    last_checkpoint_path: str
    completed_steps: int
    best_step: int
    best_validation_metrics: Mapping[str, float | bool | None]
    training_elapsed_s: float
    label_used_for_encoder_training: bool


@dataclass(frozen=True, slots=True)
class CoreRecoveryFrozenEncoding:
    sample_ids: tuple[str, ...]
    valid_mask: torch.Tensor
    sequence_embedding: torch.Tensor | None = None
    observed_sequence: torch.Tensor | None = None
    maneuver_observed_sequence: torch.Tensor | None = None
    continuous_sequence: torch.Tensor | None = None

    def select(self, sample_ids: Sequence[str]) -> "CoreRecoveryFrozenEncoding":
        positions = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
        indices = torch.tensor(
            [positions[str(sample_id)] for sample_id in sample_ids],
            dtype=torch.long,
            device=self.valid_mask.device,
        )

        def take(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.index_select(0, indices)

        return CoreRecoveryFrozenEncoding(
            sample_ids=tuple(str(value) for value in sample_ids),
            valid_mask=self.valid_mask.index_select(0, indices),
            sequence_embedding=take(self.sequence_embedding),
            observed_sequence=take(self.observed_sequence),
            maneuver_observed_sequence=take(self.maneuver_observed_sequence),
            continuous_sequence=take(self.continuous_sequence),
        )


class CoreRecoveryMethodModel(nn.Module):
    """One task-aware interface for Chronaris and the four trainable baselines."""

    def __init__(
        self,
        *,
        method_name: str,
        encoder: TrainableFusionEncoder | None,
        chronaris_residual: ObservedStateResidual | None,
        normalizer: TrainOnlyRobustNormalizer,
        physiology_target_count: int,
    ) -> None:
        super().__init__()
        if (encoder is None) == (chronaris_residual is None):
            raise ValueError("core-recovery model requires exactly one encoder path")
        if method_name == "chronaris" and chronaris_residual is None:
            raise ValueError("Chronaris requires the observed-state residual path")
        if method_name != "chronaris" and encoder is None:
            raise ValueError("trainable baseline requires its source encoder")
        self.method_name = method_name
        self.encoder = encoder
        self.chronaris_residual = chronaris_residual
        self.normalizer = normalizer
        self.heads = ChronarisCoreTaskHeads(
            physiology_target_count=physiology_target_count
        )

    def forward(
        self,
        raw: DualStreamObservationBatch,
        *,
        residual_mode: str = "full",
    ) -> CoreTaskHeadOutput:
        encoded = self.encode(raw)
        return self.forward_encoded(encoded, residual_mode=residual_mode)

    def encode(self, raw: DualStreamObservationBatch):
        device = next(self.parameters()).device
        normalized = move_observation_batch(
            self.normalizer.transform(raw),
            device=device,
        )
        if self.chronaris_residual is not None:
            return self.chronaris_residual(normalized)
        return self.encoder(normalized)

    def forward_encoded(self, encoded, *, residual_mode: str = "full"):
        if self.chronaris_residual is not None:
            return self.heads(encoded, residual_mode=residual_mode)
        return self.heads.forward_sequences(
            maneuver_sequence=encoded.sequence_embedding,
            response_sequence=encoded.sequence_embedding,
            valid_mask=encoded.modality_available_mask,
        )

    def encode_frozen(
        self,
        raw: DualStreamObservationBatch,
    ) -> CoreRecoveryFrozenEncoding:
        """Encode a full role once while the expensive source backbone is frozen."""

        device = next(self.parameters()).device
        normalized = move_observation_batch(
            self.normalizer.transform(raw),
            device=device,
        )
        if self.chronaris_residual is not None:
            observed, maneuver_observed, continuous, valid_mask, _diagnostics = (
                self.chronaris_residual.encode_paths(normalized)
            )
            return CoreRecoveryFrozenEncoding(
                sample_ids=raw.sample_ids,
                valid_mask=valid_mask.detach(),
                observed_sequence=observed.detach(),
                maneuver_observed_sequence=maneuver_observed.detach(),
                continuous_sequence=continuous.detach(),
            )
        encoded = self.encoder(normalized)
        return CoreRecoveryFrozenEncoding(
            sample_ids=raw.sample_ids,
            valid_mask=encoded.modality_available_mask.detach(),
            sequence_embedding=encoded.sequence_embedding.detach(),
        )

    def forward_frozen(
        self,
        encoded: CoreRecoveryFrozenEncoding,
        *,
        residual_mode: str = "full",
    ) -> CoreTaskHeadOutput:
        if self.chronaris_residual is not None:
            representation = self.chronaris_residual.fuse_paths(
                observed_sequence=encoded.observed_sequence,
                maneuver_observed_sequence=encoded.maneuver_observed_sequence,
                continuous_sequence=encoded.continuous_sequence,
                valid_mask=encoded.valid_mask,
            )
            return self.heads(representation, residual_mode=residual_mode)
        return self.heads.forward_sequences(
            maneuver_sequence=encoded.sequence_embedding,
            response_sequence=encoded.sequence_embedding,
            valid_mask=encoded.valid_mask,
        )


def build_core_recovery_method_model(
    *,
    source_checkpoint_path: str | Path,
    batch: DualStreamObservationBatch,
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    physiology_target_count: int,
    device: str = "cpu",
) -> tuple[CoreRecoveryMethodModel, Mapping[str, object]]:
    encoder, _pretext_heads, normalizer, payload = load_common_pretraining_checkpoint(
        source_checkpoint_path,
        device=device,
    )
    method_name = str(payload["method_name"])
    expected_train = tuple(sorted(str(value) for value in train_sample_ids))
    if normalizer.fit_sample_ids != expected_train:
        raise RepresentationContractError(
            "source checkpoint normalizer does not match the recovery inner-train role"
        )
    if bool(payload.get("label_used_for_encoder_training")):
        raise RepresentationContractError("recovery source checkpoint already used labels")
    residual = None
    retained_encoder: TrainableFusionEncoder | None = encoder
    if method_name == "chronaris":
        normalized = normalizer.transform(batch)
        projector = fit_observed_state_projector(
            normalized,
            train_sample_ids=train_sample_ids,
            held_out_sample_ids=validation_sample_ids,
            random_state=int(payload["seed"]),
        )
        maneuver_projector = fit_observed_state_projector(
            normalized,
            train_sample_ids=train_sample_ids,
            held_out_sample_ids=validation_sample_ids,
            random_state=int(payload["seed"]),
            active_stream="vehicle",
        )
        residual = ObservedStateResidual(
            continuous_encoder=encoder.backbone,
            projector=projector,
            maneuver_projector=maneuver_projector,
        )
        retained_encoder = None
    model = CoreRecoveryMethodModel(
        method_name=method_name,
        encoder=retained_encoder,
        chronaris_residual=residual,
        normalizer=normalizer,
        physiology_target_count=physiology_target_count,
    ).to(device)
    return model, payload


def train_core_recovery_method(
    *,
    model: CoreRecoveryMethodModel,
    batch: DualStreamObservationBatch,
    targets: TaskAwareTargetBundle,
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    source_checkpoint_path: str | Path,
    output_root: str | Path,
    candidate_id: str,
    config: CoreRecoveryTrainingConfig,
    precomputed_frozen_encoding: CoreRecoveryFrozenEncoding | None = None,
) -> CoreRecoveryTrainingResult:
    train_ids = tuple(str(value) for value in train_sample_ids)
    validation_ids = tuple(str(value) for value in validation_sample_ids)
    _validate_training_inputs(batch, targets, train_ids, validation_ids)
    train_targets = select_task_aware_targets(
        targets,
        train_ids,
        device=config.device,
    )
    task_head_priors = initialize_task_head_biases(model.heads, train_targets)
    root = Path(output_root) / model.method_name / candidate_id
    root.mkdir(parents=True, exist_ok=True)
    best_path = root / "best.pt"
    last_path = root / "last.pt"
    torch.manual_seed(config.seed)
    loss_config = CoreTaskLossConfig(
        high_response_weight=config.high_response_weight,
    )
    rows = []
    completed_steps = 0
    best_step = 0
    best_score = float("inf")
    best_metrics: dict[str, float | bool | None] = {}
    stale = 0
    started = time.perf_counter()
    frozen_encoding: CoreRecoveryFrozenEncoding | None = precomputed_frozen_encoding
    stages = tuple(
        row
        for row in (
        ("frozen_backbone", config.frozen_backbone_steps),
        ("partial_unfreeze", config.partial_unfreeze_steps),
        )
        if row[1] > 0
    )
    for stage_name, stage_steps in stages:
        _configure_trainable_parameters(model, stage_name=stage_name)
        if stage_name == "frozen_backbone":
            if frozen_encoding is None:
                model.eval()
                with torch.inference_mode():
                    frozen_encoding = model.encode_frozen(batch)
        else:
            frozen_encoding = None
        optimizer = _build_optimizer(model, config=config)
        for stage_step in range(1, stage_steps + 1):
            completed_steps += 1
            model.train()
            sample_ids = _step_sample_ids(
                train_ids,
                step=completed_steps,
                batch_size=config.batch_size,
                seed=config.seed,
            )
            selected_targets = select_task_aware_targets(
                targets,
                sample_ids,
                device=config.device,
            )
            optimizer.zero_grad(set_to_none=True)
            if frozen_encoding is None:
                raw = select_observation_batch(batch, sample_ids)
                output = model(raw)
            else:
                output = model.forward_frozen(frozen_encoding.select(sample_ids))
            losses = core_task_losses(output, selected_targets, config=loss_config)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("core-recovery training loss is non-finite")
            losses["total"].backward()
            gradient_norm = float(
                nn.utils.clip_grad_norm_(
                    tuple(parameter for parameter in model.parameters() if parameter.requires_grad),
                    config.gradient_clip_norm,
                )
            )
            optimizer.step()
            row = {
                "step": completed_steps,
                "stage": stage_name,
                "stage_step": stage_step,
                "sample_count": len(sample_ids),
                "gradient_norm_before_clip": gradient_norm,
                **{f"train_{key}_loss": float(value.detach()) for key, value in losses.items()},
            }
            should_validate = (
                stage_step == stage_steps
                or completed_steps % config.validation_interval_steps == 0
            )
            if should_validate:
                metrics = evaluate_core_recovery_model(
                    model=model,
                    batch=batch,
                    targets=targets,
                    sample_ids=validation_ids,
                    device=config.device,
                    frozen_encoding=frozen_encoding,
                )
                score = _selection_score(metrics)
                improved = score < best_score
                if improved:
                    best_score = score
                    best_step = completed_steps
                    best_metrics = dict(metrics)
                    stale = 0
                else:
                    stale += 1
                row.update({f"validation_{key}": value for key, value in metrics.items()})
                row["validation_selection_score"] = score
                row["improved"] = improved
                payload = _checkpoint_payload(
                    model=model,
                    source_checkpoint_path=source_checkpoint_path,
                    candidate_id=candidate_id,
                    config=config,
                    train_ids=train_ids,
                    validation_ids=validation_ids,
                    rows=rows + [row],
                    completed_steps=completed_steps,
                    best_step=best_step,
                    best_metrics=best_metrics,
                    status="running",
                    elapsed_s=time.perf_counter() - started,
                    task_head_priors=task_head_priors,
                )
                _atomic_save(last_path, payload)
                if improved:
                    _atomic_save(best_path, payload)
                if stale >= config.patience_evaluations:
                    rows.append(row)
                    break
            rows.append(row)
        if stale >= config.patience_evaluations:
            break
    if not best_path.is_file():
        raise RuntimeError("core-recovery training produced no validation checkpoint")
    best_payload = torch.load(best_path, map_location=config.device, weights_only=True)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    final_payload = _checkpoint_payload(
        model=model,
        source_checkpoint_path=source_checkpoint_path,
        candidate_id=candidate_id,
        config=config,
        train_ids=train_ids,
        validation_ids=validation_ids,
        rows=rows,
        completed_steps=completed_steps,
        best_step=best_step,
        best_metrics=best_metrics,
        status="completed",
        elapsed_s=time.perf_counter() - started,
        task_head_priors=task_head_priors,
    )
    _atomic_save(best_path, final_payload)
    _atomic_save(last_path, final_payload)
    return CoreRecoveryTrainingResult(
        method_name=model.method_name,
        status="completed",
        best_checkpoint_path=str(best_path),
        last_checkpoint_path=str(last_path),
        completed_steps=completed_steps,
        best_step=best_step,
        best_validation_metrics=best_metrics,
        training_elapsed_s=float(final_payload["training_elapsed_s"]),
        label_used_for_encoder_training=True,
    )


def evaluate_core_recovery_model(
    *,
    model: CoreRecoveryMethodModel,
    batch: DualStreamObservationBatch,
    targets: TaskAwareTargetBundle,
    sample_ids: Sequence[str],
    device: str,
    frozen_encoding: CoreRecoveryFrozenEncoding | None = None,
) -> Mapping[str, float | bool | None]:
    ids = tuple(str(value) for value in sample_ids)
    selected = select_task_aware_targets(targets, ids, device=device)
    model.eval()
    modes = ("full", "direct_only") if model.method_name == "chronaris" else ("full",)
    metrics = {}
    with torch.inference_mode():
        encoded = (
            frozen_encoding.select(ids)
            if frozen_encoding is not None
            else model.encode(select_observation_batch(batch, ids))
        )
        for mode in modes:
            output = (
                model.forward_frozen(encoded, residual_mode=mode)
                if isinstance(encoded, CoreRecoveryFrozenEncoding)
                else model.forward_encoded(encoded, residual_mode=mode)
            )
            prefix = "" if mode == "full" else "direct_"
            prediction = output.maneuver_logits.argmax(dim=-1).cpu().numpy()
            truth = selected.maneuver_class.cpu().numpy()
            metrics[prefix + "maneuver_macro_f1"] = float(
                f1_score(truth, prediction, labels=(0, 1, 2), average="macro", zero_division=0)
            )
            available = selected.response_available
            response_truth = selected.response_value[available].cpu().numpy()
            response_prediction = (
                torch.expm1(output.response_log1p[available])
                .clamp(min=0, max=10)
                .cpu()
                .numpy()
            )
            metrics[prefix + "response_rmse"] = float(
                math.sqrt(mean_squared_error(response_truth, response_prediction))
            )
            high_truth = selected.high_response[available].cpu().numpy()
            high_probability = torch.sigmoid(output.high_response_logit[available]).cpu().numpy()
            metrics[prefix + "high_response_auprc"] = (
                float(average_precision_score(high_truth, high_probability))
                if len(np.unique(high_truth)) == 2
                else None
            )
    if model.method_name == "chronaris":
        metrics["paired_no_harm_passed"] = bool(
            metrics["maneuver_macro_f1"] + 0.02 >= metrics["direct_maneuver_macro_f1"]
            and metrics["response_rmse"] <= metrics["direct_response_rmse"] + 0.02
            and (
                metrics["high_response_auprc"] is None
                or metrics["direct_high_response_auprc"] is None
                or metrics["high_response_auprc"] + 0.02 >= metrics["direct_high_response_auprc"]
            )
        )
    else:
        metrics["paired_no_harm_passed"] = True
    return metrics


def _configure_trainable_parameters(model: CoreRecoveryMethodModel, *, stage_name: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
    backbone_prefix = "chronaris_residual.continuous_encoder" if model.method_name == "chronaris" else "encoder.backbone"
    for name, parameter in model.named_parameters():
        if not name.startswith(backbone_prefix):
            continue
        parameter.requires_grad = False
        if stage_name == "partial_unfreeze" and any(
            token in name
            for token in ("observation_update", "projection", "output", "scale_gate", "gate")
        ):
            parameter.requires_grad = True
    if stage_name not in {"frozen_backbone", "partial_unfreeze"}:
        raise ValueError("unknown core-recovery training stage")


def _build_optimizer(model, *, config):
    backbone_prefix = "chronaris_residual.continuous_encoder" if model.method_name == "chronaris" else "encoder.backbone"
    backbone = []
    adapters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if name.startswith(backbone_prefix) else adapters).append(parameter)
    groups = [{"params": adapters, "lr": config.adapter_learning_rate}]
    if backbone:
        groups.append(
            {
                "params": backbone,
                "lr": config.adapter_learning_rate * config.backbone_learning_rate_ratio,
            }
        )
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def _step_sample_ids(train_ids, *, step, batch_size, seed):
    generator = torch.Generator().manual_seed(seed * 1_000_000 + step)
    if len(train_ids) <= batch_size:
        positions = torch.randperm(len(train_ids), generator=generator).tolist()
    else:
        positions = torch.randperm(len(train_ids), generator=generator)[:batch_size].tolist()
    return tuple(train_ids[index] for index in positions)


def _selection_score(metrics):
    auprc = metrics["high_response_auprc"]
    score = 1 - metrics["maneuver_macro_f1"] + metrics["response_rmse"] + (1 - auprc if auprc is not None else 1)
    if not metrics["paired_no_harm_passed"]:
        score += 10
    return float(score)


def _validate_training_inputs(batch, targets, train_ids, validation_ids):
    expected = set(train_ids) | set(validation_ids)
    if set(batch.sample_ids) != expected or set(targets.sample_ids) != expected:
        raise RepresentationContractError("core-recovery train/validation roles do not cover inputs")
    if set(train_ids) & set(validation_ids):
        raise RepresentationContractError("core-recovery train and validation roles overlap")


def _checkpoint_payload(**values):
    model = values["model"]
    return {
        "format": CORE_RECOVERY_CHECKPOINT_FORMAT,
        "training_status": values["status"],
        "method_name": model.method_name,
        "candidate_id": values["candidate_id"],
        "config": asdict(values["config"]),
        "source_checkpoint_path": str(values["source_checkpoint_path"]),
        "source_checkpoint_sha256": sha256_file(values["source_checkpoint_path"]),
        "model_state_dict": model.state_dict(),
        "normalizer": dict(model.normalizer.to_manifest()),
        "train_sample_ids": list(values["train_ids"]),
        "validation_sample_ids": list(values["validation_ids"]),
        "completed_steps": values["completed_steps"],
        "sample_exposure_count": values["completed_steps"] * values["config"].batch_size,
        "best_step": values["best_step"],
        "best_validation_metrics": dict(values["best_metrics"]),
        "training_rows": list(values["rows"]),
        "training_elapsed_s": values["elapsed_s"],
        "task_head_priors": dict(values["task_head_priors"]),
        "representation_family": "task_aware_core_recovery_v1",
        "label_used_for_encoder_training": True,
        "outer_test_accessed": False,
        "reader_visible_model_name": "Chronaris" if model.method_name == "chronaris" else model.method_name,
    }


def _atomic_save(path, payload):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
