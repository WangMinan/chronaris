"""Label-aware auxiliary fine-tuning kept separate from frozen representations."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    ApplicationConsumerSmokeTargets,
)
from chronaris.evaluation.application_tasks.application_consumers import (
    CausalTCNEmissionModel,
    TCNConsumerConfig,
)
from chronaris.modeling.fusion_encoders import NaiveTimeSyncEncoder
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training import TrainableFusionEncoder
from chronaris.modeling.training.rng import isolated_training_rng, capture_rng_state, restore_rng_state
from chronaris.modeling.training.candidate_checkpoint import candidate_source_code_sha256
from chronaris.representation import (
    DualStreamObservationBatch,
    TrainOnlyRobustNormalizer,
    select_observation_batch,
)
from chronaris.representation.contracts import FUSION_OUTPUT_DIM, RepresentationContractError
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


from chronaris.evaluation.application_tasks.application_task_heads import (
    ApplicationTaskDefinition, ApplicationTaskTargets, SIMULATION_TASKS, application_targets,
    build_application_heads, fit_application_task_parameters, select_application_targets, application_task_losses,
)

FINETUNING_FORMAT = "chronaris.application_end_to_end_finetuning.v3"


@dataclass(frozen=True, slots=True)
class EndToEndFineTuningConfig:
    learning_rate: float = 1e-4
    max_epochs: int = 20
    patience: int = 5
    batch_size: int = 128
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    seed: int = 17
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(self.learning_rate, self.gradient_clip_norm) <= 0:
            raise ValueError("fine-tuning learning rate and gradient clip must be positive")
        if min(self.max_epochs, self.patience, self.batch_size) <= 0:
            raise ValueError("fine-tuning epochs, patience, and batch size must be positive")
        if self.weight_decay < 0 or self.device not in {"cpu", "cuda"}:
            raise ValueError("fine-tuning optimizer or device configuration is invalid")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("fine-tuning requested unavailable CUDA device")


@dataclass(frozen=True, slots=True)
class EndToEndFineTuningResult:
    method_name: str
    status: str
    best_checkpoint_path: str
    last_checkpoint_path: str
    protocol_sha256: str
    best_epoch: int
    completed_epochs: int
    stopped_early: bool
    training_elapsed_s: float
    encoder_update_mode: str
    training_device_history: tuple[str, ...]
    epoch_rows: tuple[Mapping[str, object], ...]


class EndToEndApplicationModel(nn.Module):
    """A common linear workload head and two-block causal segmentation head."""

    def __init__(
        self,
        *,
        method_name: str,
        encoder: TrainableFusionEncoder | None,
        normalizer: TrainOnlyRobustNormalizer | None,
        naive_encoder: NaiveTimeSyncEncoder | None,
        task_definitions: tuple[ApplicationTaskDefinition, ...] = SIMULATION_TASKS,
    ) -> None:
        super().__init__()
        if (encoder is None) == (naive_encoder is None):
            raise ValueError("fine-tuning model requires exactly one encoder kind")
        if encoder is not None and encoder.method_name != method_name:
            raise ValueError("fine-tuning encoder method mismatch")
        if encoder is not None and normalizer is None:
            raise ValueError("trainable fine-tuning encoder requires a normalizer")
        self.method_name = method_name
        self.encoder = encoder
        self.normalizer = normalizer
        self.naive_encoder = naive_encoder
        self.task_definitions = task_definitions
        self.task_heads = build_application_heads(task_definitions)

    @property
    def workload_classifier(self):
        return self.task_heads["classification"]

    @property
    def workload_regressor(self):
        return self.task_heads["regression"]

    @property
    def segmentation_head(self):
        return self.task_heads["segmentation"]

    @property
    def encoder_update_mode(self) -> str:
        return "full_backbone" if self.encoder is not None else "head_only_nonparametric_encoder"

    def encode_with_mask(self, raw: DualStreamObservationBatch) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        if self.encoder is not None:
            normalized = move_observation_batch(
                self.normalizer.transform(raw),
                device=device,
            )
            encoded = self.encoder(normalized)
            return encoded.sequence_embedding, encoded.modality_available_mask
        sequence, available = self.naive_encoder.encode(raw)
        return sequence.to(device), available.to(device)

    def encode(self, raw: DualStreamObservationBatch) -> torch.Tensor:
        return self.encode_with_mask(raw)[0]

    def forward(self, raw: DualStreamObservationBatch) -> Mapping[str, torch.Tensor]:
        sequence, valid = self.encode_with_mask(raw)
        sequence = sequence.masked_fill(~valid.unsqueeze(-1), 0)
        pooled = sequence.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        predictions = {task.name: self.task_heads[task.name](
            sequence if task.kind == "sequence_classification" else pooled) for task in self.task_definitions}
        output = {"sequence_embedding": sequence, "pooled_embedding": pooled,
                  "valid_mask": valid, "task_predictions": predictions}
        if self.task_definitions == SIMULATION_TASKS:
            output.update(workload_logits=predictions["classification"],
                workload_regression_z=predictions["regression"].squeeze(-1), maneuver_logits=predictions["segmentation"])
        return output


def train_end_to_end_application_method(
    *,
    model: EndToEndApplicationModel,
    batch: DualStreamObservationBatch,
    targets: ApplicationConsumerSmokeTargets,
    role_sample_ids: Mapping[str, Sequence[str]],
    source_checkpoint_path: str | Path,
    output_root: str | Path,
    config: EndToEndFineTuningConfig,
    resume: bool = True,
) -> EndToEndFineTuningResult:
    targets = application_targets(targets)
    with isolated_training_rng(config.seed):
        last_path = Path(output_root) / model.method_name / "last.pt"
        if not (resume and last_path.is_file()):
            for head in model.task_heads.values():
                for module in head.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()
        return _train_end_to_end_application_method(
            model=model, batch=batch, targets=targets, role_sample_ids=role_sample_ids,
            source_checkpoint_path=source_checkpoint_path, output_root=output_root,
            config=config, resume=resume,
        )


def _train_end_to_end_application_method(
    *, model, batch, targets, role_sample_ids, source_checkpoint_path,
    output_root, config, resume,
):
    """Tune one method on train labels and select epochs on validation labels only."""
    source_path = Path(source_checkpoint_path)
    _validate_inputs(model, batch, targets, role_sample_ids, source_path)
    root = Path(output_root) / model.method_name
    best_path = root / "best.pt"
    last_path = root / "last.pt"
    protocol_hash = _protocol_hash(
        model=model,
        config=config,
        targets=targets,
        role_sample_ids=role_sample_ids,
        source_checkpoint_path=source_path,
    )
    root.mkdir(parents=True, exist_ok=True)
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    resume_payload = None
    if resume and last_path.is_file():
        resume_payload = torch.load(last_path, map_location="cpu", weights_only=True)
        if resume_payload.get("format") != FINETUNING_FORMAT or "rng_state" not in resume_payload:
            raise RepresentationContractError("legacy fine-tuning checkpoint is inference-only")
        if (
            resume_payload.get("protocol_sha256") != protocol_hash
            and not _resume_is_device_only_migration(
                resume_payload,
                config=config,
                source_path=source_path,
                role_sample_ids=role_sample_ids,
                targets=targets,
                model=model,
            )
        ):
            raise RepresentationContractError("fine-tuning protocol changed")
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        restore_rng_state(resume_payload["rng_state"])
        if resume_payload.get("training_status") == "completed":
            best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
            model.load_state_dict(best_payload["model_state_dict"], strict=True)
            return _result(resume_payload, best_path, last_path, status="resumed")
    target_index = {sample_id: index for index, sample_id in enumerate(targets.sample_ids)}
    train_ids = tuple(role_sample_ids["train"])
    validation_ids = tuple(role_sample_ids["validation"])
    task_parameters = fit_application_task_parameters(targets, model.task_definitions, train_ids)
    legacy_regression = task_parameters["tasks"].get("regression", {})
    regression_mean = legacy_regression.get("center", [0.])[0]
    regression_std = legacy_regression.get("scale", [1.])[0]
    epoch_rows = list(resume_payload.get("epoch_rows", ())) if resume_payload else []
    best_loss = float(resume_payload.get("best_validation_loss", float("inf"))) if resume_payload else float("inf")
    best_epoch = int(resume_payload.get("best_epoch", 0)) if resume_payload else 0
    stale = int(resume_payload.get("epochs_without_improvement", 0)) if resume_payload else 0
    completed_epochs = int(resume_payload.get("completed_epochs", 0)) if resume_payload else 0
    step_count = int(resume_payload.get("step_count", 0)) if resume_payload else 0
    elapsed_offset = float(resume_payload.get("training_elapsed_s", 0.0)) if resume_payload else 0.0
    device_history = list(
        resume_payload.get(
            "training_device_history",
            [resume_payload.get("config", {}).get("device", "cpu")],
        )
        if resume_payload
        else [config.device]
    )
    if device_history[-1] != config.device:
        device_history.append(config.device)
    started = time.perf_counter()
    for epoch in range(completed_epochs + 1, config.max_epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(config.seed * 10_000 + epoch)
        permutation = torch.randperm(len(train_ids), generator=generator).tolist()
        train_totals = {task.name: 0. for task in model.task_definitions}
        train_counts = dict(train_totals)
        for offset in range(0, len(permutation), config.batch_size):
            positions = permutation[offset : offset + config.batch_size]
            sample_ids = tuple(train_ids[index] for index in positions)
            raw = select_observation_batch(batch, sample_ids)
            selected = select_application_targets(targets, sample_ids, config.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(raw)
            losses = application_task_losses(output, selected, model.task_definitions, task_parameters)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("non-finite task-guided loss; unit stopped")
            losses["total"].backward()
            gradient_norm = float(
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True)
            )
            optimizer.step()
            step_count += 1
            count = len(sample_ids)
            for key in train_totals:
                train_totals[key] += float(losses[key].detach()) * losses["counts"][key]
                train_counts[key] += losses["counts"][key]
        validation_losses = _evaluate_losses(
            model=model,
            batch=batch,
            targets=targets,
            sample_ids=validation_ids,
            target_index=target_index,
            batch_size=config.batch_size,
            device=config.device,
            task_parameters=task_parameters,
        )
        selection_loss = sum(validation_losses.values()) / len(validation_losses)
        improved = selection_loss < best_loss
        if improved:
            best_loss = selection_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        completed_epochs = epoch
        epoch_rows.append(
            {
                "epoch": epoch,
                **{f"train_{key}_loss": value / max(train_counts[key], 1e-12) for key, value in train_totals.items()},
                **{f"validation_{key}_loss": value for key, value in validation_losses.items()},
                "validation_selection_loss": selection_loss,
                "gradient_norm_before_clip_last_step": gradient_norm,
                "improved": improved,
                "epochs_without_improvement": stale,
            }
        )
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            protocol_hash=protocol_hash,
            source_path=source_path,
            role_sample_ids=role_sample_ids,
            target_manifest=targets.manifest,
            task_parameters=task_parameters,
            target_sha256=_task_target_sha256(targets),
            regression_mean=regression_mean,
            regression_std=regression_std,
            epoch_rows=epoch_rows,
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            completed_epochs=completed_epochs,
            step_count=step_count,
            stale=stale,
            elapsed=elapsed_offset + time.perf_counter() - started,
            training_status="running",
            device_history=device_history,
        )
        if improved:
            _atomic_save(best_path, payload)
        _atomic_save(last_path, payload)
        if stale >= config.patience:
            break
    final_payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        protocol_hash=protocol_hash,
        source_path=source_path,
        role_sample_ids=role_sample_ids,
        target_manifest=targets.manifest,
        task_parameters=task_parameters,
        target_sha256=_task_target_sha256(targets),
        regression_mean=regression_mean,
        regression_std=regression_std,
        epoch_rows=epoch_rows,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        completed_epochs=completed_epochs,
        step_count=step_count,
        stale=stale,
        elapsed=elapsed_offset + time.perf_counter() - started,
        training_status="completed",
        device_history=device_history,
    )
    _atomic_save(last_path, final_payload)
    best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
    best_payload.update(training_status="completed", training_elapsed_s=final_payload["training_elapsed_s"])
    _atomic_save(best_path, best_payload)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    return _result(final_payload, best_path, last_path, status="completed")


def _evaluate_losses(**values):
    model = values["model"]
    model.eval()
    totals = {task.name: 0. for task in model.task_definitions}
    counts = dict(totals)
    ids = tuple(values["sample_ids"])
    with torch.inference_mode():
        for offset in range(0, len(ids), values["batch_size"]):
            sample_ids = ids[offset:offset + values["batch_size"]]
            raw = select_observation_batch(values["batch"], sample_ids)
            selected = select_application_targets(values["targets"], sample_ids, values["device"])
            losses = application_task_losses(model(raw), selected, model.task_definitions, values["task_parameters"])
            for key in totals:
                totals[key] += float(losses[key]) * losses["counts"][key]
                counts[key] += losses["counts"][key]
    if not any(counts.values()):
        raise RepresentationContractError("internal validation has no valid task labels")
    return {key: value / counts[key] for key, value in totals.items() if counts[key] > 0}


def _validate_inputs(model, batch, targets, roles, source_path):
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_payload = torch.load(source_path, map_location="cpu", weights_only=True)
    if bool(source_payload.get("label_used_for_encoder_training")):
        raise RepresentationContractError(
            "fine-tuning source checkpoint already used downstream labels"
        )
    source_method = source_payload.get("method_name")
    if source_method is not None and source_method != model.method_name:
        raise RepresentationContractError("fine-tuning source checkpoint method mismatch")
    expected = tuple(value for role in ("train", "validation", "held_out") for value in roles[role])
    if set(expected) != set(batch.sample_ids) or set(expected) != set(targets.sample_ids):
        raise RepresentationContractError("fine-tuning roles do not cover the fixed context set")
    if len(expected) != len(set(expected)):
        raise RepresentationContractError("fine-tuning roles overlap")
    if bool(targets.manifest.get("smoke_only", True)):
        return
    if model.method_name not in {"physiology_only", "vehicle_only", "naive_time_sync", "mult", "contiformer", "chronaris"}:
        raise ValueError("fine-tuning method is outside the fixed six-method set")


def _task_target_sha256(targets):
    target_hash = hashlib.sha256()
    for name in sorted(targets.values):
        target_hash.update(name.encode())
        target_hash.update(targets.values[name].detach().cpu().numpy().tobytes())
        target_hash.update(targets.valid_masks[name].detach().cpu().numpy().tobytes())
    if targets.sample_weights is not None:
        target_hash.update(targets.sample_weights.detach().cpu().numpy().tobytes())
    return target_hash.hexdigest()


def _protocol_hash(*, model, config, targets, role_sample_ids, source_checkpoint_path):
    payload = {
        "format": FINETUNING_FORMAT,
        "implementation_revision": "causal_fusion_v4",
        "source_code_sha256": _finetuning_source_sha256(),
        "method_name": model.method_name,
        "config": asdict(config),
        "source_checkpoint_sha256": sha256_file(source_checkpoint_path),
        "roles": {key: list(role_sample_ids[key]) for key in ("train", "validation", "held_out")},
        "target_sha256": _task_target_sha256(targets),
        "representation_family": "end_to_end_finetuned_v1",
        "task_definitions": [asdict(task) for task in model.task_definitions],
        "loss_reduction": "valid_samples_per_task_then_equal_tasks",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _checkpoint_payload(**values):
    model = values["model"]
    return {
        "format": FINETUNING_FORMAT,
        "implementation_revision": "causal_fusion_v4",
        "source_code_sha256": _finetuning_source_sha256(),
        "training_status": values["training_status"],
        "method_name": model.method_name,
        "seed": values["config"].seed,
        "protocol_sha256": values["protocol_hash"],
        "config": asdict(values["config"]),
        "source_checkpoint_path": str(values["source_path"]),
        "source_checkpoint_sha256": sha256_file(values["source_path"]),
        "representation_family": "end_to_end_finetuned_v1",
        "label_used_for_encoder_training": True,
        "encoder_update_mode": model.encoder_update_mode,
        "training_device_history": list(values["device_history"]),
        "role_sample_ids": {key: list(values["role_sample_ids"][key]) for key in ("train", "validation", "held_out")},
        "target_manifest": dict(values["target_manifest"]),
        "task_parameters": values["task_parameters"],
        "target_sha256": values["target_sha256"],
        "task_definitions": [asdict(task) for task in model.task_definitions],
        "encoder_backprop_uses_labels": model.encoder is not None,
        "selection_uses_validation_labels": True,
        "regression_train_mean": values["regression_mean"],
        "regression_train_std": values["regression_std"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": values["optimizer"].state_dict(),
        "rng_state": capture_rng_state(),
        "step_count": values["step_count"],
        "epoch_rows": list(values["epoch_rows"]),
        "best_epoch": values["best_epoch"],
        "best_validation_loss": values["best_validation_loss"],
        "completed_epochs": values["completed_epochs"],
        "epochs_without_improvement": values["stale"],
        "stopped_early": values["completed_epochs"] < values["config"].max_epochs,
        "training_elapsed_s": values["elapsed"],
    }


def _atomic_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _result(payload, best_path, last_path, *, status):
    return EndToEndFineTuningResult(
        method_name=str(payload["method_name"]),
        status=status,
        best_checkpoint_path=str(best_path),
        last_checkpoint_path=str(last_path),
        protocol_sha256=str(payload["protocol_sha256"]),
        best_epoch=int(payload["best_epoch"]),
        completed_epochs=int(payload["completed_epochs"]),
        stopped_early=bool(payload["stopped_early"]),
        training_elapsed_s=float(payload["training_elapsed_s"]),
        encoder_update_mode=str(payload["encoder_update_mode"]),
        training_device_history=tuple(
            payload.get(
                "training_device_history",
                [payload.get("config", {}).get("device", "cpu")],
            )
        ),
        epoch_rows=tuple(payload["epoch_rows"]),
    )


def _resume_is_device_only_migration(
    payload,
    *,
    config,
    source_path,
    role_sample_ids,
    targets,
    model,
) -> bool:
    stored_config = dict(payload.get("config", {}))
    expected_config = asdict(config)
    stored_config.pop("device", None)
    expected_config.pop("device", None)
    expected_roles = {
        key: list(role_sample_ids[key])
        for key in ("train", "validation", "held_out")
    }
    return all(
        (
            stored_config == expected_config,
            payload.get("source_code_sha256") == _finetuning_source_sha256(),
            payload.get("source_checkpoint_sha256") == sha256_file(source_path),
            payload.get("role_sample_ids") == expected_roles,
            payload.get("target_manifest") == dict(targets.manifest),
            payload.get("target_sha256") == _task_target_sha256(targets),
            payload.get("task_definitions") == [asdict(task) for task in model.task_definitions],
            payload.get("representation_family") == "end_to_end_finetuned_v1",
            payload.get("label_used_for_encoder_training") is True,
        )
    )


def _finetuning_source_sha256():
    digest = hashlib.sha256(candidate_source_code_sha256().encode())
    for name in ("application_finetuning.py", "application_finetuning_export.py", "application_consumers.py", "application_task_heads.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()
