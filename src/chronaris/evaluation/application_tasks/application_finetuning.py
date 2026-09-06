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
from chronaris.modeling.training.candidate_checkpoint import candidate_source_code_sha256, candidate_data_sha256
from chronaris.modeling.training.candidate_step import pretext_micro_step
from chronaris.modeling.training.candidate_validation import _load_batch
from chronaris.modeling.training.sample_schedule import training_sample_schedule
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
    effective_task_counts,
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
    max_updates: int | None = None
    effective_batch_size: int | None = None
    head_warmup_updates: int = 50
    head_learning_rate: float = 3e-4
    validation_interval: int = 50
    minimum_updates: int = 200
    checkpoint_interval: int = 25
    early_stopping: bool = True
    self_supervised_weight: float = .2
    sampling_hierarchy: Mapping[str, tuple[str, ...]] | None = None
    retained_updates: tuple[int, ...] = ()
    data_manifest_sha256: str | None = None
    cache_head_encodings: bool = True

    def __post_init__(self) -> None:
        if min(self.learning_rate, self.gradient_clip_norm) <= 0:
            raise ValueError("fine-tuning learning rate and gradient clip must be positive")
        if min(self.max_epochs, self.patience, self.batch_size) <= 0:
            raise ValueError("fine-tuning epochs, patience, and batch size must be positive")
        if self.weight_decay < 0 or self.device not in {"cpu", "cuda"}:
            raise ValueError("fine-tuning optimizer or device configuration is invalid")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("fine-tuning requested unavailable CUDA device")
        if self.max_updates is not None and self.max_updates <= 0:
            raise ValueError("fine-tuning max_updates must be positive")
        if self.effective_batch_size is not None and (self.max_updates is None
            or self.effective_batch_size < self.batch_size or self.effective_batch_size % self.batch_size):
            raise ValueError("fine-tuning effective batch must be a multiple of actual batch")
        if min(self.validation_interval, self.checkpoint_interval, self.head_learning_rate) <= 0:
            raise ValueError("invalid task-guided update schedule")
        if min(self.head_warmup_updates, self.minimum_updates, self.self_supervised_weight) < 0:
            raise ValueError("task-guided budgets/weights cannot be negative")
        if any(update <= 0 for update in self.retained_updates):
            raise ValueError("retained task-guided updates must be positive")


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
    optimizer_updates: int = 0
    head_warmup_updates: int = 0
    joint_updates: int = 0
    best_update: int = 0


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
        self.pretext_heads = None
        self.explicit_shift_head = None

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

    def forward(self, raw: DualStreamObservationBatch, *, encoding=None) -> Mapping[str, torch.Tensor]:
        sequence, valid = self.encode_with_mask(raw) if encoding is None else encoding
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
    batch_provider=None,
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
        from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
        with _periodic_training_heartbeat(model.method_name, 30., root=last_path.parent) as progress:
            result = _train_end_to_end_application_method(
                model=model, batch=batch, targets=targets, role_sample_ids=role_sample_ids,
                source_checkpoint_path=source_checkpoint_path, output_root=output_root,
                config=config, resume=resume, batch_provider=batch_provider, progress=progress,
            )
            progress.update(optimizer_updates=result.optimizer_updates, best_update=result.best_update,
                checkpoint=result.last_checkpoint_path, head_warmup_updates=result.head_warmup_updates,
                joint_updates=result.joint_updates)
            return result


def _train_end_to_end_application_method(
    *, model, batch, targets, role_sample_ids, source_checkpoint_path,
    output_root, config, resume, batch_provider, progress,
):
    """Tune one method on train labels and select epochs on validation labels only."""
    source_path = Path(source_checkpoint_path)
    if (batch is None) == (batch_provider is None):
        raise ValueError("fine-tuning requires exactly one observation source")
    _validate_inputs(model, batch, targets, role_sample_ids, source_path)
    update_mode = config.max_updates is not None
    source_objectives = _initialize_guided_objectives(model, source_path, config, role_sample_ids) if update_mode else None
    if source_objectives is not None:
        from chronaris.representation import FoldLineage
        fold = FoldLineage(fold_id=source_objectives["fold_id"],
            train_sample_ids=tuple(role_sample_ids["train"]), validation_sample_ids=tuple(role_sample_ids["validation"]),
            held_out_sample_ids=tuple(role_sample_ids["held_out"]),
            development_only=source_objectives["development_only"])
        if candidate_data_sha256(batch, batch_provider, fold, config.batch_size) != source_objectives["source_data_sha256"]:
            raise RepresentationContractError("task-guided source observations changed")
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
    parameters = ([{"params": model.task_heads.parameters(), "lr": config.head_learning_rate},
                   {"params": [parameter for name, parameter in model.named_parameters()
                               if not name.startswith("task_heads.")], "lr": config.learning_rate}]
                  if update_mode else model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
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
            progress.update(optimizer_updates=resume_payload["step_count"], checkpoint=str(last_path))
            return _result(resume_payload, best_path, last_path, status="resumed")
    target_index = {sample_id: index for index, sample_id in enumerate(targets.sample_ids)}
    train_ids = tuple(role_sample_ids["train"])
    validation_ids = tuple(role_sample_ids["validation"])
    task_parameters = fit_application_task_parameters(targets, model.task_definitions, train_ids)
    legacy_regression = task_parameters["tasks"].get("regression", {})
    regression_mean = legacy_regression.get("center", [0.])[0]
    regression_std = legacy_regression.get("scale", [1.])[0]
    epoch_rows = list(resume_payload.get("epoch_rows", ())) if resume_payload else []
    update_rows = list(resume_payload.get("update_rows", ())) if resume_payload else []
    best_loss = float(resume_payload.get("best_validation_loss", float("inf"))) if resume_payload else float("inf")
    best_epoch = int(resume_payload.get("best_epoch", 0)) if resume_payload else 0
    best_update = int(resume_payload.get("best_update", 0)) if resume_payload else 0
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
    accumulation = (config.effective_batch_size or config.batch_size) // config.batch_size
    cursor = dict(resume_payload.get("data_cursor", {})) if resume_payload else {}
    samples_seen, micro_batches_seen = int(cursor.get("samples_seen", 0)), int(cursor.get("micro_batches_seen", 0))
    if update_mode and len(train_ids) < config.batch_size:
        raise ValueError("actual task batch exceeds distinct training contexts")
    schedule = training_sample_schedule(batch, batch_provider, train_ids, config.batch_size, config.sampling_hierarchy) if update_mode else None
    if cursor and schedule is not None and cursor.get("sampling_order_sha256") != schedule.sha256:
        raise RepresentationContractError("task-guided resume sampling order changed")
    maximum = config.head_warmup_updates + config.max_updates if update_mode else config.max_epochs
    start_iteration = step_count + 1 if update_mode else completed_epochs + 1
    started = time.perf_counter()
    head_encodings = {}
    for epoch in range(start_iteration, maximum + 1):
        model.train()
        warming_head = update_mode and step_count < config.head_warmup_updates
        if update_mode:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith("task_heads.") or not warming_head)
            if warming_head and model.encoder is not None:
                model.encoder.eval()
            active_batches = tuple(schedule.draw(samples_seen + i * config.batch_size, config.batch_size) for i in range(accumulation))
        else:
            generator = torch.Generator().manual_seed(config.seed * 10_000 + epoch)
            permutation = torch.randperm(len(train_ids), generator=generator).tolist()
            active_batches = tuple(tuple(train_ids[index] for index in permutation[offset:offset + config.batch_size])
                                   for offset in range(0, len(permutation), config.batch_size))
        train_totals = {task.name: 0. for task in model.task_definitions}
        train_counts = dict(train_totals)
        effective_counts = effective_task_counts(targets, model.task_definitions, active_batches,
            batch=batch, provider=batch_provider, method_name=model.method_name) if update_mode else None
        active_tasks = sum(value > 0 for value in effective_counts.values()) if update_mode else 0
        optimizer.zero_grad(set_to_none=True)
        for micro_index, sample_ids in enumerate(active_batches):
            raw = _load_batch(batch, batch_provider, sample_ids)
            selected = select_application_targets(targets, sample_ids, config.device)
            encoding = None
            if warming_head and config.cache_head_encodings:
                # Exact batches retain the same padding and FP32 kernel shapes
                # when this ephemeral cache is rebuilt after an interruption.
                key = (raw.sample_ids, raw.source_sample_hashes)
                if key not in head_encodings:
                    with torch.no_grad():
                        head_encodings[key] = model.encode_with_mask(raw)
                encoding = head_encodings[key]
            else:
                head_encodings.clear()
            output = model(raw, encoding=encoding)
            losses = application_task_losses(output, selected, model.task_definitions, task_parameters)
            task_contribution = (sum(losses[name] * losses["counts"][name] / count
                for name, count in effective_counts.items() if count > 0) / active_tasks
                if update_mode and active_tasks else losses["total"] / (accumulation if update_mode else 1))
            total_loss = task_contribution
            public_loss, mechanism_loss, augmentation_ids = None, None, ()
            if update_mode and not warming_head and model.encoder is not None:
                public, mechanism, augmented, _data_wait = pretext_micro_step(
                    encoder=model.encoder, heads=model.pretext_heads, shift_head=model.explicit_shift_head,
                    batch=raw, batch_provider=None, sample_ids=sample_ids, normalizer=model.normalizer,
                    resolved=source_objectives["config"], policy=source_objectives["policy"],
                    method_name=model.method_name, epoch=micro_batches_seen + 1,
                    optimizer_updates=200, **source_objectives["mechanism_arguments"],
                )
                total_loss = total_loss + (config.self_supervised_weight * public.total_loss + mechanism.additional_loss) / accumulation
                public_loss, mechanism_loss = float(public.total_loss.detach()), float(mechanism.additional_loss.detach())
                augmentation_ids = augmented.augmentation_ids
            if not torch.isfinite(total_loss):
                raise FloatingPointError("non-finite task-guided loss; unit stopped")
            total_loss.backward()
            update_rows.append({"optimizer_update": step_count + 1, "micro_batch_index": micro_batches_seen,
                "stage": "head_warmup" if warming_head else "joint_adaptation", "sample_ids": list(sample_ids),
                "augmentation_ids": list(augmentation_ids), "task_loss": float(losses["total"].detach()),
                "task_valid_counts": losses["counts"],
                "task_update_contribution": float(task_contribution.detach()),
                "effective_task_counts": effective_counts,
                "public_loss": public_loss, "public_weight": config.self_supervised_weight if public_loss is not None else 0.,
                "mechanism_loss": mechanism_loss, "total_loss": float(total_loss.detach())})
            samples_seen += len(sample_ids)
            micro_batches_seen += 1
            if not update_mode or micro_index + 1 == accumulation:
                gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True))
                optimizer.step()
                step_count += 1
                optimizer.zero_grad(set_to_none=True)
                progress.update(optimizer_updates=step_count, stage="head_warmup" if warming_head else "joint_adaptation",
                    checkpoint=str(last_path), best_update=best_update, samples_seen=samples_seen,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if config.device == "cuda" else 0)
            count = len(sample_ids)
            for key in train_totals:
                train_totals[key] += float(losses[key].detach()) * losses["counts"][key]
                train_counts[key] += losses["counts"][key]
        completed_epochs = samples_seen // len(train_ids) if update_mode else epoch
        joint_updates = max(0, step_count - config.head_warmup_updates) if update_mode else step_count
        improved = False
        validate = (not update_mode or (not warming_head and
            (joint_updates % config.validation_interval == 0 or joint_updates == config.max_updates)))
        if validate:
            validation_losses = _evaluate_losses(
                model=model,
                batch=batch,
                batch_provider=batch_provider,
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
                best_epoch = completed_epochs
                best_update = step_count
                stale = 0
            else:
                stale += 1
            epoch_rows.append(
                {
                    "epoch": completed_epochs,
                    "optimizer_updates": step_count,
                    "joint_updates": joint_updates,
                    **{f"train_{key}_loss": value / max(train_counts[key], 1e-12) for key, value in train_totals.items()},
                    **{f"validation_{key}_loss": value for key, value in validation_losses.items()},
                    "validation_selection_loss": selection_loss,
                    "gradient_norm_before_clip_last_step": gradient_norm,
                    "improved": improved,
                    "epochs_without_improvement": stale,
                }
            )
        if update_mode and not (validate or step_count % config.checkpoint_interval == 0 or joint_updates in config.retained_updates):
            continue
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
            update_rows=update_rows,
            source_fold_id=source_objectives["fold_id"] if source_objectives else None,
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
        payload.update(_guided_update_metadata(config, step_count, samples_seen, micro_batches_seen, best_update))
        if schedule is not None:
            payload["data_cursor"]["sampling_order_sha256"] = schedule.sha256
        if improved:
            _atomic_save(best_path, payload)
        _atomic_save(last_path, payload)
        if joint_updates in config.retained_updates:
            _atomic_save(root / f"joint_update_{joint_updates:06d}.pt", payload)
        if (validate and config.early_stopping and stale >= config.patience
            and (not update_mode or joint_updates >= config.minimum_updates)):
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
        update_rows=update_rows,
        source_fold_id=source_objectives["fold_id"] if source_objectives else None,
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
    final_payload.update(_guided_update_metadata(config, step_count, samples_seen, micro_batches_seen, best_update))
    if schedule is not None:
        final_payload["data_cursor"]["sampling_order_sha256"] = schedule.sha256
    _atomic_save(last_path, final_payload)
    best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
    best_payload.update(training_status="completed", training_elapsed_s=final_payload["training_elapsed_s"])
    best_payload.update(total_optimizer_updates=step_count,
        total_stage_update_counts=final_payload["stage_update_counts"], stopped_early=final_payload["stopped_early"])
    _atomic_save(best_path, best_payload)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    return _result(final_payload, best_path, last_path, status="completed")


def _guided_update_metadata(config, updates, samples_seen, micro_batches_seen, best_update):
    warmup = min(updates, config.head_warmup_updates) if config.max_updates is not None else 0
    joint = updates - warmup
    return {"optimizer_updates": updates, "total_optimizer_updates": updates, "best_update": best_update,
        "stage_update_counts": {"pretraining": 0, "head_warmup": warmup, "joint_adaptation": joint},
        "data_cursor": {"samples_seen": samples_seen, "micro_batches_seen": micro_batches_seen},
        **({"stopped_early": joint < config.max_updates} if config.max_updates is not None else {})}


def _initialize_guided_objectives(model, source_path, config, roles):
    from chronaris.modeling.training import CandidateScreenConfig, load_common_pretraining_checkpoint
    from chronaris.modeling.training.pretext import ExplicitTimeShiftHead
    from chronaris.representation import AugmentationPolicy
    if model.encoder is None:
        return None
    source_encoder, heads, normalizer, payload = load_common_pretraining_checkpoint(source_path)
    if payload.get("source_code_sha256") != candidate_source_code_sha256() or heads.modality_feature_counts is None:
        raise RepresentationContractError("task-guided initialization requires the current v4 objective/source revision")
    if payload["seed"] != config.seed or payload["config"].get("max_updates") is None:
        raise RepresentationContractError("task-guided updates require the same-seed v4 pretraining source")
    if payload["config"].get("data_manifest_sha256") != config.data_manifest_sha256:
        raise RepresentationContractError("task-guided prepared data fingerprint changed")
    if model.encoder.config_manifest() != source_encoder.config_manifest():
        raise RepresentationContractError("task-guided source encoder architecture changed")
    if model.normalizer.to_manifest() != normalizer.to_manifest():
        raise RepresentationContractError("task-guided source normalization changed")
    for role in ("train", "validation"):
        if set(roles[role]) != set(payload["fold"][f"{role}_sample_ids"]):
            raise RepresentationContractError("task-guided source data roles changed")
    model.encoder.load_state_dict(source_encoder.state_dict(), strict=True)
    model.pretext_heads = heads
    if payload.get("chronaris_explicit_shift_enabled"):
        model.explicit_shift_head = ExplicitTimeShiftHead(FUSION_OUTPUT_DIM)
        model.explicit_shift_head.load_state_dict(payload["explicit_time_shift_head_state_dict"], strict=True)
    return {"config": CandidateScreenConfig(**(payload["config"] | {"device": config.device})),
        "policy": AugmentationPolicy(**payload["augmentation_policy"]),
        "source_data_sha256": payload["source_data_sha256"],
        "source_updates": payload["optimizer_updates"],
        "fold_id": payload["fold"]["fold_id"],
        "development_only": bool(payload["fold"].get("development_only", False)),
        "mechanism_arguments": {key: payload[key] for key in (
            "chronaris_lag_aware_weight", "chronaris_mechanism_enabled",
            "chronaris_explicit_shift_weight", "chronaris_event_pair_weight")}}


def _evaluate_losses(**values):
    model = values["model"]
    model.eval()
    totals = {task.name: 0. for task in model.task_definitions}
    counts = dict(totals)
    ids = tuple(values["sample_ids"])
    with torch.inference_mode():
        for offset in range(0, len(ids), values["batch_size"]):
            sample_ids = ids[offset:offset + values["batch_size"]]
            raw = _load_batch(values["batch"], values.get("batch_provider"), sample_ids)
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
    required = set(roles["train"]) | set(roles["validation"])
    if not required <= set(targets.sample_ids) <= set(expected):
        raise RepresentationContractError("fine-tuning targets do not cover allowed development roles")
    if batch is not None and not required <= set(batch.sample_ids) <= set(expected):
        raise RepresentationContractError("fine-tuning observations do not cover allowed development roles")
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
        "representation_family": "task_guided_v4" if config.max_updates is not None else "end_to_end_finetuned_v1",
        "task_definitions": [asdict(task) for task in model.task_definitions],
        "loss_reduction": "valid_samples_per_task_then_equal_tasks",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _checkpoint_payload(**values):
    model = values["model"]
    guided = values["config"].max_updates is not None
    encoder_labels = model.encoder is not None and any(row["stage"] == "joint_adaptation"
        and any(row["task_valid_counts"].values()) for row in values["update_rows"])
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
        "representation_family": "task_guided_v4" if guided else "end_to_end_finetuned_v1",
        "label_used_for_encoder_training": encoder_labels,
        "encoder_update_mode": model.encoder_update_mode,
        "training_device_history": list(values["device_history"]),
        "role_sample_ids": {key: list(values["role_sample_ids"][key]) for key in ("train", "validation", "held_out")},
        "target_manifest": dict(values["target_manifest"]),
        "task_parameters": values["task_parameters"],
        "target_sha256": values["target_sha256"],
        "task_definitions": [asdict(task) for task in model.task_definitions],
        "encoder_backprop_uses_labels": encoder_labels,
        "selection_uses_validation_labels": True,
        "fold_id": values.get("source_fold_id") or f"simulation_g1_to_g2_end_to_end__seed_{values['config'].seed}",
        "normalizer": model.normalizer.to_manifest() if model.normalizer is not None else None,
        "regression_train_mean": values["regression_mean"],
        "regression_train_std": values["regression_std"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": values["optimizer"].state_dict(),
        "rng_state": capture_rng_state(),
        "step_count": values["step_count"],
        "epoch_rows": list(values["epoch_rows"]),
        "update_rows": list(values["update_rows"]),
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
        optimizer_updates=int(payload.get("total_optimizer_updates", payload["step_count"])),
        head_warmup_updates=int(payload.get("stage_update_counts", {}).get("head_warmup", 0)),
        joint_updates=int(payload.get("stage_update_counts", {}).get("joint_adaptation", 0)),
        best_update=int(payload.get("best_update", 0)),
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
            payload.get("representation_family") == ("task_guided_v4" if config.max_updates is not None else "end_to_end_finetuned_v1"),
            payload.get("selection_uses_validation_labels") is True,
        )
    )


def _finetuning_source_sha256():
    digest = hashlib.sha256(candidate_source_code_sha256().encode())
    for name in ("application_finetuning.py", "application_finetuning_export.py", "application_consumers.py", "application_task_heads.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()
