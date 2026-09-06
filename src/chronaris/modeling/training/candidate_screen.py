"""Early-stopped, public-pretext-only encoder candidate screening."""
from __future__ import annotations
from contextlib import contextmanager
import math
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import nn
from chronaris.modeling.training.candidate_checkpoint import (
    atomic_save_candidate,
    build_candidate_checkpoint_payload,
    candidate_checkpoint_is_compatible,
    candidate_protocol_hash,
    candidate_source_code_sha256,
    candidate_data_sha256,
    load_candidate_payload,
    training_configs_match_ignoring_device,
)
from chronaris.modeling.training.candidate_mechanisms import (
    evaluate_candidate_mechanisms,
    interleaved_group_batch_ids,
    parameter_gradient_norm,
)
from chronaris.modeling.training.pretext import (
    CommonPretextHeadBundle,
    ExplicitTimeShiftHead,
    pretext_loss_terms_to_rows,
)
from chronaris.modeling.training.pretraining_encoders import (
    TRAINABLE_FUSION_METHODS,
    EncoderCandidateConfig,
    build_trainable_fusion_encoder,
)
from chronaris.modeling.training.rng import (
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
)
from chronaris.representation.contracts import FUSION_OUTPUT_DIM, RepresentationContractError


from chronaris.modeling.training.candidate_step import pretext_micro_step
from chronaris.modeling.training.candidate_validation import (
    PUBLIC_SELECTION_WEIGHTS, _evaluate_public_losses, _empty_loss_totals,
    _accumulate_loss_terms, _finalize_loss_totals, _public_selection_loss, _load_batch, _batch_ids,
)
_training_configs_match_ignoring_device = training_configs_match_ignoring_device

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
    max_ode_step_s: float | None = None
    ode_method: str = "euler"
    semantic_event_enabled: bool = False
    learnable_semantic_queries: bool = False
    heartbeat_interval_s: float = 30.0
    physics_calibration: Mapping[str, object] | None = None
    physics_weight: float = 0.1
    max_updates: int | None = None
    effective_batch_size: int | None = None
    validation_interval: int = 100
    validation_updates: tuple[int, ...] = ()
    minimum_updates: int = 500
    checkpoint_interval: int = 25
    early_stopping: bool = True

    def __post_init__(self) -> None:
        if self.max_epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("candidate screen epoch/batch/patience must be positive")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("candidate screen optimizer configuration is invalid")
        if self.physics_weight < 0:
            raise ValueError("candidate physics weight must be non-negative")
        if self.minimum_delta < 0:
            raise ValueError("candidate screen minimum delta must be non-negative")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("candidate screen device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("candidate screen requested unavailable CUDA device")
        if self.max_ode_step_s is not None and (
            not math.isfinite(self.max_ode_step_s) or self.max_ode_step_s <= 0
        ):
            raise ValueError("max_ode_step_s must be finite and positive when set")
        if self.ode_method not in {"euler", "midpoint", "rk4", "dopri5"}:
            raise ValueError("unsupported candidate Chronaris ODE method")
        if self.learnable_semantic_queries and not self.semantic_event_enabled:
            raise ValueError("learnable semantic queries require semantic_event_enabled")
        if not 0 < self.heartbeat_interval_s <= 60:
            raise ValueError("heartbeat_interval_s must be in (0,60]")
        if self.max_updates is not None and self.max_updates <= 0:
            raise ValueError("max_updates must be positive")
        if self.effective_batch_size is not None and (
            self.max_updates is None or self.effective_batch_size < self.batch_size
            or self.effective_batch_size % self.batch_size
        ):
            raise ValueError("effective batch must be a multiple of actual batch in update mode")
        if min(self.validation_interval, self.checkpoint_interval) <= 0 or self.minimum_updates < 0:
            raise ValueError("update validation/checkpoint schedule is invalid")
        if any(update <= 0 for update in self.validation_updates):
            raise ValueError("explicit validation updates must be positive")


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
    optimizer_updates: int = 0
    best_update: int = 0


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
    chronaris_variant: str = "full",
    chronaris_fusion_kind: str = "multiscale",
    chronaris_lag_aware_weight: float = 0.0,
    chronaris_mechanism_enabled: bool = False,
    chronaris_explicit_shift_enabled: bool = False,
    chronaris_explicit_shift_weight: float = 0.0,
    chronaris_event_pair_weight: float = 0.0,
    include_candidate_subdirectory: bool = True,
    resume: bool = True,
) -> CandidateScreenResult:
    resolved = config or CandidateScreenConfig()
    if resolved.physics_calibration is not None and (
        resolved.physics_calibration.get("fit_sample_hash") != normalizer.fit_sample_hash
        or set(resolved.physics_calibration.get("fit_sample_ids", ())) != set(fold.train_sample_ids)
    ):
        raise RepresentationContractError("physics calibration crossed the training fold")
    root = Path(output_root) / method_name
    if include_candidate_subdirectory:
        root /= candidate.candidate_id
    with _periodic_training_heartbeat(method_name, resolved.heartbeat_interval_s, root=root) as progress:
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
                chronaris_variant=chronaris_variant,
                chronaris_fusion_kind=chronaris_fusion_kind,
                chronaris_lag_aware_weight=chronaris_lag_aware_weight,
                chronaris_mechanism_enabled=chronaris_mechanism_enabled,
                chronaris_explicit_shift_enabled=chronaris_explicit_shift_enabled,
                chronaris_explicit_shift_weight=chronaris_explicit_shift_weight,
                chronaris_event_pair_weight=chronaris_event_pair_weight,
                include_candidate_subdirectory=include_candidate_subdirectory,
                resume=resume,
                progress=progress,
            )


@contextmanager
def _periodic_training_heartbeat(method_name: str, interval_s: float, *, root=None):
    stopped = threading.Event()
    started = time.perf_counter()
    progress = {"method": method_name, "status": "running", "optimizer_updates": 0}

    def emit_until_stopped() -> None:
        while not stopped.wait(interval_s):
            snapshot = dict(progress, wall_elapsed_s=time.perf_counter() - started)
            if root is not None:
                root.mkdir(parents=True, exist_ok=True)
                temporary = root / "progress.json.tmp"
                temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
                temporary.replace(root / "progress.json")
            print(
                f"[candidate-heartbeat] method={method_name} status=alive "
                f"updates={snapshot['optimizer_updates']} wall_elapsed_s={snapshot['wall_elapsed_s']:.1f}",
                flush=True,
            )

    thread = threading.Thread(target=emit_until_stopped, daemon=True)
    thread.start()
    try:
        yield progress
    except BaseException as error:
        progress.update(status="failed", error_type=type(error).__name__, reason=str(error))
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            (root / "failure.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
        raise
    finally:
        stopped.set()
        thread.join()
        if progress["status"] == "running":
            progress["status"] = "completed"
        progress["wall_elapsed_s"] = time.perf_counter() - started
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            (root / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")


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
    chronaris_variant: str = "full",
    chronaris_fusion_kind: str = "multiscale",
    chronaris_lag_aware_weight: float = 0.0,
    chronaris_mechanism_enabled: bool = False,
    chronaris_explicit_shift_enabled: bool = False,
    chronaris_explicit_shift_weight: float = 0.0,
    chronaris_event_pair_weight: float = 0.0,
    include_candidate_subdirectory: bool = True,
    resume: bool = True,
    progress: dict | None = None,
) -> CandidateScreenResult:
    """Train one frozen candidate without opening task labels or simulation truth."""

    if method_name not in TRAINABLE_FUSION_METHODS:
        raise ValueError(f"unsupported trainable method: {method_name}")
    if min(
        chronaris_lag_aware_weight,
        chronaris_explicit_shift_weight,
        chronaris_event_pair_weight,
    ) < 0:
        raise ValueError("Chronaris objective weights must be non-negative")
    if method_name != "chronaris" and (
        chronaris_variant != "full"
        or chronaris_fusion_kind != "multiscale"
        or chronaris_lag_aware_weight > 0
        or chronaris_mechanism_enabled
        or chronaris_explicit_shift_enabled
        or chronaris_explicit_shift_weight > 0
        or chronaris_event_pair_weight > 0
    ):
        raise ValueError("Chronaris-specific options require method_name='chronaris'")
    if not fold.train_sample_ids or not fold.validation_sample_ids:
        raise ValueError("candidate screen requires non-empty train and validation roles")
    if (batch is None) == (batch_provider is None):
        raise ValueError("provide exactly one of batch or batch_provider")
    resolved = config or CandidateScreenConfig()
    explicit_shift_enabled = (
        chronaris_explicit_shift_enabled or chronaris_explicit_shift_weight > 0
    )
    if chronaris_event_pair_weight > 0 and not resolved.semantic_event_enabled:
        raise ValueError("event-pair objective requires semantic event fusion")
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
    source_code_sha256 = candidate_source_code_sha256()
    source_data_sha256 = candidate_data_sha256(batch, batch_provider, fold, resolved.batch_size)
    protocol_hash = candidate_protocol_hash(
        source_data_sha256=source_data_sha256,
        source_code_sha256=source_code_sha256,
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
        chronaris_variant=chronaris_variant,
        chronaris_lag_aware_weight=chronaris_lag_aware_weight,
        chronaris_mechanism_enabled=chronaris_mechanism_enabled,
        chronaris_explicit_shift_enabled=explicit_shift_enabled,
        chronaris_explicit_shift_weight=chronaris_explicit_shift_weight,
        chronaris_event_pair_weight=chronaris_event_pair_weight,
    )
    resume_payload = None
    if resume and last_path.exists():
        last_payload = load_candidate_payload(last_path)
        if last_payload.get("source_data_sha256") != source_data_sha256:
            raise RepresentationContractError("candidate source data changed; refusing resume")
        if last_payload.get("format") != "chronaris.common_pretraining_checkpoint.v2":
            raise RepresentationContractError(
                "v1 pretraining checkpoints are inference-only and cannot resume"
            )
        if (
            last_payload.get("protocol_sha256") != protocol_hash
            and not candidate_checkpoint_is_compatible(
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
                chronaris_variant=chronaris_variant,
                chronaris_lag_aware_weight=chronaris_lag_aware_weight,
                chronaris_mechanism_enabled=chronaris_mechanism_enabled,
                chronaris_explicit_shift_enabled=explicit_shift_enabled,
                chronaris_explicit_shift_weight=chronaris_explicit_shift_weight,
                chronaris_event_pair_weight=chronaris_event_pair_weight,
            )
        ):
            raise RepresentationContractError(
                f"candidate screen checkpoint protocol changed for {method_name}/{candidate.candidate_id}"
            )
        if last_payload.get("training_status") == "completed":
            payload = load_candidate_payload(best_path)
            if progress is not None:
                progress.update(optimizer_updates=last_payload["step_count"],
                                best_update=last_payload.get("best_update", 0), checkpoint=str(last_path))
            return _result(payload, best_path, last_path, status="resumed")
        resume_payload = last_payload

    encoder = build_trainable_fusion_encoder(
        method_name,
        physiology_feature_names=physiology_feature_names,
        vehicle_feature_names=vehicle_feature_names,
        vehicle_field_labels=vehicle_field_labels,
        candidate_config=candidate,
        chronaris_fusion_kind=chronaris_fusion_kind,
        chronaris_variant=chronaris_variant,
        chronaris_max_ode_step_s=resolved.max_ode_step_s,
        chronaris_ode_method=resolved.ode_method,
        chronaris_semantic_event_enabled=resolved.semantic_event_enabled,
        chronaris_learnable_semantic_queries=resolved.learnable_semantic_queries,
        chronaris_physics_calibration=resolved.physics_calibration,
        chronaris_physics_weight=resolved.physics_weight,
    ).to(resolved.device)
    heads = CommonPretextHeadBundle(
        representation_dim=FUSION_OUTPUT_DIM,
        target_feature_count=len(physiology_feature_names) + len(vehicle_feature_names),
        modality_feature_counts=(len(physiology_feature_names), len(vehicle_feature_names)) if resolved.max_updates is not None else None,
        input_streams=(method_name.removesuffix("_only"),) if method_name.endswith("_only") else ("physiology", "vehicle"),
    ).to(resolved.device)
    shift_head = (
        ExplicitTimeShiftHead(FUSION_OUTPUT_DIM).to(resolved.device)
        if explicit_shift_enabled
        else None
    )
    trainable_parameters = tuple(encoder.parameters()) + tuple(heads.parameters())
    if shift_head is not None:
        trainable_parameters += tuple(shift_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=candidate.learning_rate,
        weight_decay=resolved.weight_decay,
    )
    transfer_initialization = None
    if resume_payload is not None:
        encoder.load_state_dict(resume_payload["encoder_state_dict"], strict=True)
        heads.load_state_dict(resume_payload["head_state_dict"], strict=True)
        if shift_head is not None:
            shift_head.load_state_dict(
                resume_payload["explicit_time_shift_head_state_dict"],
                strict=True,
            )
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
    best_update = int(resume_payload.get("best_update", 0)) if resume_payload else 0
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
    epochs_without_improvement = int(resume_payload.get("validation_checks_without_improvement",
        epoch_rows[-1]["epochs_without_improvement"] if epoch_rows else 0)) if resume_payload else 0
    step_count = int(resume_payload["step_count"]) if resume_payload is not None else 0
    total_data_wait_s = float(resume_payload.get("data_wait_s", 0)) if resume_payload else 0.
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
    update_mode = resolved.max_updates is not None
    start_epoch = (step_count + 1 if update_mode else
                   int(resume_payload["completed_epochs"]) + 1 if resume_payload else 1)
    limit = resolved.max_updates if update_mode else resolved.max_epochs
    accumulation = (resolved.effective_batch_size or resolved.batch_size) // resolved.batch_size
    data_cursor = dict(resume_payload.get("data_cursor", {})) if resume_payload else {}
    samples_seen = int(data_cursor.get("samples_seen", 0))
    micro_batches_seen = int(data_cursor.get("micro_batches_seen", 0))
    train_batches = (
        interleaved_group_batch_ids(
            batch,
            batch_provider,
            fold.train_sample_ids,
            resolved.batch_size,
        )
        if chronaris_event_pair_weight > 0 or update_mode
        else _batch_ids(fold.train_sample_ids, resolved.batch_size)
    )
    started = time.perf_counter()
    sampling_order = tuple(value for ids in train_batches for value in ids)
    if update_mode and len(sampling_order) < resolved.batch_size:
        raise ValueError("actual batch exceeds distinct training sample count")
    order_hash = candidate_protocol_hash(sample_ids=sampling_order)
    if data_cursor and data_cursor["sampling_order_sha256"] != order_hash:
        raise RepresentationContractError("resume sampling order changed")
    for epoch in range(start_epoch, limit + 1):
        encoder.train()
        heads.train()
        if shift_head is not None:
            shift_head.train()
        train_totals = _empty_loss_totals()
        gradient_norms = []
        if update_mode:
            active_batches = tuple(tuple(sampling_order[(samples_seen + i * resolved.batch_size + j)
                % len(sampling_order)] for j in range(resolved.batch_size)) for i in range(accumulation))
        else:
            active_batches = train_batches
        optimizer.zero_grad(set_to_none=True)
        update_rows_start = len(training_rows)
        for micro_index, sample_ids in enumerate(active_batches):
            augmentation_epoch = micro_batches_seen + 1 if update_mode else epoch
            output, mechanism_step, augmented, data_wait_s = pretext_micro_step(
                encoder=encoder, heads=heads, shift_head=shift_head, batch=batch,
                batch_provider=batch_provider, sample_ids=sample_ids, normalizer=normalizer,
                resolved=resolved, policy=policy, method_name=method_name, epoch=augmentation_epoch,
                chronaris_lag_aware_weight=chronaris_lag_aware_weight,
                chronaris_mechanism_enabled=chronaris_mechanism_enabled,
                chronaris_explicit_shift_weight=chronaris_explicit_shift_weight,
                chronaris_event_pair_weight=chronaris_event_pair_weight,
                optimizer_updates=step_count + 1 if update_mode else None,
            )
            total_loss = output.total_loss + mechanism_step.additional_loss
            total_data_wait_s += data_wait_s
            if not torch.isfinite(total_loss):
                root.mkdir(parents=True, exist_ok=True)
                (root / "failure.json").write_text(json.dumps({"reason": "nonfinite_loss",
                    "optimizer_updates": step_count, "sample_ids": sample_ids,
                    "micro_batches_seen": micro_batches_seen, "protocol_sha256": protocol_hash}))
                raise FloatingPointError("non-finite pretraining loss; unit stopped")
            (total_loss / (accumulation if update_mode else 1)).backward()
            global_step = step_count + 1
            loss_rows = [dict(row) for row in pretext_loss_terms_to_rows(output.terms)]
            loss_rows.extend(dict(row) for row in mechanism_step.rows)
            training_rows.extend(
                {
                    "method_name": method_name,
                    "candidate_id": candidate.candidate_id,
                    "epoch": epoch,
                    "step": global_step,
                    "batch_sample_ids": list(sample_ids),
                    "augmentation_ids": list(augmented.augmentation_ids),
                    "gradient_norm_before_clip": None,
                    "micro_batch_index": micro_batches_seen,
                    "augmentation_iteration": augmentation_epoch,
                    "related_parameter_gradient_norm": None,
                    "mechanism_metrics": mechanism_step.metrics_by_term.get(
                        row["term_name"]
                    ),
                    **row,
                }
                for row in loss_rows
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
            samples_seen += len(sample_ids)
            micro_batches_seen += 1
            complete_update = not update_mode or micro_index + 1 == accumulation
            if complete_update:
                norm = float(nn.utils.clip_grad_norm_(trainable_parameters,
                    resolved.gradient_clip_norm, error_if_nonfinite=True))
                gradient_norms.append(norm)
                encoder_norm = parameter_gradient_norm(encoder.parameters())
                shift_norm = parameter_gradient_norm(shift_head.parameters()) if shift_head is not None else None
                query_bank = getattr(getattr(getattr(encoder, "backbone", None), "semantic_event_fusion", None), "query_bank", None)
                semantic_norm = parameter_gradient_norm((query_bank.query_residual,)) if query_bank is not None else None
                for row in training_rows[update_rows_start:]:
                    row["gradient_norm_before_clip"] = norm
                    row["related_parameter_gradient_norm"] = (shift_norm if row["term_name"] == "explicit_time_shift"
                        else semantic_norm if row["term_name"] == "event_response_pairing" else encoder_norm)
                optimizer.step()
                step_count += 1
                if progress is not None:
                    progress.update(optimizer_updates=step_count, best_update=best_update,
                        sample_ids=list(sample_ids), samples_seen=samples_seen,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated() if resolved.device == "cuda" else 0,
                        checkpoint=str(last_path), device=resolved.device)
                    progress["data_wait_s"] = total_data_wait_s
                optimizer.zero_grad(set_to_none=True)
                update_rows_start = len(training_rows)
        improved = False
        validate = (not update_mode or step_count % resolved.validation_interval == 0
                    or step_count in resolved.validation_updates or step_count == limit)
        if validate:
            train_losses = _finalize_loss_totals(train_totals, allow_unavailable=update_mode)
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
            mechanism_validation = evaluate_candidate_mechanisms(
                encoder=encoder,
                shift_head=shift_head,
                batch=batch,
                batch_provider=batch_provider,
                sample_ids=fold.validation_sample_ids,
                batch_size=resolved.batch_size,
                normalizer=normalizer,
                policy=policy,
                seed=resolved.seed,
                device=resolved.device,
                mechanism_enabled=chronaris_mechanism_enabled,
                lag_aware_weight=chronaris_lag_aware_weight,
                explicit_shift_weight=chronaris_explicit_shift_weight,
                event_pair_weight=chronaris_event_pair_weight,
            )
            score = _public_selection_loss(validation_losses) + float(
                mechanism_validation["weighted_total"]
            )
            improved = score < best_score - resolved.minimum_delta
            if improved:
                best_score = score
                best_epoch = samples_seen // len(sampling_order) if update_mode else epoch
                best_update = step_count
                best_losses = dict(validation_losses)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            row = {
                "method_name": method_name,
                "candidate_id": candidate.candidate_id,
                "epoch": samples_seen // len(sampling_order) if update_mode else epoch,
                "optimizer_updates": step_count,
                "train_losses": train_losses,
                "validation_losses": validation_losses,
                "mechanism_validation": mechanism_validation,
                "public_selection_loss": score,
                "improved": improved,
                "epochs_without_improvement": epochs_without_improvement,
                "mean_gradient_norm_before_clip": sum(gradient_norms) / len(gradient_norms),
            }
            epoch_rows.append(row)
        if update_mode and not (validate or step_count % resolved.checkpoint_interval == 0):
            continue
        payload = build_candidate_checkpoint_payload(
            method_name=method_name,
            candidate=candidate,
            config=resolved,
            policy=policy,
            fold=fold,
            normalizer=normalizer,
            encoder=encoder,
            heads=heads,
            explicit_time_shift_head=shift_head,
            optimizer=optimizer,
            protocol_hash=protocol_hash,
            source_code_sha256=source_code_sha256,
            physiology_feature_names=physiology_feature_names,
            vehicle_feature_names=vehicle_feature_names,
            vehicle_field_labels=vehicle_field_labels,
            best_epoch=best_epoch,
            completed_epochs=samples_seen // len(sampling_order) if update_mode else epoch,
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
            chronaris_variant=chronaris_variant,
            chronaris_lag_aware_weight=chronaris_lag_aware_weight,
            chronaris_mechanism_enabled=chronaris_mechanism_enabled,
            chronaris_explicit_shift_enabled=(shift_head is not None),
            chronaris_explicit_shift_weight=chronaris_explicit_shift_weight,
            chronaris_event_pair_weight=chronaris_event_pair_weight,
            training_rows=training_rows,
            augmentation_rows=augmentation_rows,
            selection_weights=PUBLIC_SELECTION_WEIGHTS,
        )
        payload.update({
            "source_data_sha256": source_data_sha256,
            "data_wait_s": total_data_wait_s,
            "optimizer_updates": step_count, "total_optimizer_updates": step_count,
            "best_update": best_update, "validation_checks_without_improvement": epochs_without_improvement,
            "stage_update_counts": {"pretraining": step_count, "head_warmup": 0, "joint_adaptation": 0},
            "data_cursor": {"samples_seen": samples_seen, "micro_batches_seen": micro_batches_seen,
                            "sampling_order_sha256": order_hash},
            "actual_batch_size": resolved.batch_size,
            "effective_batch_size": resolved.effective_batch_size or resolved.batch_size,
            "update_counting": "optimizer.step",
        })
        if improved:
            atomic_save_candidate(best_path, payload)
        atomic_save_candidate(last_path, payload)
        if (validate and resolved.early_stopping and epochs_without_improvement >= resolved.patience
            and (not update_mode or step_count >= resolved.minimum_updates)):
            break
    elapsed = elapsed_offset + time.perf_counter() - started
    stopped_early = (step_count < limit) if update_mode else (epoch < limit)
    final_payload = load_candidate_payload(best_path)
    completion = {
            "training_status": "completed",
            "training_elapsed_s": elapsed,
            "completed_epochs": samples_seen // len(sampling_order) if update_mode else epoch,
            "total_optimizer_updates": step_count,
            "stopped_early": stopped_early,
            "epoch_rows": epoch_rows,
            "config": asdict(resolved),
            "training_device_history": device_history,
            "training_rows": training_rows,
            "augmentation_rows": augmentation_rows,
        }
    if not update_mode:
        completion["step_count"] = step_count
    final_payload.update(completion)
    atomic_save_candidate(best_path, final_payload)
    last_payload = load_candidate_payload(last_path)
    last_payload.update(completion)
    atomic_save_candidate(last_path, last_payload)
    return _result(final_payload, best_path, last_path, status="completed")


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
        optimizer_updates=int(payload.get("total_optimizer_updates", payload["step_count"])),
        best_update=int(payload.get("best_update", 0)),
    )
