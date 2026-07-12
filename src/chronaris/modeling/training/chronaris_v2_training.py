"""Task-independent phased training for Chronaris v2 candidates."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn

from chronaris.modeling.fusion_encoders.chronaris_v2 import (
    ChronarisV2EncoderConfig,
    ChronarisV2FusionEncoder,
)
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.candidate_screen import (
    PUBLIC_SELECTION_WEIGHTS,
)
from chronaris.modeling.training.chronaris_v2_curriculum import (
    chronaris_v2_augmentation_policy,
    chronaris_v2_curriculum_stage,
)
from chronaris.modeling.training.chronaris_v2_checkpointing import (
    ChronarisV2TrainingResult,
    build_v2_checkpoint_payload,
    load_v2_checkpoint,
    save_v2_checkpoint,
    v2_protocol_hash,
    v2_training_result,
)
from chronaris.modeling.training.chronaris_v2_distillation import (
    initialization_manifest as build_initialization_manifest,
    initialize_v2_repair_candidate,
    physiology_teacher_manifest,
    select_physiology_teacher_targets,
)
from chronaris.modeling.training.chronaris_v2_objectives import (
    ChronarisV2ObjectiveOutput,
    ChronarisV2ObjectiveWeights,
    ChronarisV2ObjectiveHeads,
    build_chronaris_v2_objective_targets,
    build_known_lag_batch,
    chronaris_v2_objective_weight_schedule,
)
from chronaris.modeling.training.chronaris_v2_selection import (
    chronaris_v2_structure_candidate,
    chronaris_v2_structure_candidates,
)
from chronaris.modeling.training.pcgrad import (
    GradientConflictController,
    loss_gradient_cosines,
    pcgrad_backward,
)
from chronaris.modeling.training.pretext import (
    CommonPretextHeadBundle,
    CommonPretextWeights,
)
from chronaris.modeling.training.pretraining_encoders import TrainableFusionEncoder
from chronaris.representation import (
    DualStreamObservationBatch,
    FoldLineage,
    FusionStreamBatch,
    TrainOnlyRobustNormalizer,
    apply_augmentation_realizations,
    build_batch_augmentation_realizations,
    build_common_pretext_targets,
    build_lag_discrimination_inputs,
    move_common_pretext_targets,
    select_observation_batch,
)
from chronaris.representation.contracts import RepresentationContractError


@dataclass(frozen=True, slots=True)
class ChronarisV2CandidateConfig:
    candidate_id: str
    internal_hidden_dim: int
    lag_mode: str
    ode_method: str
    learning_rate: float
    dropout: float = 0.1
    structure_candidate_id: str = "structure_08_complete_v2"
    physiology_teacher_mode: str = "none"
    physiology_teacher_weight: float = 0.0
    phase_epoch_offset: int = 0
    physiology_residual_mode: str = "learned"

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("v2 candidate_id is required")
        if self.internal_hidden_dim not in {64, 96, 128}:
            raise ValueError("v2 screen hidden dim must be 64, 96, or 128")
        if self.lag_mode not in {"fixed_five", "continuous_basis"}:
            raise ValueError("v2 candidate lag mode is invalid")
        if self.ode_method not in {"euler", "rk4"}:
            raise ValueError("v2 candidate ODE method is invalid")
        if self.learning_rate not in {1e-3, 3e-4}:
            raise ValueError("v2 candidate learning rate is outside the locked grid")
        if not 0 <= self.dropout < 1:
            raise ValueError("v2 candidate dropout is invalid")
        if self.physiology_teacher_mode not in {
            "none",
            "private",
            "physiology_path",
        }:
            raise ValueError("v2 physiology teacher mode is invalid")
        if self.physiology_teacher_weight < 0:
            raise ValueError("v2 physiology teacher weight must be non-negative")
        if (self.physiology_teacher_mode == "none") != (
            self.physiology_teacher_weight == 0
        ):
            raise ValueError("v2 physiology teacher mode/weight mismatch")
        if self.phase_epoch_offset not in {0, 20}:
            raise ValueError("v2 phase epoch offset must be 0 or 20")
        if self.physiology_residual_mode not in {
            "learned",
            "direct_causal_query",
        }:
            raise ValueError("v2 physiology residual mode is invalid")
        structure = chronaris_v2_structure_candidate(self.structure_candidate_id)
        if structure.architecture_version != "v2":
            raise ValueError("v1 structure candidate must use the v1 training pipeline")

    @property
    def num_heads(self) -> int:
        return 4 if self.internal_hidden_dim == 64 else 8


def chronaris_v2_hyperparameter_grid(
    *,
    physiology_residual_mode: str = "learned",
) -> tuple[ChronarisV2CandidateConfig, ...]:
    candidates = []
    for hidden in (64, 96, 128):
        for lag_mode in ("fixed_five", "continuous_basis"):
            for ode_method in ("euler", "rk4"):
                for learning_rate in (1e-3, 3e-4):
                    learning_rate_id = "1e3" if learning_rate == 1e-3 else "3e4"
                    candidates.append(
                        ChronarisV2CandidateConfig(
                            candidate_id=(
                                f"v2_h{hidden}_{lag_mode}_{ode_method}_lr{learning_rate_id}"
                            ),
                            internal_hidden_dim=hidden,
                            lag_mode=lag_mode,
                            ode_method=ode_method,
                            learning_rate=learning_rate,
                            physiology_residual_mode=physiology_residual_mode,
                        )
                    )
    return tuple(candidates)


def chronaris_v2_structure_training_grid() -> tuple[ChronarisV2CandidateConfig, ...]:
    """Return seven executable v2 candidates; v1 remains the external reference."""

    return tuple(
        ChronarisV2CandidateConfig(
            candidate_id=structure.candidate_id,
            internal_hidden_dim=64,
            lag_mode="fixed_five",
            ode_method="euler",
            learning_rate=1e-3,
            structure_candidate_id=structure.candidate_id,
        )
        for structure in chronaris_v2_structure_candidates()
        if structure.architecture_version == "v2"
    )


@dataclass(frozen=True, slots=True)
class ChronarisV2TrainingConfig:
    max_epochs: int = 50
    batch_size: int = 128
    patience: int = 8
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    seed: int = 17
    device: str = "cpu"
    minimum_epochs_before_early_stopping: int = 30

    def __post_init__(self) -> None:
        if min(
            self.max_epochs,
            self.batch_size,
            self.patience,
            self.minimum_epochs_before_early_stopping,
        ) <= 0:
            raise ValueError("v2 training epoch/batch/patience must be positive")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("v2 optimizer configuration is invalid")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("v2 training device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("v2 training requested unavailable CUDA")


def train_chronaris_v2_candidate(
    *,
    candidate: ChronarisV2CandidateConfig,
    batch: DualStreamObservationBatch | None,
    fold: FoldLineage,
    physiology_feature_names: tuple[str, ...],
    vehicle_feature_names: tuple[str, ...],
    vehicle_field_labels: tuple[tuple[str, str], ...],
    normalizer: TrainOnlyRobustNormalizer,
    output_root: str | Path,
    config: ChronarisV2TrainingConfig | None = None,
    batch_provider: Callable[[Sequence[str]], DualStreamObservationBatch] | None = None,
    physiology_teacher_targets: FusionStreamBatch | None = None,
    initialization_checkpoint: str | Path | None = None,
    resume: bool = True,
) -> ChronarisV2TrainingResult:
    resolved = config or ChronarisV2TrainingConfig()
    if (batch is None) == (batch_provider is None):
        raise ValueError("provide exactly one of batch or batch_provider")
    if (candidate.physiology_teacher_mode == "none") != (
        physiology_teacher_targets is None
    ):
        raise ValueError("physiology teacher targets do not match candidate protocol")
    teacher_manifest = physiology_teacher_manifest(physiology_teacher_targets)
    initialization_manifest = build_initialization_manifest(
        initialization_checkpoint,
        normalizer=normalizer,
    )
    root = Path(output_root) / candidate.candidate_id
    best_path = root / "best.pt"
    last_path = root / "last.pt"
    protocol_hash = v2_protocol_hash(
        candidate=asdict(candidate),
        config=asdict(resolved),
        fold=fold.to_dict(),
        normalizer=normalizer.to_manifest(),
        physiology_feature_names=physiology_feature_names,
        vehicle_feature_names=vehicle_feature_names,
        vehicle_field_labels=vehicle_field_labels,
        data_access_mode="provider" if batch_provider else "materialized",
        physiology_teacher=teacher_manifest,
        initialization=initialization_manifest,
    )
    resume_payload = None
    if resume and last_path.exists():
        resume_payload = load_v2_checkpoint(last_path, device=resolved.device)
        if resume_payload.get("protocol_sha256") != protocol_hash:
            raise RepresentationContractError("Chronaris v2 checkpoint protocol changed")
        if resume_payload.get("training_status") == "completed":
            return v2_training_result(
                load_v2_checkpoint(best_path),
                best_path,
                last_path,
                status="resumed",
            )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(resolved.seed)
        structure = chronaris_v2_structure_candidate(
            candidate.structure_candidate_id
        )
        physiology_hidden_dim = (
            max(16, candidate.internal_hidden_dim // 2)
            if structure.learned_causal_attention
            else candidate.internal_hidden_dim
        )
        backbone = ChronarisV2FusionEncoder(
            ChronarisV2EncoderConfig(
                physiology_feature_names=physiology_feature_names,
                vehicle_feature_names=vehicle_feature_names,
                field_labels=vehicle_field_labels,
                internal_hidden_dim=candidate.internal_hidden_dim,
                physiology_hidden_dim=physiology_hidden_dim,
                vehicle_hidden_dim=candidate.internal_hidden_dim,
                num_heads=candidate.num_heads,
                lag_mode=candidate.lag_mode,
                ode_method=candidate.ode_method,
                dropout=candidate.dropout,
                physics_enabled=structure.corrected_physics,
                learned_causal_attention=structure.learned_causal_attention,
                private_shared_subspaces=structure.private_shared_subspaces,
                corrected_physics=structure.corrected_physics,
                physiology_residual_mode=candidate.physiology_residual_mode,
            )
        )
        backbone.attach_normalizer(normalizer)
        encoder = TrainableFusionEncoder(
            method_name="chronaris",
            backbone=backbone,
        ).to(resolved.device)
        common_heads = CommonPretextHeadBundle(
            representation_dim=64,
            target_feature_count=(
                len(physiology_feature_names) + len(vehicle_feature_names)
            ),
        ).to(resolved.device)
        v2_heads = ChronarisV2ObjectiveHeads(
            physiology_feature_count=len(physiology_feature_names),
            vehicle_feature_count=len(vehicle_feature_names),
            physiology_teacher_mode=candidate.physiology_teacher_mode,
        ).to(resolved.device)
    if resume_payload is None and initialization_checkpoint is not None:
        initialize_v2_repair_candidate(
            initialization_checkpoint,
            encoder=encoder,
            common_heads=common_heads,
            v2_heads=v2_heads,
            device=resolved.device,
        )
    optimizer = torch.optim.AdamW(
        (
            {
                "params": tuple(backbone.physiology_stream.parameters())
                + tuple(backbone.vehicle_stream.parameters()),
                "lr": 1e-4,
            },
            {
                "params": tuple(backbone.vehicle_private_projection.parameters())
                + tuple(backbone.physiology_private_projection.parameters())
                + (
                    backbone.vehicle_private_missing,
                    backbone.physiology_private_missing,
                )
                + tuple(backbone.causal_fusion.parameters()),
                "lr": candidate.learning_rate,
            },
            {
                "params": tuple(common_heads.parameters()) + tuple(v2_heads.parameters()),
                "lr": candidate.learning_rate,
            },
        ),
        weight_decay=resolved.weight_decay,
    )
    if backbone.mixed_output_projection is not None:
        optimizer.add_param_group(
            {
                "params": tuple(backbone.mixed_output_projection.parameters()),
                "lr": candidate.learning_rate,
            }
        )
    controller = GradientConflictController()
    epoch_rows = list(resume_payload.get("epoch_rows", ())) if resume_payload else []
    gradient_rows = list(resume_payload.get("gradient_rows", ())) if resume_payload else []
    best_score = (
        float(resume_payload["best_public_selection_loss"])
        if resume_payload
        else float("inf")
    )
    best_epoch = int(resume_payload.get("best_epoch", 0)) if resume_payload else 0
    without_improvement = (
        int(epoch_rows[-1]["epochs_without_improvement"]) if epoch_rows else 0
    )
    start_epoch = int(resume_payload.get("completed_epochs", 0)) + 1 if resume_payload else 1
    elapsed_offset = float(resume_payload.get("training_elapsed_s", 0.0)) if resume_payload else 0.0
    if resume_payload:
        encoder.load_state_dict(resume_payload["encoder_state_dict"], strict=True)
        common_heads.load_state_dict(resume_payload["common_head_state_dict"], strict=True)
        v2_heads.load_state_dict(resume_payload["v2_head_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        controller.conflict_history.extend(resume_payload.get("conflict_history", ()))
    started = time.perf_counter()
    for epoch in range(start_epoch, resolved.max_epochs + 1):
        effective_epoch = epoch + candidate.phase_epoch_offset
        phase = _set_training_phase(backbone, effective_epoch)
        encoder.train()
        common_heads.train()
        v2_heads.train()
        epoch_total = 0.0
        step_count = 0
        curriculum = chronaris_v2_augmentation_policy(
            effective_epoch if structure.missingness_curriculum else 1
        )
        for sample_ids in _batch_ids(fold.train_sample_ids, resolved.batch_size):
            raw = _load_batch(batch, batch_provider, sample_ids)
            normalized_cpu = normalizer.transform(raw)
            plans = build_batch_augmentation_realizations(
                sample_ids,
                epoch=epoch,
                global_seed=resolved.seed,
                policy=curriculum,
            )
            augmented_cpu = apply_augmentation_realizations(
                normalized_cpu,
                plans,
                policy=curriculum,
            )
            common_targets = build_common_pretext_targets(
                normalized_cpu,
                augmented_cpu,
            )
            lag_negative_cpu = build_lag_discrimination_inputs(
                augmented_cpu.batch,
                augmented_cpu.augmentation_ids,
            )
            clean = move_observation_batch(normalized_cpu, device=resolved.device)
            augmented = move_observation_batch(
                augmented_cpu.batch,
                device=resolved.device,
            )
            negative = move_observation_batch(
                lag_negative_cpu.negative_batch,
                device=resolved.device,
            )
            common_targets = move_common_pretext_targets(
                common_targets,
                device=resolved.device,
            )
            optimizer.zero_grad(set_to_none=True)
            positive = encoder(augmented)
            negative_output = encoder(negative)
            common_output = common_heads(
                positive.sequence_embedding,
                negative_output.sequence_embedding,
                common_targets,
                weights=CommonPretextWeights(),
            )
            objective_enabled = (
                structure.lag_conditioned_objective and effective_epoch > 10
            )
            if objective_enabled:
                lag_labels = _known_lag_labels(
                    sample_ids,
                    epoch,
                    resolved.seed,
                ).to(resolved.device)
                known_lag = build_known_lag_batch(clean, lag_labels)
                clean_encoding = backbone(clean, compute_diagnostics=True)
                lag_encoding = backbone(known_lag, compute_diagnostics=False)
                v2_targets = build_chronaris_v2_objective_targets(clean)
                objective_weights = chronaris_v2_objective_weight_schedule(
                    effective_epoch,
                    physiology_teacher_weight=(
                        candidate.physiology_teacher_weight
                    ),
                )
                if not structure.corrected_physics:
                    objective_weights = replace(
                        objective_weights,
                        physical_consistency=0.0,
                    )
                teacher_sequence, teacher_mask = select_physiology_teacher_targets(
                    physiology_teacher_targets,
                    sample_ids,
                    device=resolved.device,
                )
                v2_output = v2_heads(
                    clean_encoding,
                    v2_targets,
                    weights=objective_weights,
                    corrupted_encoding=positive.auxiliary["chronaris_encoding"],
                    lag_encoding=lag_encoding,
                    lag_labels=lag_labels,
                    physiology_teacher_sequence=teacher_sequence,
                    physiology_teacher_mask=teacher_mask,
                )
            else:
                v2_output = ChronarisV2ObjectiveOutput(
                    total_loss=positive.sequence_embedding.sum() * 0.0,
                    terms=(),
                )
            named_losses = _grouped_optimization_losses(
                common_output,
                v2_output,
            )
            total_loss = common_output.total_loss + v2_output.total_loss
            shared_parameters = tuple(encoder.parameters())
            exclusive_parameters = tuple(
                (*common_heads.parameters(), *v2_heads.parameters())
            )
            cosines = {}
            if len(named_losses) > 1:
                if controller.use_pcgrad:
                    cosines = _shared_pcgrad_backward(
                        named_losses,
                        total_loss=total_loss,
                        shared_parameters=shared_parameters,
                        exclusive_parameters=exclusive_parameters,
                    )
                else:
                    cosines = loss_gradient_cosines(
                        named_losses,
                        shared_parameters,
                    )
                controller.observe_step(tuple(cosines.values()))
                gradient_rows.extend(
                    {
                        "epoch": epoch,
                        "step": step_count,
                        "phase": phase,
                        "loss_a": left,
                        "loss_b": right,
                        "cosine": value,
                        "conflict": value < controller.conflict_threshold,
                    }
                    for (left, right), value in cosines.items()
                )
                if controller.use_pcgrad and not any(
                    parameter.grad is not None for parameter in shared_parameters
                ):
                    _shared_pcgrad_backward(
                        named_losses,
                        total_loss=total_loss,
                        shared_parameters=shared_parameters,
                        exclusive_parameters=exclusive_parameters,
                    )
            if not controller.use_pcgrad or len(named_losses) <= 1:
                total_loss.backward()
            nn.utils.clip_grad_norm_(
                (*encoder.parameters(), *common_heads.parameters(), *v2_heads.parameters()),
                resolved.gradient_clip_norm,
            )
            optimizer.step()
            epoch_total += float(
                total_loss.detach()
            )
            step_count += 1
        validation_losses = _validation_losses(
            encoder=encoder,
            heads=common_heads,
            batch=batch,
            batch_provider=batch_provider,
            sample_ids=fold.validation_sample_ids,
            batch_size=resolved.batch_size,
            normalizer=normalizer,
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
            without_improvement = 0
        else:
            without_improvement += 1
        epoch_rows.append(
            {
                "epoch": epoch,
                "effective_epoch": effective_epoch,
                "phase": phase,
                "train_total_loss": epoch_total / max(step_count, 1),
                "validation_losses": validation_losses,
                "public_selection_loss": score,
                "improved": improved,
                "epochs_without_improvement": without_improvement,
                "curriculum_stage": chronaris_v2_curriculum_stage(
                    effective_epoch
                ).stage_index,
                "effective_curriculum_stage": (
                    chronaris_v2_curriculum_stage(effective_epoch).stage_index
                    if structure.missingness_curriculum
                    else 1
                ),
                "pcgrad_active": controller.use_pcgrad,
                "gradient_conflict_rate": controller.conflict_rate,
            }
        )
        payload = build_v2_checkpoint_payload(
            encoder=encoder,
            common_heads=common_heads,
            v2_heads=v2_heads,
            optimizer=optimizer,
            normalizer=normalizer,
            fold=fold,
            candidate=candidate,
            config=resolved,
            protocol_hash=protocol_hash,
            best_epoch=best_epoch,
            best_score=best_score,
            completed_epochs=epoch,
            epoch_rows=epoch_rows,
            gradient_rows=gradient_rows,
            controller=controller,
            elapsed=elapsed_offset + time.perf_counter() - started,
            physiology_teacher_manifest=teacher_manifest,
            initialization_manifest=initialization_manifest,
        )
        save_v2_checkpoint(last_path, payload)
        if improved:
            save_v2_checkpoint(best_path, payload)
        if (
            _early_stopping_allowed(epoch, resolved)
            and without_improvement >= resolved.patience
        ):
            break
    elapsed = elapsed_offset + time.perf_counter() - started
    final = load_v2_checkpoint(best_path)
    final.update(
        training_status="completed",
        completed_epochs=len(epoch_rows),
        stopped_early=len(epoch_rows) < resolved.max_epochs,
        training_elapsed_s=elapsed,
        epoch_rows=epoch_rows,
        gradient_rows=gradient_rows,
        conflict_history=controller.conflict_history,
    )
    save_v2_checkpoint(best_path, final)
    last = load_v2_checkpoint(last_path)
    last.update(
        training_status="completed",
        stopped_early=final["stopped_early"],
    )
    save_v2_checkpoint(last_path, last)
    return v2_training_result(final, best_path, last_path, status="completed")


def _set_training_phase(backbone: ChronarisV2FusionEncoder, epoch: int) -> str:
    if epoch <= 10:
        stream_trainable, fusion_trainable = True, False
        phase = "stream_pretraining"
    elif epoch <= 20:
        stream_trainable, fusion_trainable = False, True
        phase = "causal_fusion_training"
    else:
        stream_trainable = fusion_trainable = True
        phase = "joint_unfreeze"
    for module in (backbone.physiology_stream, backbone.vehicle_stream):
        for parameter in module.parameters():
            parameter.requires_grad_(stream_trainable)
    for module in (
        backbone.vehicle_private_projection,
        backbone.physiology_private_projection,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(stream_trainable)
    for parameter in backbone.causal_fusion.parameters():
        parameter.requires_grad_(fusion_trainable)
    if backbone.mixed_output_projection is not None:
        for parameter in backbone.mixed_output_projection.parameters():
            parameter.requires_grad_(fusion_trainable)
    return phase


def _early_stopping_allowed(
    epoch: int,
    config: ChronarisV2TrainingConfig,
) -> bool:
    return epoch >= min(
        config.max_epochs,
        config.minimum_epochs_before_early_stopping,
    )


def _grouped_optimization_losses(common_output, v2_output):
    """Group objectives as public, fidelity, causal, and physical losses."""

    common_terms = tuple(
        term.weighted_loss
        for term in common_output.terms
        if term.weighted_loss is not None
    )
    v2_terms = {
        term.name: term.weighted_loss
        for term in v2_output.terms
        if term.weight > 0
    }
    groups = {}
    if common_terms:
        groups["public_pretext"] = torch.stack(common_terms).sum()
    fidelity_names = (
        "vehicle_private_retention",
        "physiology_private_retention",
        "physiology_teacher_distillation",
        "clean_corruption_consistency",
    )
    causal_names = ("future_physiology_delta", "lag_bin_classification")
    fidelity = tuple(v2_terms[name] for name in fidelity_names if name in v2_terms)
    causal = tuple(v2_terms[name] for name in causal_names if name in v2_terms)
    if fidelity:
        groups["modality_fidelity"] = torch.stack(fidelity).sum()
    if causal:
        groups["lagged_causality"] = torch.stack(causal).sum()
    if "physical_consistency" in v2_terms:
        groups["physical_consistency"] = v2_terms["physical_consistency"]
    return groups


def _shared_pcgrad_backward(
    losses,
    *,
    total_loss,
    shared_parameters,
    exclusive_parameters,
):
    """Project shared gradients while leaving task-exclusive heads unprojected."""

    pairwise = pcgrad_backward(losses, shared_parameters)
    exclusive_gradients = torch.autograd.grad(
        total_loss,
        exclusive_parameters,
        allow_unused=True,
    )
    for parameter, gradient in zip(
        exclusive_parameters,
        exclusive_gradients,
        strict=True,
    ):
        parameter.grad = None if gradient is None else gradient.clone()
    return pairwise


def _validation_losses(**values):
    encoder, heads = values["encoder"], values["heads"]
    encoder.eval()
    heads.eval()
    totals = {name: [0.0, 0] for name in PUBLIC_SELECTION_WEIGHTS}
    policy = chronaris_v2_augmentation_policy(20)
    with torch.inference_mode():
        for sample_ids in _batch_ids(values["sample_ids"], values["batch_size"]):
            raw = _load_batch(values["batch"], values["batch_provider"], sample_ids)
            normalized = values["normalizer"].transform(raw)
            plans = build_batch_augmentation_realizations(
                sample_ids,
                epoch=0,
                global_seed=values["seed"],
                policy=policy,
            )
            augmented = apply_augmentation_realizations(normalized, plans, policy=policy)
            targets = build_common_pretext_targets(normalized, augmented)
            negative = build_lag_discrimination_inputs(
                augmented.batch,
                augmented.augmentation_ids,
            )
            positive_batch = move_observation_batch(
                augmented.batch,
                device=values["device"],
            )
            negative_batch = move_observation_batch(
                negative.negative_batch,
                device=values["device"],
            )
            targets = move_common_pretext_targets(targets, device=values["device"])
            output = heads(
                encoder(positive_batch).sequence_embedding,
                encoder(negative_batch).sequence_embedding,
                targets,
                weights=CommonPretextWeights(),
            )
            for term in output.terms:
                if term.raw_loss is not None:
                    totals[term.term_name][0] += float(term.raw_loss) * term.count
                    totals[term.term_name][1] += term.count
    return {
        name: total / count
        for name, (total, count) in totals.items()
    }


def _known_lag_labels(sample_ids, epoch, seed):
    return torch.tensor(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}\0{epoch}\0{sample_id}".encode()).digest()[:4],
                "little",
            )
            % 5
            for sample_id in sample_ids
        ],
        dtype=torch.long,
    )


def _load_batch(batch, provider, sample_ids):
    loaded = (
        provider(sample_ids)
        if provider is not None
        else select_observation_batch(batch, sample_ids)
    )
    if tuple(loaded.sample_ids) != tuple(sample_ids):
        raise RepresentationContractError("v2 batch provider changed sample order")
    return loaded


def _batch_ids(sample_ids, batch_size):
    values = tuple(sample_ids)
    return tuple(
        values[index : index + batch_size]
        for index in range(0, len(values), batch_size)
    )
