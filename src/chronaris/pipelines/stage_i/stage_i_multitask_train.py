"""Unified multitask training entry for the Stage I thesis mainline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import torch

from chronaris.dataset.stage_i_private_contracts import (
    StageIPrivateTaskEntry,
    dump_stage_i_private_task_entries,
)
from chronaris.features.experiment_input import E0ExperimentSample
from chronaris.models.alignment import (
    StageITaskHeadBatch,
    StageITaskHeadSet,
    StageITaskHeadSpec,
    build_stage_e_objective,
    build_task_loss_breakdown,
    split_e0_samples_chronologically,
)
from chronaris.models.fusion import (
    CausalFusionConfig,
    CausalFusionTensorInput,
    CausalMaskedCrossModalFusion,
    attention_entropy,
)
from chronaris.pipelines.alignment_preview import (
    AlignmentPreviewConfig,
    AlignmentPreviewPipeline,
)
from chronaris.pipelines.alignment_physics import build_batch_physics_context


def _default_multitask_preview_config() -> AlignmentPreviewConfig:
    return AlignmentPreviewConfig(
        epoch_count=3,
        batch_size=8,
        learning_rate=1e-3,
        device="auto",
        reconstruction_loss_mode="relative_mse",
        input_normalization_mode="zscore_train",
        alignment_loss_mode="mse",
        enable_physics_constraints=True,
        physics_constraint_mode="feature_first_with_latent_fallback",
        physics_constraint_family="full",
        vehicle_physics_weight=0.1,
        physiology_physics_weight=0.1,
        export_intermediate_states=False,
        intermediate_sample_limit=None,
        intermediate_partition="all",
    )


@dataclass(frozen=True, slots=True)
class StageIMultitaskTrainConfig:
    """Configuration for one multitask joint-training run."""

    run_id: str
    output_root: str | Path = "docs/artifacts/assets/stage_i_multitask"
    preview_config: AlignmentPreviewConfig = field(default_factory=_default_multitask_preview_config)
    task_head_hidden_dim: int = 32
    retrieval_embedding_dim: int = 16
    task_loss_weight: float = 1.0
    task_weights: Mapping[str, float] = field(default_factory=dict)
    causal_weight: float = 0.1
    causal_attention_temperature: float = 1.0
    causal_event_bias_weight: float = 0.25
    causal_lag_window_points: int | None = None
    checkpoint_filename: str = "multitask_checkpoint.pt"
    summary_filename: str = "multitask_summary.json"
    task_manifest_filename: str = "thesis_task_manifest.jsonl"

    def __post_init__(self) -> None:
        if self.task_head_hidden_dim <= 0:
            raise ValueError("task_head_hidden_dim must be positive.")
        if self.retrieval_embedding_dim <= 0:
            raise ValueError("retrieval_embedding_dim must be positive.")
        if self.task_loss_weight < 0:
            raise ValueError("task_loss_weight must be non-negative.")
        if self.causal_weight < 0:
            raise ValueError("causal_weight must be non-negative.")


@dataclass(frozen=True, slots=True)
class StageIMultitaskMetrics:
    """Average multitask losses for one partition."""

    sample_count: int
    batch_count: int
    physiology_reconstruction: float
    vehicle_reconstruction: float
    reconstruction_total: float
    alignment: float
    vehicle_physics: float
    physiology_physics: float
    physics_total: float
    causal_total: float
    task_total: float
    total: float
    physics_components: dict[str, float] = field(default_factory=dict)
    causal_components: dict[str, float] = field(default_factory=dict)
    task_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StageIMultitaskTrainRunResult:
    """Artifacts written by one multitask train run."""

    artifact_root: str
    checkpoint_path: str
    summary_path: str
    task_manifest_path: str
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ResolvedTaskDefinition:
    task_name: str
    task_type: str
    spec: StageITaskHeadSpec
    label_to_id: Mapping[str, int]


@dataclass(slots=True)
class StageIMultitaskTrainPipeline:
    """Train one shared alignment/causal backbone with thesis-facing task heads."""

    config: StageIMultitaskTrainConfig

    def run(
        self,
        *,
        samples: tuple[E0ExperimentSample, ...],
        task_entries: Sequence[StageIPrivateTaskEntry],
        source_summary: Mapping[str, object] | None = None,
    ) -> StageIMultitaskTrainRunResult:
        if not samples:
            raise ValueError("StageIMultitaskTrainPipeline requires at least one sample.")
        if not task_entries:
            raise ValueError("StageIMultitaskTrainPipeline requires at least one task entry.")

        preview_pipeline = AlignmentPreviewPipeline(config=self.config.preview_config)
        chronological_split = split_e0_samples_chronologically(
            samples,
            config=self.config.preview_config.split_config,
        )
        if not chronological_split.train:
            raise ValueError("multitask training requires at least one training sample.")

        normalization_stats = preview_pipeline._build_input_normalization_stats(chronological_split.train)
        model = preview_pipeline._build_model(
            chronological_split.train,
            prototype_config=self.config.preview_config.prototype_config,
        )
        task_definitions = _resolve_task_definitions(
            task_entries,
            input_dim=self.config.preview_config.prototype_config.projection_dim * 3,
            hidden_dim=self.config.task_head_hidden_dim,
            retrieval_embedding_dim=self.config.retrieval_embedding_dim,
        )
        task_heads = StageITaskHeadSet([definition.spec for definition in task_definitions.values()]).to(
            device=preview_pipeline._resolved_device_name(),
            dtype=self.config.preview_config.dtype,
        )
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(task_heads.parameters()),
            lr=self.config.preview_config.learning_rate,
        )
        physics_stats = preview_pipeline._build_physics_constraint_stats(
            chronological_split.train,
            normalization_stats=normalization_stats,
        )
        causal_fusion = CausalMaskedCrossModalFusion(
            CausalFusionConfig(
                attention_temperature=self.config.causal_attention_temperature,
                event_bias_weight=self.config.causal_event_bias_weight,
                use_causal_mask=True,
                lag_window_points=self.config.causal_lag_window_points,
            )
        ).to(device=preview_pipeline._resolved_device_name())

        train_history: list[StageIMultitaskMetrics] = []
        validation_history: list[StageIMultitaskMetrics] = []
        for _epoch_index in range(self.config.preview_config.epoch_count):
            train_history.append(
                self._run_partition(
                    model=model,
                    task_heads=task_heads,
                    causal_fusion=causal_fusion,
                    preview_pipeline=preview_pipeline,
                    samples=chronological_split.train,
                    task_entries=task_entries,
                    task_definitions=task_definitions,
                    optimizer=optimizer,
                    training=True,
                    normalization_stats=normalization_stats,
                    physics_stats=physics_stats,
                )
            )
            validation_history.append(
                self._run_partition(
                    model=model,
                    task_heads=task_heads,
                    causal_fusion=causal_fusion,
                    preview_pipeline=preview_pipeline,
                    samples=chronological_split.validation,
                    task_entries=task_entries,
                    task_definitions=task_definitions,
                    optimizer=None,
                    training=False,
                    normalization_stats=normalization_stats,
                    physics_stats=physics_stats,
                )
            )

        evaluation_samples = (
            chronological_split.test
            or chronological_split.validation
            or chronological_split.train
        )
        test_metrics = self._run_partition(
            model=model,
            task_heads=task_heads,
            causal_fusion=causal_fusion,
            preview_pipeline=preview_pipeline,
            samples=evaluation_samples,
            task_entries=task_entries,
            task_definitions=task_definitions,
            optimizer=None,
            training=False,
            normalization_stats=normalization_stats,
            physics_stats=physics_stats,
        )

        artifact_root = Path(self.config.output_root) / self.config.run_id
        artifact_root.mkdir(parents=True, exist_ok=True)
        task_manifest_path = artifact_root / self.config.task_manifest_filename
        dump_stage_i_private_task_entries(task_entries, path=task_manifest_path)
        checkpoint_path = artifact_root / self.config.checkpoint_filename
        summary_path = artifact_root / self.config.summary_filename
        checkpoint_metadata = self._save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            task_heads=task_heads,
            task_definitions=task_definitions,
            normalization_stats=normalization_stats,
            train_history=train_history,
            validation_history=validation_history,
            test_metrics=test_metrics,
        )
        summary = {
            "run_id": self.config.run_id,
            "artifact_root": str(artifact_root),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_metadata": checkpoint_metadata,
            "task_manifest_path": str(task_manifest_path),
            "sample_count": len(samples),
            "task_entry_count": len(task_entries),
            "split_counts": {
                "train": len(chronological_split.train),
                "validation": len(chronological_split.validation),
                "test": len(chronological_split.test),
            },
            "task_heads": {
                task_name: {
                    "task_type": definition.task_type,
                    "output_dim": definition.spec.output_dim,
                    "label_to_id": dict(definition.label_to_id),
                }
                for task_name, definition in task_definitions.items()
            },
            "preview_config": {
                "input_normalization_mode": self.config.preview_config.input_normalization_mode,
                "physics_constraints_enabled": self.config.preview_config.enable_physics_constraints,
                "physics_constraint_family": self.config.preview_config.physics_constraint_family,
                "alignment_weight": self.config.preview_config.alignment_weight,
            },
            "multitask_config": {
                "task_loss_weight": self.config.task_loss_weight,
                "task_weights": dict(self.config.task_weights),
                "causal_weight": self.config.causal_weight,
                "causal_attention_temperature": self.config.causal_attention_temperature,
                "causal_event_bias_weight": self.config.causal_event_bias_weight,
                "causal_lag_window_points": self.config.causal_lag_window_points,
            },
            "train_metrics": asdict(train_history[-1]) if train_history else None,
            "validation_metrics": asdict(validation_history[-1]) if validation_history else None,
            "test_metrics": asdict(test_metrics),
            "source_summary": dict(source_summary or {}),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return StageIMultitaskTrainRunResult(
            artifact_root=str(artifact_root),
            checkpoint_path=str(checkpoint_path),
            summary_path=str(summary_path),
            task_manifest_path=str(task_manifest_path),
            summary=summary,
        )

    def _run_partition(
        self,
        *,
        model,
        task_heads: StageITaskHeadSet,
        causal_fusion: CausalMaskedCrossModalFusion,
        preview_pipeline: AlignmentPreviewPipeline,
        samples: tuple[E0ExperimentSample, ...],
        task_entries: Sequence[StageIPrivateTaskEntry],
        task_definitions: Mapping[str, _ResolvedTaskDefinition],
        optimizer: torch.optim.Optimizer | None,
        training: bool,
        normalization_stats,
        physics_stats,
    ) -> StageIMultitaskMetrics:
        if not samples:
            return StageIMultitaskMetrics(
                sample_count=0,
                batch_count=0,
                physiology_reconstruction=0.0,
                vehicle_reconstruction=0.0,
                reconstruction_total=0.0,
                alignment=0.0,
                vehicle_physics=0.0,
                physiology_physics=0.0,
                physics_total=0.0,
                causal_total=0.0,
                task_total=0.0,
                total=0.0,
            )

        if training and optimizer is None:
            raise ValueError("optimizer is required when training=True.")

        model.train(training)
        task_heads.train(training)
        weighted_totals = {
            "physiology_reconstruction": 0.0,
            "vehicle_reconstruction": 0.0,
            "reconstruction_total": 0.0,
            "alignment": 0.0,
            "vehicle_physics": 0.0,
            "physiology_physics": 0.0,
            "physics_total": 0.0,
            "causal_total": 0.0,
            "task_total": 0.0,
            "total": 0.0,
        }
        weighted_physics_components: dict[str, float] = {}
        weighted_causal_components: dict[str, float] = {}
        weighted_task_components: dict[str, float] = {}
        sample_count = 0
        batch_count = 0

        for batch_samples in _iterate_sample_batches(samples, batch_size=self.config.preview_config.batch_size):
            torch_batch = preview_pipeline._build_torch_batch(batch_samples)
            torch_batch = preview_pipeline._apply_input_normalization(
                torch_batch,
                normalization_stats=normalization_stats,
            )
            physics_context = None
            if physics_stats is not None:
                physics_context = build_batch_physics_context(
                    torch_batch,
                    stats=physics_stats,
                    normalization_stats=normalization_stats,
                )
            reference_offsets_s = preview_pipeline._build_reference_offsets_s_tensor(batch_samples)
            task_batches = _build_task_batches(
                sample_ids=tuple(sample.sample_id for sample in batch_samples),
                task_entries=task_entries,
                task_definitions=task_definitions,
                device=preview_pipeline._resolved_device_name(),
            )

            with torch.set_grad_enabled(training):
                output = model(torch_batch, reference_offsets_s=reference_offsets_s)
                fused_output = _build_causal_fusion_output(output, causal_fusion=causal_fusion)
                pooled_representation = fused_output.fused_states.mean(dim=1)
                task_outputs = task_heads(pooled_representation, task_batches)
                task_loss = (
                    build_task_loss_breakdown(
                        task_outputs,
                        task_weights=self.config.task_weights,
                    )
                    if task_outputs
                    else None
                )
                causal_entropy = attention_entropy(
                    fused_output.attention_weights,
                    fused_output.causal_mask,
                ).mean()
                objective = build_stage_e_objective(
                    output,
                    torch_batch,
                    reconstruction_mode=self.config.preview_config.reconstruction_loss_mode,
                    reconstruction_scale_epsilon=self.config.preview_config.reconstruction_scale_epsilon,
                    alignment_mode=self.config.preview_config.alignment_loss_mode,
                    physiology_weight=self.config.preview_config.physiology_reconstruction_weight,
                    vehicle_weight=self.config.preview_config.vehicle_reconstruction_weight,
                    alignment_weight=self.config.preview_config.alignment_weight,
                    enable_physics_constraints=self.config.preview_config.enable_physics_constraints,
                    physics_constraint_mode=self.config.preview_config.physics_constraint_mode,
                    physics_constraint_family=self.config.preview_config.physics_constraint_family,
                    vehicle_physics_weight=self.config.preview_config.vehicle_physics_weight,
                    physiology_physics_weight=self.config.preview_config.physiology_physics_weight,
                    physics_huber_delta=self.config.preview_config.physics_huber_delta,
                    physics_context=physics_context,
                    causal_regularization=causal_entropy,
                    causal_weight=self.config.causal_weight,
                    causal_components={"attention_entropy": causal_entropy},
                    task_loss_breakdown=task_loss,
                    task_weight=self.config.task_loss_weight,
                )

                if training:
                    assert optimizer is not None
                    optimizer.zero_grad()
                    objective.total.backward()
                    optimizer.step()

            current_batch_size = len(batch_samples)
            sample_count += current_batch_size
            batch_count += 1
            weighted_totals["physiology_reconstruction"] += float(objective.physiology_reconstruction.detach()) * current_batch_size
            weighted_totals["vehicle_reconstruction"] += float(objective.vehicle_reconstruction.detach()) * current_batch_size
            weighted_totals["reconstruction_total"] += float(objective.reconstruction_total.detach()) * current_batch_size
            weighted_totals["alignment"] += float(objective.alignment.detach()) * current_batch_size
            weighted_totals["vehicle_physics"] += float(objective.vehicle_physics.detach()) * current_batch_size
            weighted_totals["physiology_physics"] += float(objective.physiology_physics.detach()) * current_batch_size
            weighted_totals["physics_total"] += float(objective.physics_total.detach()) * current_batch_size
            weighted_totals["causal_total"] += float(objective.causal_total.detach()) * current_batch_size
            weighted_totals["task_total"] += float(objective.task_total.detach()) * current_batch_size
            weighted_totals["total"] += float(objective.total.detach()) * current_batch_size
            for component_name, component_value in objective.physics_components.items():
                weighted_physics_components[component_name] = (
                    weighted_physics_components.get(component_name, 0.0)
                    + float(component_value.detach()) * current_batch_size
                )
            for component_name, component_value in objective.causal_components.items():
                weighted_causal_components[component_name] = (
                    weighted_causal_components.get(component_name, 0.0)
                    + float(component_value.detach()) * current_batch_size
                )
            for component_name, component_value in objective.task_components.items():
                weighted_task_components[component_name] = (
                    weighted_task_components.get(component_name, 0.0)
                    + float(component_value.detach()) * current_batch_size
                )

        return StageIMultitaskMetrics(
            sample_count=sample_count,
            batch_count=batch_count,
            physiology_reconstruction=weighted_totals["physiology_reconstruction"] / sample_count,
            vehicle_reconstruction=weighted_totals["vehicle_reconstruction"] / sample_count,
            reconstruction_total=weighted_totals["reconstruction_total"] / sample_count,
            alignment=weighted_totals["alignment"] / sample_count,
            vehicle_physics=weighted_totals["vehicle_physics"] / sample_count,
            physiology_physics=weighted_totals["physiology_physics"] / sample_count,
            physics_total=weighted_totals["physics_total"] / sample_count,
            causal_total=weighted_totals["causal_total"] / sample_count,
            task_total=weighted_totals["task_total"] / sample_count,
            total=weighted_totals["total"] / sample_count,
            physics_components={
                component_name: component_value / sample_count
                for component_name, component_value in weighted_physics_components.items()
            },
            causal_components={
                component_name: component_value / sample_count
                for component_name, component_value in weighted_causal_components.items()
            },
            task_components={
                component_name: component_value / sample_count
                for component_name, component_value in weighted_task_components.items()
            },
        )

    def _save_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        model,
        task_heads: StageITaskHeadSet,
        task_definitions: Mapping[str, _ResolvedTaskDefinition],
        normalization_stats,
        train_history: Sequence[StageIMultitaskMetrics],
        validation_history: Sequence[StageIMultitaskMetrics],
        test_metrics: StageIMultitaskMetrics,
    ) -> dict[str, object]:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.config.run_id,
            "feature_schema": {
                "physiology_feature_names": tuple(
                    normalization_stats.physiology.feature_names
                    if normalization_stats is not None
                    else ()
                ),
                "vehicle_feature_names": tuple(
                    normalization_stats.vehicle.feature_names
                    if normalization_stats is not None
                    else ()
                ),
            },
            "preview_config": {
                "prototype_config": asdict(self.config.preview_config.prototype_config),
                "split_config": asdict(self.config.preview_config.split_config),
                "reference_grid_config": asdict(self.config.preview_config.reference_grid_config),
                "training": {
                    "epoch_count": self.config.preview_config.epoch_count,
                    "batch_size": self.config.preview_config.batch_size,
                    "learning_rate": self.config.preview_config.learning_rate,
                    "device": self.config.preview_config.device,
                    "input_normalization_mode": self.config.preview_config.input_normalization_mode,
                    "enable_physics_constraints": self.config.preview_config.enable_physics_constraints,
                    "physics_constraint_family": self.config.preview_config.physics_constraint_family,
                },
            },
            "multitask_config": {
                "task_head_hidden_dim": self.config.task_head_hidden_dim,
                "retrieval_embedding_dim": self.config.retrieval_embedding_dim,
                "task_loss_weight": self.config.task_loss_weight,
                "task_weights": dict(self.config.task_weights),
                "causal_weight": self.config.causal_weight,
                "causal_attention_temperature": self.config.causal_attention_temperature,
                "causal_event_bias_weight": self.config.causal_event_bias_weight,
                "causal_lag_window_points": self.config.causal_lag_window_points,
            },
            "task_heads": {
                task_name: {
                    "task_type": definition.task_type,
                    "spec": asdict(definition.spec),
                    "label_to_id": dict(definition.label_to_id),
                }
                for task_name, definition in task_definitions.items()
            },
            "input_normalization_stats": _serialize_input_normalization_stats(normalization_stats),
            "train_metrics": asdict(train_history[-1]) if train_history else None,
            "validation_metrics": asdict(validation_history[-1]) if validation_history else None,
            "test_metrics": asdict(test_metrics),
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "task_head_state_dict": {key: value.detach().cpu() for key, value in task_heads.state_dict().items()},
        }
        torch.save(payload, checkpoint_path)
        return {
            "checkpoint_path": str(checkpoint_path),
            "task_head_count": len(task_definitions),
        }


def run_stage_i_multitask_train(
    config: StageIMultitaskTrainConfig,
    *,
    samples: tuple[E0ExperimentSample, ...],
    task_entries: Sequence[StageIPrivateTaskEntry],
    source_summary: Mapping[str, object] | None = None,
) -> StageIMultitaskTrainRunResult:
    return StageIMultitaskTrainPipeline(config=config).run(
        samples=samples,
        task_entries=task_entries,
        source_summary=source_summary,
    )


def _resolve_task_definitions(
    task_entries: Sequence[StageIPrivateTaskEntry],
    *,
    input_dim: int,
    hidden_dim: int,
    retrieval_embedding_dim: int,
) -> dict[str, _ResolvedTaskDefinition]:
    grouped: dict[str, list[StageIPrivateTaskEntry]] = {}
    for entry in task_entries:
        grouped.setdefault(entry.task_name, []).append(entry)

    resolved: dict[str, _ResolvedTaskDefinition] = {}
    for task_name, entries in grouped.items():
        task_type = entries[0].task_type
        if task_type == "classification":
            ordered_labels = tuple(
                sorted({str(entry.label_value) for entry in entries if entry.label_value is not None})
            )
            label_to_id = {label: index for index, label in enumerate(ordered_labels)}
            output_dim = max(len(label_to_id), 1)
        elif task_type == "regression":
            label_to_id = {}
            output_dim = 1
        elif task_type == "retrieval":
            label_to_id = {}
            output_dim = retrieval_embedding_dim
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
        resolved[task_name] = _ResolvedTaskDefinition(
            task_name=task_name,
            task_type=task_type,
            spec=StageITaskHeadSpec(
                task_name=task_name,
                task_type=task_type,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
            ),
            label_to_id=label_to_id,
        )
    return resolved


def _build_task_batches(
    *,
    sample_ids: tuple[str, ...],
    task_entries: Sequence[StageIPrivateTaskEntry],
    task_definitions: Mapping[str, _ResolvedTaskDefinition],
    device: str,
) -> tuple[StageITaskHeadBatch, ...]:
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    grouped_entries: dict[str, list[StageIPrivateTaskEntry]] = {}
    for entry in task_entries:
        if entry.sample_id not in sample_index:
            continue
        grouped_entries.setdefault(entry.task_name, []).append(entry)

    task_batches: list[StageITaskHeadBatch] = []
    for task_name, entries in grouped_entries.items():
        definition = task_definitions[task_name]
        resolved_entries: list[StageIPrivateTaskEntry] = []
        paired_sample_ids: list[str | None] = []
        targets: torch.Tensor | None = None
        if definition.task_type == "classification":
            label_ids: list[int] = []
            for entry in entries:
                if entry.label_value is None:
                    continue
                label_ids.append(definition.label_to_id[str(entry.label_value)])
                resolved_entries.append(entry)
            if label_ids:
                targets = torch.as_tensor(label_ids, dtype=torch.long, device=device)
        elif definition.task_type == "regression":
            values: list[float] = []
            for entry in entries:
                if entry.label_value is None:
                    continue
                values.append(float(entry.label_value))
                resolved_entries.append(entry)
            if values:
                targets = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1, 1)
        elif definition.task_type == "retrieval":
            for entry in entries:
                if not entry.paired_sample_id or entry.paired_sample_id not in sample_index:
                    continue
                resolved_entries.append(entry)
                paired_sample_ids.append(entry.paired_sample_id)
        else:
            raise ValueError(f"Unsupported task type: {definition.task_type}")

        if not resolved_entries:
            continue
        sample_subset = tuple(entry.sample_id for entry in resolved_entries)
        sample_indices = tuple(sample_index[entry.sample_id] for entry in resolved_entries)
        task_batches.append(
            StageITaskHeadBatch(
                task_name=task_name,
                task_type=definition.task_type,
                sample_ids=sample_subset,
                sample_indices=sample_indices,
                targets=targets,
                paired_sample_ids=tuple(paired_sample_ids),
            )
        )
    return tuple(task_batches)


def _build_causal_fusion_output(output, *, causal_fusion: CausalMaskedCrossModalFusion):
    physiology_projection = output.physiology.reference_projected_states
    vehicle_projection = output.vehicle.reference_projected_states
    physiology_offsets = output.physiology.reference_offsets_s
    vehicle_offsets = output.vehicle.reference_offsets_s
    if physiology_projection is None or vehicle_projection is None:
        raise ValueError("multitask training requires reference_projected_states for both streams.")
    if physiology_offsets is None or vehicle_offsets is None:
        raise ValueError("multitask training requires reference_offsets_s for both streams.")
    return causal_fusion(
        CausalFusionTensorInput(
            physiology_states=physiology_projection,
            vehicle_states=vehicle_projection,
            physiology_offsets_s=physiology_offsets,
            vehicle_offsets_s=vehicle_offsets,
        )
    )


def _iterate_sample_batches(
    samples: Sequence[E0ExperimentSample],
    *,
    batch_size: int,
) -> Sequence[tuple[E0ExperimentSample, ...]]:
    for batch_start in range(0, len(samples), batch_size):
        yield tuple(samples[batch_start: batch_start + batch_size])


def _serialize_input_normalization_stats(stats) -> dict[str, object] | None:
    if stats is None:
        return None
    return {
        "mode": stats.mode,
        "physiology": {
            "feature_names": stats.physiology.feature_names,
            "mean": stats.physiology.mean.detach().cpu(),
            "std": stats.physiology.std.detach().cpu(),
        },
        "vehicle": {
            "feature_names": stats.vehicle.feature_names,
            "mean": stats.vehicle.mean.detach().cpu(),
            "std": stats.vehicle.std.detach().cpu(),
        },
    }
