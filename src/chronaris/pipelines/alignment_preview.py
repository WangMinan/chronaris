"""Minimal Stage E preview training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import torch

from chronaris.features.experiment_input import E0ExperimentSample
from chronaris.models.alignment import (
    AlignmentPrototypeConfig,
    ChronologicalSampleSplit,
    ChronologicalSplitConfig,
    DualStreamODERNNPrototype,
    ReferenceGridConfig,
    StreamPrototypeOutput,
    build_reference_grids,
    build_stage_e_objective,
    build_torch_alignment_batch,
    build_alignment_batch,
    split_e0_samples_chronologically,
)


@dataclass(frozen=True, slots=True)
class AlignmentPreviewConfig:
    """Controls the minimal Stage E preview training pipeline."""

    prototype_config: AlignmentPrototypeConfig = field(default_factory=AlignmentPrototypeConfig)
    split_config: ChronologicalSplitConfig = field(default_factory=ChronologicalSplitConfig)
    reference_grid_config: ReferenceGridConfig = field(default_factory=ReferenceGridConfig)
    epoch_count: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-3
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    alignment_loss_mode: str = "mse"
    physiology_reconstruction_weight: float = 1.0
    vehicle_reconstruction_weight: float = 1.0
    alignment_weight: float = 1.0
    export_intermediate_states: bool = True
    intermediate_sample_limit: int = 3
    intermediate_partition: str = "test"

    def __post_init__(self) -> None:
        if self.epoch_count <= 0:
            raise ValueError("epoch_count must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.physiology_reconstruction_weight < 0:
            raise ValueError("physiology_reconstruction_weight must be non-negative.")
        if self.vehicle_reconstruction_weight < 0:
            raise ValueError("vehicle_reconstruction_weight must be non-negative.")
        if self.alignment_weight < 0:
            raise ValueError("alignment_weight must be non-negative.")
        if self.intermediate_sample_limit <= 0:
            raise ValueError("intermediate_sample_limit must be positive.")
        if self.intermediate_partition not in {"train", "validation", "test"}:
            raise ValueError("intermediate_partition must be one of: train, validation, test.")


@dataclass(frozen=True, slots=True)
class AlignmentPreviewMetrics:
    """Averaged losses for one preview partition over one epoch."""

    sample_count: int
    batch_count: int
    physiology_reconstruction: float
    vehicle_reconstruction: float
    reconstruction_total: float
    alignment: float
    total: float


@dataclass(frozen=True, slots=True)
class StreamIntermediateSnapshot:
    """Sample-level intermediate states exported for one stream."""

    feature_names: tuple[str, ...]
    point_count: int
    observation_offsets_s: tuple[float, ...]
    reference_offsets_s: tuple[float, ...]
    observation_hidden_states: tuple[tuple[float, ...], ...]
    reference_hidden_states: tuple[tuple[float, ...], ...]
    reference_projected_states: tuple[tuple[float, ...], ...]
    mean_observation_hidden_l2: float
    mean_reference_hidden_l2: float
    mean_reference_projection_l2: float


@dataclass(frozen=True, slots=True)
class AlignmentPreviewSampleIntermediate:
    """Sample-level dual-stream intermediate export on the shared reference grid."""

    sample_id: str
    physiology: StreamIntermediateSnapshot
    vehicle: StreamIntermediateSnapshot
    mean_reference_projection_cosine: float


@dataclass(frozen=True, slots=True)
class AlignmentPreviewIntermediateExport:
    """Intermediate-state export for one preview partition."""

    partition: str
    sample_count: int
    reference_point_count: int
    samples: tuple[AlignmentPreviewSampleIntermediate, ...]


@dataclass(frozen=True, slots=True)
class AlignmentPreviewRunResult:
    """Outputs from the minimal Stage E preview training pipeline."""

    split: ChronologicalSampleSplit
    model: DualStreamODERNNPrototype
    train_history: tuple[AlignmentPreviewMetrics, ...]
    validation_history: tuple[AlignmentPreviewMetrics, ...]
    test_metrics: AlignmentPreviewMetrics
    intermediate_export: AlignmentPreviewIntermediateExport | None = None


@dataclass(slots=True)
class AlignmentPreviewPipeline:
    """Train the minimal deterministic Stage E prototype on preview samples."""

    config: AlignmentPreviewConfig = field(default_factory=AlignmentPreviewConfig)

    def run(self, samples: tuple[E0ExperimentSample, ...]) -> AlignmentPreviewRunResult:
        """Split samples chronologically and run a minimal preview train/validation loop."""

        split = split_e0_samples_chronologically(samples, config=self.config.split_config)
        if not split.train:
            raise ValueError("AlignmentPreviewPipeline requires at least one training sample.")

        model = self._build_model(split.train)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        train_history: list[AlignmentPreviewMetrics] = []
        validation_history: list[AlignmentPreviewMetrics] = []

        for _epoch_index in range(self.config.epoch_count):
            train_history.append(
                self._run_partition(
                    model,
                    split.train,
                    optimizer=optimizer,
                    training=True,
                )
            )
            validation_history.append(
                self._run_partition(
                    model,
                    split.validation,
                    optimizer=None,
                    training=False,
                )
            )

        test_metrics = self._run_partition(
            model,
            split.test,
            optimizer=None,
            training=False,
        )

        intermediate_export = None
        if self.config.export_intermediate_states:
            partition_name, partition_samples = self._resolve_intermediate_partition_samples(split)
            intermediate_export = self._export_partition_intermediates(
                model,
                partition_samples,
                partition_name=partition_name,
            )

        return AlignmentPreviewRunResult(
            split=split,
            model=model,
            train_history=tuple(train_history),
            validation_history=tuple(validation_history),
            test_metrics=test_metrics,
            intermediate_export=intermediate_export,
        )

    def _build_model(self, samples: tuple[E0ExperimentSample, ...]) -> DualStreamODERNNPrototype:
        torch_batch = self._build_torch_batch(samples)
        model = DualStreamODERNNPrototype.from_torch_alignment_batch(
            torch_batch,
            config=self.config.prototype_config,
        )
        return model.to(device=self.config.device, dtype=self.config.dtype)

    def _run_partition(
        self,
        model: DualStreamODERNNPrototype,
        samples: tuple[E0ExperimentSample, ...],
        *,
        optimizer: torch.optim.Optimizer | None,
        training: bool,
    ) -> AlignmentPreviewMetrics:
        if not samples:
            return AlignmentPreviewMetrics(
                sample_count=0,
                batch_count=0,
                physiology_reconstruction=0.0,
                vehicle_reconstruction=0.0,
                reconstruction_total=0.0,
                alignment=0.0,
                total=0.0,
            )

        if training and optimizer is None:
            raise ValueError("optimizer is required when training=True.")

        model.train(training)
        weighted_totals = {
            "physiology_reconstruction": 0.0,
            "vehicle_reconstruction": 0.0,
            "reconstruction_total": 0.0,
            "alignment": 0.0,
            "total": 0.0,
        }
        sample_count = 0
        batch_count = 0

        for batch_samples in _iterate_sample_batches(samples, batch_size=self.config.batch_size):
            torch_batch = self._build_torch_batch(batch_samples)
            reference_offsets_s = self._build_reference_offsets_s_tensor(batch_samples)

            with torch.set_grad_enabled(training):
                output = model(torch_batch, reference_offsets_s=reference_offsets_s)
                objective = build_stage_e_objective(
                    output,
                    torch_batch,
                    alignment_mode=self.config.alignment_loss_mode,
                    physiology_weight=self.config.physiology_reconstruction_weight,
                    vehicle_weight=self.config.vehicle_reconstruction_weight,
                    alignment_weight=self.config.alignment_weight,
                )

                if training:
                    assert optimizer is not None
                    optimizer.zero_grad()
                    objective.total.backward()
                    optimizer.step()

            current_batch_size = len(batch_samples)
            sample_count += current_batch_size
            batch_count += 1
            weighted_totals["physiology_reconstruction"] += (
                float(objective.physiology_reconstruction.detach()) * current_batch_size
            )
            weighted_totals["vehicle_reconstruction"] += (
                float(objective.vehicle_reconstruction.detach()) * current_batch_size
            )
            weighted_totals["reconstruction_total"] += (
                float(objective.reconstruction_total.detach()) * current_batch_size
            )
            weighted_totals["alignment"] += float(objective.alignment.detach()) * current_batch_size
            weighted_totals["total"] += float(objective.total.detach()) * current_batch_size

        return AlignmentPreviewMetrics(
            sample_count=sample_count,
            batch_count=batch_count,
            physiology_reconstruction=weighted_totals["physiology_reconstruction"] / sample_count,
            vehicle_reconstruction=weighted_totals["vehicle_reconstruction"] / sample_count,
            reconstruction_total=weighted_totals["reconstruction_total"] / sample_count,
            alignment=weighted_totals["alignment"] / sample_count,
            total=weighted_totals["total"] / sample_count,
        )

    def _resolve_intermediate_partition_samples(
        self,
        split: ChronologicalSampleSplit,
    ) -> tuple[str, tuple[E0ExperimentSample, ...]]:
        partitions = {
            "train": split.train,
            "validation": split.validation,
            "test": split.test,
        }
        requested_partition = self.config.intermediate_partition
        requested_samples = partitions[requested_partition]
        if requested_samples:
            return requested_partition, requested_samples

        for fallback_partition in ("test", "validation", "train"):
            fallback_samples = partitions[fallback_partition]
            if fallback_samples:
                return fallback_partition, fallback_samples

        return requested_partition, ()

    def _export_partition_intermediates(
        self,
        model: DualStreamODERNNPrototype,
        samples: tuple[E0ExperimentSample, ...],
        *,
        partition_name: str,
    ) -> AlignmentPreviewIntermediateExport:
        if not samples:
            return AlignmentPreviewIntermediateExport(
                partition=partition_name,
                sample_count=0,
                reference_point_count=0,
                samples=(),
            )

        export_samples = samples[: self.config.intermediate_sample_limit]
        torch_batch = self._build_torch_batch(export_samples)
        reference_offsets_s = self._build_reference_offsets_s_tensor(export_samples)

        previous_training_mode = model.training
        model.train(False)
        with torch.no_grad():
            output = model(torch_batch, reference_offsets_s=reference_offsets_s)
        model.train(previous_training_mode)

        physiology_reference_projected = output.physiology.reference_projected_states
        vehicle_reference_projected = output.vehicle.reference_projected_states
        physiology_reference_offsets = output.physiology.reference_offsets_s
        if (
            physiology_reference_projected is None
            or vehicle_reference_projected is None
            or physiology_reference_offsets is None
        ):
            raise RuntimeError("Model output does not contain reference-grid intermediate states.")

        sample_exports: list[AlignmentPreviewSampleIntermediate] = []
        for sample_index, sample_id in enumerate(output.sample_ids):
            physiology_snapshot = self._build_stream_intermediate_snapshot(
                output.physiology,
                sample_index=sample_index,
            )
            vehicle_snapshot = self._build_stream_intermediate_snapshot(
                output.vehicle,
                sample_index=sample_index,
            )
            projection_cosine = torch.nn.functional.cosine_similarity(
                physiology_reference_projected[sample_index],
                vehicle_reference_projected[sample_index],
                dim=-1,
            )
            sample_exports.append(
                AlignmentPreviewSampleIntermediate(
                    sample_id=sample_id,
                    physiology=physiology_snapshot,
                    vehicle=vehicle_snapshot,
                    mean_reference_projection_cosine=float(projection_cosine.mean().detach().cpu()),
                )
            )

        return AlignmentPreviewIntermediateExport(
            partition=partition_name,
            sample_count=len(sample_exports),
            reference_point_count=int(physiology_reference_offsets.shape[1]),
            samples=tuple(sample_exports),
        )

    def _build_stream_intermediate_snapshot(
        self,
        stream_output: StreamPrototypeOutput,
        *,
        sample_index: int,
    ) -> StreamIntermediateSnapshot:
        reference_hidden_states = stream_output.reference_hidden_states
        reference_projected_states = stream_output.reference_projected_states
        reference_offsets_s = stream_output.reference_offsets_s
        if (
            reference_hidden_states is None
            or reference_projected_states is None
            or reference_offsets_s is None
        ):
            raise RuntimeError("Stream output does not contain reference-grid intermediate states.")

        point_count = int(stream_output.point_counts[sample_index].item())
        observation_offsets_s = stream_output.offsets_s[sample_index, :point_count]
        observation_hidden_states = stream_output.updated_hidden_states[sample_index, :point_count]
        sample_reference_offsets_s = reference_offsets_s[sample_index]
        sample_reference_hidden_states = reference_hidden_states[sample_index]
        sample_reference_projected_states = reference_projected_states[sample_index]

        return StreamIntermediateSnapshot(
            feature_names=stream_output.feature_names,
            point_count=point_count,
            observation_offsets_s=_tensor_1d_to_tuple(observation_offsets_s),
            reference_offsets_s=_tensor_1d_to_tuple(sample_reference_offsets_s),
            observation_hidden_states=_tensor_2d_to_tuple(observation_hidden_states),
            reference_hidden_states=_tensor_2d_to_tuple(sample_reference_hidden_states),
            reference_projected_states=_tensor_2d_to_tuple(sample_reference_projected_states),
            mean_observation_hidden_l2=_mean_row_l2_norm(observation_hidden_states),
            mean_reference_hidden_l2=_mean_row_l2_norm(sample_reference_hidden_states),
            mean_reference_projection_l2=_mean_row_l2_norm(sample_reference_projected_states),
        )

    def _build_torch_batch(self, samples: tuple[E0ExperimentSample, ...]):
        numpy_batch = build_alignment_batch(samples)
        return build_torch_alignment_batch(
            numpy_batch,
            device=self.config.device,
            dtype=self.config.dtype,
        )

    def _build_reference_offsets_s_tensor(self, samples: tuple[E0ExperimentSample, ...]) -> torch.Tensor:
        grids = build_reference_grids(samples, config=self.config.reference_grid_config)
        reference_offsets = [grid.relative_offsets_s for grid in grids]
        return torch.as_tensor(
            reference_offsets,
            dtype=self.config.dtype,
            device=self.config.device,
        )


def _iterate_sample_batches(
    samples: tuple[E0ExperimentSample, ...],
    *,
    batch_size: int,
) -> tuple[E0ExperimentSample, ...]:
    batch_total = ceil(len(samples) / batch_size)
    return tuple(
        samples[batch_index * batch_size : (batch_index + 1) * batch_size]
        for batch_index in range(batch_total)
    )


def _tensor_1d_to_tuple(values: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(item) for item in values.detach().cpu().tolist())


def _tensor_2d_to_tuple(values: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in values.detach().cpu().tolist())


def _mean_row_l2_norm(values: torch.Tensor) -> float:
    if values.ndim != 2:
        raise ValueError("values must be a 2D tensor.")
    if values.shape[0] == 0:
        return 0.0
    row_norms = torch.linalg.vector_norm(values, ord=2, dim=-1)
    return float(row_norms.mean().detach().cpu())
