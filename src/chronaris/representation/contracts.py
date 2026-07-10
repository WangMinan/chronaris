"""Fail-closed tensor contracts for fair downstream representation comparison."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import torch


FUSION_OUTPUT_DIM = 64
QUERY_POINT_COUNT = 96
CHECKPOINT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_REPRESENTATION_FIELDS = frozenset(
    {
        "attention_entropy",
        "candidate_id",
        "diagnostics",
        "label",
        "labels",
        "logit",
        "logits",
        "maneuver_state",
        "physical_residual",
        "prediction",
        "predictions",
        "probability",
        "rank",
        "response_target",
        "task_name",
        "workload",
        "y_pred",
        "y_true",
    }
)
ALLOWED_LINEAGE_FIELDS = frozenset({"label_used_for_encoder_training"})


class RepresentationContractError(ValueError):
    """Raised when a batch or lineage record violates the frozen protocol."""


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    """Feature order and source roles kept outside the tensor batch."""

    schema_id: str
    source_kind: str
    physiology_feature_names: tuple[str, ...]
    vehicle_feature_names: tuple[str, ...]
    physiology_feature_roles: tuple[str, ...]
    vehicle_feature_roles: tuple[str, ...]
    excluded_feature_names: tuple[str, ...] = ()
    source_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.schema_id or not self.source_kind:
            raise RepresentationContractError("schema_id and source_kind are required")
        _validate_feature_axis(
            self.physiology_feature_names,
            self.physiology_feature_roles,
            "physiology",
        )
        _validate_feature_axis(
            self.vehicle_feature_names,
            self.vehicle_feature_roles,
            "vehicle",
        )
        overlap = set(self.physiology_feature_names) & set(self.vehicle_feature_names)
        if overlap:
            raise RepresentationContractError(
                f"stream feature names must be globally unique: {sorted(overlap)[:5]}"
            )

    @property
    def schema_sha256(self) -> str:
        structural = self.to_dict()
        structural.pop("source_manifest_sha256", None)
        payload = json.dumps(structural, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in payload.items()
        }


@dataclass(frozen=True, slots=True)
class DualStreamObservationBatch:
    """Padded raw asynchronous observations with a shared query grid."""

    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    physiology_values: torch.Tensor
    physiology_timestamps_s: torch.Tensor
    physiology_point_mask: torch.Tensor
    physiology_feature_mask: torch.Tensor
    physiology_observation_age_s: torch.Tensor
    vehicle_values: torch.Tensor
    vehicle_timestamps_s: torch.Tensor
    vehicle_point_mask: torch.Tensor
    vehicle_feature_mask: torch.Tensor
    vehicle_observation_age_s: torch.Tensor
    query_timestamps_s: torch.Tensor
    source_sample_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        batch_size = len(self.sample_ids)
        if batch_size == 0:
            raise RepresentationContractError("observation batch must not be empty")
        if len(set(self.sample_ids)) != batch_size:
            raise RepresentationContractError("sample_ids must be unique within a batch")
        for name, values in (
            ("group_ids", self.group_ids),
            ("source_sample_hashes", self.source_sample_hashes),
        ):
            if len(values) != batch_size:
                raise RepresentationContractError(
                    f"{name} count {len(values)} does not match batch size {batch_size}"
                )
        _validate_stream_tensors(
            stream_name="physiology",
            batch_size=batch_size,
            values=self.physiology_values,
            timestamps=self.physiology_timestamps_s,
            point_mask=self.physiology_point_mask,
            feature_mask=self.physiology_feature_mask,
            observation_age=self.physiology_observation_age_s,
        )
        _validate_stream_tensors(
            stream_name="vehicle",
            batch_size=batch_size,
            values=self.vehicle_values,
            timestamps=self.vehicle_timestamps_s,
            point_mask=self.vehicle_point_mask,
            feature_mask=self.vehicle_feature_mask,
            observation_age=self.vehicle_observation_age_s,
        )
        _validate_query_axis(self.query_timestamps_s, batch_size=batch_size)
        for value in self.source_sample_hashes:
            if not CHECKPOINT_HASH_PATTERN.fullmatch(value):
                raise RepresentationContractError(
                    "source_sample_hashes must contain lowercase SHA-256 values"
                )


@dataclass(frozen=True, slots=True)
class FusionStreamBatch:
    """Task-independent 64-dimensional representation exported by one method."""

    sample_ids: tuple[str, ...]
    timestamps_s: torch.Tensor
    sequence_embedding: torch.Tensor
    valid_mask: torch.Tensor
    pooled_embedding: torch.Tensor
    method_name: str
    fold_id: str
    checkpoint_sha256: str
    source_sample_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        batch_size = len(self.sample_ids)
        if batch_size == 0 or len(set(self.sample_ids)) != batch_size:
            raise RepresentationContractError(
                "fusion sample_ids must be non-empty and unique"
            )
        if len(self.source_sample_hashes) != batch_size:
            raise RepresentationContractError(
                "fusion source hash count does not match sample count"
            )
        if not self.method_name or not self.fold_id:
            raise RepresentationContractError("method_name and fold_id are required")
        if not CHECKPOINT_HASH_PATTERN.fullmatch(self.checkpoint_sha256):
            raise RepresentationContractError(
                "checkpoint_sha256 must be a lowercase SHA-256 value"
            )
        if self.sequence_embedding.ndim != 3:
            raise RepresentationContractError(
                "sequence_embedding must have shape [B,Tq,64]"
            )
        expected = (batch_size, QUERY_POINT_COUNT, FUSION_OUTPUT_DIM)
        if tuple(self.sequence_embedding.shape) != expected:
            raise RepresentationContractError(
                f"sequence_embedding shape {tuple(self.sequence_embedding.shape)} != {expected}"
            )
        if tuple(self.pooled_embedding.shape) != (batch_size, FUSION_OUTPUT_DIM):
            raise RepresentationContractError(
                "pooled_embedding must have shape [B,64]"
            )
        if tuple(self.valid_mask.shape) != (batch_size, QUERY_POINT_COUNT):
            raise RepresentationContractError("valid_mask must have shape [B,Tq]")
        if self.valid_mask.dtype != torch.bool:
            raise RepresentationContractError("valid_mask must use torch.bool")
        _validate_query_axis(self.timestamps_s, batch_size=batch_size)
        if not torch.isfinite(self.sequence_embedding).all():
            raise RepresentationContractError("sequence_embedding contains non-finite values")
        if not torch.isfinite(self.pooled_embedding).all():
            raise RepresentationContractError("pooled_embedding contains non-finite values")
        valid_count = self.valid_mask.sum(dim=1, keepdim=True)
        if bool((valid_count == 0).any()):
            raise RepresentationContractError(
                "each fusion sample must have at least one valid query point"
            )
        expected_pool = (
            self.sequence_embedding
            * self.valid_mask.unsqueeze(-1).to(self.sequence_embedding.dtype)
        ).sum(dim=1) / valid_count.to(self.sequence_embedding.dtype)
        if not torch.allclose(
            self.pooled_embedding,
            expected_pool,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RepresentationContractError(
                "pooled_embedding must be the valid-mask mean of sequence_embedding"
            )


@runtime_checkable
class FusionStreamEncoder(Protocol):
    """Minimal task-independent adapter interface used by the OOF exporter."""

    method_name: str
    output_dim: int

    def __call__(self, batch: DualStreamObservationBatch) -> FusionStreamBatch: ...


def validate_fusion_method_alignment(
    outputs: Sequence[FusionStreamBatch],
) -> str:
    """Require identical sample/query/mask lineage and return a stable alignment hash."""

    if not outputs:
        raise RepresentationContractError("at least one fusion output is required")
    first = outputs[0]
    methods: set[str] = set()
    for output in outputs:
        if output.method_name in methods:
            raise RepresentationContractError(
                f"duplicate method output: {output.method_name}"
            )
        methods.add(output.method_name)
        if output.sample_ids != first.sample_ids:
            raise RepresentationContractError("fusion method sample order mismatch")
        if output.source_sample_hashes != first.source_sample_hashes:
            raise RepresentationContractError("fusion method source lineage mismatch")
        if not torch.equal(output.timestamps_s, first.timestamps_s):
            raise RepresentationContractError("fusion method query timestamps mismatch")
        if not torch.equal(output.valid_mask, first.valid_mask):
            raise RepresentationContractError("fusion method valid mask mismatch")
    digest = hashlib.sha256()
    digest.update(json.dumps(first.sample_ids, ensure_ascii=False).encode("utf-8"))
    digest.update(first.timestamps_s.detach().cpu().numpy().tobytes())
    digest.update(first.valid_mask.detach().cpu().numpy().tobytes())
    for value in first.source_sample_hashes:
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def validate_representation_mapping_fields(payload: Mapping[str, object]) -> None:
    """Reject task predictions, labels and method-only diagnostics on load."""

    normalized = {
        str(key).lower()
        for key in payload
        if str(key).lower() not in ALLOWED_LINEAGE_FIELDS
    }
    forbidden = sorted(
        field
        for field in normalized
        if any(field == token or field.startswith(token + "_") for token in FORBIDDEN_REPRESENTATION_FIELDS)
    )
    if forbidden:
        raise RepresentationContractError(
            f"representation payload contains forbidden fields: {forbidden}"
        )


def _validate_feature_axis(
    names: tuple[str, ...],
    roles: tuple[str, ...],
    stream_name: str,
) -> None:
    if not names or len(names) != len(roles):
        raise RepresentationContractError(
            f"{stream_name} feature names and roles must be non-empty and aligned"
        )
    if len(set(names)) != len(names) or any(not value for value in names):
        raise RepresentationContractError(
            f"{stream_name} feature names must be unique and non-empty"
        )


def _validate_stream_tensors(
    *,
    stream_name: str,
    batch_size: int,
    values: torch.Tensor,
    timestamps: torch.Tensor,
    point_mask: torch.Tensor,
    feature_mask: torch.Tensor,
    observation_age: torch.Tensor,
) -> None:
    if values.ndim != 3 or values.shape[0] != batch_size:
        raise RepresentationContractError(
            f"{stream_name}_values must have shape [B,T,F]"
        )
    expected_time_shape = tuple(values.shape[:2])
    if tuple(timestamps.shape) != expected_time_shape:
        raise RepresentationContractError(
            f"{stream_name}_timestamps_s shape mismatch"
        )
    if tuple(point_mask.shape) != expected_time_shape:
        raise RepresentationContractError(f"{stream_name}_point_mask shape mismatch")
    if tuple(feature_mask.shape) != tuple(values.shape):
        raise RepresentationContractError(f"{stream_name}_feature_mask shape mismatch")
    if tuple(observation_age.shape) != tuple(values.shape):
        raise RepresentationContractError(
            f"{stream_name}_observation_age_s shape mismatch"
        )
    if point_mask.dtype != torch.bool or feature_mask.dtype != torch.bool:
        raise RepresentationContractError(f"{stream_name} masks must use torch.bool")
    if not torch.equal(point_mask, feature_mask.any(dim=-1)):
        raise RepresentationContractError(
            f"{stream_name} point mask must equal any(feature mask)"
        )
    if not torch.isfinite(values[feature_mask]).all():
        raise RepresentationContractError(
            f"{stream_name} valid values contain non-finite entries"
        )
    if not torch.isfinite(timestamps[point_mask]).all():
        raise RepresentationContractError(
            f"{stream_name} valid timestamps contain non-finite entries"
        )
    if bool((observation_age[feature_mask] < 0).any()):
        raise RepresentationContractError(
            f"{stream_name} observation ages must be non-negative"
        )
    if not torch.isfinite(observation_age[feature_mask]).all():
        raise RepresentationContractError(
            f"{stream_name} valid observation ages must be finite"
        )
    for row in range(batch_size):
        valid_times = timestamps[row][point_mask[row]]
        if valid_times.numel() > 1 and bool((valid_times[1:] < valid_times[:-1]).any()):
            raise RepresentationContractError(
                f"{stream_name} timestamps must be monotonic non-decreasing"
            )


def _validate_query_axis(timestamps: torch.Tensor, *, batch_size: int) -> None:
    if tuple(timestamps.shape) != (batch_size, QUERY_POINT_COUNT):
        raise RepresentationContractError(
            f"query timestamps must have shape [B,{QUERY_POINT_COUNT}]"
        )
    if not torch.isfinite(timestamps).all():
        raise RepresentationContractError("query timestamps contain non-finite values")
    if bool((timestamps[:, 1:] <= timestamps[:, :-1]).any()):
        raise RepresentationContractError("query timestamps must be strictly increasing")
