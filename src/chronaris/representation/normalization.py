"""Training-fold-only robust normalization and PCA projection registries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from chronaris.dataset.application_evaluation.contracts import stable_sample_hash
from chronaris.representation.contracts import (
    FUSION_OUTPUT_DIM,
    DualStreamObservationBatch,
    RepresentationContractError,
)


@dataclass(frozen=True, slots=True)
class StreamRobustStatistics:
    center: torch.Tensor
    scale: torch.Tensor
    active_mask: torch.Tensor
    valid_count: torch.Tensor


class TrainOnlyRobustNormalizer:
    """Median/IQR transform whose fit lineage is explicit and immutable after fit."""

    def __init__(self, *, minimum_scale: float = 1e-6) -> None:
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        self.minimum_scale = float(minimum_scale)
        self.physiology: StreamRobustStatistics | None = None
        self.vehicle: StreamRobustStatistics | None = None
        self.fit_sample_ids: tuple[str, ...] = ()
        self.fit_sample_hash: str | None = None

    def fit(
        self,
        batch: DualStreamObservationBatch,
        *,
        train_sample_ids: Sequence[str],
        held_out_sample_ids: Sequence[str] = (),
    ) -> "TrainOnlyRobustNormalizer":
        if self.fit_sample_hash is not None:
            raise RepresentationContractError("normalizer is already fitted")
        train_ids = tuple(sorted(set(str(value) for value in train_sample_ids)))
        held_out = set(str(value) for value in held_out_sample_ids)
        if not train_ids:
            raise RepresentationContractError("normalizer train sample list is empty")
        overlap = sorted(set(train_ids) & held_out)
        if overlap:
            raise RepresentationContractError(
                f"held-out samples cannot fit normalization: {overlap[:5]}"
            )
        batch_index = {sample_id: index for index, sample_id in enumerate(batch.sample_ids)}
        missing = sorted(set(train_ids) - set(batch_index))
        if missing:
            raise RepresentationContractError(
                f"normalizer fit samples are absent from the supplied batch: {missing[:5]}"
            )
        indices = torch.tensor(
            [batch_index[value] for value in train_ids],
            dtype=torch.long,
            device=batch.physiology_values.device,
        )
        vehicle_indices = indices.to(batch.vehicle_values.device)
        self.physiology = _fit_stream_statistics(
            batch.physiology_values.index_select(0, indices),
            batch.physiology_feature_mask.index_select(0, indices),
            minimum_scale=self.minimum_scale,
        )
        self.vehicle = _fit_stream_statistics(
            batch.vehicle_values.index_select(0, vehicle_indices),
            batch.vehicle_feature_mask.index_select(0, vehicle_indices),
            minimum_scale=self.minimum_scale,
        )
        self.fit_sample_ids = train_ids
        self.fit_sample_hash = stable_sample_hash(train_ids)
        return self

    def fit_from_batch_provider(
        self,
        batch_provider: Callable[[Sequence[str]], DualStreamObservationBatch],
        *,
        train_sample_ids: Sequence[str],
        held_out_sample_ids: Sequence[str] = (),
        batch_size: int = 2,
    ) -> "TrainOnlyRobustNormalizer":
        """Fit exact train-only statistics without materializing every sample at once."""
        if self.fit_sample_hash is not None:
            raise RepresentationContractError("normalizer is already fitted")
        train_ids = tuple(sorted(set(str(value) for value in train_sample_ids)))
        held_out = set(str(value) for value in held_out_sample_ids)
        if not train_ids:
            raise RepresentationContractError("normalizer train sample list is empty")
        if batch_size <= 0:
            raise ValueError("normalizer provider batch size must be positive")
        overlap = sorted(set(train_ids) & held_out)
        if overlap:
            raise RepresentationContractError(
                f"held-out samples cannot fit normalization: {overlap[:5]}"
            )
        physiology_parts: list[list[torch.Tensor]] | None = None
        vehicle_parts: list[list[torch.Tensor]] | None = None
        for offset in range(0, len(train_ids), batch_size):
            sample_ids = train_ids[offset : offset + batch_size]
            batch = batch_provider(sample_ids)
            if tuple(batch.sample_ids) != sample_ids:
                raise RepresentationContractError(
                    "normalizer batch provider changed sample order"
                )
            physiology_parts = _append_observed_feature_parts(
                physiology_parts,
                batch.physiology_values,
                batch.physiology_feature_mask,
            )
            vehicle_parts = _append_observed_feature_parts(
                vehicle_parts,
                batch.vehicle_values,
                batch.vehicle_feature_mask,
            )
        self.physiology = _fit_stream_statistics_from_observed_parts(
            physiology_parts,
            minimum_scale=self.minimum_scale,
        )
        self.vehicle = _fit_stream_statistics_from_observed_parts(
            vehicle_parts,
            minimum_scale=self.minimum_scale,
        )
        self.fit_sample_ids = train_ids
        self.fit_sample_hash = stable_sample_hash(train_ids)
        return self

    def transform(self, batch: DualStreamObservationBatch) -> DualStreamObservationBatch:
        physiology, vehicle = self._require_fitted()
        return replace(
            batch,
            physiology_values=_transform_stream(
                batch.physiology_values,
                batch.physiology_feature_mask,
                physiology,
            ),
            vehicle_values=_transform_stream(
                batch.vehicle_values,
                batch.vehicle_feature_mask,
                vehicle,
            ),
        )

    @classmethod
    def from_manifest(
        cls,
        payload: Mapping[str, object],
    ) -> "TrainOnlyRobustNormalizer":
        if payload.get("transform") != "median_iqr":
            raise RepresentationContractError("unsupported robust normalizer manifest")
        normalizer = cls(minimum_scale=float(payload["minimum_scale"]))
        normalizer.fit_sample_ids = tuple(str(value) for value in payload["fit_sample_ids"])
        normalizer.fit_sample_hash = str(payload["fit_sample_hash"])
        if normalizer.fit_sample_hash != stable_sample_hash(normalizer.fit_sample_ids):
            raise RepresentationContractError("normalizer manifest fit hash mismatch")
        normalizer.physiology = _statistics_from_payload(payload["physiology"])
        normalizer.vehicle = _statistics_from_payload(payload["vehicle"])
        manifest_without_hash = dict(payload)
        expected_transform_hash = str(manifest_without_hash.pop("transform_sha256"))
        if _mapping_hash(manifest_without_hash) != expected_transform_hash:
            raise RepresentationContractError("normalizer manifest transform hash mismatch")
        return normalizer

    def to_manifest(self) -> Mapping[str, object]:
        physiology, vehicle = self._require_fitted()
        payload: dict[str, object] = {
            "transform": "median_iqr",
            "minimum_scale": self.minimum_scale,
            "fit_sample_ids": list(self.fit_sample_ids),
            "fit_sample_hash": self.fit_sample_hash,
            "physiology": _statistics_payload(physiology),
            "vehicle": _statistics_payload(vehicle),
        }
        payload["transform_sha256"] = _mapping_hash(payload)
        return payload

    def _require_fitted(
        self,
    ) -> tuple[StreamRobustStatistics, StreamRobustStatistics]:
        if self.physiology is None or self.vehicle is None or self.fit_sample_hash is None:
            raise RepresentationContractError("normalizer must be fitted before use")
        return self.physiology, self.vehicle


class TrainOnlyPCAProjector:
    """Unsupervised SVD projection fitted only on explicitly listed train samples."""

    def __init__(
        self,
        *,
        output_dim: int = FUSION_OUTPUT_DIM,
        solver: str = "full",
        random_state: int = 17,
    ) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if solver not in {"full", "randomized"}:
            raise ValueError("PCA solver must be full or randomized")
        self.output_dim = int(output_dim)
        self.solver = solver
        self.random_state = int(random_state)
        self.center: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.explained_variance_ratio: np.ndarray | None = None
        self.fit_sample_ids: tuple[str, ...] = ()
        self.fit_sample_hash: str | None = None

    def fit(
        self,
        values: np.ndarray,
        *,
        row_sample_ids: Sequence[str],
        train_sample_ids: Sequence[str],
        held_out_sample_ids: Sequence[str] = (),
    ) -> "TrainOnlyPCAProjector":
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) != len(row_sample_ids):
            raise RepresentationContractError(
                "PCA values and row_sample_ids must align as [N,D]"
            )
        if not np.isfinite(matrix).all():
            raise RepresentationContractError("PCA input values must be finite")
        train_ids = tuple(sorted(set(str(value) for value in train_sample_ids)))
        held_out = set(str(value) for value in held_out_sample_ids)
        if not train_ids or set(train_ids) & held_out:
            raise RepresentationContractError(
                "PCA fit samples must be non-empty and disjoint from held-out samples"
            )
        row_ids = np.asarray([str(value) for value in row_sample_ids], dtype=str)
        select = np.isin(row_ids, np.asarray(train_ids, dtype=str))
        observed_train_ids = set(row_ids[select])
        missing = sorted(set(train_ids) - observed_train_ids)
        if missing:
            raise RepresentationContractError(
                f"PCA train samples have no rows: {missing[:5]}"
            )
        fitted = matrix[select]
        self.center = fitted.mean(axis=0)
        centered = fitted - self.center
        component_count = min(self.output_dim, *centered.shape)
        if self.solver == "randomized" and component_count < min(centered.shape):
            from sklearn.utils.extmath import randomized_svd

            _u, singular_values, vh = randomized_svd(
                centered,
                n_components=component_count,
                n_iter=4,
                random_state=self.random_state,
                flip_sign=True,
            )
        else:
            _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        component_count = min(component_count, vh.shape[0])
        self.components = vh[:component_count].copy()
        variance = singular_values**2
        total = float(np.square(centered).sum())
        self.explained_variance_ratio = (
            variance[:component_count] / total
            if total > 0
            else np.zeros(component_count, dtype=np.float64)
        )
        self.fit_sample_ids = train_ids
        self.fit_sample_hash = stable_sample_hash(train_ids)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.center is None or self.components is None:
            raise RepresentationContractError("PCA projector must be fitted before use")
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.center):
            raise RepresentationContractError("PCA transform input dimension mismatch")
        projected = (matrix - self.center) @ self.components.T
        output = np.zeros((len(matrix), self.output_dim), dtype=np.float32)
        output[:, : projected.shape[1]] = projected.astype(np.float32)
        return output

    def to_manifest(self) -> Mapping[str, object]:
        if (
            self.center is None
            or self.components is None
            or self.explained_variance_ratio is None
            or self.fit_sample_hash is None
        ):
            raise RepresentationContractError("PCA projector must be fitted before export")
        payload: dict[str, object] = {
            "transform": f"train_only_{self.solver}_svd_pca",
            "solver": self.solver,
            "random_state": self.random_state,
            "input_dim": int(self.components.shape[1]),
            "component_count": int(self.components.shape[0]),
            "output_dim": self.output_dim,
            "fit_sample_ids": list(self.fit_sample_ids),
            "fit_sample_hash": self.fit_sample_hash,
            "center_sha256": hashlib.sha256(self.center.tobytes()).hexdigest(),
            "components_sha256": hashlib.sha256(self.components.tobytes()).hexdigest(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
        }
        payload["transform_sha256"] = _mapping_hash(payload)
        return payload

    def state_dict(self) -> Mapping[str, object]:
        if (
            self.center is None
            or self.components is None
            or self.explained_variance_ratio is None
            or self.fit_sample_hash is None
        ):
            raise RepresentationContractError("PCA projector must be fitted before save")
        return {
            "output_dim": self.output_dim,
            "solver": self.solver,
            "random_state": self.random_state,
            "center": torch.from_numpy(self.center.copy()),
            "components": torch.from_numpy(self.components.copy()),
            "explained_variance_ratio": torch.from_numpy(
                self.explained_variance_ratio.copy()
            ),
            "fit_sample_ids": list(self.fit_sample_ids),
            "fit_sample_hash": self.fit_sample_hash,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "TrainOnlyPCAProjector":
        projector = cls(
            output_dim=int(payload["output_dim"]),
            solver=str(payload.get("solver", "full")),
            random_state=int(payload.get("random_state", 17)),
        )
        projector.center = _tensor_to_numpy(payload["center"])
        projector.components = _tensor_to_numpy(payload["components"])
        projector.explained_variance_ratio = _tensor_to_numpy(
            payload["explained_variance_ratio"]
        )
        projector.fit_sample_ids = tuple(
            str(value) for value in payload["fit_sample_ids"]
        )
        projector.fit_sample_hash = str(payload["fit_sample_hash"])
        if projector.fit_sample_hash != stable_sample_hash(projector.fit_sample_ids):
            raise RepresentationContractError("PCA state fit hash mismatch")
        if projector.center.ndim != 1 or projector.components.ndim != 2:
            raise RepresentationContractError("PCA state array dimensions are invalid")
        if projector.components.shape[1] != len(projector.center):
            raise RepresentationContractError("PCA state input dimensions are inconsistent")
        if projector.components.shape[0] > projector.output_dim:
            raise RepresentationContractError("PCA state exceeds configured output dimension")
        return projector


def _fit_stream_statistics(
    values: torch.Tensor,
    feature_mask: torch.Tensor,
    *,
    minimum_scale: float,
) -> StreamRobustStatistics:
    feature_count = int(values.shape[-1])
    center = torch.zeros(feature_count, dtype=values.dtype, device=values.device)
    scale = torch.ones(feature_count, dtype=values.dtype, device=values.device)
    active = torch.zeros(feature_count, dtype=torch.bool, device=values.device)
    counts = torch.zeros(feature_count, dtype=torch.int64, device=values.device)
    for index in range(feature_count):
        observed = values[..., index][feature_mask[..., index]]
        counts[index] = observed.numel()
        if observed.numel() == 0:
            continue
        median = torch.quantile(observed, 0.5)
        iqr = torch.quantile(observed, 0.75) - torch.quantile(observed, 0.25)
        center[index] = median
        if bool(torch.isfinite(iqr)) and float(iqr) >= minimum_scale:
            scale[index] = iqr
            active[index] = True
    return StreamRobustStatistics(center, scale, active, counts)


def _append_observed_feature_parts(
    parts: list[list[torch.Tensor]] | None,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> list[list[torch.Tensor]]:
    feature_count = int(values.shape[-1])
    if parts is None:
        parts = [[] for _ in range(feature_count)]
    if len(parts) != feature_count:
        raise RepresentationContractError("normalizer provider feature count changed")
    cpu_values = values.detach().cpu()
    cpu_mask = mask.detach().cpu()
    for feature_index in range(feature_count):
        observed = cpu_values[..., feature_index][cpu_mask[..., feature_index]]
        if observed.numel():
            parts[feature_index].append(observed.clone())
    return parts


def _fit_stream_statistics_from_observed_parts(
    parts: list[list[torch.Tensor]] | None,
    *,
    minimum_scale: float,
) -> StreamRobustStatistics:
    if parts is None:
        raise RepresentationContractError("normalizer provider returned no chunks")
    feature_count = len(parts)
    dtype = next(
        (part.dtype for feature_parts in parts for part in feature_parts),
        torch.float32,
    )
    center = torch.zeros(feature_count, dtype=dtype)
    scale = torch.ones(feature_count, dtype=dtype)
    active = torch.zeros(feature_count, dtype=torch.bool)
    counts = torch.zeros(feature_count, dtype=torch.int64)
    for feature_index in range(feature_count):
        observed_parts = parts[feature_index]
        if not observed_parts:
            continue
        observed = torch.cat(observed_parts)
        counts[feature_index] = observed.numel()
        median = torch.quantile(observed, 0.5)
        iqr = torch.quantile(observed, 0.75) - torch.quantile(observed, 0.25)
        center[feature_index] = median
        if bool(torch.isfinite(iqr)) and float(iqr) >= minimum_scale:
            scale[feature_index] = iqr
            active[feature_index] = True
    return StreamRobustStatistics(center, scale, active, counts)


def _transform_stream(
    values: torch.Tensor,
    feature_mask: torch.Tensor,
    statistics: StreamRobustStatistics,
) -> torch.Tensor:
    if values.shape[-1] != len(statistics.center):
        raise RepresentationContractError("normalizer feature dimension mismatch")
    center = statistics.center.to(device=values.device, dtype=values.dtype)
    scale = statistics.scale.to(device=values.device, dtype=values.dtype)
    active_mask = statistics.active_mask.to(device=values.device)
    transformed = (values - center) / scale
    transformed = torch.where(
        feature_mask & active_mask.view(1, 1, -1),
        transformed,
        torch.zeros_like(transformed),
    )
    return transformed


def _statistics_payload(statistics: StreamRobustStatistics) -> Mapping[str, object]:
    return {
        "center": statistics.center.detach().cpu().tolist(),
        "scale": statistics.scale.detach().cpu().tolist(),
        "active_mask": statistics.active_mask.detach().cpu().tolist(),
        "valid_count": statistics.valid_count.detach().cpu().tolist(),
    }


def _statistics_from_payload(payload: object) -> StreamRobustStatistics:
    resolved = dict(payload)
    center = torch.tensor(resolved["center"], dtype=torch.float32)
    scale = torch.tensor(resolved["scale"], dtype=torch.float32)
    active = torch.tensor(resolved["active_mask"], dtype=torch.bool)
    counts = torch.tensor(resolved["valid_count"], dtype=torch.int64)
    if not (center.shape == scale.shape == active.shape == counts.shape):
        raise RepresentationContractError("normalizer statistics shape mismatch")
    if bool((scale <= 0).any()) or not torch.isfinite(center).all() or not torch.isfinite(scale).all():
        raise RepresentationContractError("normalizer statistics contain invalid values")
    return StreamRobustStatistics(center, scale, active, counts)


def _mapping_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tensor_to_numpy(value: object) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise RepresentationContractError("serialized PCA arrays must be tensors")
    array = value.detach().cpu().numpy().astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise RepresentationContractError("serialized PCA arrays contain non-finite values")
    return array
