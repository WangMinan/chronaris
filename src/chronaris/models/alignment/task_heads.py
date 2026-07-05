"""Unified task heads for task evaluation multitask alignment training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import nn


_TASK_TYPES = {"classification", "regression", "retrieval"}


@dataclass(frozen=True, slots=True)
class StageITaskHeadSpec:
    """One task-head configuration over the shared task evaluation backbone."""

    task_name: str
    task_type: str
    input_dim: int
    output_dim: int
    hidden_dim: int = 32
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not self.task_name:
            raise ValueError("task_name must be non-empty.")
        if self.task_type not in _TASK_TYPES:
            raise ValueError(f"task_type must be one of {sorted(_TASK_TYPES)!r}.")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")


@dataclass(frozen=True, slots=True)
class StageITaskHeadBatch:
    """Task labels for the current batch of shared sample representations."""

    task_name: str
    task_type: str
    sample_ids: tuple[str, ...]
    sample_indices: tuple[int, ...]
    targets: torch.Tensor | None = None
    paired_sample_ids: tuple[str | None, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_type not in _TASK_TYPES:
            raise ValueError(f"task_type must be one of {sorted(_TASK_TYPES)!r}.")
        if len(self.sample_ids) != len(self.sample_indices):
            raise ValueError("sample_ids and sample_indices must have the same length.")
        if self.paired_sample_ids and len(self.paired_sample_ids) != len(self.sample_ids):
            raise ValueError("paired_sample_ids must match sample_ids length when provided.")
        if self.targets is not None and int(self.targets.shape[0]) != len(self.sample_ids):
            raise ValueError("targets must align with the task sample count.")


@dataclass(frozen=True, slots=True)
class StageITaskHeadOutput:
    """Forward output from one task head on top of shared backbone features."""

    task_name: str
    task_type: str
    sample_ids: tuple[str, ...]
    logits: torch.Tensor
    sample_indices: tuple[int, ...]
    targets: torch.Tensor | None = None
    paired_sample_ids: tuple[str | None, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


class ClassificationTaskHead(nn.Module):
    """Small MLP classifier over pooled shared states."""

    def __init__(self, spec: StageITaskHeadSpec) -> None:
        super().__init__()
        self.spec = spec
        self.network = _build_mlp(
            input_dim=spec.input_dim,
            hidden_dim=spec.hidden_dim,
            output_dim=spec.output_dim,
            dropout=spec.dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class RegressionTaskHead(nn.Module):
    """Small MLP regressor over pooled shared states."""

    def __init__(self, spec: StageITaskHeadSpec) -> None:
        super().__init__()
        self.spec = spec
        self.network = _build_mlp(
            input_dim=spec.input_dim,
            hidden_dim=spec.hidden_dim,
            output_dim=spec.output_dim,
            dropout=spec.dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class RetrievalTaskHead(nn.Module):
    """Project pooled shared states into one retrieval embedding space."""

    def __init__(self, spec: StageITaskHeadSpec) -> None:
        super().__init__()
        self.spec = spec
        self.network = _build_mlp(
            input_dim=spec.input_dim,
            hidden_dim=spec.hidden_dim,
            output_dim=spec.output_dim,
            dropout=spec.dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embeddings = self.network(features)
        return torch.nn.functional.normalize(embeddings, dim=-1)


class StageITaskHeadSet(nn.Module):
    """A registry of task heads sharing one backbone representation space."""

    def __init__(self, specs: Sequence[StageITaskHeadSpec]) -> None:
        super().__init__()
        if not specs:
            raise ValueError("StageITaskHeadSet requires at least one task spec.")
        self.specs = {spec.task_name: spec for spec in specs}
        if len(self.specs) != len(tuple(specs)):
            raise ValueError("task specs must use unique task_name values.")
        self.heads = nn.ModuleDict(
            {
                spec.task_name: _build_head(spec)
                for spec in specs
            }
        )

    def forward(
        self,
        shared_representations: torch.Tensor,
        task_batches: Sequence[StageITaskHeadBatch],
    ) -> tuple[StageITaskHeadOutput, ...]:
        outputs: list[StageITaskHeadOutput] = []
        for batch in task_batches:
            head = self.heads[batch.task_name]
            index_tensor = torch.as_tensor(
                batch.sample_indices,
                dtype=torch.long,
                device=shared_representations.device,
            )
            task_features = shared_representations.index_select(0, index_tensor)
            logits = head(task_features)
            targets = None
            if batch.targets is not None:
                targets = batch.targets.to(device=shared_representations.device)
            outputs.append(
                StageITaskHeadOutput(
                    task_name=batch.task_name,
                    task_type=batch.task_type,
                    sample_ids=batch.sample_ids,
                    logits=logits,
                    sample_indices=batch.sample_indices,
                    targets=targets,
                    paired_sample_ids=batch.paired_sample_ids,
                    metadata=dict(batch.metadata),
                )
            )
        return tuple(outputs)


def _build_head(spec: StageITaskHeadSpec) -> nn.Module:
    if spec.task_type == "classification":
        return ClassificationTaskHead(spec)
    if spec.task_type == "regression":
        return RegressionTaskHead(spec)
    if spec.task_type == "retrieval":
        return RetrievalTaskHead(spec)
    raise ValueError(f"Unsupported task type: {spec.task_type}")


def _build_mlp(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)
