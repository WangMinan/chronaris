"""Small deterministic PCGrad implementation and automatic conflict switch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch


@dataclass(slots=True)
class GradientConflictController:
    conflict_threshold: float = -0.1
    switch_rate: float = 0.20
    minimum_pair_count: int = 10
    conflict_history: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not -1 < self.conflict_threshold < 0:
            raise ValueError("conflict_threshold must be in (-1,0)")
        if not 0 < self.switch_rate < 1 or self.minimum_pair_count <= 0:
            raise ValueError("PCGrad switch configuration is invalid")

    def observe(self, pairwise_cosines: Sequence[float]) -> None:
        """Record pair-level outcomes for backward-compatible audits."""

        self.conflict_history.extend(
            float(value) < self.conflict_threshold for value in pairwise_cosines
        )

    def observe_step(self, pairwise_cosines: Sequence[float]) -> None:
        """Record one conflict decision for one optimizer step."""

        values = tuple(float(value) for value in pairwise_cosines)
        if not values:
            return
        self.conflict_history.append(min(values) < self.conflict_threshold)

    @property
    def conflict_rate(self) -> float:
        return (
            sum(self.conflict_history) / len(self.conflict_history)
            if self.conflict_history
            else 0.0
        )

    @property
    def use_pcgrad(self) -> bool:
        return (
            len(self.conflict_history) >= self.minimum_pair_count
            and self.conflict_rate > self.switch_rate
        )


def loss_gradient_cosines(
    losses: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
) -> dict[tuple[str, str], float]:
    vectors, _ = _loss_gradient_vectors(losses, parameters)
    names = tuple(vectors)
    result = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            result[(left, right)] = _cosine(vectors[left], vectors[right])
    return result


def pcgrad_backward(
    losses: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
) -> dict[tuple[str, str], float]:
    """Project conflicting loss gradients and write the mean into `.grad`."""

    vectors, active_parameters = _loss_gradient_vectors(losses, parameters)
    names = tuple(vectors)
    pairwise = {
        (left, right): _cosine(vectors[left], vectors[right])
        for left_index, left in enumerate(names)
        for right in names[left_index + 1 :]
    }
    projected = []
    for left in names:
        gradient = vectors[left].clone()
        for right in names:
            if right == left:
                continue
            other = vectors[right]
            dot = torch.dot(gradient, other)
            if float(dot) < 0:
                gradient = gradient - dot * other / other.square().sum().clamp_min(1e-15)
        projected.append(gradient)
    merged = torch.stack(projected).mean(dim=0)
    offset = 0
    for parameter in active_parameters:
        count = parameter.numel()
        parameter.grad = merged[offset : offset + count].reshape_as(parameter).clone()
        offset += count
    return pairwise


def _loss_gradient_vectors(losses, parameters):
    active_parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    active_losses = {
        name: loss for name, loss in losses.items() if loss.requires_grad
    }
    if not active_parameters or not active_losses:
        raise ValueError("PCGrad requires trainable parameters and differentiable losses")
    vectors = {}
    for name, loss in active_losses.items():
        gradients = torch.autograd.grad(
            loss,
            active_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        vectors[name] = torch.cat(
            tuple(
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
                for parameter, gradient in zip(active_parameters, gradients, strict=True)
            )
        )
    return vectors, active_parameters


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return (
        float(torch.dot(left, right) / denominator.clamp_min(1e-15))
        if float(denominator) > 0
        else 0.0
    )
