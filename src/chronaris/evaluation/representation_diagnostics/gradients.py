"""Per-loss gradient norms and pairwise conflict audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.nn import functional as F


def gradient_conflict_rows(
    losses: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[dict[str, object], ...]:
    """Return norms and pairwise cosines without mutating parameter gradients."""

    active_parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not losses or not active_parameters:
        raise ValueError("gradient audit requires losses and trainable parameters")
    vectors: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for name, loss in losses.items():
        gradients = torch.autograd.grad(
            loss,
            active_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        vector = torch.cat(
            tuple(
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
                for parameter, gradient in zip(active_parameters, gradients, strict=True)
            )
        )
        vectors[name] = vector
        rows.append(
            {
                "row_type": "gradient_norm",
                "loss_a": name,
                "loss_b": None,
                "value": float(torch.linalg.vector_norm(vector)),
                "conflict": False,
            }
        )
    names = tuple(vectors)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_vector = vectors[left]
            right_vector = vectors[right]
            if float(torch.linalg.vector_norm(left_vector)) == 0 or float(
                torch.linalg.vector_norm(right_vector)
            ) == 0:
                cosine = 0.0
            else:
                cosine = float(F.cosine_similarity(left_vector, right_vector, dim=0))
            rows.append(
                {
                    "row_type": "gradient_cosine",
                    "loss_a": left,
                    "loss_b": right,
                    "value": cosine,
                    "conflict": cosine < -0.1,
                }
            )
    return tuple(rows)
