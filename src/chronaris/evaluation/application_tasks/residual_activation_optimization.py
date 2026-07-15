"""Optimization utilities for task-decoupled residual activation."""

from __future__ import annotations

import copy
from typing import Mapping

import torch


def shared_gradient_cosines(model, task_groups):
    """Audit gradients only when the candidate intentionally shares an adapter."""

    if model.candidate.adapter_mode != "shared":
        return {
            "gradient_cosine_maneuver_response": float("nan"),
            "gradient_cosine_maneuver_high_response": float("nan"),
            "gradient_cosine_response_high_response": float("nan"),
        }
    parameters = tuple(model.adapters["shared"].parameters())
    gradients = {
        name: torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        for name, loss in task_groups.items()
    }

    def cosine(left, right):
        first = torch.cat(
            [
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
                for parameter, gradient in zip(
                    parameters, gradients[left], strict=True
                )
            ]
        )
        second = torch.cat(
            [
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
                for parameter, gradient in zip(
                    parameters, gradients[right], strict=True
                )
            ]
        )
        denominator = first.norm() * second.norm()
        if float(denominator) <= 1e-12:
            return float("nan")
        return float(torch.dot(first, second).div(denominator).detach())

    return {
        "gradient_cosine_maneuver_response": cosine("maneuver", "response"),
        "gradient_cosine_maneuver_high_response": cosine(
            "maneuver", "high_response"
        ),
        "gradient_cosine_response_high_response": cosine(
            "response", "high_response"
        ),
    }


def compose_task_checkpoint(
    *,
    base_state: Mapping[str, torch.Tensor],
    task_states: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Compose non-overlapping task adapters, deltas, and gates into one model."""

    output = copy.deepcopy(dict(base_state))
    task_members = {
        "maneuver": ("maneuver", "maneuver_score"),
        "response": ("response",),
        "high_response": ("high_response",),
    }
    for selection_task, model_tasks in task_members.items():
        state = task_states[selection_task]
        for model_task in model_tasks:
            prefixes = (
                f"adapters.{model_task}.",
                f"delta_layers.{model_task}.",
                f"global_gate_logits.{model_task}",
                f"conditional_gates.{model_task}.",
            )
            for name, value in state.items():
                if name.startswith(prefixes):
                    output[name] = copy.deepcopy(value)
    return output


def update_task_best_states(
    *,
    model,
    metrics,
    direct_metrics,
    task_best_states,
    task_best_values,
    task_best_epochs,
    epoch,
):
    """Select each task branch independently while enforcing its safety bound."""

    safe = {
        "maneuver": metrics["maneuver"]["macro_f1"]
        >= direct_metrics["maneuver"]["macro_f1"] - 0.005,
        "response": metrics["response"]["rmse"]
        <= direct_metrics["response"]["rmse"] * 1.01,
        "high_response": metrics["high_response"]["auprc"]
        >= direct_metrics["high_response"]["auprc"] - 0.005,
    }
    current = {
        "maneuver": float(metrics["maneuver"]["macro_f1"]),
        "response": float(metrics["response"]["rmse"]),
        "high_response": float(metrics["high_response"]["normalized_ap"]),
    }
    improved = {
        "maneuver": safe["maneuver"]
        and current["maneuver"] > task_best_values["maneuver"] + 1e-9,
        "response": safe["response"]
        and current["response"] < task_best_values["response"] - 1e-9,
        "high_response": safe["high_response"]
        and current["high_response"] > task_best_values["high_response"] + 1e-9,
    }
    for task, changed in improved.items():
        if changed:
            task_best_states[task] = copy.deepcopy(model.state_dict())
            task_best_values[task] = current[task]
            task_best_epochs[task] = epoch
    return improved
