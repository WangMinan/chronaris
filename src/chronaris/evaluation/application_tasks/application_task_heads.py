"""Task dimensions, partial labels and train-only calibration for application tuning."""
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from chronaris.evaluation.application_tasks.application_consumers import CausalTCNEmissionModel, TCNConsumerConfig
from chronaris.representation.contracts import FUSION_OUTPUT_DIM, RepresentationContractError


@dataclass(frozen=True)
class ApplicationTaskDefinition:
    name: str
    kind: str
    output_dim: int

    def __post_init__(self):
        if not self.name.isidentifier() or self.kind not in {"classification", "regression", "sequence_classification"}:
            raise ValueError("invalid application task name/kind")
        if self.output_dim < (1 if self.kind == "regression" else 2):
            raise ValueError("invalid application task output dimension")


SIMULATION_TASKS = (
    ApplicationTaskDefinition("classification", "classification", 3),
    ApplicationTaskDefinition("regression", "regression", 1),
    ApplicationTaskDefinition("segmentation", "sequence_classification", 5),
)


@dataclass(frozen=True)
class ApplicationTaskTargets:
    sample_ids: tuple[str, ...]
    values: Mapping[str, torch.Tensor]
    valid_masks: Mapping[str, torch.Tensor]
    manifest: Mapping[str, object]
    sample_weights: torch.Tensor | None = None

    def __post_init__(self):
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids):
            raise RepresentationContractError("task sample identifiers must be unique")
        if not self.values or set(self.values) != set(self.valid_masks):
            raise RepresentationContractError("every task needs an explicit target mask")
        for name, value in self.values.items():
            mask = self.valid_masks[name]
            if value.shape[0] != len(self.sample_ids) or value.shape != mask.shape or mask.dtype != torch.bool:
                raise RepresentationContractError("task target/mask dimensions differ")
            if not torch.isfinite(value[mask]).all():
                raise RepresentationContractError("observed task targets must be finite")
        if self.sample_weights is not None and (
            self.sample_weights.shape != (len(self.sample_ids),)
            or not torch.isfinite(self.sample_weights).all() or (self.sample_weights <= 0).any()
        ):
            raise RepresentationContractError("task sample weights must be finite and positive")


def application_targets(targets):
    if isinstance(targets, ApplicationTaskTargets):
        return targets
    values = {"classification": targets.workload_class, "regression": targets.future_workload_mean,
              "segmentation": targets.maneuver_state}
    return ApplicationTaskTargets(tuple(targets.sample_ids), values,
        {name: torch.ones_like(value, dtype=torch.bool) for name, value in values.items()}, targets.manifest)


def build_application_heads(definitions):
    if not definitions or len({task.name for task in definitions}) != len(definitions):
        raise ValueError("application tasks must be nonempty and uniquely named")
    return nn.ModuleDict({task.name: (
        CausalTCNEmissionModel(TCNConsumerConfig(input_dim=FUSION_OUTPUT_DIM, hidden_channels=64,
            class_count=task.output_dim, kernel_size=5, dilations=(1, 2), dropout=.1, epochs=1, device="cpu"))
        if task.kind == "sequence_classification" else nn.Linear(FUSION_OUTPUT_DIM, task.output_dim)
    ) for task in definitions})


def fit_application_task_parameters(targets, definitions, train_ids):
    if set(targets.values) != {task.name for task in definitions}:
        raise RepresentationContractError("task definitions and targets differ")
    index = {value: i for i, value in enumerate(targets.sample_ids)}
    positions = [index[value] for value in train_ids]
    if not positions or len(set(train_ids)) != len(train_ids):
        raise RepresentationContractError("task calibration requires distinct internal train samples")
    parameters = {}
    for task in definitions:
        value, mask = targets.values[task.name][positions], targets.valid_masks[task.name][positions]
        if not mask.any():
            raise RepresentationContractError(f"task has no internal training labels: {task.name}")
        if task.kind == "regression":
            if value.ndim == 1:
                value, mask = value[:, None], mask[:, None]
            if value.ndim != 2 or value.shape[1] != task.output_dim:
                raise RepresentationContractError("regression target dimensions differ from head")
            if not mask.any(dim=0).all():
                raise RepresentationContractError("regression field lacks internal training support")
            center = torch.stack([value[:, j][mask[:, j]].float().mean() for j in range(task.output_dim)])
            scale = torch.stack([value[:, j][mask[:, j]].float().std(unbiased=False) for j in range(task.output_dim)]).clamp_min(1e-6)
            parameters[task.name] = {"center": center.tolist(), "scale": scale.tolist()}
        else:
            if value.ndim != (2 if task.kind == "sequence_classification" else 1):
                raise RepresentationContractError("classification target dimensions differ from head")
            labels = value[mask]
            if labels.dtype != torch.long or (labels < 0).any() or (labels >= task.output_dim).any():
                raise RepresentationContractError("classification labels exceed declared classes")
            count = torch.bincount(labels.cpu(), minlength=task.output_dim).float()
            # A class absent from inner training may occur in validation; keep its loss visible.
            weight = torch.where(count > 0, count.sum() / (task.output_dim * count.clamp_min(1)), 1.)
            parameters[task.name] = {"class_weights": weight.tolist()}
    return {"fit_sample_ids": list(train_ids), "tasks": parameters}


def select_application_targets(targets, sample_ids, device):
    index = {value: i for i, value in enumerate(targets.sample_ids)}
    positions = [index[value] for value in sample_ids]
    return dict(values={name: value[positions].to(device) for name, value in targets.values.items()},
        valid_masks={name: value[positions].to(device) for name, value in targets.valid_masks.items()},
        sample_weights=(targets.sample_weights[positions].to(device) if targets.sample_weights is not None
                        else torch.ones(len(positions), device=device)))


def application_task_losses(output, selected, definitions, parameters):
    losses, counts = {}, {}
    for task in definitions:
        prediction = output["task_predictions"][task.name]
        target, mask = selected["values"][task.name], selected["valid_masks"][task.name]
        params = parameters["tasks"][task.name]
        if task.kind == "regression":
            if target.ndim == 1:
                target, mask = target[:, None], mask[:, None]
            center, scale = (prediction.new_tensor(params[name]) for name in ("center", "scale"))
            safe_target = target.masked_fill(~mask, 0)
            element_loss = (prediction - (safe_target - center) / scale).square()
        else:
            if task.kind == "sequence_classification":
                mask = mask & output["valid_mask"]
            safe_target = target.masked_fill(~mask, 0)
            if ((safe_target < 0) | (safe_target >= task.output_dim)).any():
                raise RepresentationContractError("target class exceeds task head")
            element_loss = F.cross_entropy(prediction.reshape(-1, task.output_dim), safe_target.reshape(-1),
                weight=prediction.new_tensor(params["class_weights"]), reduction="none").reshape(mask.shape)
        mask = mask.reshape(mask.shape[0], -1)
        element_loss = element_loss.reshape(mask.shape)
        per_sample = element_loss.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        weights = selected["sample_weights"] * mask.any(dim=1)
        losses[task.name] = (per_sample * weights).sum() / weights.sum().clamp_min(1e-12)
        counts[task.name] = float(weights.sum())
    active = [loss for name, loss in losses.items() if counts[name] > 0]
    total = torch.stack(active).mean() if active else output["sequence_embedding"].sum() * 0
    return losses | {"total": total, "counts": counts}
