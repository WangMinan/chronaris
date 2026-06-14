"""Torch model and tensor helpers for the public UAB opt branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from chronaris.pipelines.stage_i.public.opt_shared import sanitize_public_opt_regression_outputs
from chronaris.pipelines.stage_i.public.opt_torch_catalog import TorchUABCandidateSpec


@dataclass(frozen=True, slots=True)
class _TorchFeatureBundle:
    feature_profile: str
    feature_columns: tuple[str, ...]
    physiology_indices: tuple[int, ...]
    context_indices: tuple[int, ...]
    feature_matrix: np.ndarray
    feature_frame: pd.DataFrame
    subset_bundles: Mapping[str, "_TorchSubsetBundle"]


@dataclass(frozen=True, slots=True)
class _TorchSubsetBundle:
    subset_id: str
    subset_frame: pd.DataFrame
    subset_matrix: np.ndarray
    split_groups: np.ndarray
    targets: np.ndarray
    loso_splits: tuple[object, ...]


class _TabularRegressionDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        physiology_indices: Sequence[int],
        context_indices: Sequence[int],
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.physiology = torch.as_tensor(
            features[:, physiology_indices],
            dtype=torch.float32,
        )
        self.context = torch.as_tensor(features[:, context_indices], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.features[index],
            self.physiology[index],
            self.context[index],
            self.targets[index],
        )


class _HuberMLP(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        first_hidden, second_hidden = tuple(hidden_dims)
        self.network = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.LayerNorm(second_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        physiology_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        del physiology_features, context_features
        return self.network(features)


class _LinearHuber(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        physiology_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        del physiology_features, context_features
        return self.head(features)


class _ResidualGatedMLP(nn.Module):
    def __init__(
        self,
        physiology_dim: int,
        context_dim: int,
        *,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        first_hidden, second_hidden = tuple(hidden_dims)
        self.physiology_encoder = nn.Sequential(
            nn.Linear(physiology_dim, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(first_hidden * 2, first_hidden),
            nn.GELU(),
            nn.Linear(first_hidden, first_hidden),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.LayerNorm(second_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        physiology_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        del features
        encoded_physiology = self.physiology_encoder(physiology_features)
        encoded_context = self.context_encoder(context_features)
        gate = self.gate(torch.cat((encoded_physiology, encoded_context), dim=-1))
        fused = gate * encoded_physiology + (1.0 - gate) * encoded_context
        return self.head(fused)


def _build_torch_uab_model(
    *,
    candidate: TorchUABCandidateSpec,
    input_dim: int,
    physiology_dim: int,
    context_dim: int,
) -> nn.Module:
    if candidate.model_family == "linear_huber":
        return _LinearHuber(input_dim)
    if candidate.model_family == "mlp_huber_small":
        return _HuberMLP(
            input_dim,
            hidden_dims=candidate.hidden_dims,
            dropout=candidate.dropout,
        )
    if candidate.model_family == "mlp_huber_wide":
        return _HuberMLP(
            input_dim,
            hidden_dims=candidate.hidden_dims,
            dropout=candidate.dropout,
        )
    if candidate.model_family == "residual_gated_mlp":
        return _ResidualGatedMLP(
            physiology_dim,
            context_dim,
            hidden_dims=candidate.hidden_dims,
            dropout=candidate.dropout,
        )
    raise ValueError(f"unsupported torch UAB model family: {candidate.model_family}")


def _fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64)
    std = np.std(values, axis=0, dtype=np.float64)
    std = np.where(std <= 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_standardizer(
    values: np.ndarray,
    standardizer: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    mean, std = standardizer
    normalized = (values - mean.reshape(1, -1)) / std.reshape(1, -1)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _split_train_validation_groups(
    split_groups: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(split_groups) <= 1:
        indices = np.arange(len(split_groups), dtype=int)
        return indices, None
    unique_groups = np.unique(split_groups)
    if unique_groups.size < 2:
        indices = np.arange(len(split_groups), dtype=int)
        return indices, None
    rng = np.random.default_rng(seed)
    validation_group = str(rng.choice(unique_groups, size=1, replace=False)[0])
    validation_mask = split_groups == validation_group
    if not np.any(validation_mask) or np.all(validation_mask):
        indices = np.arange(len(split_groups), dtype=int)
        return indices, None
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    return train_indices.astype(int), validation_indices.astype(int)


def _validation_rmse(
    *,
    model: nn.Module,
    dataset: _TabularRegressionDataset | None,
    device: str,
    fallback_value: float,
) -> float:
    if dataset is None or len(dataset) == 0:
        return 0.0
    non_blocking = device == "cuda"
    predictions = _predict_torch_uab_from_tensors(
        model=model,
        feature_tensor=dataset.features.to(device=device, non_blocking=non_blocking),
        physiology_tensor=dataset.physiology.to(device=device, non_blocking=non_blocking),
        context_tensor=dataset.context.to(device=device, non_blocking=non_blocking),
        fallback_value=fallback_value,
    )
    truth = dataset.targets.cpu().numpy()
    return float(np.sqrt(np.mean(np.square(predictions - truth), dtype=np.float64)))


def _predict_torch_uab_from_tensors(
    *,
    model: nn.Module,
    feature_tensor: torch.Tensor,
    physiology_tensor: torch.Tensor,
    context_tensor: torch.Tensor,
    fallback_value: float,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        outputs = model(feature_tensor, physiology_tensor, context_tensor).reshape(-1)
    predictions, _ = sanitize_public_opt_regression_outputs(
        outputs.detach().cpu().numpy(),
        fallback_value=fallback_value,
    )
    return predictions


def _predict_torch_uab(
    *,
    model: nn.Module,
    features: np.ndarray,
    physiology_indices: Sequence[int],
    context_indices: Sequence[int],
    device: str,
    fallback_value: float,
    use_pre_sliced_branches: bool = False,
    physiology_values: np.ndarray | None = None,
    context_values: np.ndarray | None = None,
) -> np.ndarray:
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    if use_pre_sliced_branches:
        if physiology_values is None or context_values is None:
            raise ValueError("pre-sliced branch prediction requires physiology/context arrays.")
        physiology_tensor = torch.as_tensor(
            physiology_values,
            dtype=torch.float32,
            device=device,
        )
        context_tensor = torch.as_tensor(
            context_values,
            dtype=torch.float32,
            device=device,
        )
    else:
        physiology_tensor = feature_tensor[:, physiology_indices]
        context_tensor = feature_tensor[:, context_indices]
    return _predict_torch_uab_from_tensors(
        model=model,
        feature_tensor=feature_tensor,
        physiology_tensor=physiology_tensor,
        context_tensor=context_tensor,
        fallback_value=fallback_value,
    )
