"""Candidate and config helpers for torch-native UAB public-opt runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class TorchUABCandidateSpec:
    candidate_id: str
    model_family: str
    feature_profile: str
    hidden_dims: tuple[int, int]
    dropout: float
    learning_rate: float
    weight_decay: float


def validate_torch_uab_config(
    *,
    dataset_id: str,
    expected_dataset_id: str,
    profile: str,
    expected_profile: str,
    feature_profiles: tuple[str, ...],
    learning_rates: tuple[float, ...],
    weight_decays: tuple[float, ...],
) -> None:
    if dataset_id != expected_dataset_id:
        raise ValueError(f"torch UAB runner only supports {expected_dataset_id}.")
    if profile != expected_profile:
        raise ValueError(f"torch UAB runner only supports profile={expected_profile}.")
    if not feature_profiles:
        raise ValueError("torch UAB runner requires at least one feature profile.")
    if set(feature_profiles) - {"full", "residual_only"}:
        raise ValueError(
            "torch UAB runner only supports feature profiles full/residual_only."
        )
    if not learning_rates or not weight_decays:
        raise ValueError("torch UAB runner requires non-empty lr and weight decay grids.")


def build_torch_uab_candidates(
    *,
    feature_profiles: tuple[str, ...],
    learning_rates: tuple[float, ...],
    weight_decays: tuple[float, ...],
) -> tuple[TorchUABCandidateSpec, ...]:
    families = (
        ("mlp_huber_small", (128, 64), 0.1),
        ("mlp_huber_wide", (256, 128), 0.2),
        ("residual_gated_mlp", (128, 64), 0.1),
    )
    rows: list[TorchUABCandidateSpec] = []
    for feature_profile in feature_profiles:
        for model_family, hidden_dims, dropout in families:
            for learning_rate in learning_rates:
                for weight_decay in weight_decays:
                    candidate_id = (
                        f"{model_family}__{feature_profile}"
                        f"__lr{_slug_float(learning_rate)}"
                        f"__wd{_slug_float(weight_decay)}"
                    )
                    rows.append(
                        TorchUABCandidateSpec(
                            candidate_id=candidate_id,
                            model_family=model_family,
                            feature_profile=feature_profile,
                            hidden_dims=hidden_dims,
                            dropout=dropout,
                            learning_rate=float(learning_rate),
                            weight_decay=float(weight_decay),
                        )
                    )
    return tuple(rows)


def is_better_torch_screen_row(
    row: Mapping[str, object],
    best_row: Mapping[str, object] | None,
) -> bool:
    if best_row is None:
        return True
    return (
        float(row["screen_mean_rmse"]),
        float(row["screen_mean_mae"]),
        str(row.get("candidate_id", "")),
    ) < (
        float(best_row["screen_mean_rmse"]),
        float(best_row["screen_mean_mae"]),
        str(best_row.get("candidate_id", "")),
    )


def _slug_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")
