"""Candidate selection and ensemble helpers for public UAB torch opt."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.evaluation import evaluate_regression_predictions
from chronaris.evaluation.public_datasets.pipelines.opt_postprocess import validate_public_opt_prediction_aggregation_policy
from chronaris.evaluation.public_datasets.pipelines.opt_shared import sanitize_public_opt_metrics
from chronaris.evaluation.public_datasets.pipelines.opt_torch_supervision import validate_torch_uab_supervision_granularity


def _build_torch_uab_full_shortlist_rows(
    *,
    leaderboard: pd.DataFrame,
    full_candidate_limit: int,
    group_winner_limit: int,
    ensemble_policy: str,
    selected_subsets: tuple[str, ...],
    candidate_catalog: str,
) -> list[dict[str, object]]:
    shortlist_size = max(
        full_candidate_limit,
        2 if ensemble_policy == "mean_top2" else 1,
    )
    selected_candidate_ids: list[str] = []
    _extend_shortlist_with_ordered_ids(
        selected_candidate_ids,
        leaderboard["candidate_id"].head(shortlist_size).astype(str).tolist(),
    )
    if candidate_catalog == "heat_specialist":
        _extend_shortlist_with_sorted_candidates(
            selected_candidate_ids,
            leaderboard=leaderboard,
            sort_columns=("screen_mean_mae", "screen_mean_rmse", "candidate_id"),
        )
        _extend_shortlist_with_sorted_candidates(
            selected_candidate_ids,
            leaderboard=leaderboard.loc[
                leaderboard["candidate_id"]
                .astype(str)
                .str.startswith("heat_affine_calibrated_blend")
            ].reset_index(drop=True),
            sort_columns=("screen_mean_rmse", "screen_mean_mae", "candidate_id"),
        )
    if group_winner_limit > 0:
        for subset_id in selected_subsets:
            metric_name = f"{subset_id}_rmse"
            if metric_name not in leaderboard.columns:
                continue
            ordered_group = leaderboard.sort_values(
                [metric_name, "screen_mean_rmse", "screen_mean_mae", "candidate_id"],
                ascending=[True, True, True, True],
            )
            _extend_shortlist_with_ordered_ids(
                selected_candidate_ids,
                ordered_group["candidate_id"]
                .head(group_winner_limit)
                .astype(str)
                .tolist(),
            )
    by_candidate_id = {
        str(row["candidate_id"]): row
        for row in leaderboard.to_dict(orient="records")
    }
    return [by_candidate_id[candidate_id] for candidate_id in selected_candidate_ids]


def _extend_shortlist_with_sorted_candidates(
    selected_candidate_ids: list[str],
    *,
    leaderboard: pd.DataFrame,
    sort_columns: tuple[str, ...],
) -> None:
    if leaderboard.empty:
        return
    ordered = leaderboard.sort_values(
        list(sort_columns),
        ascending=[True] * len(sort_columns),
    )
    _extend_shortlist_with_ordered_ids(
        selected_candidate_ids,
        ordered["candidate_id"].head(1).astype(str).tolist(),
    )


def _extend_shortlist_with_ordered_ids(
    selected_candidate_ids: list[str],
    candidate_ids: Sequence[str],
) -> None:
    for candidate_id in candidate_ids:
        if candidate_id not in selected_candidate_ids:
            selected_candidate_ids.append(candidate_id)


def _validate_torch_uab_runtime_config(config: Any) -> None:
    if config.full_candidate_limit < 1:
        raise ValueError("torch UAB full_candidate_limit must be >= 1.")
    if config.full_group_winner_limit < 0:
        raise ValueError("torch UAB full_group_winner_limit must be >= 0.")
    if config.ensemble_policy not in {"none", "mean_top2"}:
        raise ValueError(
            f"unsupported torch UAB ensemble_policy: {config.ensemble_policy}"
        )
    validate_public_opt_prediction_aggregation_policy(
        config.prediction_aggregation_policy
    )
    validate_torch_uab_supervision_granularity(config.supervision_granularity)
    selected_subsets = _normalize_torch_uab_selected_subsets(config.selected_subsets)
    if config.candidate_catalog == "heat_specialist" and selected_subsets != (
        "heat_the_chair",
    ):
        raise ValueError(
            "candidate_catalog=heat_specialist only supports selected_subsets=('heat_the_chair',)."
        )


def _normalize_torch_uab_selected_subsets(
    selected_subsets: Sequence[str],
) -> tuple[str, ...]:
    if not selected_subsets:
        raise ValueError("torch UAB selected_subsets must not be empty.")
    normalized = tuple(dict.fromkeys(str(value) for value in selected_subsets))
    allowed = {"n_back", "heat_the_chair"}
    unsupported = sorted(set(normalized) - allowed)
    if unsupported:
        raise ValueError(
            "unsupported torch UAB selected_subsets: " + ", ".join(unsupported)
        )
    return normalized


def _group_metric_or_nan(
    groups: Mapping[str, object],
    subset_id: str,
    metric_name: str,
) -> float:
    payload = groups.get(subset_id)
    if not isinstance(payload, Mapping):
        return float("nan")
    value = payload.get(metric_name)
    return float(value) if value is not None else float("nan")


def _select_terminal_torch_uab_result(
    *,
    candidate_results: Mapping[str, Mapping[str, object]],
    candidate_predictions: Mapping[str, pd.DataFrame],
    ensemble_policy: str,
    selected_subsets: tuple[str, ...],
) -> tuple[dict[str, object], pd.DataFrame]:
    selected_groups: dict[str, object] = {}
    selected_predictions: list[pd.DataFrame] = []
    selection_details: dict[str, object] = {}
    for subset_id in selected_subsets:
        candidate_metrics = {
            candidate_id: result["groups"][subset_id]
            for candidate_id, result in candidate_results.items()
        }
        candidate_frames = {
            candidate_id: predictions.loc[
                predictions["subset_id"].astype(str) == subset_id
            ].reset_index(drop=True)
            for candidate_id, predictions in candidate_predictions.items()
        }
        best_candidate_id = min(
            candidate_metrics,
            key=lambda candidate_id: (
                float(candidate_metrics[candidate_id]["rmse"]),
                float(candidate_metrics[candidate_id]["mae"]),
                candidate_id,
            ),
        )
        selected_metrics = dict(candidate_metrics[best_candidate_id])
        selected_frame = candidate_frames[best_candidate_id].copy()
        selection_payload = {
            "selected_source_type": "candidate",
            "selected_source_id": best_candidate_id,
            "selected_metrics": {
                "rmse": float(selected_metrics["rmse"]),
                "mae": float(selected_metrics["mae"]),
            },
        }
        if ensemble_policy == "mean_top2":
            ensemble_frame, ensemble_metrics, ensemble_members = _build_torch_regression_ensemble(
                candidate_prediction_frames=candidate_frames,
                candidate_metrics=candidate_metrics,
            )
            if (
                ensemble_frame is not None
                and ensemble_metrics is not None
                and _is_better_regression_metrics(ensemble_metrics, selected_metrics)
            ):
                selected_metrics = ensemble_metrics
                selected_frame = ensemble_frame
                selection_payload = {
                    "selected_source_type": "mean_top2_ensemble",
                    "selected_source_id": "mean_top2_ensemble",
                    "selected_members": list(ensemble_members),
                    "selected_metrics": {
                        "rmse": float(selected_metrics["rmse"]),
                        "mae": float(selected_metrics["mae"]),
                    },
                }
        selected_groups[subset_id] = selected_metrics
        selected_predictions.append(selected_frame)
        selection_details[subset_id] = selection_payload
    merged_predictions = pd.concat(selected_predictions, axis=0, ignore_index=True)
    mean_rmse = float(
        np.mean([float(selected_groups[subset_id]["rmse"]) for subset_id in selected_subsets], dtype=np.float64)
    )
    mean_mae = float(
        np.mean([float(selected_groups[subset_id]["mae"]) for subset_id in selected_subsets], dtype=np.float64)
    )
    return {
        "selection_policy": {
            "ensemble_policy": ensemble_policy,
            "selection_scope": "per_subset_best_of_full_candidates",
        },
        "groups": selected_groups,
        "group_selections": selection_details,
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
    }, merged_predictions


def _build_torch_regression_ensemble(
    *,
    candidate_prediction_frames: Mapping[str, pd.DataFrame],
    candidate_metrics: Mapping[str, Mapping[str, object]],
) -> tuple[pd.DataFrame | None, dict[str, object] | None, tuple[str, str] | None]:
    if len(candidate_prediction_frames) < 2:
        return None, None, None
    top_two = tuple(
        sorted(
            candidate_metrics,
            key=lambda candidate_id: (
                float(candidate_metrics[candidate_id]["rmse"]),
                float(candidate_metrics[candidate_id]["mae"]),
                candidate_id,
            ),
        )[:2]
    )
    merged = _merge_prediction_frames(
        [candidate_prediction_frames[candidate_id] for candidate_id in top_two]
    )
    if merged.empty:
        return None, None, None
    ensemble = merged.loc[
        :,
        [
            "track",
            "dataset_id",
            "profile",
            "evaluation_group",
            "subset_id",
            "split_group",
            "sample_id",
            "subject_id",
            "session_id",
            "y_true",
        ],
    ].copy()
    ensemble["candidate_id"] = "mean_top2_ensemble"
    ensemble["model_name"] = "mean_top2_ensemble"
    ensemble["feature_profile"] = "ensemble"
    ensemble["y_pred"] = merged[
        [f"y_pred__{index}" for index in range(len(top_two))]
    ].mean(axis=1)
    ensemble = ensemble[
        [
            "track",
            "dataset_id",
            "profile",
            "evaluation_group",
            "subset_id",
            "candidate_id",
            "model_name",
            "feature_profile",
            "split_group",
            "sample_id",
            "subject_id",
            "session_id",
            "y_true",
            "y_pred",
        ]
    ]
    metrics = sanitize_public_opt_metrics(evaluate_regression_predictions(ensemble))
    return ensemble, metrics, top_two


def _merge_prediction_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    base_keys = [
        "track",
        "dataset_id",
        "profile",
        "evaluation_group",
        "subset_id",
        "split_group",
        "sample_id",
        "subject_id",
        "session_id",
        "y_true",
    ]
    merged = None
    for index, frame in enumerate(frames):
        renamed = frame.loc[:, base_keys + ["y_pred"]].rename(
            columns={"y_pred": f"y_pred__{index}"}
        )
        merged = renamed if merged is None else merged.merge(renamed, on=base_keys, how="inner")
    if merged is None:
        return pd.DataFrame()
    return merged


def _is_better_regression_metrics(
    candidate_metrics: Mapping[str, object],
    incumbent_metrics: Mapping[str, object],
) -> bool:
    return (
        float(candidate_metrics["rmse"]),
        float(candidate_metrics["mae"]),
    ) < (
        float(incumbent_metrics["rmse"]),
        float(incumbent_metrics["mae"]),
    )
