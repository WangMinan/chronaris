"""Post-processing helpers for task evaluation public-opt predictions."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

SUPPORTED_PUBLIC_OPT_PREDICTION_AGGREGATION_POLICIES = (
    "none",
    "session_mean_broadcast",
    "session_median_broadcast",
)


def validate_public_opt_prediction_aggregation_policy(policy: str) -> None:
    if policy not in set(SUPPORTED_PUBLIC_OPT_PREDICTION_AGGREGATION_POLICIES):
        raise ValueError(
            "unsupported public opt prediction_aggregation_policy: "
            f"{policy}"
        )


def apply_public_opt_regression_prediction_aggregation(
    predictions: pd.DataFrame,
    *,
    policy: str,
) -> pd.DataFrame:
    validate_public_opt_prediction_aggregation_policy(policy)
    if policy == "none" or predictions.empty:
        return predictions

    reducer = "mean" if policy == "session_mean_broadcast" else "median"
    group_keys = [
        column_name
        for column_name in (
            "track",
            "dataset_id",
            "profile",
            "evaluation_group",
            "subset_id",
            "head_name",
            "candidate_id",
            "model_name",
            "feature_profile",
            "split_group",
            "subject_id",
            "session_id",
        )
        if column_name in predictions.columns
    ]
    aggregated = predictions.copy()
    aggregated["y_pred"] = aggregated.groupby(group_keys, sort=False)["y_pred"].transform(
        reducer
    )
    return aggregated


def merge_public_opt_prediction_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    base_keys = [
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
        selected_columns = base_keys + ["y_pred"]
        if "prediction_confidence" in frame.columns:
            selected_columns.append("prediction_confidence")
        renamed = frame.loc[:, selected_columns].rename(
            columns={
                "y_pred": f"y_pred__{index}",
                "prediction_confidence": f"prediction_confidence__{index}",
            }
        )
        merged = (
            renamed
            if merged is None
            else merged.merge(renamed, on=base_keys, how="inner")
        )
    if merged is None:
        return pd.DataFrame()
    return merged
