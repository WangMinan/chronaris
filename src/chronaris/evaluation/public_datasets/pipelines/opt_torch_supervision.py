"""Supervision-shape helpers for torch-native UAB public-opt runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SUPPORTED_TORCH_UAB_SUPERVISION_GRANULARITIES = (
    "window",
    "session_pooled_broadcast",
)


@dataclass(frozen=True, slots=True)
class TorchUABSupervisionView:
    frame: pd.DataFrame
    matrix: np.ndarray
    targets: np.ndarray
    split_groups: np.ndarray


def validate_torch_uab_supervision_granularity(granularity: str) -> None:
    if granularity not in set(SUPPORTED_TORCH_UAB_SUPERVISION_GRANULARITIES):
        raise ValueError(
            "unsupported torch UAB supervision_granularity: "
            f"{granularity}"
        )


def build_torch_uab_supervision_view(
    *,
    subset_frame: pd.DataFrame,
    subset_matrix: np.ndarray,
    targets: np.ndarray,
    supervision_granularity: str,
) -> TorchUABSupervisionView:
    validate_torch_uab_supervision_granularity(supervision_granularity)
    subset_frame = subset_frame.reset_index(drop=True)
    if supervision_granularity == "window":
        return TorchUABSupervisionView(
            frame=subset_frame,
            matrix=np.asarray(subset_matrix, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.float32),
            split_groups=subset_frame["split_group"].astype(str).to_numpy(),
        )

    grouped_frames: list[dict[str, object]] = []
    pooled_matrices: list[np.ndarray] = []
    pooled_targets: list[float] = []
    for session_id, session_frame in subset_frame.groupby("session_id", sort=False):
        session_indices = session_frame.index.to_numpy(dtype=int, copy=True)
        grouped_frames.append(
            {
                "session_id": str(session_id),
                "split_group": str(session_frame["split_group"].iloc[0]),
                "subject_id": str(session_frame["subject_id"].iloc[0]),
                "member_count": int(len(session_indices)),
            }
        )
        pooled_matrices.append(
            np.mean(subset_matrix[session_indices], axis=0, dtype=np.float64).astype(
                np.float32
            )
        )
        pooled_targets.append(float(session_frame["y_true"].iloc[0]))

    pooled_frame = pd.DataFrame(grouped_frames)
    pooled_matrix = np.vstack(pooled_matrices).astype(np.float32, copy=False)
    pooled_target_array = np.asarray(pooled_targets, dtype=np.float32)
    return TorchUABSupervisionView(
        frame=pooled_frame,
        matrix=pooled_matrix,
        targets=pooled_target_array,
        split_groups=pooled_frame["split_group"].astype(str).to_numpy(),
    )


def broadcast_torch_uab_session_predictions(
    *,
    target_frame: pd.DataFrame,
    pooled_frame: pd.DataFrame,
    pooled_predictions: np.ndarray,
) -> np.ndarray:
    by_session_id = {
        str(session_id): float(prediction)
        for session_id, prediction in zip(
            pooled_frame["session_id"].astype(str).tolist(),
            np.asarray(pooled_predictions, dtype=np.float32).tolist(),
            strict=True,
        )
    }
    return (
        target_frame["session_id"]
        .astype(str)
        .map(by_session_id)
        .to_numpy(dtype=np.float32, copy=True)
    )
