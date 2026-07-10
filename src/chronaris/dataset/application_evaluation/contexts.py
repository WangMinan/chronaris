"""Thirty-second context construction over aligned Dingxin window records."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from chronaris.dataset.application_evaluation.contracts import ApplicationContextRecord


REQUIRED_RECORD_COLUMNS = frozenset(
    {
        "sample_id",
        "sortie_id",
        "view_id",
        "pilot_id",
        "window_index",
        "start_offset_ms",
        "end_offset_ms",
    }
)


def build_application_contexts(
    records: pd.DataFrame,
    *,
    history_window_count: int = 6,
    expected_stride_ms: int = 5_000,
    stride_tolerance_ms: int = 10,
) -> tuple[ApplicationContextRecord, ...]:
    """Build all candidate contexts while retaining structured exclusions."""

    missing = sorted(REQUIRED_RECORD_COLUMNS - set(records.columns))
    if missing:
        raise ValueError(f"application context records are missing columns: {', '.join(missing)}")
    if history_window_count <= 0:
        raise ValueError("history_window_count must be positive.")
    if expected_stride_ms <= 0 or stride_tolerance_ms < 0:
        raise ValueError("stride settings are invalid.")

    contexts: list[ApplicationContextRecord] = []
    for view_id, frame in records.groupby("view_id", sort=True):
        ordered = frame.sort_values(["window_index", "sample_id"], kind="mergesort").reset_index(drop=True)
        for end_position in range(history_window_count - 1, len(ordered)):
            context_rows = ordered.iloc[end_position - history_window_count + 1 : end_position + 1]
            end_row = ordered.iloc[end_position]
            classification_reason = _context_exclusion_reason(
                context_rows,
                expected_stride_ms=expected_stride_ms,
                stride_tolerance_ms=stride_tolerance_ms,
            )
            target_row = ordered.iloc[end_position + 1] if end_position + 1 < len(ordered) else None
            response_reason = classification_reason
            if response_reason is None:
                response_reason = _target_exclusion_reason(
                    end_row,
                    target_row,
                    expected_stride_ms=expected_stride_ms,
                    stride_tolerance_ms=stride_tolerance_ms,
                )
            contexts.append(
                ApplicationContextRecord(
                    context_id=f"{view_id}::context_end_{int(end_row['window_index']):04d}",
                    sortie_id=str(end_row["sortie_id"]),
                    view_id=str(view_id),
                    pilot_id=int(end_row["pilot_id"]),
                    start_window_index=int(context_rows.iloc[0]["window_index"]),
                    end_window_index=int(end_row["window_index"]),
                    source_sample_ids=tuple(str(value) for value in context_rows["sample_id"]),
                    end_sample_id=str(end_row["sample_id"]),
                    target_sample_id=None if target_row is None else str(target_row["sample_id"]),
                    start_offset_ms=int(context_rows.iloc[0]["start_offset_ms"]),
                    end_offset_ms=int(end_row["end_offset_ms"]),
                    classification_eligible=classification_reason is None,
                    response_eligible=response_reason is None,
                    classification_exclusion_reason=classification_reason,
                    response_exclusion_reason=response_reason,
                )
            )
    return tuple(contexts)


def contexts_to_frame(contexts: Sequence[ApplicationContextRecord]) -> pd.DataFrame:
    """Return a stable manifest frame for context candidates."""

    frame = pd.DataFrame(context.to_dict() for context in contexts)
    if frame.empty:
        return frame
    return frame.sort_values(["sortie_id", "view_id", "end_window_index"]).reset_index(drop=True)


def _context_exclusion_reason(
    rows: pd.DataFrame,
    *,
    expected_stride_ms: int,
    stride_tolerance_ms: int,
) -> str | None:
    if rows.empty:
        return "empty_context"
    if rows["sortie_id"].astype(str).nunique() != 1 or rows["view_id"].astype(str).nunique() != 1:
        return "mixed_group_context"
    window_indices = rows["window_index"].astype(int).to_list()
    if any(right != left + 1 for left, right in zip(window_indices[:-1], window_indices[1:])):
        return "non_consecutive_window_index"
    starts = rows["start_offset_ms"].astype(int).to_list()
    if any(
        abs((right - left) - expected_stride_ms) > stride_tolerance_ms
        for left, right in zip(starts[:-1], starts[1:])
    ):
        return "non_consecutive_window_time"
    return None


def _target_exclusion_reason(
    end_row: pd.Series,
    target_row: pd.Series | None,
    *,
    expected_stride_ms: int,
    stride_tolerance_ms: int,
) -> str | None:
    if target_row is None:
        return "future_window_unavailable"
    if str(target_row["view_id"]) != str(end_row["view_id"]):
        return "future_window_group_mismatch"
    if int(target_row["window_index"]) != int(end_row["window_index"]) + 1:
        return "future_window_index_gap"
    time_delta = int(target_row["start_offset_ms"]) - int(end_row["start_offset_ms"])
    if abs(time_delta - expected_stride_ms) > stride_tolerance_ms:
        return "future_window_time_gap"
    return None
