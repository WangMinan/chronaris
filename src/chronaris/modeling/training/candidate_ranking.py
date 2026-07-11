"""Deterministic within-method ranking for the frozen four-candidate screen."""

from __future__ import annotations

from typing import Mapping, Sequence

from chronaris.modeling.training.candidate_screen import (
    PUBLIC_SELECTION_WEIGHTS,
    CandidateScreenResult,
)


def rank_encoder_candidates(
    results: Sequence[CandidateScreenResult],
) -> tuple[Mapping[str, object], ...]:
    grouped: dict[str, list[CandidateScreenResult]] = {}
    for result in results:
        grouped.setdefault(result.method_name, []).append(result)
    rows = []
    for method_name, values in sorted(grouped.items()):
        by_id = {value.candidate_id: value for value in values}
        if set(by_id) != {"A", "B", "C", "D"} or len(values) != 4:
            raise ValueError(f"candidate ranking requires A-D exactly once for {method_name}")
        normalized_by_loss = {}
        for loss_name in PUBLIC_SELECTION_WEIGHTS:
            raw = {
                key: value.best_validation_losses[loss_name]
                for key, value in by_id.items()
            }
            lower = min(raw.values())
            upper = max(raw.values())
            normalized_by_loss[loss_name] = {
                key: (0.0 if upper == lower else (score - lower) / (upper - lower))
                for key, score in raw.items()
            }
        method_rows = []
        for candidate_id, result in sorted(by_id.items()):
            normalized = {
                name: normalized_by_loss[name][candidate_id]
                for name in PUBLIC_SELECTION_WEIGHTS
            }
            score = sum(
                PUBLIC_SELECTION_WEIGHTS[name] * normalized[name]
                for name in PUBLIC_SELECTION_WEIGHTS
            )
            method_rows.append(
                {
                    "method_name": method_name,
                    "candidate_id": candidate_id,
                    "raw_validation_losses": dict(result.best_validation_losses),
                    "normalized_validation_losses": normalized,
                    "selection_loss": score,
                    "parameter_count": result.parameter_count,
                    "best_epoch": result.best_epoch,
                    "completed_epochs": result.completed_epochs,
                    "checkpoint_path": result.best_checkpoint_path,
                }
            )
        method_rows.sort(
            key=lambda row: (
                row["selection_loss"],
                row["parameter_count"],
                row["candidate_id"],
            )
        )
        for rank, row in enumerate(method_rows, start=1):
            rows.append({**row, "rank": rank, "selected": rank == 1})
    return tuple(rows)
