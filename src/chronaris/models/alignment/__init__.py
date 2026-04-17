"""Continuous-time alignment models."""

from typing import Any

from chronaris.models.alignment.splits import (
    ChronologicalSampleSplit,
    ChronologicalSplitConfig,
    split_e0_samples_chronologically,
)
from chronaris.models.alignment.reference_grid import (
    ReferenceGrid,
    ReferenceGridConfig,
    build_reference_grid,
    build_reference_grids,
)

_BATCHING_EXPORTS = {
    "AlignmentBatch",
    "AlignmentStreamBatch",
    "build_alignment_batch",
}

__all__ = [
    "ChronologicalSampleSplit",
    "ChronologicalSplitConfig",
    "ReferenceGrid",
    "ReferenceGridConfig",
    "build_reference_grid",
    "build_reference_grids",
    "split_e0_samples_chronologically",
] + sorted(_BATCHING_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _BATCHING_EXPORTS:
        from chronaris.models.alignment.batching import AlignmentBatch, AlignmentStreamBatch, build_alignment_batch

        exports = {
            "AlignmentBatch": AlignmentBatch,
            "AlignmentStreamBatch": AlignmentStreamBatch,
            "build_alignment_batch": build_alignment_batch,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
