"""Tests for Stage E reference-grid builders."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix
from chronaris.models.alignment import ReferenceGridConfig, build_reference_grid, build_reference_grids
from chronaris.schema.models import StreamKind


def _empty_stream(kind: StreamKind) -> NumericStreamMatrix:
    return NumericStreamMatrix(
        stream_kind=kind,
        point_count=0,
        feature_names=(),
        point_offsets_ms=(),
        point_measurements=(),
        values=(),
        dropped_fields=(),
    )


def _sample(sample_id: str, start_offset_ms: int, end_offset_ms: int) -> E0ExperimentSample:
    return E0ExperimentSample(
        sample_id=sample_id,
        sortie_id="sortie-001",
        start_offset_ms=start_offset_ms,
        end_offset_ms=end_offset_ms,
        physiology=_empty_stream(StreamKind.PHYSIOLOGY),
        vehicle=_empty_stream(StreamKind.VEHICLE),
    )


class AlignmentReferenceGridTest(unittest.TestCase):
    def test_build_reference_grid_spans_full_window_when_include_end_is_true(self) -> None:
        grid = build_reference_grid(
            _sample("sample-001", 0, 5000),
            config=ReferenceGridConfig(point_count=5, include_end=True),
        )

        self.assertEqual(grid.duration_ms, 5000)
        self.assertEqual(grid.relative_offsets_ms, (0.0, 1250.0, 2500.0, 3750.0, 5000.0))
        self.assertEqual(grid.absolute_offsets_ms, (0.0, 1250.0, 2500.0, 3750.0, 5000.0))
        self.assertEqual(grid.relative_offsets_s, (0.0, 1.25, 2.5, 3.75, 5.0))

    def test_build_reference_grid_uses_left_closed_spacing_when_include_end_is_false(self) -> None:
        grid = build_reference_grid(
            _sample("sample-002", 1000, 5000),
            config=ReferenceGridConfig(point_count=4, include_end=False),
        )

        self.assertEqual(grid.duration_ms, 4000)
        self.assertEqual(grid.relative_offsets_ms, (0.0, 1000.0, 2000.0, 3000.0))
        self.assertEqual(grid.absolute_offsets_ms, (1000.0, 2000.0, 3000.0, 4000.0))

    def test_build_reference_grid_supports_single_reference_point(self) -> None:
        grid = build_reference_grid(
            _sample("sample-003", 2000, 7000),
            config=ReferenceGridConfig(point_count=1),
        )

        self.assertEqual(grid.relative_offsets_ms, (0.0,))
        self.assertEqual(grid.absolute_offsets_ms, (2000.0,))

    def test_build_reference_grids_preserves_sample_order(self) -> None:
        grids = build_reference_grids(
            (
                _sample("sample-010", 0, 5000),
                _sample("sample-011", 5000, 10000),
            ),
            config=ReferenceGridConfig(point_count=3),
        )

        self.assertEqual(tuple(grid.sample_id for grid in grids), ("sample-010", "sample-011"))
        self.assertEqual(grids[1].absolute_offsets_ms, (5000.0, 7500.0, 10000.0))


if __name__ == "__main__":
    unittest.main()
