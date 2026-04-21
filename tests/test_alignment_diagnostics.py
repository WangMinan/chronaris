"""Tests for sample-level alignment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.alignment_diagnostics import (
    AlignmentProjectionThresholdConfig,
    evaluate_alignment_projection_thresholds,
    render_alignment_projection_diagnostics_markdown,
    summarize_alignment_projection_diagnostics,
)


@dataclass(frozen=True, slots=True)
class _FakeStreamSnapshot:
    reference_projected_states: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class _FakeSampleIntermediate:
    sample_id: str
    physiology: _FakeStreamSnapshot
    vehicle: _FakeStreamSnapshot


@dataclass(frozen=True, slots=True)
class _FakeExport:
    sample_count: int
    reference_point_count: int
    samples: tuple[_FakeSampleIntermediate, ...]


class AlignmentDiagnosticsTest(unittest.TestCase):
    def test_summarize_alignment_projection_diagnostics_returns_expected_values(self) -> None:
        export = _FakeExport(
            sample_count=2,
            reference_point_count=2,
            samples=(
                _FakeSampleIntermediate(
                    sample_id="sample-001",
                    physiology=_FakeStreamSnapshot(
                        reference_projected_states=((1.0, 0.0), (1.0, 0.0)),
                    ),
                    vehicle=_FakeStreamSnapshot(
                        reference_projected_states=((0.0, 1.0), (1.0, 0.0)),
                    ),
                ),
                _FakeSampleIntermediate(
                    sample_id="sample-002",
                    physiology=_FakeStreamSnapshot(
                        reference_projected_states=((1.0, 1.0), (1.0, 1.0)),
                    ),
                    vehicle=_FakeStreamSnapshot(
                        reference_projected_states=((1.0, 1.0), (1.0, 1.0)),
                    ),
                ),
            ),
        )

        summary = summarize_alignment_projection_diagnostics(export)

        self.assertEqual(summary.sample_count, 2)
        self.assertEqual(summary.reference_point_count, 2)
        self.assertAlmostEqual(summary.mean_projection_cosine, 0.75, places=6)
        self.assertAlmostEqual(summary.min_projection_cosine, 0.0, places=6)
        self.assertAlmostEqual(summary.max_projection_cosine, 1.0, places=6)
        self.assertAlmostEqual(summary.std_projection_cosine, 0.25, places=6)
        self.assertAlmostEqual(summary.cv_projection_cosine, 1.0 / 3.0, places=6)
        self.assertEqual(len(summary.samples), 2)
        self.assertEqual(summary.samples[0].sample_id, "sample-001")
        self.assertAlmostEqual(summary.samples[0].mean_projection_cosine, 0.5, places=6)
        self.assertAlmostEqual(summary.samples[1].mean_projection_cosine, 1.0, places=6)

    def test_render_alignment_projection_diagnostics_markdown_contains_table(self) -> None:
        export = _FakeExport(
            sample_count=1,
            reference_point_count=1,
            samples=(
                _FakeSampleIntermediate(
                    sample_id="sample-001",
                    physiology=_FakeStreamSnapshot(reference_projected_states=((1.0, 0.0),)),
                    vehicle=_FakeStreamSnapshot(reference_projected_states=((1.0, 0.0),)),
                ),
            ),
        )
        summary = summarize_alignment_projection_diagnostics(export)

        markdown = render_alignment_projection_diagnostics_markdown(summary)

        self.assertIn("## Sample-Level Projection Diagnostics", markdown)
        self.assertIn("| sample id | mean cosine |", markdown)
        self.assertIn("sample-001", markdown)

    def test_threshold_evaluation_defaults_and_optional_min_cosine_gate(self) -> None:
        export = _FakeExport(
            sample_count=2,
            reference_point_count=2,
            samples=(
                _FakeSampleIntermediate(
                    sample_id="sample-001",
                    physiology=_FakeStreamSnapshot(reference_projected_states=((1.0, 0.0), (1.0, 0.0))),
                    vehicle=_FakeStreamSnapshot(reference_projected_states=((0.0, 1.0), (1.0, 0.0))),
                ),
                _FakeSampleIntermediate(
                    sample_id="sample-002",
                    physiology=_FakeStreamSnapshot(reference_projected_states=((1.0, 1.0), (1.0, 1.0))),
                    vehicle=_FakeStreamSnapshot(reference_projected_states=((1.0, 1.0), (1.0, 1.0))),
                ),
            ),
        )
        summary = summarize_alignment_projection_diagnostics(export)

        default_eval = evaluate_alignment_projection_thresholds(summary)
        self.assertFalse(default_eval.passed)
        self.assertEqual(default_eval.verdict, "WARN")
        self.assertNotIn("min_projection_cosine", tuple(check.name for check in default_eval.checks))

        relaxed_eval = evaluate_alignment_projection_thresholds(
            summary,
            config=AlignmentProjectionThresholdConfig(max_projection_cosine_cv=0.5),
        )
        self.assertTrue(relaxed_eval.passed)
        self.assertEqual(relaxed_eval.verdict, "PASS")

        strict_eval = evaluate_alignment_projection_thresholds(
            summary,
            config=AlignmentProjectionThresholdConfig(enforce_min_projection_cosine=True),
        )
        self.assertFalse(strict_eval.passed)
        self.assertEqual(strict_eval.verdict, "WARN")
        self.assertIn("min_projection_cosine", tuple(check.name for check in strict_eval.checks))

        markdown = render_alignment_projection_diagnostics_markdown(
            summary,
            threshold_evaluation=relaxed_eval,
        )
        self.assertIn("### Threshold Evaluation", markdown)
        self.assertIn("| check | actual | operator | expected | result |", markdown)


if __name__ == "__main__":
    unittest.main()
