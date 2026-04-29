"""Stage I Phase 2 case-study tests."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.features import load_stage_i_case_study_run
from chronaris.pipelines.stage_i_case_study import (
    StageICaseStudyConfig,
    render_stage_i_case_study_report,
    run_stage_i_case_study,
)


class StageICaseStudyPipelineTest(unittest.TestCase):
    def test_case_study_loader_and_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_manifest_path = _write_fake_stage_h_case_run(root)

            run_input = load_stage_i_case_study_run(run_manifest_path)
            self.assertEqual(len(run_input.views), 2)
            self.assertEqual(run_input.views[0].case_partition_sample_count, 2)
            self.assertEqual(run_input.views[1].projection_diagnostics_verdict, "WARN")

            result = run_stage_i_case_study(
                StageICaseStudyConfig(
                    run_id="stage-i-phase2-test",
                    stage_h_run_manifest_path=str(run_manifest_path),
                    output_root=str(root / "artifacts" / "stage_i"),
                    report_path=str(root / "docs" / "reports" / "stage-i-phase2-test.md"),
                    top_k_windows=1,
                )
            )

            self.assertEqual(len(result.view_results), 2)
            self.assertEqual(len(result.pilot_comparisons), 1)
            warn_view = next(item for item in result.view_results if item.view_summary.verdict == "WARN")
            self.assertIsNotNone(warn_view.warn_explanation)
            self.assertEqual(
                [ablation.name for ablation in warn_view.ablations],
                [
                    "projection_refusion_baseline",
                    "no_event_bias",
                    "no_state_normalization",
                    "vehicle_delta_suppressed",
                ],
            )
            suppressed = next(item for item in warn_view.ablations if item.name == "vehicle_delta_suppressed")
            self.assertAlmostEqual(suppressed.mean_top_event_score, 0.0, places=6)
            self.assertEqual(len(warn_view.top_windows), 1)

            report = render_stage_i_case_study_report(result)
            self.assertIn("WARN View Interpretation", report)
            self.assertIn("Same-Sortie Pilot Comparison", report)
            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.view_summary_csv_path).exists())
            self.assertTrue(Path(result.ablation_summary_csv_path).exists())
            self.assertTrue(Path(result.window_rankings_csv_path).exists())


def _write_fake_stage_h_case_run(root: Path) -> Path:
    artifact_root = root / "artifacts" / "stage_h" / "stage-h-case"
    sortie_root = artifact_root / "sorties" / "sortie-001"
    view_a_dir = sortie_root / "views" / "sortie-001__pilot_10035"
    view_b_dir = sortie_root / "views" / "sortie-001__pilot_10033"
    for directory in (view_a_dir, view_b_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _write_fake_view(
        view_dir=view_a_dir,
        view_id="sortie-001__pilot_10035",
        sortie_id="sortie-001",
        pilot_id=10035,
        verdict="PASS",
        projection_mean=0.72,
        projection_cv=0.12,
        l2_gap=0.08,
        l2_gap_cv=0.20,
    )
    _write_fake_view(
        view_dir=view_b_dir,
        view_id="sortie-001__pilot_10033",
        sortie_id="sortie-001",
        pilot_id=10033,
        verdict="WARN",
        projection_mean=0.56,
        projection_cv=0.31,
        l2_gap=0.11,
        l2_gap_cv=0.56,
    )

    sortie_manifest_path = sortie_root / "sortie_manifest.json"
    sortie_manifest_path.write_text(
        json.dumps(
            {
                "sortie_id": "sortie-001",
                "view_manifest_paths": {
                    "sortie-001__pilot_10035": str(view_a_dir / "view_manifest.json"),
                    "sortie-001__pilot_10033": str(view_b_dir / "view_manifest.json"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_manifest_path = artifact_root / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "output_root": str(artifact_root),
                "sortie_manifest_paths": {
                    "sortie-001": str(sortie_manifest_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_manifest_path


def _write_fake_view(
    *,
    view_dir: Path,
    view_id: str,
    sortie_id: str,
    pilot_id: int,
    verdict: str,
    projection_mean: float,
    projection_cv: float,
    l2_gap: float,
    l2_gap_cv: float,
) -> None:
    physiology = np.asarray(
        [
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    vehicle = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.8, 0.0], [0.8, 1.0]],
        ],
        dtype=np.float32,
    )
    fused = np.concatenate((physiology, vehicle, physiology - vehicle), axis=-1)
    offsets = np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    attention = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.6, 0.4, 0.0], [0.2, 0.3, 0.5]],
            [[1.0, 0.0, 0.0], [0.7, 0.3, 0.0], [0.1, 0.3, 0.6]],
        ],
        dtype=np.float32,
    )
    event_scores = np.asarray([[0.0, 0.5, 1.0], [0.0, 0.4, 1.0]], dtype=np.float32)
    np.savez(
        view_dir / "feature_bundle.npz",
        physiology_reference_projection=physiology,
        vehicle_reference_projection=vehicle,
        fused_representation=fused,
        reference_offsets_s=offsets,
        attention_weights=attention,
        vehicle_event_scores=event_scores,
    )

    sample_ids = [f"{sortie_id}:0001", f"{sortie_id}:0002"]
    projection_summary = {
        "summary": {
            "sample_count": 2,
            "reference_point_count": 3,
            "mean_projection_cosine": projection_mean,
            "cv_projection_cosine": projection_cv,
            "mean_projection_l2_gap": l2_gap,
            "cv_projection_l2_gap": l2_gap_cv,
            "samples": [
                {
                    "sample_id": sample_ids[0],
                    "reference_point_count": 3,
                    "mean_projection_cosine": projection_mean,
                    "min_projection_cosine": projection_mean - 0.1,
                    "max_projection_cosine": projection_mean + 0.1,
                    "physiology_projection_l2_mean": 1.0,
                    "vehicle_projection_l2_mean": 1.0 + l2_gap,
                    "projection_l2_gap_mean": l2_gap,
                    "projection_l2_ratio_mean": 1.0 + l2_gap,
                },
                {
                    "sample_id": sample_ids[1],
                    "reference_point_count": 3,
                    "mean_projection_cosine": projection_mean,
                    "min_projection_cosine": projection_mean - 0.1,
                    "max_projection_cosine": projection_mean + 0.1,
                    "physiology_projection_l2_mean": 1.0,
                    "vehicle_projection_l2_mean": 1.0 + l2_gap,
                    "projection_l2_gap_mean": l2_gap,
                    "projection_l2_ratio_mean": 1.0 + l2_gap,
                },
            ],
        },
        "threshold_evaluation": {
            "verdict": verdict,
            "checks": [
                {
                    "name": "mean_projection_cosine",
                    "passed": verdict == "PASS",
                    "actual": projection_mean,
                    "operator": ">=",
                    "expected": 0.65,
                },
                {
                    "name": "projection_cosine_cv",
                    "passed": verdict == "PASS",
                    "actual": projection_cv,
                    "operator": "<=",
                    "expected": 0.15,
                },
                {
                    "name": "projection_l2_gap_cv",
                    "passed": verdict == "PASS",
                    "actual": l2_gap_cv,
                    "operator": "<=",
                    "expected": 0.25,
                },
            ],
        },
    }
    (view_dir / "projection_diagnostics_summary.json").write_text(
        json.dumps(projection_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    causal_summary = {
        "sample_count": 2,
        "reference_point_count": 3,
        "state_dim": 2,
        "fused_dim": 6,
        "mean_attention_entropy": 0.5,
        "mean_max_attention": 0.8,
        "mean_causal_option_count": 2.0,
        "mean_top_event_score": 1.0,
        "mean_top_contribution_score": 1.5,
        "samples": [
            {
                "sample_id": sample_ids[0],
                "reference_point_count": 3,
                "state_dim": 2,
                "fused_dim": 6,
                "mean_attention_entropy": 0.5,
                "mean_max_attention": 0.8,
                "mean_causal_option_count": 2.0,
                "top_event_offset_s": 2.0,
                "top_event_score": 1.0,
                "top_contribution_offset_s": 1.0,
                "top_contribution_score": 1.5,
                "attention_weights": attention[0].tolist(),
                "vehicle_event_scores": event_scores[0].tolist(),
            },
            {
                "sample_id": sample_ids[1],
                "reference_point_count": 3,
                "state_dim": 2,
                "fused_dim": 6,
                "mean_attention_entropy": 0.5,
                "mean_max_attention": 0.8,
                "mean_causal_option_count": 2.0,
                "top_event_offset_s": 2.0,
                "top_event_score": 1.0,
                "top_contribution_offset_s": 1.0,
                "top_contribution_score": 1.5,
                "attention_weights": attention[1].tolist(),
                "vehicle_event_scores": event_scores[1].tolist(),
            },
        ],
    }
    (view_dir / "causal_fusion_summary.json").write_text(
        json.dumps(causal_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (view_dir / "intermediate_summary.json").write_text(
        json.dumps({"partition": "test", "sample_count": 2, "reference_point_count": 3}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    window_rows = [
        {
            "sample_id": f"{sortie_id}:0000",
            "sortie_id": sortie_id,
            "window_index": 0,
            "start_offset_ms": 0,
            "end_offset_ms": 5000,
            "physiology_point_count": 25,
            "vehicle_point_count": 120,
            "selected_for_model": True,
        },
        {
            "sample_id": sample_ids[0],
            "sortie_id": sortie_id,
            "window_index": 1,
            "start_offset_ms": 5000,
            "end_offset_ms": 10000,
            "physiology_point_count": 25,
            "vehicle_point_count": 120,
            "selected_for_model": True,
        },
        {
            "sample_id": sample_ids[1],
            "sortie_id": sortie_id,
            "window_index": 2,
            "start_offset_ms": 10000,
            "end_offset_ms": 15000,
            "physiology_point_count": 25,
            "vehicle_point_count": 120,
            "selected_for_model": True,
        },
    ]
    (view_dir / "window_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in window_rows),
        encoding="utf-8",
    )

    view_manifest = {
        "view_id": view_id,
        "sortie_id": sortie_id,
        "pilot_id": pilot_id,
        "projection_diagnostics_verdict": verdict,
        "artifact_paths": {
            "feature_bundle_npz": str(view_dir / "feature_bundle.npz"),
            "intermediate_summary_json": str(view_dir / "intermediate_summary.json"),
            "projection_diagnostics_summary_json": str(view_dir / "projection_diagnostics_summary.json"),
            "causal_fusion_summary_json": str(view_dir / "causal_fusion_summary.json"),
            "window_manifest_jsonl": str(view_dir / "window_manifest.jsonl"),
        },
    }
    (view_dir / "view_manifest.json").write_text(
        json.dumps(view_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
