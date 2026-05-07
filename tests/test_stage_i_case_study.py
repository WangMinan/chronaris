"""Stage I Phase 2 case-study tests."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.features import load_stage_i_case_study_run
from chronaris.pipelines import StageIAnchorConfig, run_stage_i_anchor
from chronaris.pipelines.stage_i.stage_i_case_study import (
    StageICaseStudyConfig,
    render_stage_i_case_study_report,
    run_stage_i_case_study,
)
from chronaris.serving import StageIRuntimeDemoConfig, run_stage_i_runtime_demo


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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_case_study_pipeline_supports_cuda_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_manifest_path = _write_fake_stage_h_case_run(root)

            result = run_stage_i_case_study(
                StageICaseStudyConfig(
                    run_id="stage-i-phase2-cuda-test",
                    stage_h_run_manifest_path=str(run_manifest_path),
                    output_root=str(root / "artifacts" / "stage_i"),
                    report_path=str(root / "docs" / "reports" / "stage-i-phase2-cuda-test.md"),
                    top_k_windows=1,
                    device="cuda",
                )
            )
            self.assertEqual(len(result.view_results), 2)


class StageIRuntimeDemoTest(unittest.TestCase):
    def test_runtime_demo_summarizes_stage_h_case_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_manifest_path = _write_fake_stage_h_case_run(root)

            result = run_stage_i_runtime_demo(
                StageIRuntimeDemoConfig(
                    run_id="runtime-demo-stage-h",
                    source_path=str(run_manifest_path),
                    artifact_root=str(root / "artifacts" / "runtime_demo"),
                    report_root=str(root / "reports"),
                )
            )

            self.assertEqual(result.source_type, "stage_h_run_manifest")
            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertTrue(Path(result.window_csv_path).exists())
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["stage_h"]["view_count"], 2)
            self.assertEqual(summary["stage_h"]["case_window_count"], 4)
            self.assertEqual(summary["stage_h"]["view_verdict_counts"]["WARN"], 1)

            window_rows = json.loads(
                Path(result.summary_path).read_text(encoding="utf-8")
            )["stage_h"]["views"]
            self.assertEqual(len(window_rows), 2)
            report = Path(result.report_path).read_text(encoding="utf-8")
            self.assertIn("Stage H Runtime Overview", report)
            self.assertIn("View Summary", report)

    def test_runtime_demo_summarizes_optimized_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_path = _write_fake_optimized_package(root)

            result = run_stage_i_runtime_demo(
                StageIRuntimeDemoConfig(
                    run_id="runtime-demo-package",
                    source_path=str(package_path),
                    artifact_root=str(root / "artifacts" / "runtime_demo"),
                    report_root=str(root / "reports"),
                )
            )

            self.assertEqual(result.source_type, "optimized_candidate_package")
            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertIsNone(result.window_csv_path)
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(
                summary["optimized_package"]["target_variant_name"],
                "chronaris_opt",
            )
            self.assertEqual(len(summary["optimized_package"]["tasks"]), 3)
            self.assertTrue(
                summary["optimized_package"]["tasks"][0]["prediction_contract_available"]
            )
            report = Path(result.report_path).read_text(encoding="utf-8")
            self.assertIn("Optimized Package Overview", report)
            self.assertIn("Task Export Summary", report)


class StageIAnchorPipelineTest(unittest.TestCase):
    def test_anchor_pipeline_exports_ranked_windows_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_manifest_path = _write_fake_stage_h_case_run(root)
            private_summary_path = _write_fake_private_benchmark_summary(root)

            result = run_stage_i_anchor(
                StageIAnchorConfig(
                    run_id="anchor-test",
                    stage_h_run_manifest_path=str(run_manifest_path),
                    output_root=str(root / "artifacts" / "stage_i_anchor"),
                    report_root=str(root / "reports"),
                    private_benchmark_summary_path=str(private_summary_path),
                    top_k_windows=1,
                )
            )

            self.assertTrue(Path(result.anchor_manifest_path).exists())
            self.assertTrue(Path(result.anchor_windows_csv_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            summary = json.loads(Path(result.anchor_manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["overview"]["selected_view_count"], 2)
            self.assertEqual(summary["overview"]["selected_anchor_count"], 2)
            self.assertEqual(summary["anchors"][0]["view_verdict"], "WARN")
            self.assertEqual(
                summary["private_no_mask_summary"]["tasks"]["T1_maneuver_intensity_class"][
                    "target_beats_no_mask"
                ],
                True,
            )
            report = Path(result.report_path).read_text(encoding="utf-8")
            self.assertIn("Anchor Windows", report)
            self.assertIn("Private No-Mask Comparison", report)

    def test_anchor_pipeline_supports_warn_only_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_manifest_path = _write_fake_stage_h_case_run(root)
            private_summary_path = _write_fake_private_benchmark_summary(root)

            result = run_stage_i_anchor(
                StageIAnchorConfig(
                    run_id="anchor-warn-only",
                    stage_h_run_manifest_path=str(run_manifest_path),
                    output_root=str(root / "artifacts" / "stage_i_anchor"),
                    report_root=str(root / "reports"),
                    private_benchmark_summary_path=str(private_summary_path),
                    top_k_windows=1,
                    view_verdict_filter="warn_only",
                )
            )

            summary = json.loads(Path(result.anchor_manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["overview"]["selected_view_count"], 1)
            self.assertEqual(summary["overview"]["selected_anchor_count"], 1)
            self.assertEqual(summary["anchors"][0]["view_verdict"], "WARN")


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


def _write_fake_optimized_package(root: Path) -> Path:
    package_path = root / "optimized_candidate_package.json"
    package_path.write_text(
        json.dumps(
            {
                "package_version": "v1",
                "run_id": "private-opt-package",
                "target_variant_name": "chronaris_opt",
                "source_manifests": {
                    "e_run_manifest_path": "e-run.json",
                    "f_run_manifest_path": "f-run.json",
                },
                "dependency_contracts": {
                    "requires_stage_h_all_window_contract": True,
                    "requires_f_full_reference_hidden": True,
                    "requires_stage_g_causal_fusion": True,
                    "use_causal_mask": True,
                },
                "records_summary": {
                    "sample_count": 111,
                    "view_count": 3,
                    "sortie_count": 2,
                },
                "selected_vehicle_fields": ["BUS001.speed", "BUS001.altitude"],
                "selected_physiology_fields": ["eeg.alpha", "spo2"],
                "tasks": {
                    "T1_maneuver_intensity_class": {
                        "status": "exported",
                        "task_type": "classification",
                        "head_family": "class_balanced_threshold",
                        "thresholds": {"low_threshold": 1.0, "high_threshold": 2.0},
                        "cross_validated_best_metrics": {
                            "macro_f1": 1.0,
                            "balanced_accuracy": 1.0,
                        },
                    },
                    "T2_next_window_physiology_response": {
                        "status": "exported",
                        "task_type": "regression",
                        "recommended_head": "physiology_persistence",
                        "available_heads": {
                            "physiology_persistence": {"head_family": "physiology_persistence"}
                        },
                        "cross_validated_best_metrics": {
                            "rmse": 10.0,
                            "mae": 5.0,
                        },
                    },
                    "T3_paired_pilot_window_retrieval": {
                        "status": "exported",
                        "task_type": "retrieval",
                        "head_family": "chronaris_time_residual_retrieval",
                        "feature_columns": ["feat__ctx__window_index"],
                        "cross_validated_metrics": {
                            "top1_accuracy": 1.0,
                            "mrr": 1.0,
                        },
                    },
                },
                "diagnostics": {
                    "mean_attention_entropy": 0.93,
                    "mean_top_event_concentration": 0.88,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return package_path


def _write_fake_private_benchmark_summary(root: Path) -> Path:
    summary_path = root / "private_benchmark_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "target_variant_name": "chronaris_opt",
                "tasks": {
                    "T1_maneuver_intensity_class": {
                        "task_type": "classification",
                        "variants": {
                            "chronaris_opt": {
                                "best_metrics": {
                                    "macro_f1": 1.0,
                                    "balanced_accuracy": 1.0,
                                }
                            },
                            "chronaris_opt_no_causal_mask": {
                                "best_metrics": {
                                    "macro_f1": 0.2,
                                    "balanced_accuracy": 0.3,
                                }
                            },
                        },
                    },
                    "T2_next_window_physiology_response": {
                        "task_type": "regression",
                        "variants": {
                            "chronaris_opt": {
                                "best_metrics": {
                                    "rmse": 10.0,
                                    "mae": 5.0,
                                }
                            },
                            "chronaris_opt_no_causal_mask": {
                                "best_metrics": {
                                    "rmse": 20.0,
                                    "mae": 8.0,
                                }
                            },
                        },
                    },
                    "T3_paired_pilot_window_retrieval": {
                        "task_type": "retrieval",
                        "variants": {
                            "chronaris_opt": {
                                "top1_accuracy": 1.0,
                                "mrr": 1.0,
                            },
                            "chronaris_opt_no_causal_mask": {
                                "top1_accuracy": 0.1,
                                "mrr": 0.2,
                            },
                        },
                    },
                },
                "conclusion": {
                    "target_variant_name": "chronaris_opt",
                    "no_mask_variant_name": "chronaris_opt_no_causal_mask",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path
