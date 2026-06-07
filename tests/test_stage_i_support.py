"""Stage I support aggregation tests."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines import (  # noqa: E402
    StageIMidtermEvidenceConfig,
    StageISupportConfig,
    run_stage_i_midterm_evidence,
    run_stage_i_support,
)


class StageISupportTest(unittest.TestCase):
    def test_run_stage_i_support_writes_summary_matrix_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sources = _write_support_sources(temp_root)
            result = run_stage_i_support(
                StageISupportConfig(
                    run_id="stage-i-support-test",
                    artifact_root=str(temp_root / "artifacts"),
                    report_root=str(temp_root / "reports"),
                    alignment_e_summary_path=str(sources["alignment_e"]),
                    alignment_f_summary_path=str(sources["alignment_f"]),
                    causal_g_summary_path=str(sources["causal_g"]),
                    stage_h_run_manifest_path=str(sources["stage_h"]),
                    case_study_summary_path=str(sources["case_study_summary"]),
                    case_study_ablation_csv_path=str(sources["case_study_ablation"]),
                    private_benchmark_summary_path=str(sources["private_summary"]),
                    deep_comparison_summary_path=str(sources["deep_summary"]),
                )
            )

            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.matrix_path).exists())
            self.assertTrue(Path(result.main_matrix_path).exists())
            self.assertTrue(Path(result.alignment_report_path).exists())
            self.assertTrue(Path(result.causal_report_path).exists())
            self.assertTrue(Path(result.ablation_report_path).exists())
            if result.overview_plot_path is not None:
                self.assertTrue(Path(result.overview_plot_path).exists())

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            alignment_delta = summary["alignment_support"]["alignment_chain"]["delta_f_minus_e"]
            self.assertAlmostEqual(alignment_delta["mean_projection_cosine"], 0.05, places=6)
            self.assertEqual(
                summary["causal_support"]["case_study"]["strongest_ablation"]["name"],
                "vehicle_delta_suppressed",
            )
            self.assertTrue(
                summary["causal_support"]["private_no_mask"]["tasks"][
                    "T1_maneuver_intensity_class"
                ]["target_beats_no_mask"]
            )

            matrix = pd.read_csv(result.matrix_path)
            self.assertIn("stage_h_export", set(matrix["variant"]))
            self.assertIn("vehicle_delta_suppressed", set(matrix["variant"]))
            self.assertIn("chronaris_opt_no_causal_mask", set(matrix["variant"]))

            main_matrix = pd.read_csv(result.main_matrix_path)
            self.assertEqual(
                list(main_matrix["variant"]),
                [
                    "e_baseline",
                    "f_full",
                    "g_min",
                    "g_no_causal_mask",
                    "vehicle_delta_suppressed",
                    "no_event_bias",
                ],
            )
            self.assertNotIn("no_state_normalization", set(main_matrix["variant"]))

            alignment_report = Path(result.alignment_report_path).read_text(encoding="utf-8")
            causal_report = Path(result.causal_report_path).read_text(encoding="utf-8")
            ablation_report = Path(result.ablation_report_path).read_text(encoding="utf-8")
            self.assertIn("Stage H Export Stability", alignment_report)
            self.assertIn("本结论能支撑什么", alignment_report)
            self.assertIn("vehicle_delta_suppressed", causal_report)
            self.assertIn("Private No-Mask Comparison", causal_report)
            self.assertIn("Semantic Event Fusion", causal_report)
            self.assertIn("事件级归因", causal_report)
            self.assertIn("固定六路径主矩阵", ablation_report)
            self.assertIn("G(no causal mask)", ablation_report)


class StageIMidtermEvidenceTest(unittest.TestCase):
    def test_run_stage_i_midterm_evidence_writes_manifest_metrics_figures_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sources = _write_midterm_sources(temp_root)
            result = run_stage_i_midterm_evidence(
                StageIMidtermEvidenceConfig(
                    run_id="stage-i-midterm-test",
                    artifact_root=str(temp_root / "midterm_assets"),
                    report_root=str(sources["report_root"]),
                    workspace_root=str(sources["workspace_root"]),
                    private_summary_path=str(sources["private_midterm_summary"]),
                    support_summary_path=str(sources["support_summary"]),
                    public_mainline_summary_path=str(sources["public_mainline_summary"]),
                    anchor_manifest_path=str(sources["anchor_manifest"]),
                    deep_comparison_summary_path=str(sources["deep_comparison_summary"]),
                    public_fusion_screen_summary_path=str(sources["public_fusion_screen_summary"]),
                    nasa_public_opt_summary_path=str(sources["nasa_public_opt_summary"]),
                    uab_public_opt_summary_path=str(sources["uab_public_opt_summary"]),
                    uab_fairness_summary_path=str(sources["uab_fairness_summary"]),
                    nasa_fusion_confirm_summary_path=str(sources["nasa_fusion_confirm_summary"]),
                )
            )

            artifact_root = Path(result.artifact_root)
            self.assertTrue(Path(result.manifest_path).exists())
            self.assertTrue(Path(result.metrics_path).exists())
            self.assertTrue(Path(result.figure_index_path).exists())
            self.assertTrue(Path(result.cleanup_audit_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertTrue((artifact_root / "run.log").exists())
            self.assertTrue((artifact_root / "progress.json").exists())
            for plot_name in (
                "thesis_chain_status.png",
                "private_opt_task_metrics.png",
                "public_mainline_metrics.png",
                "support_ablation_metrics.png",
                "anchor_window_scores.png",
                "runtime_progress_timeline.png",
            ):
                self.assertTrue((artifact_root / "plots" / plot_name).exists())

            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"]["public_mainline_status"], "public opt closed")
            self.assertFalse(manifest["status"]["nasa_fusion_gate_passed"])
            self.assertEqual(manifest["counts"]["figure_count"], 6)

            metrics = pd.read_csv(result.metrics_path)
            self.assertIn("fairness", set(metrics["section"]))
            self.assertIn("public", set(metrics["section"]))

            cleanup_audit = json.loads(Path(result.cleanup_audit_path).read_text(encoding="utf-8"))
            self.assertEqual(cleanup_audit["summary"]["outdated_doc_candidate_count"], 2)
            self.assertEqual(cleanup_audit["summary"]["unreferenced_code_candidate_count"], 1)
            report_candidates = cleanup_audit["outdated_report_candidates"]
            self.assertEqual(
                report_candidates[0]["reference_hits"][0]["path"],
                "docs/README.md",
            )

            report_text = Path(result.report_path).read_text(encoding="utf-8")
            self.assertIn("negative evidence", report_text)
            self.assertIn("fonts-wqy-zenhei", report_text)


def _write_support_sources(temp_root: Path) -> dict[str, Path]:
    temp_root.mkdir(parents=True, exist_ok=True)
    alignment_e = temp_root / "alignment_e.json"
    alignment_f = temp_root / "alignment_f.json"
    causal_g = temp_root / "causal_g.json"
    stage_h = temp_root / "stage_h_run_manifest.json"
    case_study_summary = temp_root / "case_study_summary.json"
    case_study_ablation = temp_root / "ablation_summary.csv"
    private_summary = temp_root / "private_benchmark_summary.json"
    deep_summary = temp_root / "deep_comparison_summary.json"

    alignment_e.write_text(
        json.dumps(
            {
                "summary": {
                    "sample_count": 3,
                    "mean_projection_cosine": 0.70,
                    "mean_projection_l2_gap": 0.21,
                },
                "threshold_evaluation": {"verdict": "PASS"},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    alignment_f.write_text(
        json.dumps(
            {
                "summary": {
                    "sample_count": 3,
                    "mean_projection_cosine": 0.75,
                    "mean_projection_l2_gap": 0.19,
                },
                "threshold_evaluation": {"verdict": "PASS"},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    causal_g.write_text(
        json.dumps(
            {
                "sample_count": 3,
                "mean_attention_entropy": 0.93,
                "mean_max_attention": 0.22,
                "mean_top_event_score": 1.0,
                "mean_top_contribution_score": 2.6,
                "semantic_event": {
                    "query_names": ["risk_proxy", "workload_proxy", "event_replay_tag"],
                    "query_count": 3,
                    "mean_event_token_count": 2.0,
                    "mean_query_entropy": 0.31,
                    "mean_top_query_score": 0.62,
                    "mean_top_event_attribution": 0.88,
                    "samples": [
                        {
                            "sample_id": "sample-1",
                            "event_token_count": 2,
                            "top_query_name": "risk_proxy",
                            "top_query_score": 0.62,
                            "top_query_event_offset_s": 2.0,
                            "top_event_attribution": 0.88,
                        }
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    stage_h.write_text(
        json.dumps(
            {
                "run_id": "stage-h-test",
                "sortie_ids": ["sortie-1", "sortie-2"],
                "generated_view_count": 3,
                "generated_view_ids": ["view-1", "view-2", "view-3"],
                "partial_data": {
                    "entry_count": 1,
                    "built_entry_count": 1,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    case_study_summary.write_text(
        json.dumps(
            {
                "view_results": [
                    {"view_summary": {"verdict": "PASS"}},
                    {"view_summary": {"verdict": "PASS"}},
                    {"view_summary": {"verdict": "WARN"}},
                ],
                "pilot_comparisons": [
                    {
                        "sortie_id": "sortie-2",
                        "delta_mean_projection_cosine": -0.17,
                        "delta_projection_cosine_cv": 0.19,
                        "delta_mean_top_contribution_score": -0.21,
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "ablation_name": "no_event_bias",
                "delta_mean_attention_entropy": 0.002,
                "delta_mean_top_event_score": 0.0,
                "delta_mean_top_contribution_score": -0.20,
                "delta_fused_l2_norm": -0.002,
                "delta_fused_cosine_to_projection_baseline": -0.0001,
            },
            {
                "ablation_name": "vehicle_delta_suppressed",
                "delta_mean_attention_entropy": 0.01,
                "delta_mean_top_event_score": -1.0,
                "delta_mean_top_contribution_score": -2.4,
                "delta_fused_l2_norm": 0.40,
                "delta_fused_cosine_to_projection_baseline": -0.20,
            },
        ]
    ).to_csv(case_study_ablation, index=False)
    private_summary.write_text(
        json.dumps(
            {
                "conclusion": {
                    "target_variant_name": "chronaris_opt",
                    "no_mask_variant_name": "chronaris_opt_no_causal_mask",
                    "criterion_details": {
                        "t1_chronaris_opt_beats_chronaris_opt_no_causal_mask": True
                    },
                },
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
                                    "macro_f1": 0.17,
                                    "balanced_accuracy": 0.33,
                                }
                            },
                        },
                    },
                    "T2_next_window_physiology_response": {
                        "task_type": "regression",
                        "variants": {
                            "chronaris_opt": {
                                "best_metrics": {
                                    "rmse": 201.4,
                                    "mae": 113.8,
                                }
                            },
                            "chronaris_opt_no_causal_mask": {
                                "best_metrics": {
                                    "rmse": 313.2,
                                    "mae": 173.6,
                                }
                            },
                        },
                    },
                    "T3_paired_pilot_window_retrieval": {
                        "task_type": "retrieval",
                        "variants": {
                            "chronaris_opt": {
                                "best_metrics": {
                                    "top1_accuracy": 1.0,
                                    "mrr": 1.0,
                                }
                            },
                            "chronaris_opt_no_causal_mask": {
                                "best_metrics": {
                                    "top1_accuracy": 0.02,
                                    "mrr": 0.11,
                                }
                            },
                        },
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    deep_summary.write_text(
        json.dumps(
            {
                "datasets": {
                    "stage_h_case": {
                        "status": "completed",
                        "models": {
                            "mult": {
                                "summary": {
                                    "view_metrics": [
                                        {
                                            "event_mask_interference": 0.12,
                                            "mean_attention_entropy": 2.75,
                                        },
                                        {
                                            "event_mask_interference": 0.10,
                                            "mean_attention_entropy": 2.76,
                                        },
                                    ],
                                    "pilot_metrics": [
                                        {
                                            "delta_event_mask_interference": -0.09,
                                        }
                                    ],
                                }
                            },
                            "contiformer": {
                                "summary": {
                                    "view_metrics": [
                                        {
                                            "event_mask_interference": 0.06,
                                            "mean_attention_entropy": 2.74,
                                        }
                                    ],
                                    "pilot_metrics": [
                                        {
                                            "delta_event_mask_interference": -0.02,
                                        }
                                    ],
                                }
                            },
                        },
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "alignment_e": alignment_e,
        "alignment_f": alignment_f,
        "causal_g": causal_g,
        "stage_h": stage_h,
        "case_study_summary": case_study_summary,
        "case_study_ablation": case_study_ablation,
        "private_summary": private_summary,
        "deep_summary": deep_summary,
    }


def _write_midterm_sources(temp_root: Path) -> dict[str, Path]:
    workspace_root = temp_root / "workspace"
    report_root = workspace_root / "docs" / "reports" / "stage_i"
    report_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "AGENTS.md").write_text("midterm fixture\n", encoding="utf-8")
    (workspace_root / "docs").mkdir(exist_ok=True)
    (workspace_root / "docs" / "README.md").write_text(
        "stage-i-public-mainline-history-fixture.md\n",
        encoding="utf-8",
    )
    (report_root / "README.md").write_text("stage_i report index\n", encoding="utf-8")
    (report_root / "stage-i-closure-2026-04-30.md").write_text("keep\n", encoding="utf-8")
    (report_root / "stage-i-public-mainline-history-fixture.md").write_text(
        "old mainline\n",
        encoding="utf-8",
    )
    (report_root / "stage-i-public-opt-history-fixture.md").write_text(
        "old public opt\n",
        encoding="utf-8",
    )
    unused_module_root = workspace_root / "src" / "chronaris" / "pipelines" / "stage_i"
    unused_module_root.mkdir(parents=True, exist_ok=True)
    (unused_module_root / "__init__.py").write_text("", encoding="utf-8")
    (unused_module_root / "stage_i_unused_candidate.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    for relative in ("scripts", "tests"):
        (workspace_root / relative).mkdir(parents=True, exist_ok=True)

    support_sources = _write_support_sources(temp_root / "support_inputs")
    support_result = run_stage_i_support(
        StageISupportConfig(
            run_id="support-summary",
            artifact_root=str(temp_root / "support_assets"),
            report_root=str(temp_root / "support_reports"),
            alignment_e_summary_path=str(support_sources["alignment_e"]),
            alignment_f_summary_path=str(support_sources["alignment_f"]),
            causal_g_summary_path=str(support_sources["causal_g"]),
            stage_h_run_manifest_path=str(support_sources["stage_h"]),
            case_study_summary_path=str(support_sources["case_study_summary"]),
            case_study_ablation_csv_path=str(support_sources["case_study_ablation"]),
            private_benchmark_summary_path=str(support_sources["private_summary"]),
            deep_comparison_summary_path=str(support_sources["deep_summary"]),
        )
    )

    private_midterm_summary = temp_root / "private_midterm_summary.json"
    private_midterm_summary.write_text(
        json.dumps(
            {
                "run_id": "20260504T120000Z-stage-i-private-opt-package",
                "target_variant_name": "chronaris_opt",
                "private_optimality_supported": True,
                "conclusion": {
                    "no_mask_variant_name": "chronaris_opt_no_causal_mask",
                },
                "tasks": {
                    "T1_maneuver_intensity_class": {
                        "variants": {
                            "chronaris_opt": {"best_metrics": {"macro_f1": 0.91}},
                            "chronaris_opt_no_causal_mask": {"best_metrics": {"macro_f1": 0.28}},
                            "naive_sync": {"best_metrics": {"macro_f1": 0.77}},
                        }
                    },
                    "T2_next_window_physiology_response": {
                        "variants": {
                            "chronaris_opt": {"best_metrics": {"rmse": 201.4}},
                            "chronaris_opt_no_causal_mask": {"best_metrics": {"rmse": 313.2}},
                            "naive_sync": {"best_metrics": {"rmse": 77849.5}},
                        }
                    },
                    "T3_paired_pilot_window_retrieval": {
                        "variants": {
                            "chronaris_opt": {"top1_accuracy": 1.0},
                            "chronaris_opt_no_causal_mask": {"top1_accuracy": 0.02},
                            "naive_sync": {"top1_accuracy": 0.33},
                        }
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    public_mainline_summary = temp_root / "public_mainline_summary.json"
    public_mainline_summary.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-05-08T13:01:00Z",
                "public_mainline_status": "public opt closed",
                "uab": {
                    "groups": {
                        "n_back": {
                            "best_public_head": "ridge_residual_cv",
                            "public_rmse": 4.60,
                            "rmse_margin_vs_best_deep": 0.05,
                            "best_deep_model": "contiformer",
                        },
                        "heat_the_chair": {
                            "best_public_head": "target_prior_median",
                            "public_rmse": 1.43,
                            "rmse_margin_vs_best_deep": 0.02,
                            "best_deep_model": "contiformer",
                        },
                    }
                },
                "nasa": {
                    "combined_best_head": "balanced_logistic_context",
                    "combined_macro_f1": 0.455,
                    "combined_balanced_accuracy": 0.559,
                },
                "public_fusion": {},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    anchor_manifest = temp_root / "anchor_manifest.json"
    anchor_manifest.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-05-06T16:54:35Z",
                "overview": {
                    "selected_view_count": 3,
                    "selected_anchor_count": 2,
                },
                "anchors": [
                    {
                        "anchor_rank": 1,
                        "anchor_score": 3.31,
                        "view_verdict": "WARN",
                    },
                    {
                        "anchor_rank": 2,
                        "anchor_score": 3.12,
                        "view_verdict": "PASS",
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    deep_comparison_summary = temp_root / "deep_comparison_summary.json"
    deep_comparison_summary.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-05-01T04:33:48Z",
                "datasets": {
                    "uab_workload_dataset": {
                        "models": {
                            "contiformer": {
                                "summary": {
                                    "subjective": {
                                        "groups": {
                                            "n_back": {"rmse": 4.6541},
                                            "heat_the_chair": {"rmse": 1.4568},
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    public_fusion_screen_summary = temp_root / "public_fusion_screen_summary.json"
    public_fusion_screen_summary.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-05-06T05:07:45Z",
                "runtime_device": "cuda",
                "per_dataset_rankings": {
                    "nasa_csm": [
                        {
                            "candidate_id": "fusion_round2",
                            "selection_score": 0.3558,
                            "secondary_score": 0.5163,
                        }
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    nasa_public_opt_summary = temp_root / "nasa_public_opt_summary.json"
    nasa_public_opt_summary.write_text(
        json.dumps({"track": "objective"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    uab_public_opt_summary = temp_root / "uab_public_opt_summary.json"
    uab_public_opt_summary.write_text(
        json.dumps({"track": "subjective"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    uab_fairness_summary = temp_root / "uab_fairness_summary.json"
    uab_fairness_summary.write_text(
        json.dumps(
            {
                "run_id": "20260509T010000Z-stage-i-uab-fairness",
                "subjective": {
                    "groups": {
                        "n_back": {"rmse": 4.7000},
                        "heat_the_chair": {"rmse": 1.4600},
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    nasa_fusion_confirm_summary = temp_root / "nasa_fusion_confirm_summary.json"
    nasa_fusion_confirm_summary.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-05-09T02:00:00Z",
                "runtime_device": "cuda",
                "per_dataset_rankings": {
                    "nasa_csm": [
                        {
                            "candidate_id": "fusion_balanced_confirm",
                            "selection_score": 0.3900,
                            "secondary_score": 0.5200,
                        }
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "workspace_root": workspace_root,
        "report_root": report_root,
        "support_summary": Path(support_result.summary_path),
        "private_midterm_summary": private_midterm_summary,
        "public_mainline_summary": public_mainline_summary,
        "anchor_manifest": anchor_manifest,
        "deep_comparison_summary": deep_comparison_summary,
        "public_fusion_screen_summary": public_fusion_screen_summary,
        "nasa_public_opt_summary": nasa_public_opt_summary,
        "uab_public_opt_summary": uab_public_opt_summary,
        "uab_fairness_summary": uab_fairness_summary,
        "nasa_fusion_confirm_summary": nasa_fusion_confirm_summary,
    }
