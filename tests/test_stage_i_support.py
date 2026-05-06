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

from chronaris.pipelines import StageISupportConfig, run_stage_i_support  # noqa: E402


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
            self.assertIn("固定六路径主矩阵", ablation_report)
            self.assertIn("G(no causal mask)", ablation_report)


def _write_support_sources(temp_root: Path) -> dict[str, Path]:
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
