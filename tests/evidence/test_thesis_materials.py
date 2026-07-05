"""Tests for task evaluation thesis materials data contracts."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evidence.thesis_materials_data import (  # noqa: E402
    build_llm_comparison_rows,
    build_private_component_rows,
    build_public_transfer_rows,
    build_rigid_body_rotation_rows,
    build_runtime_payload_schema_rows,
    build_runtime_service_flow_rows,
    build_runtime_semantic_case_rows,
    build_thesis_table_rows,
)
from chronaris.evidence.thesis_materials import StageIThesisMaterialsConfig  # noqa: E402


class StageIThesisMaterialsDataTest(unittest.TestCase):
    def test_default_sources_point_to_current_leakage_safe_materials(self) -> None:
        config = StageIThesisMaterialsConfig(run_id="test")

        self.assertIn("2026-06-19_dingxin-leakage-safe-ablation", config.private_component_summary_path)
        self.assertTrue(config.private_component_summary_path.endswith("ablation_summary.json"))
        self.assertIn("2026-06-19_rotation-audit-figure-refresh", config.rotation_audit_summary_path)

    def test_runtime_payload_schema_distinguishes_native_and_canonical(self) -> None:
        rows = build_runtime_payload_schema_rows(_sources())

        self.assertEqual(len(rows), 2)
        native, canonical = rows
        self.assertEqual(native["payload_name"], "native replay payload")
        self.assertEqual(native["schema_status"], "aligned")
        self.assertEqual(native["physiology_feature_count"], 12)
        self.assertEqual(native["vehicle_feature_count"], 965)
        self.assertEqual(native["missing_vehicle_feature_count"], 965)
        self.assertEqual(canonical["payload_name"], "canonical service payload")
        self.assertEqual(canonical["schema_status"], "exact")
        self.assertEqual(canonical["vehicle_feature_count"], 1930)
        self.assertEqual(canonical["missing_vehicle_feature_count"], 0)

    def test_runtime_service_flow_rows_describe_four_report_steps(self) -> None:
        rows = build_runtime_service_flow_rows(_sources())

        self.assertEqual(
            [row["step_title_cn"] for row in rows],
            ["模型参数加载", "原始回放窗口输入", "运行字段规范检查", "批量推理与结果归档"],
        )
        self.assertIn("已读取飞机状态字段965", rows[1]["detail_cn"])
        self.assertIn("1930维", rows[2]["detail_cn"])
        self.assertIn("任务输出", rows[3]["output_items_cn"])

    def test_rotation_rows_use_availability_matrix_for_missing_rates(self) -> None:
        rows = build_rigid_body_rotation_rows(_sources())
        matrix_rows = [row for row in rows if row["row_type"] == "rotation_field_matrix"]

        self.assertEqual(len(matrix_rows), 6)
        angle_rows = [row for row in matrix_rows if row["field_type"] == "angle"]
        rate_rows = [row for row in matrix_rows if row["field_type"] == "rate"]
        self.assertTrue(all(row["available"] for row in angle_rows))
        self.assertFalse(any(row["available"] for row in rate_rows))
        self.assertTrue(all(row["rotation_status"] == "disabled" for row in matrix_rows))
        self.assertTrue(all("角速度字段" in row["rotation_disabled_reason_cn"] for row in matrix_rows))

    def test_runtime_semantic_case_rows_copy_case_values(self) -> None:
        rows = build_runtime_semantic_case_rows(_sources())

        self.assertEqual(len(rows), 3)
        self.assertEqual([row["window_label"] for row in rows], ["0000", "0001", "0002"])
        self.assertEqual([row["window_label_cn"] for row in rows], ["窗口1", "窗口2", "窗口3"])
        self.assertEqual(rows[0]["view_id"], "view-runtime")
        self.assertEqual(rows[0]["native_feature_schema_status"], "aligned")
        self.assertEqual(rows[0]["canonical_feature_schema_status"], "exact")
        self.assertEqual(rows[0]["input_vehicle_feature_count"], 965)
        self.assertEqual(rows[0]["expected_vehicle_feature_count"], 1930)
        self.assertEqual(rows[0]["native_missing_measurement_group_count"], 6)
        self.assertIn("runtime_case.csv", rows[0]["source_path"])

    def test_private_component_rows_keep_task_direction_and_normalized_delta(self) -> None:
        rows = build_private_component_rows(_sources())

        by_metric = {row["primary_metric_name"]: row["direction"] for row in rows}
        self.assertEqual(by_metric["macro_f1"], "higher_is_better")
        self.assertEqual(by_metric["rmse"], "lower_is_better")
        self.assertTrue(all(0.0 <= float(row["normalized_delta_vs_full"]) <= 1.0 for row in rows))
        self.assertEqual({row["task_name_cn"] for row in rows}, {"风险预测", "工作负荷预测"})
        self.assertTrue(all(row["report_protocol_cn"] == "严格评价协议" for row in rows))
        self.assertIn("source_path", rows[0])

    def test_llm_and_transfer_rows_preserve_boundary_language(self) -> None:
        llm_rows = build_llm_comparison_rows(_sources())
        transfer_rows = build_public_transfer_rows(_sources())

        a4 = next(row for row in llm_rows if row["condition"] == "A4_human_review_packet")
        self.assertIn("待复核", a4["boundary_cn"])
        self.assertIn("human_review_completed=False", a4["metric_note"])
        self.assertEqual([row["segment_title_cn"] for row in transfer_rows], ["公开适配", "鼎新弱监督主线", "鼎新组件诊断"])
        self.assertTrue(all("不能" not in row["positive_reading_cn"] for row in transfer_rows))

    def test_table_contract_includes_required_outputs(self) -> None:
        tables = build_thesis_table_rows(_sources())

        self.assertEqual(
            set(tables),
            {
                "evidence_layer_overview.csv",
                "runtime_payload_schema.csv",
                "runtime_service_flow.csv",
                "runtime_semantic_case.csv",
                "rigid_body_rotation_audit.csv",
                "weak_label_sweep_ablation.csv",
                "chronaris_opt_component_ablation.csv",
                "model_backbone_ablation.csv",
                "task_adapter_ablation.csv",
                "public_transfer_boundary.csv",
                "semantic_event_fusion_overview.csv",
                "llm_comparison_a0_a4.csv",
            },
        )


def _sources() -> dict[str, dict[str, object]]:
    return {
        "runtime_service": {
            "path": "runtime_service.json",
            "payload": {
                "run_id": "runtime-r2",
                "input_sample_count": 12,
                "native_feature_schema_status": "aligned",
                "canonical_feature_schema_status": "exact",
                "input_vehicle_feature_count": 965,
                "canonical_vehicle_feature_count": 1930,
                "missing_vehicle_feature_count": 965,
                "missing_vehicle_measurement_group_counts": {"BUS2": 221},
            },
        },
        "runtime_schema_contract": {
            "path": "runtime_schema_contract.json",
            "payload": {
                "schema_source": "input_normalization_stats",
                "schema_hash": "abc",
                "native_input": {
                    "sample_count": 12,
                    "physiology": {"feature_count": 12},
                    "vehicle": {"feature_count": 965},
                },
                "canonical_payload": {
                    "sample_count": 12,
                    "physiology": {"feature_count": 12},
                    "vehicle": {"feature_count": 1930},
                },
                "expected_schema": {
                    "physiology": {"feature_count": 12},
                    "vehicle": {"feature_count": 1930},
                },
            },
        },
        "rigid_body": {
            "path": "rigid.json",
            "payload": {
                "run_id": "rigid-r2",
                "families": {
                    "minimal": {"test_total": 10.0, "test_alignment": 0.1, "test_physics_total": 100.0},
                    "full": {"test_total": 2.0, "test_alignment": 0.2, "test_physics_total": 3.0},
                    "rigid_body": {
                        "test_total": 3.0,
                        "test_alignment": 0.3,
                        "test_physics_total": 4.0,
                        "rigid_body_mapping_diagnostics": {"enabled_residuals": ["translation", "vertical"]},
                    },
                },
            },
        },
        "rotation_audit": {
            "path": "rotation.json",
            "payload": {
                "run_id": "rotation-r3",
                "evidence_layer": "rotation_diagnostics",
                "rotation_status": "disabled",
                "rotation_reading": "current sortie still lacks paired rate fields for pitch/roll/yaw",
                "feature_rotation_candidates": {"pitch": [1], "pitch_rate": [], "roll": [1], "roll_rate": [], "yaw": [1], "yaw_rate": []},
                "mysql_rotation_candidates": {"pitch": [1], "pitch_rate": [], "roll": [1], "roll_rate": [], "yaw": [1], "yaw_rate": []},
            },
        },
        "private_component": {
            "path": "private.json",
            "payload": {
                "run_id": "private-r2",
                "rows": [
                    {"task_name": "T1", "variant_name": "chronaris_opt", "component": "full_candidate", "primary_metric_name": "macro_f1", "primary_metric_value": 1.0, "delta_vs_full": 0.0},
                    {"task_name": "T1", "variant_name": "chronaris_opt_no_causal_mask", "component": "remove_causal_mask", "primary_metric_name": "macro_f1", "primary_metric_value": 0.5, "delta_vs_full": 0.5},
                    {"task_name": "T2", "variant_name": "chronaris_opt", "component": "full_candidate", "primary_metric_name": "rmse", "primary_metric_value": 10.0, "delta_vs_full": 0.0},
                    {"task_name": "T2", "variant_name": "chronaris_opt_no_task_head", "component": "remove_task_head", "primary_metric_name": "rmse", "primary_metric_value": 20.0, "delta_vs_full": 10.0},
                ],
                "tasks": {"T1": {}, "T2": {}},
                "variant_order": ["chronaris_opt", "chronaris_opt_no_causal_mask", "chronaris_opt_no_task_head"],
            },
        },
        "proxy_sweep": {
            "path": "proxy.json",
            "payload": _sweep_payload("proxy-run", "feature_export_window_stats_proxy", 100.0),
        },
        "live_sweep": {
            "path": "live.json",
            "payload": _sweep_payload("live-run", "live_influx", 110.0),
        },
        "live_partial": {
            "path": "partial.json",
            "payload": {
                **_sweep_payload("partial-run", "live_influx", 110.0),
                "status": "partial_blocked",
                "target_combination_count": 4,
                "combination_count_completed": 2,
                "blocked_at_run_index": 3,
                "blocked_attempt_log_paths": ["blocked.log"],
            },
        },
        "public_calibration": {
            "path": "public_calibration.json",
            "payload": {"run_id": "public-r2", "rows": [{}, {}], "best_by_category": {"public": {}}},
        },
        "public_transfer": {
            "path": "public_transfer.json",
            "payload": {"run_id": "transfer-r2", "private_mainline": {"thesis_boundary": "weak label"}},
        },
        "support": {
            "path": "support.json",
            "payload": {
                "run_id": "support-r2",
                "causal_support": {"semantic_event": _semantic_payload()},
            },
        },
        "semantic_event": {
            "path": "semantic.json",
            "payload": _semantic_payload(),
        },
        "runtime": {
            "path": "runtime.json",
            "payload": {"run_id": "runtime-replay", "sample_count": 40},
        },
        "runtime_case_table": {
            "path": "runtime_case.csv",
            "payload": {
                "rows": [
                    _runtime_case_row("0000", "workload_proxy", 4.3, 0.386, 0.195),
                    _runtime_case_row("0001", "risk_proxy", 4.6, 0.392, 0.222),
                    _runtime_case_row("0002", "risk_proxy", 8.36, 0.393, 0.222),
                ]
            },
        },
        "evidence_manifest": {
            "path": "evidence.json",
            "payload": {"run_id": "evidence-r2", "tasks": {}},
        },
        "llm_preprocessing": {
            "path": "llm_pre.json",
            "payload": {"run_id": "p20", "request_count": 8, "error_count": 0, "semantic_query_hint_count": 4, "runtime_explanation_count": 4},
        },
        "llm_comparison": {
            "path": "llm_cmp.json",
            "payload": {
                "run_id": "p21",
                "task_context": {"attached_entry_count": 333, "label_unchanged": True, "label_changed_count": 0},
                "semantic_hints": {"baseline_query_count": 3, "combined_query_count": 7, "added_query_count": 4},
                "runtime_explanations": {"llm_explained_case_count": 4, "runtime_case_count": 12, "with_llm_average_completeness_for_explained_cases": 1.0},
                "human_review_packet": {"item_count": 15, "human_review_completed": False},
            },
        },
    }


def _runtime_case_row(
    window_label: str,
    query_name: str,
    attribution: float,
    risk_confidence: float,
    workload_prediction: float,
) -> dict[str, object]:
    return {
        "sample_id": f"view-runtime::sortie:{window_label}",
        "view_id": "view-runtime",
        "semantic_top_query_name": query_name,
        "semantic_top_event_attribution": attribution,
        "semantic_top_query_event_offset_s": 0.33,
        "top_contribution_score": attribution / 2,
        "risk_proxy_prediction": "medium",
        "risk_proxy_confidence": risk_confidence,
        "workload_proxy_prediction": workload_prediction,
        "event_replay_tag_score": 1.0,
        "native_feature_schema_status": "aligned",
        "canonical_feature_schema_status": "exact",
        "expected_vehicle_feature_count": 1930,
        "input_vehicle_feature_count": 965,
        "missing_vehicle_feature_count": 965,
        "native_missing_measurement_group_count": 6,
        "schema_hash": "abc",
        "support_source_path": "support.json",
        "runtime_service_source_path": "runtime_service.json",
        "runtime_schema_contract_source_path": "runtime_schema_contract.json",
        "evidence_layer": "runtime_semantic_support",
        "case_definition": "fixture runtime case",
    }


def _sweep_payload(run_id: str, sample_source: str, test_total: float) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "completed",
        "evidence_layer": "thesis_weak_label",
        "sample_count": 111,
        "task_entry_count": 333,
        "combination_count": 2,
        "source_summary": {"sample_collection": {"sample_source": sample_source}},
        "best_run": {"child_run_id": f"{run_id}-01", "test_total": test_total},
        "rows": [
            {
                "child_run_id": f"{run_id}-01",
                "physics_constraint_family": "minimal",
                "causal_weight": 0.0,
                "task_loss_weight": 0.5,
                "causal_lag_window_points": None,
                "test_total": test_total,
                "test_task_total": 2.5,
                "test_causal_total": 0.9,
            }
        ],
    }


def _semantic_payload() -> dict[str, object]:
    return {
        "run_id": "semantic-r2",
        "query_count": 3,
        "query_names": ["risk_proxy", "workload_proxy", "event_replay_tag"],
        "view_count": 1,
        "top_view_id": "view-1",
        "view_rows": [
            {
                "view_id": "view-1",
                "sortie_id": "sortie-1",
                "pilot_id": 10033,
                "sample_count": 37,
                "dominant_query_name": "risk_proxy",
                "mean_event_token_count": 1.0,
                "mean_top_event_attribution": 7.5,
                "top_sample_id": "sample-1",
                "top_sample_query_name": "risk_proxy",
                "top_sample_event_attribution": 8.0,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
