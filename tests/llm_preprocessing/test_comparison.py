"""Tests for the P21 task evaluation LLM preprocessing comparison."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
import unittest

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.dataset import StageIPrivateTaskEntry, dump_task_eval_private_task_entries  # noqa: E402
from chronaris.llm_preprocessing.comparison import (  # noqa: E402
    StageILLMComparisonConfig,
    run_task_eval_llm_comparison,
)


class StageILLMComparisonTest(unittest.TestCase):
    def test_comparison_writes_required_outputs_and_preserves_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = _write_sources(root)
            result = run_task_eval_llm_comparison(
                StageILLMComparisonConfig(
                    run_id="task-eval-llm-comparison-test",
                    artifact_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    midterm_root=str(root / "midterm"),
                    llm_context_path=str(sources["context"]),
                    task_manifest_path=str(sources["task_manifest"]),
                    support_summary_path=str(sources["support_summary"]),
                    runtime_schema_contract_path=str(sources["runtime_schema_contract"]),
                    runtime_case_table_path=str(sources["runtime_case_table"]),
                    max_human_review_fields=2,
                    max_human_review_schema_gaps=2,
                )
            )

            self.assertEqual(result.status, "success")
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertTrue(summary["task_context"]["label_unchanged"])
            self.assertEqual(summary["task_context"]["label_unchanged_text"], "true")
            self.assertEqual(summary["task_context"]["attached_entry_count"], 6)
            self.assertEqual(summary["semantic_hints"]["baseline_query_count"], 3)
            self.assertEqual(summary["semantic_hints"]["combined_query_count"], 4)
            self.assertEqual(summary["semantic_hints"]["rejected_hint_count"], 1)
            self.assertFalse(summary["semantic_hints"]["view_ranking_recomputed"])
            self.assertEqual(summary["runtime_explanations"]["runtime_case_count"], 2)
            self.assertEqual(summary["runtime_explanations"]["llm_explained_case_count"], 1)
            self.assertEqual(summary["runtime_explanations"]["complete_with_llm_case_count"], 1)
            self.assertFalse(summary["human_review_packet"]["human_review_completed"])
            self.assertEqual(summary["human_review_packet"]["validation_status"], "pending_human_review")

            for key in (
                "condition_manifest_path",
                "task_context_comparison_path",
                "semantic_hint_comparison_path",
                "runtime_explanation_comparison_path",
                "human_review_packet_path",
                "midterm_claims_payload_path",
                "report_path",
                "midterm_summary_path",
            ):
                self.assertTrue(Path(summary[key]).exists(), key)

            with Path(summary["task_context_comparison_path"]).open(encoding="utf-8") as handle:
                task_rows = list(csv.DictReader(handle))
            self.assertTrue(task_rows)
            self.assertTrue(all(row["label_unchanged"] == "true" for row in task_rows))

            with Path(summary["runtime_explanation_comparison_path"]).open(encoding="utf-8") as handle:
                runtime_rows = list(csv.DictReader(handle))
            explained = [row for row in runtime_rows if row["has_llm_explanation"] == "true"]
            self.assertEqual(len(explained), 1)
            self.assertEqual(explained[0]["weak_label_boundary_with_llm"], "true")

            with Path(summary["human_review_packet_path"]).open(encoding="utf-8") as handle:
                review_rows = list(csv.DictReader(handle))
            self.assertTrue(review_rows)
            self.assertTrue(all(row["validation_status"] == "pending_human_review" for row in review_rows))
            self.assertTrue(all(row["human_decision"] == "" for row in review_rows))


def _write_sources(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    task_manifest = root / "thesis_task_manifest.jsonl"
    dump_task_eval_private_task_entries(_task_entries(), path=task_manifest)

    runtime_explanations = root / "runtime_llm_explanations.jsonl"
    runtime_explanations.write_text(
        json.dumps(
            {
                "sample_id": "view-1::sample-0",
                "model_prediction": "risk_proxy: medium",
                "semantic_attribution": "risk_proxy:0.8",
                "schema_gap_note": "native aligned, canonical exact",
                "weak_label_boundary": "weak_label_proxy_not_manual_ground_truth",
                "needs_human_review": True,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    context = root / "llm_preprocessing_context.json"
    context.write_text(
        json.dumps(
            {
                "run_id": "p20-context-test",
                "schema_version": "task_eval_llm_preprocessing_context.v2",
                "context_path": str(context),
                "field_semantic_dictionary_path": str(root / "field_semantic_dictionary.csv"),
                "field_semantic_dictionary": [
                    {
                        "stream_kind": "vehicle",
                        "measurement_group": "BUS1",
                        "feature_name": "BUS1.code1",
                        "llm_semantic_role": "vehicle_state",
                        "llm_unit_guess": "m/s",
                        "evidence": "field name",
                        "confidence": 0.7,
                        "needs_human_review": True,
                    },
                    {
                        "stream_kind": "physiology",
                        "measurement_group": "eeg",
                        "feature_name": "eeg.af3",
                        "llm_semantic_role": "physiological_load",
                        "llm_unit_guess": None,
                        "evidence": "field name",
                        "confidence": 0.8,
                        "needs_human_review": True,
                    },
                ],
                "weak_label_rule_review": [
                    {
                        "task_name": "risk_proxy",
                        "review_decision": "keep_current_rule",
                        "confidence": 0.7,
                        "agreement_basis": "bounded weak-label rule",
                        "needs_human_review": True,
                        "recommended_action": "audit_before_training",
                    },
                    {
                        "task_name": "workload_proxy",
                        "review_decision": "keep_current_rule",
                        "confidence": 0.7,
                        "agreement_basis": "bounded weak-label rule",
                        "needs_human_review": True,
                        "recommended_action": "audit_before_training",
                    },
                    {
                        "task_name": "event_replay_tag",
                        "review_decision": "keep_current_rule",
                        "confidence": 0.7,
                        "agreement_basis": "bounded weak-label rule",
                        "needs_human_review": True,
                        "recommended_action": "audit_before_training",
                    },
                ],
                "semantic_query_hints": [
                    {"name": "llm_gap", "recipe": "coordination_gap"},
                    {"name": "bad_prompt", "recipe": "free_text_prompt"},
                ],
                "schema_gap_policy": {
                    "status": "draft_policy",
                    "native_exact_claim_allowed": False,
                    "policies": [
                        {
                            "measurement_group": "BUS2",
                            "policy_type": "human_review_required",
                            "reason": "missing native group",
                            "needs_human_review": True,
                        },
                        {
                            "measurement_group": "BUS3",
                            "policy_type": "human_review_required",
                            "reason": "missing native group",
                            "needs_human_review": True,
                        },
                    ],
                },
                "runtime_case_explanations_path": str(runtime_explanations),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    support_summary = root / "support_summary.json"
    support_summary.write_text(
        json.dumps(
            {
                "causal_support": {
                    "semantic_event": {
                        "query_count": 3,
                        "query_names": ["risk_proxy", "workload_proxy", "event_replay_tag"],
                        "view_count": 1,
                        "top_view_id": "view-1",
                        "view_rows": [{"view_id": "view-1", "dominant_query_name": "risk_proxy"}],
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_schema_contract = root / "runtime_schema_contract.json"
    runtime_schema_contract.write_text(
        json.dumps({"native_input": {"status": "aligned"}, "canonical_payload": {"status": "exact"}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    runtime_case_table = root / "runtime_semantic_case.csv"
    runtime_case_table.write_text(
        "\n".join(
            [
                "sample_id,risk_proxy_prediction,workload_proxy_prediction,event_replay_tag_prediction,semantic_top_query_name,semantic_top_event_attribution,native_feature_schema_status,canonical_feature_schema_status,missing_vehicle_feature_count",
                "view-1::sample-0,medium,0.5,view-1::sample-1,risk_proxy,0.8,aligned,exact,5",
                "view-1::sample-1,low,0.4,view-1::sample-0,workload_proxy,0.6,aligned,exact,5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "context": context,
        "task_manifest": task_manifest,
        "support_summary": support_summary,
        "runtime_schema_contract": runtime_schema_contract,
        "runtime_case_table": runtime_case_table,
    }


def _task_entries() -> tuple[StageIPrivateTaskEntry, ...]:
    rows = []
    for index in range(2):
        base = {
            "sample_id": f"view-1::sample-{index}",
            "sortie_id": "sortie-1",
            "pilot_id": 10033,
            "view_id": "view-1",
            "window_index": index,
            "sample_partition": "train",
            "benchmark_role": "thesis_task_weak_label_benchmark",
            "task_role": "thesis_weak_label_task",
            "context_payload": {"thesis_task_boundary": "weak_label_proxy_not_manual_ground_truth"},
        }
        rows.extend(
            [
                StageIPrivateTaskEntry(
                    **base,
                    task_name="risk_proxy",
                    task_type="classification",
                    label_name="risk_proxy_class",
                    label_value="medium",
                    label_source="vehicle_intensity_plus_physiology_variation",
                    source_refs={"window_summary": "raw_window_summary.jsonl"},
                ),
                StageIPrivateTaskEntry(
                    **base,
                    task_name="workload_proxy",
                    task_type="regression",
                    label_name="workload_proxy_score",
                    label_value=0.5,
                    label_source="physiology_variation_plus_vehicle_intensity",
                    source_refs={"window_summary": "raw_window_summary.jsonl"},
                ),
                StageIPrivateTaskEntry(
                    **base,
                    task_name="event_replay_tag",
                    task_type="retrieval",
                    label_name="event_replay_pair_sample_id",
                    label_value="paired-sample",
                    label_source="derived_event_tag_group_pairing",
                    source_refs={"window_manifest": "window_manifest.jsonl"},
                    paired_sample_id="paired-sample",
                ),
            ]
        )
    return tuple(rows)


if __name__ == "__main__":
    unittest.main()
