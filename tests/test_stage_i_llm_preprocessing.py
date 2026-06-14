"""Tests for Stage I LLM preprocessing context generation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.dataset import (  # noqa: E402
    StageIPrivateTaskEntry,
    attach_llm_preprocessing_context_to_task_entries,
    dump_stage_i_private_task_entries,
)
from chronaris.llm import LLMProvider, LLMTaskRequest, LLMTaskResponse, MockLLMProvider  # noqa: E402
from chronaris.llm.prompts import build_llm_task_request  # noqa: E402
from chronaris.llm.schemas import PROMPT_VERSION, SCHEMA_VERSION  # noqa: E402
from chronaris.models.fusion import semantic_query_specs_from_llm_hints  # noqa: E402
from chronaris.pipelines import StageILLMPreprocessingConfig, run_stage_i_llm_preprocessing  # noqa: E402


class StageILLMPreprocessingTest(unittest.TestCase):
    def test_mock_provider_writes_context_and_downstream_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = _write_sources(root)
            result = run_stage_i_llm_preprocessing(
                StageILLMPreprocessingConfig(
                    run_id="stage-i-llm-preprocessing-test",
                    mode="build-and-evaluate",
                    artifact_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    provider_name="mock",
                    model="mock-stage-i-llm",
                    multitask_summary_path=str(sources["multitask_summary"]),
                    task_manifest_path=str(sources["task_manifest"]),
                    live_sweep_summary_path=str(sources["live_sweep_summary"]),
                    support_summary_path=str(sources["support_summary"]),
                    runtime_schema_contract_path=str(sources["runtime_schema_contract"]),
                    runtime_service_summary_path=str(sources["runtime_service_summary"]),
                    runtime_case_table_path=str(sources["runtime_case_table"]),
                    max_schema_fields=4,
                    max_window_cards=2,
                    max_runtime_cases=2,
                ),
                provider=MockLLMProvider(),
            )

            self.assertEqual(result.status, "success")
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            context = json.loads(Path(result.context_path).read_text(encoding="utf-8"))
            harness_summary = json.loads(Path(summary["harness_summary_path"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["field_semantic_count"], 4)
            self.assertEqual(summary["weak_label_review_count"], 3)
            self.assertEqual(summary["runtime_explanation_count"], 2)
            self.assertEqual(summary["prompt_version"], PROMPT_VERSION)
            self.assertEqual(context["schema_version"], SCHEMA_VERSION)
            self.assertEqual(harness_summary["schema_repair_attempt_count"], 0)
            self.assertEqual(harness_summary["final_invalid_task_count"], 0)
            self.assertEqual(summary["comparison"]["sample_count"], 6)
            self.assertEqual(summary["downstream_consumption"]["task_builder_context_attached_entry_count"], 6)
            self.assertIn("field_semantic_dictionary", context)
            self.assertTrue(Path(summary["field_semantic_dictionary_path"]).exists())
            self.assertTrue(Path(summary["weak_label_comparison_path"]).exists())
            self.assertTrue(Path(summary["schema_gap_policy_path"]).exists())
            self.assertTrue(Path(summary["audit_path"]).exists())
            self.assertTrue(Path(summary["report_path"]).exists())

            schema_policy = json.loads(Path(summary["schema_gap_policy_path"]).read_text(encoding="utf-8"))
            self.assertFalse(schema_policy["native_exact_claim_allowed"])
            self.assertFalse(any(row["allows_value_fabrication"] for row in schema_policy["policies"]))

    def test_provider_failure_is_recorded_as_expected_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = _write_sources(root)
            result = run_stage_i_llm_preprocessing(
                StageILLMPreprocessingConfig(
                    run_id="stage-i-llm-preprocessing-failure-test",
                    artifact_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    provider_name="deepseek",
                    model="deepseek-v4-pro",
                    multitask_summary_path=str(sources["multitask_summary"]),
                    task_manifest_path=str(sources["task_manifest"]),
                    live_sweep_summary_path=str(sources["live_sweep_summary"]),
                    support_summary_path=str(sources["support_summary"]),
                    runtime_schema_contract_path=str(sources["runtime_schema_contract"]),
                    runtime_service_summary_path=str(sources["runtime_service_summary"]),
                    runtime_case_table_path=str(sources["runtime_case_table"]),
                    max_schema_fields=2,
                    max_window_cards=1,
                    max_runtime_cases=1,
                ),
                provider=_FailingProvider(),
            )

            self.assertEqual(result.status, "expected_failure")
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            error_cases = json.loads(Path(summary["error_cases_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(error_cases), 5)
            self.assertEqual(summary["harness_summary"]["provider_failure_attempt_count"], 5)
            schema_policy = json.loads(Path(summary["schema_gap_policy_path"]).read_text(encoding="utf-8"))
            self.assertEqual(schema_policy["status"], "fallback_policy")

    def test_schema_repair_retries_invalid_local_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = _write_sources(root)
            result = run_stage_i_llm_preprocessing(
                StageILLMPreprocessingConfig(
                    run_id="stage-i-llm-preprocessing-repair-test",
                    artifact_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    provider_name="mock",
                    model="mock-stage-i-llm",
                    multitask_summary_path=str(sources["multitask_summary"]),
                    task_manifest_path=str(sources["task_manifest"]),
                    live_sweep_summary_path=str(sources["live_sweep_summary"]),
                    support_summary_path=str(sources["support_summary"]),
                    runtime_schema_contract_path=str(sources["runtime_schema_contract"]),
                    runtime_service_summary_path=str(sources["runtime_service_summary"]),
                    runtime_case_table_path=str(sources["runtime_case_table"]),
                    max_schema_fields=2,
                    max_window_cards=1,
                    max_runtime_cases=1,
                ),
                provider=_RepairingProvider(),
            )

            self.assertEqual(result.status, "success")
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["request_count"], 6)
            self.assertEqual(summary["error_count"], 0)
            self.assertEqual(summary["semantic_query_hint_count"], 1)
            self.assertEqual(summary["harness_summary"]["schema_repair_attempt_count"], 1)
            self.assertEqual(summary["harness_summary"]["failed_initial_attempt_count"], 1)

    def test_large_payloads_are_sliced_and_merged_locally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = _write_sources(root)
            result = run_stage_i_llm_preprocessing(
                StageILLMPreprocessingConfig(
                    run_id="stage-i-llm-preprocessing-slicing-test",
                    mode="build-and-evaluate",
                    artifact_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    provider_name="mock",
                    model="mock-stage-i-llm",
                    multitask_summary_path=str(sources["multitask_summary"]),
                    task_manifest_path=str(sources["task_manifest"]),
                    live_sweep_summary_path=str(sources["live_sweep_summary"]),
                    support_summary_path=str(sources["support_summary"]),
                    runtime_schema_contract_path=str(sources["runtime_schema_contract"]),
                    runtime_service_summary_path=str(sources["runtime_service_summary"]),
                    runtime_case_table_path=str(sources["runtime_case_table"]),
                    max_schema_fields=4,
                    max_window_cards=2,
                    max_runtime_cases=2,
                    schema_field_chunk_size=2,
                    weak_label_task_chunk_size=2,
                    runtime_case_chunk_size=1,
                ),
                provider=MockLLMProvider(),
            )

            self.assertEqual(result.status, "success")
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["request_count"], 8)
            self.assertEqual(summary["field_semantic_count"], 4)
            self.assertEqual(summary["weak_label_review_count"], 3)
            self.assertEqual(summary["runtime_explanation_count"], 2)
            self.assertEqual(summary["slicing_summary"]["sliced_task_count"], 3)
            self.assertEqual(summary["harness_summary"]["final_invalid_task_count"], 0)

    def test_prompt_declares_agent_protocol_and_local_gates(self) -> None:
        request = build_llm_task_request(
            run_id="prompt-test",
            request_id="prompt-test-01-field-semantics",
            task_name="field_semantics",
            provider="deepseek",
            model="deepseek-v4-pro",
            input_payload={
                "fields": [
                    {
                        "stream_kind": "vehicle",
                        "measurement_group": "BUS6000019110020",
                        "feature_name": "BUS6000019110020.code1030",
                    }
                ]
            },
        )
        payload = json.loads(request.messages[1].content)
        self.assertEqual(payload["prompt_version"], PROMPT_VERSION)
        self.assertEqual(payload["output_schema_version"], SCHEMA_VERSION)
        self.assertIn("agent_protocol", payload)
        self.assertIn("feature_name_coverage_exact", payload["local_harness_gates"])
        self.assertEqual(
            payload["response_contract"]["allowed_top_level_keys"],
            ["field_semantics"],
        )

    def test_downstream_helpers_keep_llm_context_optional_and_whitelisted(self) -> None:
        entries = _task_entries()
        context = {
            "run_id": "llm-run",
            "schema_version": "stage_i_llm_preprocessing_context.v1",
            "field_semantic_dictionary_path": "field_semantic_dictionary.csv",
            "weak_label_rule_review": [
                {"task_name": "risk_proxy", "review_decision": "keep_current_rule"}
            ],
        }
        attached = attach_llm_preprocessing_context_to_task_entries(entries, context)
        self.assertEqual(entries[0].label_value, attached[0].label_value)
        self.assertIn("llm_preprocessing_context", attached[0].context_payload)
        self.assertEqual(
            attached[0].context_payload["llm_weak_label_rule_review"]["review_decision"],
            "keep_current_rule",
        )

        specs = semantic_query_specs_from_llm_hints(
            [
                {"name": "llm_schema_gap_review", "recipe": "coordination_gap"},
                {"name": "free_text_prompt", "recipe": "arbitrary_prompt"},
            ]
        )
        names = {spec.name for spec in specs}
        self.assertIn("risk_proxy", names)
        self.assertIn("llm_schema_gap_review", names)
        self.assertNotIn("free_text_prompt", names)


class _FailingProvider(LLMProvider):
    provider_name = "deepseek"
    model = "deepseek-v4-pro"

    def generate(self, request: LLMTaskRequest) -> LLMTaskResponse:
        return LLMTaskResponse(
            request_id=request.request_id,
            provider=self.provider_name,
            model=self.model,
            status="expected_failure",
            error_summary="simulated provider failure",
        )


class _RepairingProvider(MockLLMProvider):
    def generate(self, request: LLMTaskRequest) -> LLMTaskResponse:
        if request.task_name == "semantic_query_hints" and not request.request_id.endswith("-repair"):
            return LLMTaskResponse(
                request_id=request.request_id,
                provider=self.provider_name,
                model=self.model,
                status="success",
                parsed_payload={"semantic_query_hints": []},
                content='{"semantic_query_hints":[]}',
            )
        if request.task_name == "semantic_query_hints":
            return LLMTaskResponse(
                request_id=request.request_id,
                provider=self.provider_name,
                model=self.model,
                status="success",
                parsed_payload={
                    "semantic_query_hints": [
                        {
                            "name": "llm_repaired_hint",
                            "recipe": "coordination_gap",
                            "confidence": 0.7,
                            "source": "repair",
                            "needs_human_review": True,
                        }
                    ]
                },
                content='{"semantic_query_hints":[{"name":"llm_repaired_hint","recipe":"coordination_gap"}]}',
            )
        return super().generate(request)


def _write_sources(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    task_manifest = root / "thesis_task_manifest.jsonl"
    entries = _task_entries()
    dump_stage_i_private_task_entries(entries, path=task_manifest)

    task_summary = {
        "entry_count": len(entries),
        "task_counts": {"risk_proxy": 2, "workload_proxy": 2, "event_replay_tag": 2},
        "thesis_task_boundary": "weak_label_proxy_not_manual_ground_truth",
        "thesis_task_definitions": {
            "risk_proxy": {
                "task_family": "risk_analysis",
                "label_source": "vehicle_intensity_plus_physiology_variation",
            },
            "workload_proxy": {
                "task_family": "cognitive_workload",
                "label_source": "physiology_variation_plus_vehicle_intensity",
            },
            "event_replay_tag": {
                "task_family": "event_replay",
                "label_source": "derived_event_tag_group_pairing",
            },
        },
    }
    multitask_summary = root / "multitask_summary.json"
    multitask_summary.write_text(
        json.dumps({"run_id": "multitask", "source_summary": {"task_payload_summary": task_summary}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    live_sweep_summary = root / "live_sweep_summary.json"
    live_sweep_summary.write_text(
        json.dumps({"run_id": "live", "sample_count": 2, "task_entry_count": 6}, indent=2) + "\n",
        encoding="utf-8",
    )
    support_summary = root / "support_summary.json"
    support_summary.write_text(
        json.dumps(
            {
                "causal_support": {
                    "semantic_event": {
                        "query_names": ["risk_proxy", "workload_proxy", "event_replay_tag"],
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
        json.dumps(
            {
                "schema_source": "input_normalization_stats",
                "schema_hash": "abc123",
                "expected_schema": {
                    "physiology": {"feature_names": ["eeg.af3", "spo2.toi1"]},
                    "vehicle": {
                        "feature_names": [
                            "BUS6000019110020.code1030",
                            "BUS6000019110021.code1002",
                        ]
                    },
                },
                "native_input": {
                    "comparison": {
                        "status": "aligned",
                        "vehicle": {
                            "missing_feature_names_preview": ["BUS6000019110021.code1002"],
                            "missing_measurement_group_counts": {"BUS6000019110021": 1},
                        },
                    }
                },
                "canonical_payload": {"comparison": {"status": "exact"}},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_service_summary = root / "runtime_service_summary.json"
    runtime_service_summary.write_text(json.dumps({"status": "success"}, indent=2) + "\n", encoding="utf-8")
    runtime_case_table = root / "runtime_semantic_case.csv"
    runtime_case_table.write_text(
        "\n".join(
            [
                "sample_id,risk_proxy_prediction,semantic_top_query_name,native_feature_schema_status,canonical_feature_schema_status,source_path,support_source_path,runtime_schema_contract_source_path",
                "sample-1,medium,risk_proxy,aligned,exact,runtime.json,support.json,contract.json",
                "sample-2,low,workload_proxy,aligned,exact,runtime.json,support.json,contract.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "multitask_summary": multitask_summary,
        "task_manifest": task_manifest,
        "live_sweep_summary": live_sweep_summary,
        "support_summary": support_summary,
        "runtime_schema_contract": runtime_schema_contract,
        "runtime_service_summary": runtime_service_summary,
        "runtime_case_table": runtime_case_table,
    }


def _task_entries() -> tuple[StageIPrivateTaskEntry, ...]:
    rows = []
    for index in range(2):
        sample_id = f"view-1::sample-{index}"
        base = {
            "sample_id": sample_id,
            "sortie_id": "sortie-1",
            "pilot_id": 10033,
            "view_id": "view-1",
            "window_index": index,
            "sample_partition": "train",
            "benchmark_role": "thesis_task_weak_label_benchmark",
            "task_role": "thesis_weak_label_task",
            "context_payload": {"thesis_task_boundary": "weak_label_proxy_not_manual_ground_truth"},
        }
        rows.append(
            StageIPrivateTaskEntry(
                **base,
                task_name="risk_proxy",
                task_type="classification",
                label_name="risk_proxy_class",
                label_value="medium",
                label_source="vehicle_intensity_plus_physiology_variation",
                source_refs={"window_summary": "raw_window_summary.jsonl"},
            )
        )
        rows.append(
            StageIPrivateTaskEntry(
                **base,
                task_name="workload_proxy",
                task_type="regression",
                label_name="workload_proxy_score",
                label_value=0.5,
                label_source="physiology_variation_plus_vehicle_intensity",
                source_refs={"window_summary": "raw_window_summary.jsonl"},
            )
        )
        rows.append(
            StageIPrivateTaskEntry(
                **base,
                task_name="event_replay_tag",
                task_type="retrieval",
                label_name="event_replay_pair_sample_id",
                label_value="paired-sample",
                label_source="derived_event_tag_group_pairing",
                source_refs={"window_manifest": "window_manifest.jsonl"},
                paired_sample_id="paired-sample",
            )
        )
    return tuple(rows)


if __name__ == "__main__":
    unittest.main()
