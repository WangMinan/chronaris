"""Stage I DeepSeek/LLM preprocessing context builder."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from chronaris.dataset import (
    attach_llm_preprocessing_context_to_task_entries,
    load_stage_i_private_task_entries,
)
from chronaris.llm import LLMProvider, resolve_llm_provider
from chronaris.llm.schemas import (
    SCHEMA_VERSION,
)
from chronaris.pipelines.stage_i.stage_i_llm_preprocessing_harness import run_preprocessing_llm_task
from chronaris.models.fusion import semantic_query_specs_from_llm_hints
from chronaris.pipelines.stage_i.stage_i_llm_preprocessing_reporting import (
    render_stage_i_llm_preprocessing_report,
)
from chronaris.pipelines.stage_i.stage_i_run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_llm_preprocessing"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"
DEFAULT_MULTITASK_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_multitask/"
    "20260607T-stage-i-multitask-real-closure-r2/multitask_summary.json"
)
DEFAULT_TASK_MANIFEST_PATH = (
    "docs/artifacts/assets/stage_i_multitask/"
    "20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl"
)
DEFAULT_LIVE_SWEEP_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_multitask_sweep/"
    "20260613T-stage-i-p11-live-influx-r3-resume/multitask_sweep_summary.json"
)
DEFAULT_SUPPORT_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_support/"
    "20260607T-stage-i-support-semantic-r2/support_summary.json"
)
DEFAULT_RUNTIME_SCHEMA_CONTRACT_PATH = (
    "docs/artifacts/assets/stage_i_runtime_service/"
    "20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_schema_contract.json"
)
DEFAULT_RUNTIME_SERVICE_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_runtime_service/"
    "20260613T-stage-i-runtime-service-smoke-r2-contract/runtime_service_smoke_summary.json"
)
DEFAULT_RUNTIME_CASE_TABLE_PATH = (
    "docs/artifacts/assets/stage_i_thesis_figures/"
    "20260613T-stage-i-thesis-materials-r2-p18/runtime_semantic_case.csv"
)

LLM_PREPROCESSING_MODES = ("build-context", "build-and-evaluate")


@dataclass(frozen=True, slots=True)
class StageILLMPreprocessingConfig:
    """Configuration for one Stage I LLM preprocessing run."""

    run_id: str
    mode: str = "build-context"
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    provider_name: str = "deepseek"
    model: str = "deepseek-v4-pro"
    multitask_summary_path: str = DEFAULT_MULTITASK_SUMMARY_PATH
    task_manifest_path: str = DEFAULT_TASK_MANIFEST_PATH
    live_sweep_summary_path: str = DEFAULT_LIVE_SWEEP_SUMMARY_PATH
    support_summary_path: str = DEFAULT_SUPPORT_SUMMARY_PATH
    runtime_schema_contract_path: str = DEFAULT_RUNTIME_SCHEMA_CONTRACT_PATH
    runtime_service_summary_path: str = DEFAULT_RUNTIME_SERVICE_SUMMARY_PATH
    runtime_case_table_path: str = DEFAULT_RUNTIME_CASE_TABLE_PATH
    max_schema_fields: int = 36
    max_window_cards: int = 12
    max_runtime_cases: int = 12

    def __post_init__(self) -> None:
        if self.mode not in LLM_PREPROCESSING_MODES:
            raise ValueError(f"unsupported llm preprocessing mode: {self.mode}")
        if self.max_schema_fields <= 0:
            raise ValueError("max_schema_fields must be positive.")
        if self.max_window_cards <= 0:
            raise ValueError("max_window_cards must be positive.")
        if self.max_runtime_cases <= 0:
            raise ValueError("max_runtime_cases must be positive.")


@dataclass(frozen=True, slots=True)
class StageILLMPreprocessingRunResult:
    """Artifacts written by the Stage I LLM preprocessing pipeline."""

    run_id: str
    artifact_root: str
    context_path: str
    summary_path: str
    report_path: str
    status: str
    summary: Mapping[str, object]


def run_stage_i_llm_preprocessing(
    config: StageILLMPreprocessingConfig,
    *,
    provider: LLMProvider | None = None,
) -> StageILLMPreprocessingRunResult:
    """Build audited LLM preprocessing context from existing Stage I assets."""
    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_llm_preprocessing",
        logger=LOGGER,
        initial_progress={
            "artifact_root": str(run_root),
            "mode": config.mode,
            "provider_name": config.provider_name,
            "model": config.model,
        },
    ) as progress:
        return _run_observed(
            config=config,
            run_root=run_root,
            provider=provider,
            progress=progress,
        )


def _run_observed(
    *,
    config: StageILLMPreprocessingConfig,
    run_root: Path,
    provider: LLMProvider | None,
    progress: StageIRunProgress,
) -> StageILLMPreprocessingRunResult:
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    sources = _load_sources(config)
    progress.update("sources_loaded", source_count=len(sources))

    resolved_provider = provider or resolve_llm_provider(
        provider_name=config.provider_name,
        model=config.model,
    )
    cards = _build_cards(config=config, sources=sources)
    progress.update("cards_built", card_count=len(cards))

    audit_rows: list[dict[str, object]] = []
    error_cases: list[dict[str, object]] = []
    outputs: dict[str, object] = {
        "field_semantics": [],
        "weak_label_rule_review": [],
        "semantic_query_hints": [],
        "schema_gap_policy": {},
        "runtime_case_explanations": [],
    }
    task_order = (
        "field_semantics",
        "weak_label_review",
        "semantic_query_hints",
        "schema_gap_policy",
        "runtime_explanations",
    )
    for index, task_name in enumerate(task_order, start=1):
        output_key, output_value, task_audits, task_errors = run_preprocessing_llm_task(
            run_id=config.run_id,
            provider=resolved_provider,
            index=index,
            task_name=task_name,
            input_payload=cards[task_name],
        )
        outputs[output_key] = output_value
        audit_rows.extend(task_audits)
        error_cases.extend(task_errors)
    progress.update("llm_requests_finished", request_count=len(audit_rows), error_count=len(error_cases))

    if not outputs["schema_gap_policy"]:
        outputs["schema_gap_policy"] = _fallback_schema_gap_policy(cards["schema_gap_policy"])

    paths = _write_outputs(
        config=config,
        run_root=run_root,
        report_root=report_root,
        sources=sources,
        outputs=outputs,
        audit_rows=audit_rows,
        error_cases=error_cases,
    )
    progress.finish(summary_path=paths["summary_path"], report_path=paths["report_path"])
    return StageILLMPreprocessingRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        context_path=paths["context_path"],
        summary_path=paths["summary_path"],
        report_path=paths["report_path"],
        status=str(paths["status"]),
        summary=dict(paths["summary"]),
    )


def _load_sources(config: StageILLMPreprocessingConfig) -> dict[str, object]:
    entries = load_stage_i_private_task_entries(config.task_manifest_path)
    runtime_cases = _read_csv_rows(config.runtime_case_table_path, limit=config.max_runtime_cases)
    return {
        "multitask_summary": _load_json(config.multitask_summary_path),
        "task_entries": entries,
        "live_sweep_summary": _load_json(config.live_sweep_summary_path),
        "support_summary": _load_json(config.support_summary_path),
        "runtime_schema_contract": _load_json(config.runtime_schema_contract_path),
        "runtime_service_summary": _load_json(config.runtime_service_summary_path),
        "runtime_cases": runtime_cases,
        "source_paths": {
            "multitask_summary_path": config.multitask_summary_path,
            "task_manifest_path": config.task_manifest_path,
            "live_sweep_summary_path": config.live_sweep_summary_path,
            "support_summary_path": config.support_summary_path,
            "runtime_schema_contract_path": config.runtime_schema_contract_path,
            "runtime_service_summary_path": config.runtime_service_summary_path,
            "runtime_case_table_path": config.runtime_case_table_path,
        },
    }


def _build_cards(
    *,
    config: StageILLMPreprocessingConfig,
    sources: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    task_summary = _resolve_task_payload_summary(sources)
    runtime_contract = _as_mapping(sources["runtime_schema_contract"])
    support_summary = _as_mapping(sources["support_summary"])
    runtime_cases = list(sources["runtime_cases"])
    schema_fields = _select_schema_fields(
        runtime_contract=runtime_contract,
        max_fields=config.max_schema_fields,
    )
    task_summaries = _build_task_summaries(task_summary, sources["task_entries"])
    window_cards = _build_window_cards(sources["task_entries"], limit=config.max_window_cards)
    semantic_event = _as_mapping(_as_mapping(support_summary.get("causal_support")).get("semantic_event"))
    missing_groups = (
        _as_mapping(runtime_contract.get("native_input"))
        .get("comparison", {})
        .get("vehicle", {})
        .get("missing_measurement_group_counts", {})
    )
    source_paths = dict(_as_mapping(sources["source_paths"]))
    return {
        "field_semantics": {
            "card_type": "schema_card",
            "fields": schema_fields,
            "schema_hash": runtime_contract.get("schema_hash"),
            "source_paths": source_paths,
            "boundary": "field semantics are preprocessing hints, not unit truth",
        },
        "weak_label_review": {
            "card_type": "window_summary_card",
            "task_summaries": task_summaries,
            "window_cards": window_cards,
            "task_payload_summary": task_summary,
            "source_paths": source_paths,
            "boundary": "weak-label rules are not manual ground truth",
        },
        "semantic_query_hints": {
            "card_type": "semantic_support_card",
            "allowed_recipes": [
                "coordination_gap",
                "gap_plus_event",
                "physiology_plus_gap",
                "vehicle_plus_event",
            ],
            "existing_query_names": list(semantic_event.get("query_names", [])),
            "view_rows": list(semantic_event.get("view_rows", []))[:6],
            "source_paths": source_paths,
        },
        "schema_gap_policy": {
            "card_type": "schema_gap_card",
            "native_feature_schema_status": _as_mapping(runtime_contract.get("native_input"))
            .get("comparison", {})
            .get("status"),
            "canonical_feature_schema_status": _as_mapping(runtime_contract.get("canonical_payload"))
            .get("comparison", {})
            .get("status"),
            "missing_measurement_groups": dict(missing_groups) if isinstance(missing_groups, Mapping) else {},
            "missing_feature_preview": _as_mapping(runtime_contract.get("native_input"))
            .get("comparison", {})
            .get("vehicle", {})
            .get("missing_feature_names_preview", []),
            "source_paths": source_paths,
            "forbidden": [
                "fabricate missing BUS group values",
                "claim native exact when only canonical exact is available",
            ],
        },
        "runtime_explanations": {
            "card_type": "runtime_case_card",
            "runtime_cases": runtime_cases,
            "schema_gap_summary": {
                "native_feature_schema_status": _as_mapping(runtime_contract.get("native_input"))
                .get("comparison", {})
                .get("status"),
                "canonical_feature_schema_status": _as_mapping(runtime_contract.get("canonical_payload"))
                .get("comparison", {})
                .get("status"),
                "missing_measurement_groups": dict(missing_groups) if isinstance(missing_groups, Mapping) else {},
            },
            "source_paths": source_paths,
            "boundary": "runtime explanations are explanatory preprocessing context, not expert replay truth",
        },
    }


def _write_outputs(
    *,
    config: StageILLMPreprocessingConfig,
    run_root: Path,
    report_root: Path,
    sources: Mapping[str, object],
    outputs: Mapping[str, object],
    audit_rows: Sequence[Mapping[str, object]],
    error_cases: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    field_rows = list(outputs["field_semantics"])
    weak_review_rows = list(outputs["weak_label_rule_review"])
    query_hints = list(outputs["semantic_query_hints"])
    schema_gap_policy = dict(outputs["schema_gap_policy"])
    runtime_explanations = list(outputs["runtime_case_explanations"])
    task_summary = _resolve_task_payload_summary(sources)
    comparison_rows = _build_weak_label_comparison_rows(
        task_summary=task_summary,
        task_entries=sources["task_entries"],
        weak_review_rows=weak_review_rows,
    )
    comparison_stats = _summarize_comparison_rows(comparison_rows)
    downstream_summary = _build_downstream_consumption_summary(
        config=config,
        sources=sources,
        outputs=outputs,
        comparison_stats=comparison_stats,
    )
    context_path = run_root / "llm_preprocessing_context.json"
    field_jsonl_path = run_root / "llm_field_semantics.jsonl"
    field_csv_path = run_root / "field_semantic_dictionary.csv"
    weak_review_path = run_root / "llm_weak_label_review.jsonl"
    comparison_path = run_root / "weak_label_llm_comparison.csv"
    schema_gap_path = run_root / "llm_schema_gap_policy.json"
    runtime_explanations_path = run_root / "runtime_llm_explanations.jsonl"
    summary_path = run_root / "llm_preprocessing_summary.json"
    audit_path = run_root / "llm_request_response_audit.jsonl"
    error_cases_path = run_root / "llm_error_cases.json"
    report_path = report_root / f"stage-i-llm-preprocessing-{config.run_id}.md"

    context = {
        "run_id": config.run_id,
        "schema_version": SCHEMA_VERSION,
        "status": _resolve_status(audit_rows, error_cases),
        "mode": config.mode,
        "provider": _provider_from_audit(audit_rows),
        "model": _model_from_audit(audit_rows),
        "context_path": str(context_path),
        "field_semantic_dictionary_path": str(field_csv_path),
        "field_semantic_dictionary": field_rows,
        "weak_label_rule_review": weak_review_rows,
        "semantic_query_hints": query_hints,
        "schema_gap_policy": schema_gap_policy,
        "runtime_case_explanations_path": str(runtime_explanations_path),
        "downstream_consumption": downstream_summary,
        "source_paths": dict(_as_mapping(sources["source_paths"])),
        "boundary": "llm_preprocessing_context_not_ground_truth",
    }
    _write_jsonl(field_jsonl_path, field_rows)
    _write_csv(
        field_csv_path,
        field_rows,
        (
            "stream_kind",
            "measurement_group",
            "feature_name",
            "llm_semantic_role",
            "llm_unit_guess",
            "evidence",
            "confidence",
            "needs_human_review",
        ),
    )
    _write_jsonl(weak_review_path, weak_review_rows)
    _write_csv(
        comparison_path,
        comparison_rows,
        (
            "task_name",
            "baseline_label_source",
            "sample_count",
            "review_decision",
            "agreement_count",
            "conflict_count",
            "needs_human_review_count",
            "label_agreement_rate",
            "boundary",
        ),
    )
    schema_gap_path.write_text(
        json.dumps(schema_gap_policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(runtime_explanations_path, runtime_explanations)
    _write_jsonl(audit_path, audit_rows)
    error_cases_path.write_text(
        json.dumps(list(error_cases), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    context_path.write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "run_id": config.run_id,
        "status": context["status"],
        "mode": config.mode,
        "artifact_root": str(run_root),
        "context_path": str(context_path),
        "field_semantic_dictionary_path": str(field_csv_path),
        "weak_label_review_path": str(weak_review_path),
        "weak_label_comparison_path": str(comparison_path),
        "schema_gap_policy_path": str(schema_gap_path),
        "runtime_llm_explanations_path": str(runtime_explanations_path),
        "audit_path": str(audit_path),
        "error_cases_path": str(error_cases_path),
        "report_path": str(report_path),
        "request_count": len(audit_rows),
        "error_count": len(error_cases),
        "field_semantic_count": len(field_rows),
        "weak_label_review_count": len(weak_review_rows),
        "semantic_query_hint_count": len(query_hints),
        "runtime_explanation_count": len(runtime_explanations),
        "comparison": comparison_stats,
        "downstream_consumption": downstream_summary,
        "source_paths": dict(_as_mapping(sources["source_paths"])),
        "boundary": "DeepSeek/LLM outputs are preprocessing context, not manual truth.",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        render_stage_i_llm_preprocessing_report(summary=summary, comparison_rows=comparison_rows)
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": context["status"],
        "context_path": str(context_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary,
    }

def _build_task_summaries(
    task_summary: Mapping[str, object],
    task_entries: Sequence[object],
) -> list[dict[str, object]]:
    task_counts = dict(task_summary.get("task_counts") or {})
    definitions = dict(task_summary.get("thesis_task_definitions") or {})
    if not task_counts:
        task_counts = dict(Counter(entry.task_name for entry in task_entries))
    rows = []
    for task_name in sorted(task_counts):
        definition = dict(definitions.get(task_name) or {})
        rows.append(
            {
                "task_name": task_name,
                "sample_count": int(task_counts[task_name]),
                "task_family": definition.get("task_family"),
                "label_source": definition.get("label_source") or _label_source_for_task(task_entries, task_name),
                "weak_label_note": definition.get("weak_label_note"),
            }
        )
    return rows


def _build_window_cards(task_entries: Sequence[object], *, limit: int) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for entry in task_entries:
        row = grouped.setdefault(
            entry.sample_id,
            {
                "sample_id": entry.sample_id,
                "sortie_id": entry.sortie_id,
                "pilot_id": entry.pilot_id,
                "view_id": entry.view_id,
                "window_index": entry.window_index,
                "sample_partition": entry.sample_partition,
                "labels": {},
                "boundaries": set(),
            },
        )
        row["labels"][entry.task_name] = entry.label_value
        row["boundaries"].add(entry.context_payload.get("thesis_task_boundary"))
        if len(grouped) >= limit and entry.sample_id not in grouped:
            break
    cards = []
    for row in list(grouped.values())[:limit]:
        boundaries = sorted(boundary for boundary in row.pop("boundaries") if boundary)
        row["boundaries"] = boundaries
        cards.append(row)
    return cards


def _select_schema_fields(
    *,
    runtime_contract: Mapping[str, object],
    max_fields: int,
) -> list[dict[str, object]]:
    expected = _as_mapping(runtime_contract.get("expected_schema"))
    rows: list[dict[str, object]] = []
    for stream_kind in ("physiology", "vehicle"):
        stream = _as_mapping(expected.get(stream_kind))
        for feature_name in list(stream.get("feature_names", [])):
            rows.append(
                {
                    "stream_kind": stream_kind,
                    "measurement_group": _measurement_group(str(feature_name)),
                    "feature_name": str(feature_name),
                    "schema_source": runtime_contract.get("schema_source"),
                }
            )
            if len(rows) >= max_fields:
                return rows
    return rows


def _build_weak_label_comparison_rows(
    *,
    task_summary: Mapping[str, object],
    task_entries: Sequence[object],
    weak_review_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    review_by_task = {str(row.get("task_name")): row for row in weak_review_rows}
    task_summaries = _build_task_summaries(task_summary, task_entries)
    rows = []
    for task in task_summaries:
        task_name = str(task["task_name"])
        sample_count = int(task.get("sample_count") or 0)
        review = review_by_task.get(task_name, {})
        decision = str(review.get("review_decision") or "not_reviewed")
        needs_review = bool(review.get("needs_human_review", decision != "keep_current_rule"))
        agreement_count = sample_count if decision == "keep_current_rule" else 0
        conflict_count = sample_count if decision == "revise_current_rule" else 0
        human_review_count = sample_count if needs_review or decision in {"needs_human_review", "not_reviewed"} else 0
        rows.append(
            {
                "task_name": task_name,
                "baseline_label_source": task.get("label_source"),
                "sample_count": sample_count,
                "review_decision": decision,
                "agreement_count": agreement_count,
                "conflict_count": conflict_count,
                "needs_human_review_count": human_review_count,
                "label_agreement_rate": float(agreement_count / sample_count) if sample_count else 0.0,
                "boundary": "llm_review_not_ground_truth",
            }
        )
    return rows


def _summarize_comparison_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    sample_count = sum(int(row.get("sample_count") or 0) for row in rows)
    agreement_count = sum(int(row.get("agreement_count") or 0) for row in rows)
    conflict_count = sum(int(row.get("conflict_count") or 0) for row in rows)
    human_review_count = sum(int(row.get("needs_human_review_count") or 0) for row in rows)
    return {
        "sample_count": sample_count,
        "agreement_count": agreement_count,
        "conflict_count": conflict_count,
        "needs_human_review_count": human_review_count,
        "label_agreement_rate": float(agreement_count / sample_count) if sample_count else 0.0,
    }


def _build_downstream_consumption_summary(
    *,
    config: StageILLMPreprocessingConfig,
    sources: Mapping[str, object],
    outputs: Mapping[str, object],
    comparison_stats: Mapping[str, object],
) -> dict[str, object]:
    context_stub = {
        "run_id": config.run_id,
        "schema_version": SCHEMA_VERSION,
        "field_semantic_dictionary_path": "field_semantic_dictionary.csv",
        "weak_label_rule_review": list(outputs["weak_label_rule_review"]),
    }
    attached_entries = attach_llm_preprocessing_context_to_task_entries(
        sources["task_entries"],
        context_stub,
    )
    query_specs = semantic_query_specs_from_llm_hints(outputs["semantic_query_hints"])
    return {
        "evaluation_enabled": config.mode == "build-and-evaluate",
        "task_builder_context_attached_entry_count": len(attached_entries),
        "semantic_query_spec_count": len(query_specs),
        "semantic_query_specs": [{"name": spec.name, "recipe": spec.recipe} for spec in query_specs],
        "runtime_explanation_count": len(outputs["runtime_case_explanations"]),
        "comparison": dict(comparison_stats),
        "boundary": "optional downstream context only; no label overwrite",
    }


def _fallback_schema_gap_policy(card: Mapping[str, object]) -> dict[str, object]:
    groups = card.get("missing_measurement_groups", {})
    policies = []
    if isinstance(groups, Mapping):
        for group in sorted(groups):
            policies.append(
                {
                    "measurement_group": group,
                    "policy_type": "human_review_required",
                    "reason": "LLM policy unavailable; preserve explicit schema gap",
                    "allows_value_fabrication": False,
                    "needs_human_review": True,
                }
            )
    return {
        "status": "fallback_policy",
        "canonical_exact_boundary": "canonical exact is a service payload contract, not native exact evidence",
        "native_exact_claim_allowed": False,
        "policies": policies,
    }


def _resolve_task_payload_summary(sources: Mapping[str, object]) -> dict[str, object]:
    multitask_summary = _as_mapping(sources.get("multitask_summary"))
    source_summary = _as_mapping(multitask_summary.get("source_summary"))
    payload = source_summary.get("task_payload_summary")
    if isinstance(payload, Mapping):
        return dict(payload)
    entries = sources.get("task_entries") or ()
    return {
        "task_counts": dict(Counter(entry.task_name for entry in entries)),
        "thesis_task_boundary": "weak_label_proxy_not_manual_ground_truth",
        "thesis_task_definitions": {},
    }


def _label_source_for_task(task_entries: Sequence[object], task_name: str) -> str | None:
    for entry in task_entries:
        if entry.task_name == task_name:
            return entry.label_source
    return None


def _resolve_status(
    audit_rows: Sequence[Mapping[str, object]],
    error_cases: Sequence[Mapping[str, object]],
) -> str:
    if not audit_rows:
        return "expected_failure"
    if len(error_cases) == len(audit_rows):
        return "expected_failure"
    if error_cases:
        return "partial_success"
    return "success"


def _provider_from_audit(audit_rows: Sequence[Mapping[str, object]]) -> str | None:
    if not audit_rows:
        return None
    response = _as_mapping(audit_rows[0].get("response"))
    return response.get("provider")


def _model_from_audit(audit_rows: Sequence[Mapping[str, object]]) -> str | None:
    if not audit_rows:
        return None
    response = _as_mapping(audit_rows[0].get("response"))
    return response.get("model")


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv_rows(path: str | Path, *, limit: int) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for _, row in zip(range(limit), reader)]

def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _measurement_group(feature_name: str) -> str:
    if "." in feature_name:
        return feature_name.split(".", 1)[0]
    return feature_name
