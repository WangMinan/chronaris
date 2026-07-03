"""Stage I LLM preprocessing comparison over existing evidence assets."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from chronaris.dataset import load_stage_i_private_task_entries
from chronaris.pipelines.stage_i.common.run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)
from chronaris.pipelines.stage_i.llm.comparison_builders import (
    build_condition_manifest,
    build_human_review_packet,
    build_midterm_claims_payload,
    build_runtime_explanation_comparison,
    build_semantic_hint_comparison,
    build_task_context_comparison,
)
from chronaris.pipelines.stage_i.llm.comparison_reporting import (
    render_stage_i_llm_comparison_report,
    render_stage_i_llm_midterm_summary,
)
from chronaris.pipelines.stage_i.llm.preprocessing import (
    DEFAULT_RUNTIME_CASE_TABLE_PATH,
    DEFAULT_RUNTIME_SCHEMA_CONTRACT_PATH,
    DEFAULT_SUPPORT_SUMMARY_PATH,
    DEFAULT_TASK_MANIFEST_PATH,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_LLM_COMPARISON_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_llm_comparison"
DEFAULT_LLM_COMPARISON_REPORT_ROOT = "docs/artifacts/stage_i"
DEFAULT_LLM_COMPARISON_MIDTERM_ROOT = "docs/midterm"
DEFAULT_LLM_CONTEXT_PATH = (
    "docs/artifacts/assets/stage_i_llm_preprocessing/"
    "20260614T-stage-i-p20-deepseek-llm-preprocessing-r3-sliced/"
    "llm_preprocessing_context.json"
)
DEFAULT_LLM_COMPARISON_RUN_ID = "20260614T-stage-i-p21-llm-comparison-r1"


def resolve_runtime_explanations_path(*, context: Mapping[str, object], context_path: Path) -> Path:
    raw_path = context.get("runtime_case_explanations_path")
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return path
    return context_path.parent / "runtime_llm_explanations.jsonl"


def load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    target = Path(path)
    if not target.exists():
        return []
    return [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

@dataclass(frozen=True, slots=True)
class StageILLMComparisonConfig:
    """Configuration for one P21 LLM preprocessing comparison run."""

    run_id: str = DEFAULT_LLM_COMPARISON_RUN_ID
    artifact_root: str = DEFAULT_LLM_COMPARISON_ARTIFACT_ROOT
    report_root: str = DEFAULT_LLM_COMPARISON_REPORT_ROOT
    midterm_root: str = DEFAULT_LLM_COMPARISON_MIDTERM_ROOT
    llm_context_path: str = DEFAULT_LLM_CONTEXT_PATH
    task_manifest_path: str = DEFAULT_TASK_MANIFEST_PATH
    support_summary_path: str = DEFAULT_SUPPORT_SUMMARY_PATH
    runtime_schema_contract_path: str = DEFAULT_RUNTIME_SCHEMA_CONTRACT_PATH
    runtime_case_table_path: str = DEFAULT_RUNTIME_CASE_TABLE_PATH
    max_human_review_fields: int = 6
    max_human_review_schema_gaps: int = 6

    def __post_init__(self) -> None:
        if self.max_human_review_fields <= 0:
            raise ValueError("max_human_review_fields must be positive.")
        if self.max_human_review_schema_gaps <= 0:
            raise ValueError("max_human_review_schema_gaps must be positive.")


@dataclass(frozen=True, slots=True)
class StageILLMComparisonRunResult:
    """Artifacts written by the P21 comparison pipeline."""

    run_id: str
    artifact_root: str
    summary_path: str
    condition_manifest_path: str
    report_path: str
    midterm_summary_path: str
    status: str
    summary: Mapping[str, object]


def run_stage_i_llm_comparison(
    config: StageILLMComparisonConfig | None = None,
) -> StageILLMComparisonRunResult:
    """Run A0-A4 comparison using the existing P20 LLM preprocessing context."""

    resolved = config or StageILLMComparisonConfig()
    run_root = Path(resolved.artifact_root) / resolved.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=resolved.run_id,
        stage_name="stage_i_llm_comparison",
        logger=LOGGER,
        initial_progress={
            "artifact_root": str(run_root),
            "llm_context_path": resolved.llm_context_path,
            "task_manifest_path": resolved.task_manifest_path,
        },
    ) as progress:
        return _run_observed(config=resolved, run_root=run_root, progress=progress)


def _run_observed(
    *,
    config: StageILLMComparisonConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageILLMComparisonRunResult:
    sources = _load_sources(config)
    progress.update(
        "sources_loaded",
        task_entry_count=len(sources["task_entries"]),
        runtime_case_count=len(sources["runtime_cases"]),
    )

    task_rows, task_summary = build_task_context_comparison(sources)
    semantic_rows, semantic_summary = build_semantic_hint_comparison(sources)
    runtime_rows, runtime_summary = build_runtime_explanation_comparison(sources)
    review_rows, review_summary = build_human_review_packet(config, sources)
    condition_manifest = build_condition_manifest(
        config=config,
        task_summary=task_summary,
        semantic_summary=semantic_summary,
        runtime_summary=runtime_summary,
        review_summary=review_summary,
    )
    progress.update(
        "comparisons_built",
        label_unchanged=task_summary["label_unchanged"],
        semantic_added_query_count=semantic_summary["added_query_count"],
        llm_explained_case_count=runtime_summary["llm_explained_case_count"],
        human_review_item_count=review_summary["item_count"],
    )

    paths = _write_outputs(
        config=config,
        run_root=run_root,
        sources=sources,
        condition_manifest=condition_manifest,
        task_rows=task_rows,
        semantic_rows=semantic_rows,
        runtime_rows=runtime_rows,
        review_rows=review_rows,
        task_summary=task_summary,
        semantic_summary=semantic_summary,
        runtime_summary=runtime_summary,
        review_summary=review_summary,
    )
    progress.finish(summary_path=paths["summary_path"], report_path=paths["report_path"])
    return StageILLMComparisonRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=paths["summary_path"],
        condition_manifest_path=paths["condition_manifest_path"],
        report_path=paths["report_path"],
        midterm_summary_path=paths["midterm_summary_path"],
        status=paths["status"],
        summary=paths["summary"],
    )


def _load_sources(config: StageILLMComparisonConfig) -> dict[str, object]:
    context = load_json(config.llm_context_path)
    runtime_explanations_path = resolve_runtime_explanations_path(
        context=context,
        context_path=Path(config.llm_context_path),
    )
    return {
        "llm_context": context,
        "task_entries": load_stage_i_private_task_entries(config.task_manifest_path),
        "support_summary": load_json(config.support_summary_path),
        "runtime_schema_contract": load_json(config.runtime_schema_contract_path),
        "runtime_cases": read_csv_rows(config.runtime_case_table_path),
        "runtime_explanations": read_jsonl(runtime_explanations_path),
        "runtime_explanations_path": str(runtime_explanations_path),
        "source_paths": {
            "llm_context_path": config.llm_context_path,
            "task_manifest_path": config.task_manifest_path,
            "support_summary_path": config.support_summary_path,
            "runtime_schema_contract_path": config.runtime_schema_contract_path,
            "runtime_case_table_path": config.runtime_case_table_path,
            "runtime_explanations_path": str(runtime_explanations_path),
        },
    }


def _write_outputs(
    *,
    config: StageILLMComparisonConfig,
    run_root: Path,
    sources: Mapping[str, object],
    condition_manifest: Mapping[str, object],
    task_rows: Sequence[Mapping[str, object]],
    semantic_rows: Sequence[Mapping[str, object]],
    runtime_rows: Sequence[Mapping[str, object]],
    review_rows: Sequence[Mapping[str, object]],
    task_summary: Mapping[str, object],
    semantic_summary: Mapping[str, object],
    runtime_summary: Mapping[str, object],
    review_summary: Mapping[str, object],
) -> dict[str, object]:
    report_root = Path(config.report_root)
    midterm_root = Path(config.midterm_root)
    report_root.mkdir(parents=True, exist_ok=True)
    midterm_root.mkdir(parents=True, exist_ok=True)

    condition_path = run_root / "condition_manifest.json"
    task_path = run_root / "task_context_comparison.csv"
    semantic_path = run_root / "semantic_hint_comparison.csv"
    runtime_path = run_root / "runtime_explanation_comparison.csv"
    review_path = run_root / "human_review_packet.csv"
    claims_path = run_root / "midterm_claims_payload.json"
    summary_path = run_root / "llm_comparison_summary.json"
    report_path = report_root / f"stage-i-llm-comparison-{config.run_id}.md"
    midterm_summary_path = midterm_root / "llm-preprocessing-comparison-summary-2026-06-14.md"

    midterm_claims = build_midterm_claims_payload(
        task_summary=task_summary,
        semantic_summary=semantic_summary,
        runtime_summary=runtime_summary,
        review_summary=review_summary,
        paths={
            "summary_path": str(summary_path),
            "condition_manifest_path": str(condition_path),
            "task_context_comparison_path": str(task_path),
            "semantic_hint_comparison_path": str(semantic_path),
            "runtime_explanation_comparison_path": str(runtime_path),
            "human_review_packet_path": str(review_path),
            "report_path": str(report_path),
            "midterm_summary_path": str(midterm_summary_path),
        },
    )
    summary = {
        "run_id": config.run_id,
        "status": "success",
        "artifact_root": str(run_root),
        "summary_path": str(summary_path),
        "condition_manifest_path": str(condition_path),
        "task_context_comparison_path": str(task_path),
        "semantic_hint_comparison_path": str(semantic_path),
        "runtime_explanation_comparison_path": str(runtime_path),
        "human_review_packet_path": str(review_path),
        "midterm_claims_payload_path": str(claims_path),
        "report_path": str(report_path),
        "midterm_summary_path": str(midterm_summary_path),
        "source_paths": dict(sources["source_paths"]),
        "task_context": dict(task_summary),
        "semantic_hints": dict(semantic_summary),
        "runtime_explanations": dict(runtime_summary),
        "human_review_packet": dict(review_summary),
        "midterm_claims": midterm_claims,
        "boundary": "LLM preprocessing comparison evidence only; not manual truth or core causal proof.",
    }

    write_json(condition_path, condition_manifest)
    write_csv(task_path, task_rows)
    write_csv(semantic_path, semantic_rows)
    write_csv(runtime_path, runtime_rows)
    write_csv(review_path, review_rows)
    write_json(claims_path, midterm_claims)
    write_json(summary_path, summary)
    report_path.write_text(
        render_stage_i_llm_comparison_report(
            summary=summary,
            condition_manifest=condition_manifest,
            task_rows=task_rows,
            semantic_rows=semantic_rows,
            runtime_rows=runtime_rows,
            review_rows=review_rows,
        )
        + "\n",
        encoding="utf-8",
    )
    midterm_summary_path.write_text(
        render_stage_i_llm_midterm_summary(summary=summary, condition_manifest=condition_manifest)
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "success",
        "summary_path": str(summary_path),
        "condition_manifest_path": str(condition_path),
        "report_path": str(report_path),
        "midterm_summary_path": str(midterm_summary_path),
        "summary": summary,
    }
