"""Artifact writers for controlled Chronaris optimization."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from chronaris.evaluation.dingxin.pipelines.benchmark_data import TASK_RESPONSE
from chronaris.evaluation.fusion_stream_structure.deep_baseline_representation_io import (
    json_default,
    repo_rel,
    validate_deep_baseline_representation_frame,
)
from chronaris.evaluation.fusion_stream_structure.deep_baseline_representation_types import REPO_ROOT
from chronaris.evaluation.fusion_stream_structure.chronaris_controlled_specs import (
    CONFIRM_RUN_ID,
    OLD_E3_REVIEW_ROOT,
    OLD_FOUR_METHOD_ROOT,
    REPRESENTATION_FAMILY,
    ChronarisCandidateSpec,
    ControlledOptimizationConfig,
)


def candidate_summary_base(spec: ChronarisCandidateSpec) -> dict[str, object]:
    return {
        "candidate_id": spec.candidate_id,
        "model_name": spec.model_name,
        "changed_knobs": spec.changed_knobs,
        "representation_postprocess": spec.representation_postprocess,
    }


def write_candidate_registry(path: Path, specs: Sequence[ChronarisCandidateSpec]) -> None:
    pd.DataFrame(
        [
            {
                **asdict(spec),
                "task": TASK_RESPONSE,
                "split": "leave_one_view_out",
                "seed": 17,
                "representation_family": REPRESENTATION_FAMILY,
            }
            for spec in specs
        ]
    ).to_csv(path, index=False)


def write_candidate_metric_tables(dev_root: Path, t1_t2_rows: Sequence[Mapping[str, object]], e3_rows: Sequence[Mapping[str, object]]) -> None:
    pd.DataFrame(list(t1_t2_rows)).to_csv(dev_root / "candidate_t1_t2_metrics.csv", index=False)
    pd.DataFrame(list(e3_rows)).to_csv(dev_root / "candidate_e3_metrics.csv", index=False)


def write_pareto_selection(path: Path, selection: Mapping[str, object], old_refs: Mapping[str, object]) -> None:
    lines = [
        "# Pareto Selection",
        "",
        f"- selected_candidate_id: `{selection.get('selected_candidate_id')}`",
        f"- rule: {selection.get('selection_rule')}",
        f"- old Chronaris T2 RMSE: `{old_refs['t2']['rmse']['chronaris']}`",
        "",
        "| candidate_id | T2 RMSE | T2 improved | T1 not degraded | E3 positive signal | Pareto pass |",
        "|---|---:|---|---|---|---|",
    ]
    for row in selection.get("pareto_rows", []):
        lines.append(
            f"| {row['candidate_id']} | {_fmt(row.get('t2_rmse'))} | {row['t2_improved']} | {row['t1_not_degraded']} | {row['e3_positive_signal_count']} | {row['pareto_pass']} |"
        )
    lines.extend(["", "Dev sweep rows are not thesis confirmed metrics and are not written into the protocol snapshot."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_cases(path: Path, specs: Sequence[ChronarisCandidateSpec], smoke_passed: Sequence[ChronarisCandidateSpec], selection: Mapping[str, object]) -> None:
    passed_ids = {spec.candidate_id for spec in smoke_passed}
    lines = ["# Failure Cases", ""]
    for spec in specs:
        if spec.candidate_id not in passed_ids:
            lines.append(f"- `{spec.candidate_id}`: smoke did not pass or produced no completed fold.")
    for row in selection.get("pareto_rows", []):
        if not row["pareto_pass"]:
            reasons = []
            if not row["t2_improved"]:
                reasons.append("T2 RMSE did not improve")
            if not row["t1_not_degraded"]:
                reasons.append("T1 degraded beyond tolerance")
            if not row["e3_positive_signal"]:
                reasons.append("no target E3 metric improved")
            lines.append(f"- `{row['candidate_id']}`: {', '.join(reasons) or 'Pareto rule not met'}.")
    if len(lines) == 2:
        lines.append("- No failed candidates under the recorded Pareto filters.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_boundary(path: Path, selection: Mapping[str, object]) -> None:
    location = "supplement or appendix diagnostic" if selection.get("selected_candidate_id") else "appendix or future work only"
    lines = [
        "# Paper Boundary",
        "",
        "- E3 remains an unsupervised structure diagnostic and does not enter the main experiment table.",
        "- T1/T2 confirmed metrics remain the existing protocol snapshot unless a separate thesis confirmation is approved.",
        "- Weak labels are proxy tasks, not expert truth.",
        "- Negative or tied E3 rows are retained; this run does not package them as a winner leaderboard.",
        f"- Recommended paper location for this run: `{location}`.",
        "- Next options: expand samples, collect expert event boundaries, or redesign E3 metrics before main-text use.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_locked_outputs(
    confirm_root: Path,
    spec: ChronarisCandidateSpec,
    result: Mapping[str, object],
    selection: Mapping[str, object],
    config: ControlledOptimizationConfig,
    old_refs: Mapping[str, object],
) -> None:
    (confirm_root / "locked_candidate_config.json").write_text(
        json.dumps({**asdict(spec), "selection": selection, "confirm_epochs": config.confirm_epochs, "representation_family": REPRESENTATION_FAMILY}, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(list(result.get("metric_rows", []))).to_csv(confirm_root / "locked_candidate_t1_t2_metrics.csv", index=False)
    pd.DataFrame(list(result.get("e3_metric_rows", []))).to_csv(confirm_root / "locked_candidate_e3_metrics.csv", index=False)
    report = [
        "# Locked Confirmation Report",
        "",
        f"- locked_candidate_id: `{spec.candidate_id}`",
        f"- status: `{result.get('status')}`",
        f"- completed folds: `{result.get('completed_fold_count')}` / `{result.get('expected_fold_count')}`",
        f"- T2 RMSE: `{_metric_value(result.get('metric_rows', []), 'rmse')}` vs old Chronaris `{old_refs['t2']['rmse']['chronaris']}`",
        f"- E3 target positive signals: `{result.get('e3_positive_signal_count')}`",
        "- confirmed_metrics_changed: `false`",
        "- thesis_protocol_snapshot_modified: `false`",
        "",
        "This locked confirmation fixes the selected candidate configuration after dev selection. It is still a new controlled optimization run, not a protocol-snapshot rewrite.",
    ]
    (confirm_root / "locked_confirmation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_confirm_manifest(confirm_root, config, spec, result)


def write_no_locked_candidate(path: Path, selection: Mapping[str, object], dev_results: Mapping[str, Mapping[str, object]]) -> None:
    lines = [
        "# No Locked Candidate",
        "",
        "No candidate satisfied all Pareto gates, so locked confirmation was not run.",
        "",
        f"- selection_rule: {selection.get('selection_rule')}",
        f"- dev_candidate_count: `{len(dev_results)}`",
        "- confirmed_metrics_changed: `false`",
        "- thesis_protocol_snapshot_modified: `false`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path.parent / "evidence_manifest.json").write_text(
        json.dumps({"run_id": CONFIRM_RUN_ID, "status": "no_locked_candidate", "training_invoked": False, "confirmed_metrics_changed": False, "thesis_protocol_snapshot_modified": False, "output_files": [path.name]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_representation_outputs(
    *,
    representation_root: Path,
    config: ControlledOptimizationConfig,
    specs: Sequence[ChronarisCandidateSpec],
    embedding_rows: Sequence[Mapping[str, object]],
    checkpoint_rows: Sequence[Mapping[str, object]],
    fold_status_rows: Sequence[Mapping[str, object]],
    curve_rows: Sequence[Mapping[str, object]],
    sequence_frame: pd.DataFrame,
    runtime_device: str,
) -> None:
    embedding_frame = pd.DataFrame(list(embedding_rows))
    if not embedding_frame.empty:
        validate_deep_baseline_representation_frame(embedding_frame)
    embedding_frame.to_csv(representation_root / "chronaris_oof_embeddings_long.csv", index=False)
    pd.DataFrame(list(checkpoint_rows)).to_csv(representation_root / "checkpoint_manifest.csv", index=False)
    (representation_root / "checkpoint_manifest.json").write_text(json.dumps(list(checkpoint_rows), ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    pd.DataFrame(list(fold_status_rows)).to_csv(representation_root / "fold_status.csv", index=False)
    pd.DataFrame(list(curve_rows)).to_csv(representation_root / "training_curves.csv", index=False)
    summary = {
        "run_id": config.representation_run_id,
        "status": "completed" if not embedding_frame.empty else "partial",
        "training_invoked": True,
        "confirmed_metrics_changed": False,
        "thesis_protocol_snapshot_modified": False,
        "old_deep_baseline_modified": False,
        "old_e3_validation_modified": False,
        "representation_family": REPRESENTATION_FAMILY,
        "task_name": TASK_RESPONSE,
        "split_strategy": "leave_one_view_out",
        "seed": int(config.seed),
        "runtime_device": runtime_device,
        "sequence_sample_count": int(len(sequence_frame)),
        "candidate_count": int(len(specs)),
        "embedding_row_count": int(len(embedding_frame)),
        "candidate_counts": dict(embedding_frame["candidate_id"].value_counts().sort_index()) if not embedding_frame.empty else {},
        "feature_dimensions": _feature_dimensions_by_candidate(embedding_frame),
        "source_manifest_paths": {
            "e_run_manifest_path": config.e_run_manifest_path,
            "f_run_manifest_path": config.f_run_manifest_path,
        },
    }
    (representation_root / "chronaris_oof_embeddings_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    (representation_root / "representation_manifest.json").write_text(
        json.dumps({**summary, "representation_table_path": repo_rel(representation_root / "chronaris_oof_embeddings_long.csv"), "oof_protocol": "held-out leave-one-view-out inference only"}, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    (representation_root / "evidence_manifest.json").write_text(
        json.dumps({**summary, "run_type": "chronaris_oof_representation_export", "branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"), "commit": _git_value("rev-parse", "HEAD"), "created_at_utc": _utc_now(), "output_files": ["chronaris_oof_embeddings_long.csv", "chronaris_oof_embeddings_summary.json", "checkpoint_manifest.csv", "checkpoint_manifest.json", "representation_manifest.json", "fold_status.csv", "training_curves.csv"]}, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_dev_manifest(dev_root: Path, config: ControlledOptimizationConfig, specs: Sequence[ChronarisCandidateSpec], summary: Mapping[str, object]) -> None:
    manifest = {
        "run_id": config.dev_run_id,
        "run_type": "chronaris_controlled_optimization_dev",
        "branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git_value("rev-parse", "HEAD"),
        "created_at_utc": _utc_now(),
        "training_invoked": True,
        "confirmed_metrics_changed": False,
        "thesis_protocol_snapshot_modified": False,
        "old_deep_baseline_modified": False,
        "old_e3_validation_modified": False,
        "t3_artifact_deleted": False,
        "candidate_count": len(specs),
        "summary": dict(summary),
        "source_artifacts": {
            "old_e3_review": OLD_E3_REVIEW_ROOT,
            "old_four_method_e3_validation": OLD_FOUR_METHOD_ROOT,
            "deep_baseline_root": config.deep_baseline_root,
            "thesis_protocol_root": config.thesis_protocol_root,
        },
        "output_files": [
            "candidate_registry.csv",
            "candidate_t1_t2_metrics.csv",
            "candidate_e3_metrics.csv",
            "candidate_summary.json",
            "pareto_selection.md",
            "failure_cases.md",
            "paper_boundary.md",
            "evidence_manifest.json",
        ],
    }
    (dev_root / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_confirm_manifest(confirm_root: Path, config: ControlledOptimizationConfig, spec: ChronarisCandidateSpec, result: Mapping[str, object]) -> None:
    manifest = {
        "run_id": config.confirm_run_id,
        "run_type": "chronaris_controlled_optimization_confirm",
        "branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git_value("rev-parse", "HEAD"),
        "created_at_utc": _utc_now(),
        "training_invoked": True,
        "confirmed_metrics_changed": False,
        "thesis_protocol_snapshot_modified": False,
        "old_deep_baseline_modified": False,
        "old_e3_validation_modified": False,
        "locked_candidate_id": spec.candidate_id,
        "summary": {key: value for key, value in result.items() if key not in {"embedding_rows", "checkpoint_rows", "fold_status_rows", "curve_rows", "metric_rows", "e3_metric_rows"}},
        "output_files": [
            "locked_candidate_config.json",
            "locked_candidate_t1_t2_metrics.csv",
            "locked_candidate_e3_metrics.csv",
            "locked_confirmation_report.md",
            "evidence_manifest.json",
        ],
    }
    (confirm_root / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_blocked_outputs(
    config: ControlledOptimizationConfig,
    dev_root: Path,
    confirm_root: Path,
    representation_root: Path,
    specs: Sequence[ChronarisCandidateSpec],
    runtime_device: str,
) -> dict[str, object]:
    write_candidate_registry(dev_root / "candidate_registry.csv", specs)
    for path in (dev_root / "candidate_t1_t2_metrics.csv", dev_root / "candidate_e3_metrics.csv"):
        pd.DataFrame().to_csv(path, index=False)
    reason = f"cuda_required_but_resolved_{runtime_device}"
    summary = {"status": "blocked", "blocked_reason": reason, "training_invoked": False, "candidate_count": len(specs)}
    (dev_root / "candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_paper_boundary(dev_root / "paper_boundary.md", {"selected_candidate_id": None})
    (dev_root / "pareto_selection.md").write_text(f"# Pareto Selection\n\nBlocked: `{reason}`.\n", encoding="utf-8")
    (dev_root / "failure_cases.md").write_text(f"# Failure Cases\n\nBlocked: `{reason}`.\n", encoding="utf-8")
    write_dev_manifest(dev_root, config, specs, summary)
    write_no_locked_candidate(confirm_root / "no_locked_candidate.md", {"selection_rule": "blocked"}, {})
    (representation_root / "chronaris_oof_embeddings_long.csv").write_text("", encoding="utf-8")
    (representation_root / "chronaris_oof_embeddings_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    pairs = []
    for column in frame.columns:
        text = str(column)
        if text.startswith("fusion_feature_"):
            try:
                pairs.append((int(text.rsplit("_", 1)[1]), text))
            except ValueError:
                continue
    return [column for _idx, column in sorted(pairs)]


def _feature_dimensions_by_candidate(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    dimensions: dict[str, int] = {}
    for candidate, group in frame.groupby("candidate_id"):
        columns = [
            column
            for column in _feature_columns(group)
            if not group[column].isna().all()
        ]
        dimensions[str(candidate)] = len(columns)
    return dimensions


def _metric_value(rows: Sequence[Mapping[str, object]], metric: str) -> float | None:
    for row in rows:
        if row.get("metric") == metric and row.get("value") is not None:
            return float(row["value"])
    return None


def _fmt(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True)
    except Exception as exc:  # pragma: no cover
        return f"unavailable:{exc!r}"
    return result.stdout.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "candidate_summary_base",
    "write_blocked_outputs",
    "write_candidate_metric_tables",
    "write_candidate_registry",
    "write_failure_cases",
    "write_locked_outputs",
    "write_no_locked_candidate",
    "write_paper_boundary",
    "write_pareto_selection",
    "write_representation_outputs",
    "write_dev_manifest",
]
