"""P38 thesis protocol freeze builder for Stage I evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_P30_ROOT = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_i_private_thirdparty_comparison"
    / "20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1"
)
DEFAULT_P31_ROOT = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_i_public_fusion_ablation"
    / "20260702T-stage-i-public-fusion-ablation-gpuopt-r1"
)
DEFAULT_P32_ROOT = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_i_cross_evidence_matrix"
    / "20260702T-stage-i-cross-evidence-matrix-gpuopt-r1"
)
DEFAULT_P34_ROOT = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_i_task_heads_optimization"
    / "20260702T-stage-i-task-heads-optimization-r3-confirm20"
)
DEFAULT_P35_ROOT = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_i_stream_role_fusion"
    / "20260702T-stage-i-stream-role-fusion-r4-v3-confirm20"
)
DEFAULT_P36_ROOT = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_i_optimized_reevaluation"
    / "20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20"
)
DEFAULT_P36_SUMMARY_ROOT = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_i_optimized_model_summary"
    / "20260702T-stage-i-optimized-model-summary-r4-v3-confirm20"
)
DEFAULT_P37_ROOT = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_i_optimized_final_polish"
    / "20260702T-stage-i-optimized-final-polish-r1"
)
DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_thesis_protocol"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"
PYTHON = "/home/wangminan/env/anaconda3/envs/chronaris/bin/python"

MATRIX_COLUMNS = [
    "evidence_quadrant",
    "dataset_role",
    "task",
    "split_protocol",
    "model_or_component",
    "metric",
    "value",
    "baseline",
    "delta_positive_is_better",
    "seed_count",
    "artifact_path",
    "claim_boundary",
]
EXTRA_MATRIX_COLUMNS = [
    "source_stage",
    "source_file",
    "dataset_id",
    "status",
    "delta_convention",
    "figure_path",
]


@dataclass(frozen=True, slots=True)
class StageIThesisProtocolConfig:
    run_id: str
    p30_root: str = str(DEFAULT_P30_ROOT)
    p31_root: str = str(DEFAULT_P31_ROOT)
    p32_root: str = str(DEFAULT_P32_ROOT)
    p34_root: str = str(DEFAULT_P34_ROOT)
    p35_root: str = str(DEFAULT_P35_ROOT)
    p36_root: str = str(DEFAULT_P36_ROOT)
    p36_summary_root: str = str(DEFAULT_P36_SUMMARY_ROOT)
    p37_root: str = str(DEFAULT_P37_ROOT)
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT


@dataclass(frozen=True, slots=True)
class StageIThesisProtocolResult:
    run_id: str
    artifact_root: str
    summary_path: str
    evidence_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


def build_stage_i_thesis_protocol(
    config: StageIThesisProtocolConfig,
) -> StageIThesisProtocolResult:
    """Build the P38 thesis protocol package from existing Stage I artifacts."""

    run_root = _resolve_path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    roots = {
        "P30": _resolve_path(config.p30_root),
        "P31": _resolve_path(config.p31_root),
        "P32": _resolve_path(config.p32_root),
        "P34": _resolve_path(config.p34_root),
        "P35": _resolve_path(config.p35_root),
        "P36": _resolve_path(config.p36_root),
        "P36_summary": _resolve_path(config.p36_summary_root),
        "P37": _resolve_path(config.p37_root),
    }

    registry = _experiment_registry(roots)
    registry_path = run_root / "experiment_registry.csv"
    registry.to_csv(registry_path, index=False)

    matrix = _result_matrix(roots)
    if matrix.empty:
        raise ValueError("P38 thesis protocol matrix has no rows.")
    matrix = matrix[MATRIX_COLUMNS + EXTRA_MATRIX_COLUMNS]
    matrix_path = run_root / "result_matrix_long.csv"
    matrix.to_csv(matrix_path, index=False)

    summary_table = _result_summary(matrix)
    summary_table_path = run_root / "result_matrix_summary.csv"
    summary_table.to_csv(summary_table_path, index=False)

    claim_table = _claim_boundary_table(roots)
    claim_table_path = run_root / "claim_boundary_table.csv"
    claim_table.to_csv(claim_table_path, index=False)

    resume_path = run_root / "resume_command.txt"
    resume_path.write_text(_resume_command(config, roots) + "\n", encoding="utf-8")

    run_log_path = run_root / "run.log"
    generated_at = _utc_now()
    run_log_path.write_text(
        "\n".join(
            [
                f"{generated_at} INFO stage=P38 run_id={config.run_id} status=completed",
                "Built thesis protocol package from existing P30/P31/P32/P34/P35/P36/P37 artifacts.",
                "No training, baseline rerun, artifact deletion, or git history rewrite was performed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary_path = run_root / "thesis_protocol_summary.json"
    manifest_path = run_root / "evidence_manifest.json"
    report_path = _resolve_path(config.report_root) / f"stage-i-thesis-protocol-{config.run_id}.md"
    progress_path = run_root / "progress.json"
    source_roots = {stage: str(root) for stage, root in roots.items()}
    summary = {
        "run_id": config.run_id,
        "stage": "P38-thesis-protocol",
        "status": "completed",
        "generated_at_utc": generated_at,
        "artifact_root": str(run_root),
        "experiment_registry_csv": str(registry_path),
        "result_matrix_long_csv": str(matrix_path),
        "result_matrix_summary_csv": str(summary_table_path),
        "claim_boundary_table_csv": str(claim_table_path),
        "resume_command_txt": str(resume_path),
        "run_log": str(run_log_path),
        "report_path": str(report_path),
        "evidence_manifest_path": str(manifest_path),
        "progress_path": str(progress_path),
        "source_roots": source_roots,
        "matrix_rows": int(matrix.shape[0]),
        "registry_rows": int(registry.shape[0]),
        "claim_boundary_rows": int(claim_table.shape[0]),
        "quadrant_counts": matrix["evidence_quadrant"].value_counts().to_dict(),
        "source_stage_counts": matrix["source_stage"].value_counts().to_dict(),
        "field_contract": MATRIX_COLUMNS,
        "protocol_boundary": (
            "P38 is a read-only protocol freeze over existing P30/P31/P32/P34/P35/P36/P37 artifacts; "
            "public rows remain context-proxy evidence, private rows remain proxy/weak-label evidence, "
            "and P38 does not replace expert labels or rerun experiments."
        ),
    }
    _write_json(summary_path, summary)
    _write_json(manifest_path, {**summary, "summary_path": str(summary_path)})
    _write_json(
        progress_path,
        {
            "run_id": config.run_id,
            "stage": "P38-thesis-protocol",
            "status": "completed",
            "completed": True,
            "generated_at_utc": generated_at,
            "artifact_root": str(run_root),
        },
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(summary, registry, matrix, summary_table, claim_table) + "\n", encoding="utf-8")
    return StageIThesisProtocolResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        evidence_manifest_path=str(manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def _experiment_registry(roots: Mapping[str, Path]) -> pd.DataFrame:
    rows = [
        _registry_row(
            stage="P30",
            evidence_quadrant="private_model_comparison",
            dataset_role="private_real_dual_stream",
            task_scope="T1/T2/T3",
            split_protocol="leave_one_view_out + leave_one_sortie_out",
            artifact_root=roots["P30"],
            primary_result_path=roots["P30"] / "model_comparison_long.csv",
            report_path=REPO_ROOT
            / "docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md",
            claim_boundary="Private Stage H proxy-task third-party comparison; not expert truth.",
        ),
        _registry_row(
            stage="P31",
            evidence_quadrant="public_component_ablation",
            dataset_role="public_context_proxy",
            task_scope="NASA/UAB",
            split_protocol="LOSO / fixed public split",
            artifact_root=roots["P31"],
            primary_result_path=roots["P31"] / "ablation_summary.csv",
            report_path=REPO_ROOT
            / "docs/artifacts/stage_i/stage-i-public-fusion-ablation-20260702T-stage-i-public-fusion-ablation-gpuopt-r1.md",
            claim_boundary="Public adapter context-proxy component ablation; not real aircraft-bus validation.",
        ),
        _registry_row(
            stage="P32",
            evidence_quadrant="all_four_quadrants",
            dataset_role="mixed_private_public",
            task_scope="P30/P31/P24/P27",
            split_protocol="source protocols preserved",
            artifact_root=roots["P32"],
            primary_result_path=roots["P32"] / "cross_evidence_matrix.csv",
            report_path=REPO_ROOT
            / "docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md",
            claim_boundary="Cross-evidence routing matrix; do not merge private/public/proxy metrics into one leaderboard.",
        ),
        _registry_row(
            stage="P34",
            evidence_quadrant="private_component_ablation",
            dataset_role="private_real_dual_stream",
            task_scope="T1/T2/T3",
            split_protocol="leave_one_view_out + leave_one_sortie_out",
            artifact_root=roots["P34"],
            primary_result_path=roots["P34"] / "task_head_metrics_long.csv",
            report_path=REPO_ROOT
            / "docs/artifacts/stage_i/stage-i-task-aware-heads-20260702T-stage-i-task-heads-optimization-r3-confirm20.md",
            claim_boundary="CUDA 20-epoch task-head confirm; T3 remains mixed and must not be overstated.",
        ),
        _registry_row(
            stage="P35",
            evidence_quadrant="private_component_ablation + public_component_ablation",
            dataset_role="mixed_private_public_route",
            task_scope="T1/T2/T3 + NASA/UAB",
            split_protocol="private leave-one-* + public LOSO/fixed split",
            artifact_root=roots["P35"],
            primary_result_path=roots["P35"] / "private_metrics.csv",
            report_path=REPO_ROOT
            / "docs/artifacts/stage_i/stage-i-stream-role-aware-fusion-20260702T-stage-i-stream-role-fusion-r4-v3-confirm20.md",
            claim_boundary="Stream-role v3 confirm separates private real vehicle streams from public context proxy streams.",
        ),
        _registry_row(
            stage="P36",
            evidence_quadrant="optimized_summary",
            dataset_role="mixed_private_public",
            task_scope="P30/P31/P32/P34/P35 aggregate",
            split_protocol="source protocols preserved",
            artifact_root=roots["P36"],
            primary_result_path=roots["P36"] / "optimized_cross_evidence_matrix.csv",
            report_path=REPO_ROOT
            / "docs/artifacts/stage_i/stage-i-optimized-chronaris-reevaluation-20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20.md",
            claim_boundary="Aggregation over fixed references; does not overwrite P30/P31/P32 confirmed results.",
        ),
        _registry_row(
            stage="P36_summary",
            evidence_quadrant="optimized_summary",
            dataset_role="mixed_private_public",
            task_scope="paper-facing optimized model summary",
            split_protocol="source protocols preserved",
            artifact_root=roots["P36_summary"],
            primary_result_path=roots["P36_summary"] / "optimized_model_summary.csv",
            report_path=REPO_ROOT
            / "docs/artifacts/stage_i/stage-i-optimized-model-summary-20260702T-stage-i-optimized-model-summary-r4-v3-confirm20.md",
            claim_boundary="Paper-facing summary table and claim boundaries over fixed source artifacts.",
        ),
        _registry_row(
            stage="P37",
            evidence_quadrant="private_model_comparison + public_model_comparison",
            dataset_role="mixed_private_public",
            task_scope="T1/T3 + NASA/UAB public route",
            split_protocol="fixed P30/P31/P34/P35/P36 references",
            artifact_root=roots["P37"],
            primary_result_path=roots["P37"] / "optimized_final_polish_summary.json",
            report_path=REPO_ROOT
            / "docs/artifacts/stage_i/stage-i-optimized-final-polish-20260702T-stage-i-optimized-final-polish-r1.md",
            claim_boundary="Final polish accepts T1/public route only; T3 rejected and public remains context proxy.",
        ),
    ]
    return pd.DataFrame(rows)


def _registry_row(
    *,
    stage: str,
    evidence_quadrant: str,
    dataset_role: str,
    task_scope: str,
    split_protocol: str,
    artifact_root: Path,
    primary_result_path: Path,
    report_path: Path,
    claim_boundary: str,
) -> dict[str, object]:
    status = _status_from_stage(stage, artifact_root)
    return {
        "stage": stage,
        "status": status,
        "evidence_quadrant": evidence_quadrant,
        "dataset_role": dataset_role,
        "task_scope": task_scope,
        "split_protocol": split_protocol,
        "artifact_root": str(artifact_root),
        "primary_result_path": str(primary_result_path),
        "report_path": str(report_path),
        "claim_boundary": claim_boundary,
        "exists": artifact_root.exists() and primary_result_path.exists(),
    }


def _result_matrix(roots: Mapping[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(_p32_rows(roots["P32"]))
    rows.extend(_p34_rows(roots["P34"]))
    rows.extend(_p35_private_rows(roots["P35"]))
    rows.extend(_p35_public_rows(roots["P35"]))
    rows.extend(_p36_summary_rows(roots["P36_summary"]))
    rows.extend(_p37_private_rows(roots["P37"]))
    rows.extend(_p37_public_rows(roots["P37"]))
    if not rows:
        return pd.DataFrame(columns=MATRIX_COLUMNS + EXTRA_MATRIX_COLUMNS)
    return pd.DataFrame(rows)


def _p32_rows(root: Path) -> list[dict[str, object]]:
    path = root / "cross_evidence_matrix.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows = []
    for record in frame.to_dict(orient="records"):
        quadrant = _p38_quadrant(record.get("evidence_quadrant"))
        rows.append(
            _matrix_row(
                evidence_quadrant=quadrant,
                dataset_role=record.get("dataset_role"),
                task=record.get("task_group"),
                split_protocol=_split_protocol(record.get("protocol"), record.get("dataset_id")),
                model_or_component=record.get("model_or_component"),
                metric=record.get("metric"),
                value=record.get("value"),
                baseline=record.get("baseline"),
                delta_positive_is_better=record.get("delta_abs"),
                seed_count=np.nan,
                artifact_path=record.get("artifact_path") or str(path),
                claim_boundary=record.get("wording_boundary"),
                source_stage=_p32_source_stage(record.get("evidence_quadrant")),
                source_file=str(path),
                dataset_id=record.get("dataset_id"),
                status="completed",
                delta_convention="source positive means source reference/full/Chronaris claim is better",
                figure_path=record.get("figure_path"),
            )
        )
    return rows


def _p38_quadrant(value: object) -> str:
    text = "" if _is_missing(value) else str(value)
    if text == "private_thirdparty_comparison":
        return "private_model_comparison"
    return text


def _p32_source_stage(value: object) -> str:
    text = "" if _is_missing(value) else str(value)
    if text == "private_thirdparty_comparison":
        return "P30_via_P32"
    if text == "private_component_ablation":
        return "P24_via_P32"
    if text == "public_model_comparison":
        return "P27_via_P32"
    if text == "public_component_ablation":
        return "P31_via_P32"
    return "P32"


def _p34_rows(root: Path) -> list[dict[str, object]]:
    path = root / "task_head_metrics_long.csv"
    if not path.exists():
        return []
    delta_path = root / "improvement_vs_p30.csv"
    deltas = _delta_lookup(delta_path, ("task_name", "split_strategy", "metric"))
    rows = []
    for record in pd.read_csv(path).to_dict(orient="records"):
        key = (record.get("task_name"), record.get("split_strategy"), record.get("metric"))
        delta = deltas.get(key, {}).get("delta_abs_positive_is_better") if record.get("model_name") == "chronaris_v2_task_heads" else np.nan
        baseline = "P30 chronaris_full" if record.get("model_name") == "chronaris_v2_task_heads" else "chronaris_v2_task_heads"
        rows.append(
            _matrix_row(
                evidence_quadrant="private_component_ablation",
                dataset_role="private_real_dual_stream",
                task=record.get("task_name"),
                split_protocol=record.get("split_strategy"),
                model_or_component=record.get("model_name"),
                metric=record.get("metric"),
                value=record.get("value_mean"),
                baseline=baseline,
                delta_positive_is_better=delta,
                seed_count=record.get("seed_count"),
                artifact_path=str(path),
                claim_boundary="P34 is a CUDA 20-epoch private task-head confirm; T3 remains mixed.",
                source_stage="P34",
                source_file=str(path),
                dataset_id="private_stage_h",
                status=_status_from_json(root / "task_head_optimization_summary.json"),
                delta_convention="positive means P34 improves over P30 when available",
                figure_path=str(root / "fig_p34_delta_vs_p30_heatmap.png"),
            )
        )
    return rows


def _p35_private_rows(root: Path) -> list[dict[str, object]]:
    path = root / "private_metrics.csv"
    if not path.exists():
        return []
    rows = []
    for record in pd.read_csv(path).to_dict(orient="records"):
        rows.append(
            _matrix_row(
                evidence_quadrant="private_component_ablation",
                dataset_role="private_real_dual_stream",
                task=record.get("task_name"),
                split_protocol=record.get("split_strategy"),
                model_or_component=record.get("model_name"),
                metric=record.get("metric"),
                value=record.get("value_mean"),
                baseline="P34_reference" if str(record.get("source_stage", "")).startswith("P35") else "",
                delta_positive_is_better=np.nan,
                seed_count=record.get("seed_count"),
                artifact_path=str(path),
                claim_boundary="P35 private rows are stream-role v3 confirm or fixed P34 references.",
                source_stage="P35",
                source_file=str(path),
                dataset_id="private_stage_h",
                status=record.get("p35_status", _status_from_json(root / "stream_role_fusion_summary.json")),
                delta_convention="metric row; use P35 comparison CSVs for explicit deltas",
                figure_path=str(root / "fig_p35_private_task_delta.png"),
            )
        )
    return rows


def _p35_public_rows(root: Path) -> list[dict[str, object]]:
    path = root / "public_metrics.csv"
    if not path.exists():
        return []
    rows = []
    for record in pd.read_csv(path).to_dict(orient="records"):
        rows.append(
            _matrix_row(
                evidence_quadrant="public_component_ablation",
                dataset_role="public_context_proxy",
                task=record.get("task_group"),
                split_protocol="LOSO/fixed_public_split",
                model_or_component=record.get("variant_id"),
                metric=record.get("metric"),
                value=record.get("value_mean"),
                baseline="P31 full" if not _is_missing(record.get("full_value_mean")) else "",
                delta_positive_is_better=record.get("delta_abs_mean"),
                seed_count=record.get("value_count"),
                artifact_path=str(path),
                claim_boundary="P35 public rows are route checks on public context-proxy data.",
                source_stage="P35",
                source_file=str(path),
                dataset_id=record.get("dataset_id"),
                status=record.get("p35_status", _status_from_json(root / "stream_role_fusion_summary.json")),
                delta_convention="source positive means full/P31 or route reference is better where delta exists",
                figure_path=str(root / "fig_p35_public_ablation_comparison.png"),
            )
        )
    return rows


def _p36_summary_rows(root: Path) -> list[dict[str, object]]:
    path = root / "key_metric_summary.csv"
    if not path.exists():
        return []
    rows = []
    for record in pd.read_csv(path).to_dict(orient="records"):
        scope = str(record.get("scope", ""))
        if scope == "private_stage_h":
            quadrant = "private_component_ablation"
            dataset_role = "private_real_dual_stream"
        elif scope == "public_context_proxy":
            quadrant = "public_component_ablation"
            dataset_role = "public_context_proxy"
        else:
            quadrant = "private_model_comparison"
            dataset_role = "private_proxy"
        rows.append(
            _matrix_row(
                evidence_quadrant=quadrant,
                dataset_role=dataset_role,
                task=record.get("dataset_or_task"),
                split_protocol=record.get("split_or_group"),
                model_or_component=record.get("optimized"),
                metric=record.get("metric"),
                value=record.get("optimized_value"),
                baseline=record.get("reference"),
                delta_positive_is_better=record.get("delta_positive_is_better"),
                seed_count=np.nan,
                artifact_path=str(path),
                claim_boundary=record.get("boundary"),
                source_stage="P36_summary",
                source_file=str(path),
                dataset_id=scope,
                status=record.get("status"),
                delta_convention="positive means optimized/reference status according to P36 summary",
                figure_path=str(root / "fig_model_summary_metric_delta.png"),
            )
        )
    return rows


def _p37_private_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    t1_path = root / "t1_calibration_metrics.csv"
    t3_path = root / "t3_final_polish_metrics.csv"
    p34_delta_path = root / "p37_delta_vs_p34.csv"
    p34_deltas = _delta_lookup(p34_delta_path, ("task_name", "split_strategy", "metric"))
    for path in (t1_path, t3_path):
        if not path.exists():
            continue
        for record in pd.read_csv(path).to_dict(orient="records"):
            key = (record.get("task_name"), record.get("split_strategy"), record.get("metric"))
            delta = p34_deltas.get(key, {}).get("delta_positive_is_better")
            task = str(record.get("task_name", ""))
            rows.append(
                _matrix_row(
                    evidence_quadrant="private_model_comparison",
                    dataset_role="private_real_dual_stream",
                    task=task,
                    split_protocol=record.get("split_strategy"),
                    model_or_component=record.get("model_name"),
                    metric=record.get("metric"),
                    value=record.get("value_mean"),
                    baseline="P34 confirmed retrieval/task-head reference",
                    delta_positive_is_better=delta,
                    seed_count=record.get("seed_count"),
                    artifact_path=str(path),
                    claim_boundary=(
                        "P37 T3 is rejected and keeps P34 retrieval."
                        if task.startswith("T3")
                        else "P37 T1 calibration is accepted only under fixed P34 reference."
                    ),
                    source_stage="P37",
                    source_file=str(path),
                    dataset_id="private_stage_h",
                    status="rejected_t3" if task.startswith("T3") else "accepted_t1",
                    delta_convention="positive means P37 improves over P34; zero T3 is rejected",
                    figure_path=str(root / ("fig_p37_t3_delta_vs_p34.png" if task.startswith("T3") else "fig_p37_t1_macro_f1_leaderboard.png")),
                )
            )
    return rows


def _p37_public_rows(root: Path) -> list[dict[str, object]]:
    path = root / "public_route_calibration_metrics.csv"
    delta_path = root / "p37_delta_vs_p35.csv"
    if not path.exists():
        return []
    deltas = _delta_lookup(delta_path, ("dataset_id", "metric", "p37_variant"))
    rows = []
    for record in pd.read_csv(path).to_dict(orient="records"):
        key = (record.get("dataset_id"), record.get("primary_metric"), record.get("variant_id"))
        delta_record = deltas.get(key, {})
        rows.append(
            _matrix_row(
                evidence_quadrant="public_model_comparison",
                dataset_role="public_context_proxy",
                task=record.get("dataset_id"),
                split_protocol="fixed_public_split",
                model_or_component=record.get("variant_id"),
                metric=record.get("primary_metric"),
                value=record.get("selection_score"),
                baseline=delta_record.get("reference", "P35 v3_stream_role"),
                delta_positive_is_better=delta_record.get("delta_positive_is_better"),
                seed_count=1,
                artifact_path=str(path),
                claim_boundary="P37 public route calibration is accepted, but still public context-proxy evidence.",
                source_stage="P37",
                source_file=str(path),
                dataset_id=record.get("dataset_id"),
                status="accepted_public_route",
                delta_convention="positive means P37 improves over P35 reference",
                figure_path=str(root / "fig_p37_public_route_delta_heatmap.png"),
            )
        )
    return rows


def _matrix_row(**fields: object) -> dict[str, object]:
    row = {
        "evidence_quadrant": "",
        "dataset_role": "",
        "task": "",
        "split_protocol": "",
        "model_or_component": "",
        "metric": "",
        "value": np.nan,
        "baseline": "",
        "delta_positive_is_better": np.nan,
        "seed_count": np.nan,
        "artifact_path": "",
        "claim_boundary": "",
        "source_stage": "",
        "source_file": "",
        "dataset_id": "",
        "status": "",
        "delta_convention": "",
        "figure_path": "",
    }
    row.update(fields)
    return row


def _result_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    frame = matrix.copy()
    frame["value_numeric"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["delta_numeric"] = pd.to_numeric(frame["delta_positive_is_better"], errors="coerce")
    grouped = (
        frame.groupby(["evidence_quadrant", "dataset_role", "task", "split_protocol", "metric"], dropna=False)
        .agg(
            row_count=("metric", "count"),
            numeric_row_count=("value_numeric", lambda values: int(values.notna().sum())),
            mean_value=("value_numeric", "mean"),
            best_value=("value_numeric", "max"),
            mean_delta_positive_is_better=("delta_numeric", "mean"),
            improved_row_count=("delta_numeric", lambda values: int((values > 0).sum())),
            non_improved_row_count=("delta_numeric", lambda values: int((values <= 0).sum())),
            source_stages=("source_stage", _join_unique),
            source_files=("source_file", _join_unique),
            claim_boundaries=("claim_boundary", _join_unique),
        )
        .reset_index()
    )
    return grouped


def _claim_boundary_table(roots: Mapping[str, Path]) -> pd.DataFrame:
    rows = [
        {
            "claim_id": "private_real_dual_stream_scope",
            "evidence_quadrant": "private_model_comparison",
            "dataset_role": "private_real_dual_stream",
            "allowed_claim": "Use P30/P34/P35/P37 private rows as T1/T2/T3 proxy-task evidence on existing Stage H dual-stream samples.",
            "required_boundary": "Do not describe T1/T2/T3 as expert-label ground truth or claim new Dingxin data was obtained.",
            "source_stage": "P30/P34/P35/P37",
            "artifact_path": str(roots["P37"] / "optimized_final_polish_summary.json"),
        },
        {
            "claim_id": "private_component_ablation_scope",
            "evidence_quadrant": "private_component_ablation",
            "dataset_role": "private_real_dual_stream",
            "allowed_claim": "Use leakage-safe and optimized component rows to explain task heads, residual heads, lag gates and stream-role routes.",
            "required_boundary": "Report mixed or failed rows, especially T3; do not write a blanket Chronaris superiority claim.",
            "source_stage": "P32/P34/P35/P36",
            "artifact_path": str(roots["P36_summary"] / "claim_boundary_summary.csv"),
        },
        {
            "claim_id": "public_context_proxy_scope",
            "evidence_quadrant": "public_model_comparison",
            "dataset_role": "public_context_proxy",
            "allowed_claim": "Use NASA/UAB rows as public adapter, calibration and context-proxy generalization evidence.",
            "required_boundary": "Public second stream is context proxy, not private aircraft-bus telemetry.",
            "source_stage": "P31/P35/P37",
            "artifact_path": str(roots["P37"] / "public_route_calibration_metrics.csv"),
        },
        {
            "claim_id": "public_component_ablation_scope",
            "evidence_quadrant": "public_component_ablation",
            "dataset_role": "public_context_proxy",
            "allowed_claim": "Use public ablations to discuss route sensitivity, context gate behavior and NASA/UAB calibration.",
            "required_boundary": "Do not let public component wins override private dual-stream limitations.",
            "source_stage": "P31/P35/P36",
            "artifact_path": str(roots["P31"] / "ablation_summary.csv"),
        },
        {
            "claim_id": "p37_final_polish_scope",
            "evidence_quadrant": "private_model_comparison + public_model_comparison",
            "dataset_role": "mixed_private_public",
            "allowed_claim": "P37 accepts T1 calibration and public route calibration under fixed references.",
            "required_boundary": "P37 T3 is rejected and remains P34 confirmed retrieval; P37 does not rerun or overwrite P30/P31/P34/P35/P36.",
            "source_stage": "P37",
            "artifact_path": str(roots["P37"] / "rejected_candidate_summary.json"),
        },
        {
            "claim_id": "synthetic_future_scope",
            "evidence_quadrant": "future_synthetic_stress_test",
            "dataset_role": "synthetic_stress_test",
            "allowed_claim": "Future P39 synthetic rows may validate mechanisms under simulation oracle labels.",
            "required_boundary": "Synthetic rows must remain appendix stress tests and cannot replace real data or expert truth.",
            "source_stage": "P39-future",
            "artifact_path": "",
        },
        {
            "claim_id": "llm_future_scope",
            "evidence_quadrant": "preprocessing_context",
            "dataset_role": "llm_semantic_context",
            "allowed_claim": "DeepSeek v4-pro outputs may supply preprocessing context, hints, explanations and review packets.",
            "required_boundary": "LLM output is not expert evaluation data and must not directly become truth labels.",
            "source_stage": "P20/P21/P39-future",
            "artifact_path": str(
                REPO_ROOT
                / "docs/artifacts/assets/stage_i_llm_comparison/20260614T-stage-i-p21-llm-comparison-r1/llm_comparison_summary.json"
            ),
        },
    ]
    return pd.DataFrame(rows)


def _render_report(
    summary: Mapping[str, object],
    registry: pd.DataFrame,
    matrix: pd.DataFrame,
    summary_table: pd.DataFrame,
    claim_table: pd.DataFrame,
) -> str:
    counts = pd.DataFrame(
        sorted(summary["quadrant_counts"].items()),
        columns=["evidence_quadrant", "row_count"],
    )
    source_counts = pd.DataFrame(
        sorted(summary["source_stage_counts"].items()),
        columns=["source_stage", "row_count"],
    )
    lines = [
        f"# Stage I Thesis Protocol Freeze - {summary['run_id']}",
        "",
        "## 摘要",
        "",
        "P38 已把 P30/P31/P32/P34/P35/P36/P37 的既有证据冻结为论文协议矩阵。"
        "本轮只读取已有 artifact，不重跑训练、不删除 artifact、不改写 git history。",
        "",
        f"- artifact root: `{summary['artifact_root']}`",
        f"- experiment registry: `{summary['experiment_registry_csv']}`",
        f"- result matrix long: `{summary['result_matrix_long_csv']}`",
        f"- result matrix summary: `{summary['result_matrix_summary_csv']}`",
        f"- claim boundary table: `{summary['claim_boundary_table_csv']}`",
        f"- evidence manifest: `{summary['evidence_manifest_path']}`",
        "",
        "## 行覆盖",
        "",
        _markdown_table(counts),
        "",
        "## 来源阶段覆盖",
        "",
        _markdown_table(source_counts),
        "",
        "## 实验注册表预览",
        "",
        _markdown_table(registry[["stage", "status", "evidence_quadrant", "dataset_role", "primary_result_path", "claim_boundary"]]),
        "",
        "## 矩阵字段合同",
        "",
        "P38 论文长表至少包含以下字段：",
        "",
        *[f"- `{column}`" for column in MATRIX_COLUMNS],
        "",
        "额外保留 `source_stage/source_file/dataset_id/status/delta_convention/figure_path`，用于从论文图表反查原始 artifact。",
        "",
        "## 结果矩阵预览",
        "",
        _markdown_table(matrix.head(20)),
        "",
        "## Summary 预览",
        "",
        _markdown_table(summary_table.head(20)),
        "",
        "## Claim Boundary",
        "",
        _markdown_table(claim_table),
        "",
        "## 复现与边界",
        "",
        f"- resume command: `{summary['resume_command_txt']}`",
        f"- run log: `{summary['run_log']}`",
        f"- protocol boundary: {summary['protocol_boundary']}",
        "- 后续 P41/P43 论文图表应优先从本 registry/matrix 反查到原始 source artifact。",
    ]
    return "\n".join(lines)


def _status_from_stage(stage: str, root: Path) -> str:
    if stage == "P34":
        return _status_from_json(root / "task_head_optimization_summary.json")
    if stage == "P35":
        return _status_from_json(root / "stream_role_fusion_summary.json")
    if stage == "P36":
        return _status_from_json(root / "optimized_reevaluation_summary.json")
    if stage == "P36_summary":
        return _status_from_json(root / "optimized_model_summary.json")
    if stage == "P37":
        return _status_from_json(root / "optimized_final_polish_summary.json")
    manifest = root / "evidence_manifest.json"
    if manifest.exists():
        return _status_from_json(manifest)
    return "completed" if root.exists() else "missing"


def _status_from_json(path: Path) -> str:
    if not path.exists():
        return "missing"
    data = _read_json(path)
    return str(data.get("status", "completed"))


def _delta_lookup(path: Path, columns: Iterable[str]) -> dict[tuple[object, ...], dict[str, object]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    lookup = {}
    for record in frame.to_dict(orient="records"):
        lookup[tuple(record.get(column) for column in columns)] = record
    return lookup


def _split_protocol(protocol: object, dataset_id: object) -> str:
    text = "" if _is_missing(protocol) else str(protocol)
    dataset = "" if _is_missing(dataset_id) else str(dataset_id)
    if "leave_one_view_out" in text:
        return "leave_one_view_out"
    if "leave_one_sortie_out" in text:
        return "leave_one_sortie_out"
    if "LOSO" in text or "loso" in text:
        return "LOSO"
    if dataset.startswith("nasa") or dataset.startswith("uab"):
        return "fixed_public_split"
    return text or "source_protocol_preserved"


def _join_unique(values: pd.Series) -> str:
    ordered = []
    for value in values.dropna().astype(str).tolist():
        if value and value not in ordered:
            ordered.append(value)
    return "; ".join(ordered)


def _resume_command(config: StageIThesisProtocolConfig, roots: Mapping[str, Path]) -> str:
    parts = [
        PYTHON,
        str(REPO_ROOT / "scripts/stage_i/evidence/build_thesis_protocol.py"),
        "--run-id",
        config.run_id,
        "--p30-root",
        str(roots["P30"]),
        "--p31-root",
        str(roots["P31"]),
        "--p32-root",
        str(roots["P32"]),
        "--p34-root",
        str(roots["P34"]),
        "--p35-root",
        str(roots["P35"]),
        "--p36-root",
        str(roots["P36"]),
        "--p36-summary-root",
        str(roots["P36_summary"]),
        "--p37-root",
        str(roots["P37"]),
        "--artifact-root",
        str(_resolve_path(config.artifact_root)),
        "--report-root",
        str(_resolve_path(config.report_root)),
    ]
    return " ".join(parts)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.copy()
    if len(view) > 40:
        view = view.head(40)
    columns = list(view.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in view.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, float):
                cells.append(f"{value:.6g}" if np.isfinite(value) else "")
            else:
                text = "" if value is None else str(value)
                cells.append(text.replace("\n", " "))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
