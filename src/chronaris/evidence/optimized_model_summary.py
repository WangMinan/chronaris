"""task evaluation optimized model summary package builder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from chronaris.evidence.optimized_model_summary_render import (
    render_figures,
    render_report,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_P30_ROOT = (
    REPO_ROOT
    / "docs/artifacts/runs"
    / "20260702T-task-eval-private-thirdparty-comparison-gpuopt-r1"
)
DEFAULT_P31_ROOT = (
    REPO_ROOT
    / "docs/artifacts/runs"
    / "20260702T-task-eval-public-fusion-ablation-gpuopt-r1"
)
DEFAULT_P32_ROOT = (
    REPO_ROOT
    / "docs/artifacts/runs"
    / "20260702T-task-eval-cross-evidence-matrix-gpuopt-r1"
)
DEFAULT_ARTIFACT_ROOT = "docs/artifacts/runs"
DEFAULT_REPORT_ROOT = "docs/artifacts/runs"
PYTHON = "/home/wangminan/env/anaconda3/envs/chronaris/bin/python"


@dataclass(frozen=True, slots=True)
class StageIOptimizedModelSummaryConfig:
    run_id: str
    p30_root: str = str(DEFAULT_P30_ROOT)
    p31_root: str = str(DEFAULT_P31_ROOT)
    p32_root: str = str(DEFAULT_P32_ROOT)
    p34_root: str | None = None
    p35_root: str | None = None
    p36_root: str | None = None
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT


@dataclass(frozen=True, slots=True)
class StageIOptimizedModelSummaryResult:
    run_id: str
    artifact_root: str
    summary_path: str
    evidence_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


def run_task_eval_optimized_model_summary(
    config: StageIOptimizedModelSummaryConfig,
) -> StageIOptimizedModelSummaryResult:
    run_root = _resolve_path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    p30_root = _resolve_path(config.p30_root)
    p31_root = _resolve_path(config.p31_root)
    p32_root = _resolve_path(config.p32_root)
    p34_root = _resolve_path(config.p34_root) if config.p34_root else _latest_root("task_eval_task_heads_optimization")
    p35_root = _resolve_path(config.p35_root) if config.p35_root else _latest_root("task_eval_stream_role_fusion")
    p36_root = _resolve_path(config.p36_root) if config.p36_root else _latest_root("task_eval_optimized_reevaluation")

    config_path = run_root / "optimized_model_summary_config.json"
    _write_json(
        config_path,
        {
            "run_id": config.run_id,
            "p30_root": str(p30_root),
            "p31_root": str(p31_root),
            "p32_root": str(p32_root),
            "p34_root": str(p34_root) if p34_root else None,
            "p35_root": str(p35_root) if p35_root else None,
            "p36_root": str(p36_root) if p36_root else None,
        },
    )

    model_table = _model_summary_table(p30_root, p31_root, p32_root, p34_root, p35_root, p36_root)
    model_table_path = run_root / "optimized_model_summary.csv"
    model_table.to_csv(model_table_path, index=False)

    key_metric_table = _key_metric_table(p30_root, p31_root, p34_root, p35_root, p36_root)
    key_metric_path = run_root / "key_metric_summary.csv"
    key_metric_table.to_csv(key_metric_path, index=False)

    gate_table = _gate_table(p35_root)
    gate_path = run_root / "stream_role_gate_summary.csv"
    gate_table.to_csv(gate_path, index=False)

    gpu_table = _gpu_table(p34_root, p35_root)
    gpu_path = run_root / "gpu_runtime_summary.csv"
    gpu_table.to_csv(gpu_path, index=False)

    claim_table = _claim_boundary_table()
    claim_path = run_root / "claim_boundary_summary.csv"
    claim_table.to_csv(claim_path, index=False)

    resume_path = run_root / "resume_commands.txt"
    resume_path.write_text(_resume_commands(config, p30_root, p31_root, p32_root, p34_root, p35_root, p36_root) + "\n", encoding="utf-8")

    figure_paths = render_figures(run_root, model_table, key_metric_table, gate_table, gpu_table)
    report_path = _resolve_path(config.report_root) / f"task-eval-optimized-model-summary-{config.run_id}.md"
    status = _summary_status(model_table)
    runtime_device = _summary_runtime_device(gpu_table)
    generated_at = _utc_now()
    summary = {
        "run_id": config.run_id,
        "status": status,
        "runtime_device": runtime_device,
        "generated_at_utc": generated_at,
        "artifact_root": str(run_root),
        "p30_root": str(p30_root),
        "p31_root": str(p31_root),
        "p32_root": str(p32_root),
        "p34_root": str(p34_root) if p34_root else None,
        "p35_root": str(p35_root) if p35_root else None,
        "p36_root": str(p36_root) if p36_root else None,
        "optimized_model_summary_config_json": str(config_path),
        "optimized_model_summary_csv": str(model_table_path),
        "key_metric_summary_csv": str(key_metric_path),
        "stream_role_gate_summary_csv": str(gate_path),
        "gpu_runtime_summary_csv": str(gpu_path),
        "claim_boundary_summary_csv": str(claim_path),
        "resume_commands_txt": str(resume_path),
        "figure_paths": figure_paths,
        "report_path": str(report_path),
        "protocol_boundary": _summary_protocol_boundary(status),
        "model_summary_rows": int(model_table.shape[0]),
        "key_metric_rows": int(key_metric_table.shape[0]),
        "gate_rows": int(gate_table.shape[0]),
        "gpu_rows": int(gpu_table.shape[0]),
    }
    summary_path = run_root / "optimized_model_summary.json"
    manifest_path = run_root / "evidence_manifest.json"
    _write_json(summary_path, summary)
    _write_json(manifest_path, {**summary, "summary_path": str(summary_path), "stage": "P36-summary"})
    if status != "completed":
        _write_json(run_root / "partial_summary.json", summary)
    _write_json(
        run_root / "progress.json",
        {
            "run_id": config.run_id,
            "stage": "P36-summary",
            "status": status,
            "runtime_device": runtime_device,
            "completed": True,
            "generated_at_utc": generated_at,
            "artifact_root": str(run_root),
        },
    )
    (run_root / "run.log").write_text(
        "\n".join(
            [
                f"{generated_at} INFO stage=P36-summary run_id={config.run_id} status={status} runtime_device={runtime_device}",
                "Built optimized model summary from P34/P35/P36 artifacts and fixed P30/P31/P32 references.",
                "No training or baseline rerun was performed by this summary builder.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, key_metric_table, gate_table, gpu_table) + "\n", encoding="utf-8")
    return StageIOptimizedModelSummaryResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        evidence_manifest_path=str(manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def _model_summary_table(
    p30_root: Path,
    p31_root: Path,
    p32_root: Path,
    p34_root: Path | None,
    p35_root: Path | None,
    p36_root: Path | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        _source_row("P30", "completed", "private_thirdparty_comparison", p30_root, "Fixed private T1/T2/T3 reference."),
        _source_row("P31", "completed", "public_fusion_ablation", p31_root, "Fixed public context-proxy ablation reference."),
        _source_row("P32", "completed", "cross_evidence_matrix", p32_root, "Fixed cross-layer evidence matrix reference."),
    ]
    if p34_root:
        p34_status = _status_from_json(p34_root / "task_head_optimization_summary.json")
        p34_summary = (
            "CUDA task-aware v2 head confirm."
            if p34_status == "completed"
            else "CUDA screen for task-aware v2 heads."
        )
        rows.append(_source_row("P34", p34_status, "task_aware_heads", p34_root, p34_summary))
    if p35_root:
        p35_status = _status_from_json(p35_root / "stream_role_fusion_summary.json")
        p35_summary = (
            "CUDA stream-role route check plus requested private/public v3 confirm."
            if p35_status == "completed"
            else "CUDA stream-role route check with incomplete or resumable v3 confirm."
        )
        rows.append(_source_row("P35", p35_status, "stream_role_routing", p35_root, p35_summary))
    if p36_root:
        p36_status = _status_from_json(p36_root / "optimized_reevaluation_summary.json")
        p36_summary = (
            "Aggregates completed optimized private/public v3 evidence with fixed references."
            if p36_status == "completed"
            else "Aggregates available optimized private/public evidence with fixed references."
        )
        rows.append(_source_row("P36", p36_status, "optimized_reevaluation", p36_root, p36_summary))
    return pd.DataFrame(rows)


def _source_row(stage: str, status: str, evidence_layer: str, root: Path, summary: str) -> dict[str, object]:
    return {
        "stage": stage,
        "status": status,
        "evidence_layer": evidence_layer,
        "artifact_root": str(root),
        "summary": summary,
        "boundary": _stage_boundary(stage),
    }


def _key_metric_table(
    p30_root: Path,
    p31_root: Path,
    p34_root: Path | None,
    p35_root: Path | None,
    p36_root: Path | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if p34_root and (p34_root / "improvement_vs_p30.csv").exists():
        deltas = pd.read_csv(p34_root / "improvement_vs_p30.csv")
        p34_boundary = _p34_key_metric_boundary(p34_root)
        for record in deltas.to_dict(orient="records"):
            rows.append(
                {
                    "stage": "P34",
                    "scope": "private_feature_export",
                    "dataset_or_task": record.get("task_name"),
                    "split_or_group": record.get("split_strategy"),
                    "metric": record.get("metric"),
                    "reference": "P30 Chronaris v1",
                    "reference_value": record.get("p30_chronaris_v1"),
                    "optimized": "P34 chronaris_v2_task_heads",
                    "optimized_value": record.get("p34_chronaris_v2"),
                    "delta_positive_is_better": record.get("delta_abs_positive_is_better"),
                    "status": _delta_status(record.get("delta_abs_positive_is_better")),
                    "boundary": p34_boundary,
                }
            )
    if p35_root and (p35_root / "public_metrics.csv").exists():
        public = pd.read_csv(p35_root / "public_metrics.csv")
        for record in public.to_dict(orient="records"):
            p35_status = str(record.get("p35_status", ""))
            rows.append(
                {
                    "stage": "P35",
                    "scope": "public_context_proxy",
                    "dataset_or_task": record.get("dataset_id"),
                    "split_or_group": record.get("task_group"),
                    "metric": record.get("metric"),
                    "reference": "P31 full",
                    "reference_value": record.get("full_value_mean"),
                    "optimized": record.get("variant_id"),
                    "optimized_value": record.get("value_mean"),
                    "delta_positive_is_better": record.get("delta_abs_mean"),
                    "status": p35_status,
                    "boundary": _p35_public_metric_boundary(p35_status),
                }
            )
    if p36_root and (p36_root / "model_selection_summary.json").exists():
        model = _read_json(p36_root / "model_selection_summary.json")
        rows.append(
            {
                "stage": "P36",
                "scope": "cross_evidence",
                "dataset_or_task": "optimized_reevaluation",
                "split_or_group": "all_available_partial_rows",
                "metric": "selection_status",
                "reference": "P30/P31/P32",
                "reference_value": np.nan,
                "optimized": "P34/P35/P36 aggregate",
                "optimized_value": model.get("selection_status"),
                "delta_positive_is_better": np.nan,
                "status": model.get("selection_status"),
                "boundary": model.get("positive_delta_convention"),
            }
        )
    if not rows:
        rows.append(
            {
                "stage": "summary",
                "scope": "missing",
                "dataset_or_task": "",
                "split_or_group": "",
                "metric": "no_rows",
                "reference": "",
                "reference_value": np.nan,
                "optimized": "",
                "optimized_value": np.nan,
                "delta_positive_is_better": np.nan,
                "status": "missing",
                "boundary": "No optimized artifacts were found.",
            }
        )
    return pd.DataFrame(rows)


def _gate_table(p35_root: Path | None) -> pd.DataFrame:
    if not p35_root or not (p35_root / "gate_statistics.csv").exists():
        return pd.DataFrame(columns=["dataset_id", "second_stream_role", "fusion_route", "lag_gate_mean", "context_gate_mean", "vehicle_gate_mean", "causal_gate_mean"])
    return pd.read_csv(p35_root / "gate_statistics.csv")


def _gpu_table(p34_root: Path | None, p35_root: Path | None) -> pd.DataFrame:
    rows = []
    for stage, root in (("P34", p34_root), ("P35", p35_root)):
        if not root:
            continue
        path = root / "gpu_perf_summary.json"
        if not path.exists():
            continue
        data = _read_json(path)
        snapshot = data.get("utilization_snapshot") if isinstance(data.get("utilization_snapshot"), dict) else {}
        rows.append(
            {
                "stage": stage,
                "runtime_device": data.get("runtime_device"),
                "gpu_name": data.get("gpu_name"),
                "torch_version": data.get("torch_version"),
                "cuda_version": data.get("cuda_version"),
                "tensor_cache_mode": data.get("tensor_cache_mode"),
                "best_batch_size": data.get("best_batch_size"),
                "batch_size": data.get("batch_size"),
                "amp_mode": data.get("amp_mode"),
                "torch_compile_mode": data.get("torch_compile_mode"),
                "max_gpu_memory_gb": data.get("max_gpu_memory_gb", data.get("max_memory_allocated_gb")),
                "samples_per_sec": data.get("samples_per_sec"),
                "batch_time_sec": data.get("batch_time_sec"),
                "copy_time_sec": data.get("copy_time_sec"),
                "forward_time_sec": data.get("forward_time_sec"),
                "backward_time_sec": data.get("backward_time_sec"),
                "step_time_sec": data.get("step_time_sec"),
                "gpu_util_pct": snapshot.get("gpu_util_pct"),
                "gpu_memory_used_gb": snapshot.get("gpu_memory_used_gb"),
                "gpu_memory_total_gb": snapshot.get("gpu_memory_total_gb"),
            }
        )
    return pd.DataFrame(rows)


def _claim_boundary_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim": "P34/P35 convert P30/P31 diagnostics into model changes.",
                "allowed_strength": "Use each artifact status: P34 confirm and P35 requested v3 confirm when completed; otherwise partial/route-only.",
                "required_boundary": "Do not write partial or route-only evidence as completed/full LOSO.",
            },
            {
                "claim": "Task-aware heads improve T1/T2 in current private confirm; T3 remains mixed.",
                "allowed_strength": "metric-backed but limited",
                "required_boundary": "T3 mixed/regressed metrics must be reported; do not claim broad superiority.",
            },
            {
                "claim": "Stream-role routing separates real vehicle streams from context proxy streams.",
                "allowed_strength": "route/gate smoke evidence",
                "required_boundary": "Public context proxy is not real vehicle telemetry.",
            },
            {
                "claim": "P36 summarizes optimized Chronaris against fixed references.",
                "allowed_strength": "aggregation package",
                "required_boundary": "Do not overwrite or reinterpret P30/P31/P32 confirmed artifacts.",
            },
        ]
    )


def _resume_commands(
    config: StageIOptimizedModelSummaryConfig,
    p30_root: Path,
    p31_root: Path,
    p32_root: Path,
    p34_root: Path | None,
    p35_root: Path | None,
    p36_root: Path | None,
) -> str:
    commands = []
    for root in (p34_root, p35_root, p36_root):
        if root and (root / "resume_command.txt").exists():
            commands.append((root / "resume_command.txt").read_text(encoding="utf-8").strip())
    commands.append(
        " ".join(
            [
                PYTHON,
                str(REPO_ROOT / "scripts/task_eval/evidence/build_optimized_model_summary.py"),
                "--run-id",
                config.run_id,
                "--p30-root",
                str(p30_root),
                "--p31-root",
                str(p31_root),
                "--p32-root",
                str(p32_root),
                "--p34-root",
                str(p34_root) if p34_root else "",
                "--p35-root",
                str(p35_root) if p35_root else "",
                "--p36-root",
                str(p36_root) if p36_root else "",
                "--artifact-root",
                str(_resolve_path(config.artifact_root)),
                "--report-root",
                str(_resolve_path(config.report_root)),
            ]
        )
    )
    return "\n".join(command for command in commands if command)


def _summary_status(frame: pd.DataFrame) -> str:
    statuses = set(frame["status"].astype(str)) if "status" in frame else set()
    if "blocked" in statuses or "partial_blocked" in statuses:
        return "partial"
    if "partial" in statuses:
        return "partial"
    return "completed"


def _summary_runtime_device(gpu_table: pd.DataFrame) -> str:
    if "runtime_device" not in gpu_table:
        return "unknown"
    devices = [str(device) for device in gpu_table["runtime_device"].dropna().tolist()]
    if "cuda" in devices:
        return "cuda"
    return devices[0] if devices else "unknown"


def _summary_protocol_boundary(status: str) -> str:
    if status == "completed":
        return (
            "This package summarizes completed optimized P34/P35/P36 evidence with fixed P30/P31/P32 references; "
            "public rows remain context-proxy evidence and historical artifacts are read-only."
        )
    return (
        "This package summarizes available optimized P34/P35/P36 evidence with fixed P30/P31/P32 references; "
        "partial inputs remain resumable and must not be written as completed."
    )


def _stage_boundary(stage: str) -> str:
    return {
        "P30": "Private proxy task third-party comparison; mixed result, not human truth.",
        "P31": "Public context-proxy ablation; not real vehicle stream.",
        "P32": "Evidence matrix, not a single global leaderboard.",
        "P34": "Task-aware heads CUDA confirm; separate from P35 stream-role confirm.",
        "P35": "Stream-role route evidence plus requested v3 confirm when status is completed; public stream remains context proxy.",
        "P36": "Aggregation of optimized evidence and fixed references; status mirrors P34/P35 artifact completeness.",
    }.get(stage, "task evaluation evidence artifact.")


def _p34_key_metric_boundary(p34_root: Path) -> str:
    summary_path = p34_root / "task_head_optimization_summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    seeds = summary.get("seeds")
    splits = summary.get("split_strategies")

    if isinstance(seeds, list):
        seed_text = f"{len(seeds)} seeds"
    else:
        seed_text = "seed count unknown"
    if isinstance(splits, list) and splits:
        split_text = " + ".join(str(split) for split in splits)
    else:
        split_text = "split protocol unknown"

    epoch_text = "epoch count unknown"
    training_curves_path = p34_root / "training_curves.csv"
    if training_curves_path.exists():
        try:
            training_curves = pd.read_csv(training_curves_path, usecols=["epoch"])
            epochs = pd.to_numeric(training_curves["epoch"], errors="coerce")
            max_epoch = int(epochs.max()) if not epochs.dropna().empty else 0
            if max_epoch > 0:
                epoch_text = f"{max_epoch} epochs"
        except (ValueError, OSError):
            epoch_text = "epoch count unknown"

    status = str(summary.get("status", "unknown"))
    prefix = "CUDA confirm" if status == "completed" else "partial confirm-screen"
    return f"{prefix}; {seed_text} / {split_text} / {epoch_text}; not a P35 stream-role confirm"


def _p35_public_metric_boundary(status: str) -> str:
    if status == "v3_confirm":
        return "P35 requested public v3 confirm; public stream is context proxy, not real vehicle"
    return "P31 fixed reference row; public stream is context proxy, not real vehicle"


def _status_from_json(path: Path) -> str:
    if not path.exists():
        return "missing"
    data = _read_json(path)
    return str(data.get("status", "unknown"))


def _delta_status(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if numeric > 1e-9:
        return "improved"
    if numeric < -1e-9:
        return "regressed"
    return "tie"


def _latest_root(name: str) -> Path | None:
    root = REPO_ROOT / "docs/artifacts/assets" / name
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir()]
    return sorted(candidates)[-1] if candidates else None


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _resolve_path(path_like: str | Path | None) -> Path:
    if path_like is None:
        raise ValueError("path must not be None")
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialize {type(value)!r}")
