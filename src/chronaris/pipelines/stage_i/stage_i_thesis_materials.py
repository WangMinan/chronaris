"""Build Stage I thesis-facing tables and explanatory figures."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from chronaris.pipelines.stage_i.stage_i_run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronaris-matplotlib")

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_thesis_figures"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"
DEFAULT_EVIDENCE_MANIFEST_PATH = (
    "docs/artifacts/assets/stage_i_evidence/20260607T-stage-i-evidence-closure-r2/"
    "evidence_manifest.json"
)
DEFAULT_PROXY_SWEEP_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_multitask_sweep/20260607T-stage-i-evidence-closure-r2-multitask/"
    "multitask_sweep_summary.json"
)
DEFAULT_PRIVATE_COMPONENT_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/"
    "chronaris_opt_component_ablation.json"
)
DEFAULT_PUBLIC_CALIBRATION_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_public_adapter_calibration/20260607T-stage-i-evidence-closure-r2-public-adapter/"
    "public_adapter_calibration_summary.json"
)
DEFAULT_PUBLIC_TRANSFER_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_public_transfer_boundary/20260607T-stage-i-evidence-closure-r2-transfer-boundary/"
    "public_transfer_boundary_summary.json"
)
DEFAULT_RIGID_BODY_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_rigid_body/20260607T-stage-i-rigid-body-r2/"
    "rigid_body_ablation_summary.json"
)
DEFAULT_ROTATION_AUDIT_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_rotation_audit/20260607T-stage-i-rotation-audit-r2/"
    "rigid_body_rotation_audit_summary.json"
)
DEFAULT_RUNTIME_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-service-r2/"
    "runtime_inference_summary.json"
)
DEFAULT_SUPPORT_SUMMARY_PATH = (
    "docs/artifacts/assets/stage_i_support/20260607T-stage-i-support-semantic-r2/"
    "support_summary.json"
)
DEFAULT_CJK_FONT_CANDIDATES = (
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans SC",
    "Source Han Sans SC",
    "AR PL UMing CN",
)


@dataclass(frozen=True, slots=True)
class StageIThesisMaterialsConfig:
    run_id: str
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    evidence_manifest_path: str = DEFAULT_EVIDENCE_MANIFEST_PATH
    proxy_sweep_summary_path: str = DEFAULT_PROXY_SWEEP_SUMMARY_PATH
    live_sweep_summary_path: str | None = None
    live_partial_summary_path: str | None = None
    private_component_summary_path: str = DEFAULT_PRIVATE_COMPONENT_SUMMARY_PATH
    public_calibration_summary_path: str = DEFAULT_PUBLIC_CALIBRATION_SUMMARY_PATH
    public_transfer_summary_path: str = DEFAULT_PUBLIC_TRANSFER_SUMMARY_PATH
    rigid_body_summary_path: str = DEFAULT_RIGID_BODY_SUMMARY_PATH
    rotation_audit_summary_path: str = DEFAULT_ROTATION_AUDIT_SUMMARY_PATH
    runtime_summary_path: str = DEFAULT_RUNTIME_SUMMARY_PATH
    runtime_service_summary_path: str | None = None
    runtime_schema_contract_path: str | None = None
    support_summary_path: str = DEFAULT_SUPPORT_SUMMARY_PATH


@dataclass(frozen=True, slots=True)
class StageIThesisMaterialsRunResult:
    run_id: str
    artifact_root: str
    table_manifest_path: str
    figure_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PlotFontSelection:
    family: str | None
    ascii_only: bool
    note: str


def run_stage_i_thesis_materials(
    config: StageIThesisMaterialsConfig,
) -> StageIThesisMaterialsRunResult:
    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_thesis_materials",
        logger=LOGGER,
        initial_progress={"artifact_root": str(run_root)},
    ) as progress:
        return _run_stage_i_thesis_materials_observed(
            config=config,
            run_root=run_root,
            progress=progress,
        )


def _run_stage_i_thesis_materials_observed(
    *,
    config: StageIThesisMaterialsConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageIThesisMaterialsRunResult:
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    sources = _load_sources(config)
    progress.update("sources_loaded", source_count=len(sources))

    font = _detect_plot_font()
    progress.update("font_selected", ascii_only=font.ascii_only, font_family=font.family)

    table_entries = _write_tables(run_root=run_root, sources=sources)
    progress.update("tables_written", table_count=len(table_entries))

    figure_entries = _write_figures(
        run_root=run_root,
        font=font,
        sources=sources,
        table_entries=table_entries,
    )
    progress.update("figures_written", figure_count=len(figure_entries))

    table_manifest = {
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "tables": table_entries,
    }
    table_manifest_path = run_root / "table_manifest.json"
    table_manifest_path.write_text(
        json.dumps(table_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    figure_manifest = {
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "figures": figure_entries,
    }
    figure_manifest_path = run_root / "figure_manifest.json"
    figure_manifest_path.write_text(
        json.dumps(figure_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_path = report_root / f"stage-i-thesis-materials-{config.run_id}.md"
    report_path.write_text(
        render_stage_i_thesis_materials_report(
            run_id=config.run_id,
            sources=sources,
            table_entries=table_entries,
            figure_entries=figure_entries,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "table_manifest_path": str(table_manifest_path),
        "figure_manifest_path": str(figure_manifest_path),
        "report_path": str(report_path),
    }
    progress.finish(
        table_manifest_path=str(table_manifest_path),
        figure_manifest_path=str(figure_manifest_path),
        report_path=str(report_path),
    )
    return StageIThesisMaterialsRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        table_manifest_path=str(table_manifest_path),
        figure_manifest_path=str(figure_manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def _load_sources(config: StageIThesisMaterialsConfig) -> dict[str, dict[str, object]]:
    path_map = {
        "evidence_manifest": config.evidence_manifest_path,
        "proxy_sweep": config.proxy_sweep_summary_path,
        "private_component": config.private_component_summary_path,
        "public_calibration": config.public_calibration_summary_path,
        "public_transfer": config.public_transfer_summary_path,
        "rigid_body": config.rigid_body_summary_path,
        "rotation_audit": config.rotation_audit_summary_path,
        "runtime": config.runtime_summary_path,
        "support": config.support_summary_path,
    }
    if config.live_sweep_summary_path:
        path_map["live_sweep"] = config.live_sweep_summary_path
    if config.live_partial_summary_path:
        path_map["live_partial"] = config.live_partial_summary_path
    if config.runtime_service_summary_path:
        path_map["runtime_service"] = config.runtime_service_summary_path
    if config.runtime_schema_contract_path:
        path_map["runtime_schema_contract"] = config.runtime_schema_contract_path
    sources: dict[str, dict[str, object]] = {}
    for name, path_like in path_map.items():
        path = Path(path_like)
        sources[name] = {
            "path": str(path),
            "payload": json.loads(path.read_text(encoding="utf-8")),
        }
    return sources


def _write_tables(*, run_root: Path, sources: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    tables = {
        "evidence_layer_overview.csv": pd.DataFrame(_build_evidence_rows(sources)),
        "weak_label_sweep_ablation.csv": pd.DataFrame(_build_weak_label_rows(sources)),
        "chronaris_opt_component_ablation.csv": pd.DataFrame(_build_private_rows(sources)),
        "public_transfer_boundary.csv": pd.DataFrame(_build_transfer_rows(sources)),
        "runtime_semantic_case.csv": pd.DataFrame(_build_runtime_case_rows(sources)),
        "rigid_body_rotation_audit.csv": pd.DataFrame(_build_rigid_body_rows(sources)),
    }
    entries: list[dict[str, object]] = []
    for file_name, frame in tables.items():
        path = run_root / file_name
        frame.to_csv(path, index=False)
        entries.append(
            {
                "table_id": path.stem,
                "path": str(path),
                "row_count": len(frame),
            }
        )
    return entries


def _build_evidence_rows(sources: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    manifest = sources["evidence_manifest"]["payload"]
    rows = []
    for task_name, task_payload in manifest["tasks"].items():
        outputs = task_payload.get("outputs", {})
        rows.append(
            {
                "artifact_id": task_name,
                "evidence_layer": task_payload.get("evidence_layer"),
                "status": task_payload.get("status"),
                "reused_existing": task_payload.get("reused_existing"),
                "summary_path": outputs.get("summary_path"),
                "report_path": outputs.get("report_path"),
                "artifact_root": outputs.get("artifact_root"),
                "source_path": sources["evidence_manifest"]["path"],
            }
        )
    if "live_sweep" in sources:
        live = sources["live_sweep"]
        rows.append(
            {
                "artifact_id": "multitask_live_influx",
                "evidence_layer": live["payload"].get("evidence_layer"),
                "status": "completed",
                "reused_existing": False,
                "summary_path": live["path"],
                "report_path": str(
                    Path(DEFAULT_REPORT_ROOT)
                    / f"stage-i-thesis-weak-label-multitask-sweep-{live['payload']['run_id']}.md"
                ),
                "artifact_root": live["payload"].get("artifact_root"),
                "source_path": live["path"],
            }
        )
    return rows


def _build_weak_label_rows(sources: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for source_name in ("proxy_sweep", "live_sweep"):
        if source_name not in sources:
            continue
        payload = sources[source_name]["payload"]
        sample_collection = payload.get("source_summary", {}).get("sample_collection", {})
        sample_source = sample_collection.get("sample_source", source_name)
        blocked_attempt_log_paths = payload.get("blocked_attempt_log_paths", [])
        for row in payload.get("rows", []):
            rows.append(
                {
                    "summary_status": payload.get("status", "completed"),
                    "sample_source": sample_source,
                    "sample_count": payload.get("sample_count"),
                    "task_entry_count": payload.get("task_entry_count"),
                    "combination_count": payload.get("combination_count"),
                    "best_child_run_id": payload.get("best_run", {}).get("child_run_id"),
                    "best_test_total": payload.get("best_run", {}).get("test_total"),
                    "derived_from_run_id": payload.get("derived_from_run_id"),
                    "blocked_at_run_index": payload.get("blocked_at_run_index"),
                    "blocked_attempt_log_path_count": len(blocked_attempt_log_paths),
                    "physics_constraint_family": row.get("physics_constraint_family"),
                    "causal_weight": row.get("causal_weight"),
                    "task_loss_weight": row.get("task_loss_weight"),
                    "causal_lag_window_points": row.get("causal_lag_window_points"),
                    "test_total": row.get("test_total"),
                    "test_task_total": row.get("test_task_total"),
                    "test_causal_total": row.get("test_causal_total"),
                    "source_path": sources[source_name]["path"],
                    "evidence_layer": payload.get("evidence_layer"),
                    "metric_definition": "test_total/test_task_total/test_causal_total from multitask sweep summary",
                }
            )
    if "live_partial" in sources:
        partial_payload = sources["live_partial"]["payload"]
        rows.append(
            {
                "summary_status": partial_payload.get("status", "partial_blocked"),
                "sample_source": partial_payload.get("source_summary", {}).get("sample_collection", {}).get("sample_source", "live_influx"),
                "sample_count": partial_payload.get("sample_count"),
                "task_entry_count": partial_payload.get("task_entry_count"),
                "combination_count": partial_payload.get("target_combination_count"),
                "best_child_run_id": partial_payload.get("best_run", {}).get("child_run_id"),
                "best_test_total": partial_payload.get("best_run", {}).get("test_total"),
                "derived_from_run_id": partial_payload.get("derived_from_run_id"),
                "blocked_at_run_index": partial_payload.get("blocked_at_run_index"),
                "blocked_attempt_log_path_count": len(partial_payload.get("blocked_attempt_log_paths", [])),
                "physics_constraint_family": "partial_resume_state",
                "causal_weight": None,
                "task_loss_weight": None,
                "causal_lag_window_points": None,
                "test_total": None,
                "test_task_total": None,
                "test_causal_total": None,
                "source_path": sources["live_partial"]["path"],
                "evidence_layer": partial_payload.get("evidence_layer"),
                "metric_definition": "partial summary row capturing completed child runs and blocked-at index for resume/blocker evidence",
            }
        )
    return rows


def _build_private_rows(sources: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    source_path = sources["private_component"]["path"]
    return [
        {
            **dict(row),
            "source_path": source_path,
            "metric_definition": "delta_vs_full uses the full chronaris_opt candidate as reference",
        }
        for row in sources["private_component"]["payload"].get("rows", [])
    ]


def _build_transfer_rows(sources: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    payload = sources["public_transfer"]["payload"]
    rows: list[dict[str, object]] = []
    for row in payload.get("data_boundary_rows", []):
        rows.append({**dict(row), "row_type": "data_boundary", "source_path": sources["public_transfer"]["path"]})
    for row in payload.get("task_boundary_rows", []):
        rows.append({**dict(row), "row_type": "task_boundary", "source_path": sources["public_transfer"]["path"]})
    for row in payload.get("performance_rows", []):
        rows.append({**dict(row), "row_type": "performance", "source_path": sources["public_transfer"]["path"]})
    return rows


def _build_runtime_case_rows(sources: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    runtime = sources["runtime"]["payload"]
    runtime_service = dict(sources.get("runtime_service", {}).get("payload") or {})
    runtime_schema_contract = dict(sources.get("runtime_schema_contract", {}).get("payload") or {})
    rows = pd.DataFrame(runtime.get("samples", []))
    rows["view_id"] = rows["sample_id"].map(_infer_view_id)
    selected_view = rows["view_id"].mode().iloc[0]
    case_rows = rows.loc[rows["view_id"] == selected_view].head(12).copy()
    case_rows["source_path"] = sources["runtime"]["path"]
    case_rows["support_source_path"] = sources["support"]["path"]
    case_rows["runtime_service_source_path"] = sources.get("runtime_service", {}).get("path")
    case_rows["runtime_schema_contract_source_path"] = sources.get("runtime_schema_contract", {}).get("path")
    case_rows["evidence_layer"] = "runtime_semantic_support"
    case_rows["case_definition"] = (
        "top 12 windows from the most frequent runtime replay view, keeping predictions and semantic attributions"
    )
    case_rows["native_feature_schema_status"] = runtime_service.get("native_feature_schema_status")
    case_rows["canonical_feature_schema_status"] = runtime_service.get("canonical_feature_schema_status")
    case_rows["expected_vehicle_feature_count"] = runtime_service.get("expected_vehicle_feature_count")
    case_rows["input_vehicle_feature_count"] = runtime_service.get("input_vehicle_feature_count")
    case_rows["missing_vehicle_feature_count"] = runtime_service.get("missing_vehicle_feature_count")
    case_rows["native_missing_measurement_group_count"] = len(
        (runtime_service.get("missing_vehicle_measurement_group_counts") or {}).keys()
    )
    case_rows["schema_hash"] = runtime_schema_contract.get("schema_hash")
    return case_rows.to_dict(orient="records")


def _build_rigid_body_rows(sources: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    rigid_body = sources["rigid_body"]["payload"]
    rotation = sources["rotation_audit"]["payload"]
    rows: list[dict[str, object]] = []
    for family, payload in rigid_body.get("families", {}).items():
        components = payload.get("physics_components", {})
        rows.append(
            {
                "row_type": "family_metric",
                "family": family,
                "test_total": payload.get("test_total"),
                "test_alignment": payload.get("test_alignment"),
                "test_physics_total": payload.get("test_physics_total"),
                "vehicle_rigid_body_translation": components.get("vehicle_rigid_body_translation"),
                "vehicle_rigid_body_vertical": components.get("vehicle_rigid_body_vertical"),
                "vehicle_rigid_body_rotation": components.get("vehicle_rigid_body_rotation"),
                "source_path": sources["rigid_body"]["path"],
                "evidence_layer": "rigid_body_rotation_diagnostics",
            }
        )
    for name in ("pitch", "pitch_rate", "roll", "roll_rate", "yaw", "yaw_rate"):
        rows.append(
            {
                "row_type": "rotation_requirement",
                "family": name,
                "feature_candidate_count": len(rotation.get("feature_rotation_candidates", {}).get(name, [])),
                "mysql_candidate_count": len(rotation.get("mysql_rotation_candidates", {}).get(name, [])),
                "rotation_status": rotation.get("rotation_status"),
                "rotation_enabled": rotation.get("rotation_enabled"),
                "source_path": sources["rotation_audit"]["path"],
                "evidence_layer": rotation.get("evidence_layer"),
            }
        )
    return rows


def _write_figures(
    *,
    run_root: Path,
    font: PlotFontSelection,
    sources: Mapping[str, Mapping[str, object]],
    table_entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    table_paths = {Path(entry["path"]).stem: entry["path"] for entry in table_entries}
    return [
        _plot_evidence_layer_overview(run_root / "evidence_layer_overview.png", sources, font, table_paths),
        _plot_weak_label_sweep(run_root / "weak_label_sweep_ablation.png", sources, font, table_paths),
        _plot_private_component(run_root / "chronaris_opt_component_ablation.png", sources, font, table_paths),
        _plot_public_transfer(run_root / "public_transfer_boundary.png", sources, font, table_paths),
        _plot_runtime_semantic_case(run_root / "runtime_semantic_case.png", sources, font, table_paths),
        _plot_rigid_body_rotation(run_root / "rigid_body_rotation_audit.png", sources, font, table_paths),
    ]


def _plot_evidence_layer_overview(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(_build_evidence_rows(sources))
    plt, _ = _import_matplotlib(font)
    colors = {
        "thesis_weak_label": "#40634d",
        "private_proxy": "#87523b",
        "public_adapter_calibration": "#355070",
        "rotation_diagnostics": "#7a6f36",
        "runtime_semantic_support": "#6c4f7d",
    }
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(frame["artifact_id"], [1.0] * len(frame), color=[colors.get(value, "#7a8793") for value in frame["evidence_layer"]])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("artifact present")
    ax.set_title(_pick_label(font, "证据层级与产物总览", "Evidence Layer Overview"))
    for index, row in frame.reset_index(drop=True).iterrows():
        ax.text(0.02, index, f"{row['evidence_layer']} | {Path(str(row['summary_path'] or '')).name}", va="center", fontsize=8, color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="evidence_layer_overview",
        path=path,
        source_paths=[sources["evidence_manifest"]["path"]],
        evidence_layer="cross_layer_index",
        table_path=table_paths["evidence_layer_overview"],
        metric_definition="One bar per stable artifact/task with color-coded evidence layer.",
    )


def _plot_weak_label_sweep(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(_build_weak_label_rows(sources))
    best = (
        frame.loc[frame["test_total"].notna()]
        .sort_values("test_total")
        .groupby("sample_source", as_index=False)
        .first()[["sample_source", "sample_count", "task_entry_count", "best_test_total"]]
    )
    plt, _ = _import_matplotlib(font)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(best["sample_source"], best["best_test_total"], color=["#7a8793", "#40634d"][: len(best)])
    axes[0].set_title("best test_total", fontsize=10)
    for index, row in best.reset_index(drop=True).iterrows():
        axes[0].text(index, row["best_test_total"], f"samples={int(row['sample_count'])}\ntasks={int(row['task_entry_count'])}", ha="center", va="bottom", fontsize=8)
    completed_rows = frame.loc[frame["test_total"].notna()].copy()
    completed_rows["label"] = completed_rows["sample_source"].str.replace("_", "\n") + "\n" + completed_rows["physics_constraint_family"]
    axes[1].bar(
        completed_rows["label"],
        completed_rows["test_total"],
        color=["#7a8793" if value == "stage_h_window_stats_proxy" else "#40634d" for value in completed_rows["sample_source"]],
    )
    axes[1].tick_params(axis="x", labelrotation=25, labelsize=8)
    axes[1].set_title("all sweep rows", fontsize=10)
    if "live_partial" in sources:
        partial_payload = sources["live_partial"]["payload"]
        axes[1].text(
            0.98,
            0.95,
            f"partial_blocked @ run_index={partial_payload.get('blocked_at_run_index')}\ncompleted_child_runs={len(partial_payload.get('completed_child_runs', []))}",
            ha="right",
            va="top",
            fontsize=8,
            transform=axes[1].transAxes,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f7efe0", "edgecolor": "#9a6b39"},
        )
    fig.suptitle(_pick_label(font, "weak-label 小网格对比", "Weak-Label Sweep Ablation"), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    source_paths = [sources["proxy_sweep"]["path"]]
    if "live_sweep" in sources:
        source_paths.append(sources["live_sweep"]["path"])
    if "live_partial" in sources:
        source_paths.append(sources["live_partial"]["path"])
    return _figure_entry(
        figure_id="weak_label_sweep_ablation",
        path=path,
        source_paths=source_paths,
        evidence_layer="thesis_weak_label",
        table_path=table_paths["weak_label_sweep_ablation"],
        metric_definition="Compare proxy and live_influx sweep sample counts plus test_total rows, with partial/resume blocker note.",
    )


def _plot_private_component(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(_build_private_rows(sources))
    selected = frame.loc[
        frame["variant_name"].isin(
            (
                "chronaris_opt",
                "chronaris_opt_no_causal_mask",
                "chronaris_opt_no_time_residual",
                "chronaris_opt_no_task_head",
            )
        )
    ].copy()
    pivot = selected.pivot(index="task_name", columns="variant_name", values="delta_vs_full").fillna(0.0)
    plt, _ = _import_matplotlib(font)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x_positions = range(len(pivot.index))
    width = 0.18
    for offset, column in enumerate(pivot.columns):
        ax.bar(
            [x + (offset - 1.5) * width for x in x_positions],
            pivot[column],
            width=width,
            label=column.replace("chronaris_opt_", ""),
        )
    ax.set_xticks(list(x_positions), [name.replace("_", "\n") for name in pivot.index])
    ax.set_ylabel("delta_vs_full")
    ax.legend(fontsize=8)
    ax.set_title(_pick_label(font, "chronaris_opt 组件贡献", "chronaris_opt Component Ablation"), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="chronaris_opt_component_ablation",
        path=path,
        source_paths=[sources["private_component"]["path"]],
        evidence_layer="private_proxy",
        table_path=table_paths["chronaris_opt_component_ablation"],
        metric_definition="delta_vs_full uses the optimized private proxy candidate as reference.",
    )


def _plot_public_transfer(path: Path, sources, font, table_paths):
    transfer = sources["public_transfer"]["payload"]
    plt, _ = _import_matplotlib(font)
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 4.8))
    boxes = [
        (0.5, 2.2, 3.2, 1.0, "#dae8f5", "Public Adapter", "UAB/NASA\ncontext proxy"),
        (4.4, 2.2, 3.2, 1.0, "#dcebdc", "Thesis Weak-Label", "risk/workload/event\nreal Stage H"),
        (8.3, 2.2, 3.0, 1.0, "#f5e3d6", "Private Proxy", "T1/T2/T3\nmechanism diagnostics"),
    ]
    for x0, y0, width, height, color, title, subtitle in boxes:
        ax.add_patch(FancyBboxPatch((x0, y0), width, height, boxstyle="round,pad=0.12", facecolor=color, edgecolor="#334155"))
        ax.text(x0 + width / 2, y0 + 0.62, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x0 + width / 2, y0 + 0.26, subtitle, ha="center", va="center", fontsize=9)
    ax.text(6.0, 1.5, transfer["boundary_note"], ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(4.3, 2.7), xytext=(3.8, 2.7), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#334155"})
    ax.annotate("", xy=(8.2, 2.7), xytext=(7.7, 2.7), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#334155"})
    ax.set_xlim(0, 11.6)
    ax.set_ylim(1.0, 3.7)
    ax.set_axis_off()
    ax.set_title(_pick_label(font, "公开迁移边界", "Public Transfer Boundary"), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="public_transfer_boundary",
        path=path,
        source_paths=[sources["public_transfer"]["path"], sources["public_calibration"]["path"]],
        evidence_layer="public_adapter_calibration",
        table_path=table_paths["public_transfer_boundary"],
        metric_definition="Boundary diagram separating public adapter, thesis weak-label, and private proxy evidence.",
    )


def _plot_runtime_semantic_case(path: Path, sources, font, table_paths):
    frame = pd.DataFrame(_build_runtime_case_rows(sources))
    frame["window_index"] = range(len(frame))
    plt, _ = _import_matplotlib(font)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.6), sharex=False)
    axes[0].plot(frame["window_index"], frame["workload_proxy_prediction"], marker="o", label="workload_proxy_prediction")
    axes[0].plot(frame["window_index"], frame["risk_proxy_confidence"], marker="s", label="risk_proxy_confidence")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("score")
    axes[1].bar(frame["window_index"], frame["semantic_top_event_attribution"], color="#6c4f7d")
    axes[1].set_xticks(frame["window_index"], [Path(sample_id).name[-4:] for sample_id in frame["sample_id"]], rotation=20)
    axes[1].set_ylabel("semantic attribution")
    for _, row in frame.iterrows():
        axes[1].text(row["window_index"], row["semantic_top_event_attribution"], str(row["semantic_top_query_name"]), rotation=90, va="bottom", ha="center", fontsize=7)
    axes[2].bar(["native input", "expected/canonical"], [frame["input_vehicle_feature_count"].iloc[0], frame["expected_vehicle_feature_count"].iloc[0]], color=["#d37c5c", "#40634d"])
    axes[2].set_ylabel("vehicle feature count")
    axes[2].set_title(
        f"native={frame['native_feature_schema_status'].iloc[0]} | canonical={frame['canonical_feature_schema_status'].iloc[0]} | gap={int(frame['input_vehicle_feature_count'].iloc[0])}->{int(frame['expected_vehicle_feature_count'].iloc[0])}",
        fontsize=10,
    )
    fig.suptitle(_pick_label(font, "runtime 与语义事件案例", "Runtime Semantic Case"), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    source_paths = [sources["runtime"]["path"], sources["support"]["path"]]
    if "runtime_service" in sources:
        source_paths.append(sources["runtime_service"]["path"])
    if "runtime_schema_contract" in sources:
        source_paths.append(sources["runtime_schema_contract"]["path"])
    return _figure_entry(
        figure_id="runtime_semantic_case",
        path=path,
        source_paths=source_paths,
        evidence_layer="runtime_semantic_support",
        table_path=table_paths["runtime_semantic_case"],
        case_definition="Top 12 windows from the most frequent replay view; compare workload/risk outputs, semantic attributions, and native aligned vs canonical exact schema counts.",
    )


def _plot_rigid_body_rotation(path: Path, sources, font, table_paths):
    rigid_frame = pd.DataFrame(_build_rigid_body_rows(sources))
    family_rows = rigid_frame.loc[rigid_frame["row_type"] == "family_metric"].copy()
    requirement_rows = rigid_frame.loc[rigid_frame["row_type"] == "rotation_requirement"].copy()
    plt, _ = _import_matplotlib(font)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x_positions = range(len(family_rows))
    width = 0.22
    for offset, column in enumerate(
        (
            "vehicle_rigid_body_translation",
            "vehicle_rigid_body_vertical",
            "vehicle_rigid_body_rotation",
        )
    ):
        axes[0].bar(
            [x + (offset - 1) * width for x in x_positions],
            family_rows[column],
            width=width,
            label=column.replace("vehicle_rigid_body_", ""),
        )
    axes[0].set_xticks(list(x_positions), family_rows["family"])
    axes[0].legend(fontsize=8)
    axes[0].set_title("family components", fontsize=10)
    axes[1].bar(requirement_rows["family"], requirement_rows["feature_candidate_count"], color="#7a8793", label="stage_h")
    axes[1].bar(requirement_rows["family"], requirement_rows["mysql_candidate_count"], color="#d9ae61", alpha=0.6, label="mysql")
    axes[1].tick_params(axis="x", labelrotation=25, labelsize=8)
    axes[1].legend(fontsize=8)
    axes[1].set_title("rotation field candidates", fontsize=10)
    fig.suptitle(_pick_label(font, "刚体约束与 rotation 审计", "Rigid-Body Rotation Audit"), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="rigid_body_rotation_audit",
        path=path,
        source_paths=[sources["rigid_body"]["path"], sources["rotation_audit"]["path"]],
        evidence_layer="rigid_body_rotation_diagnostics",
        table_path=table_paths["rigid_body_rotation_audit"],
        metric_definition="Compare family-level rigid-body components and pitch/roll/yaw candidate coverage.",
    )


def _figure_entry(
    *,
    figure_id: str,
    path: Path,
    source_paths: list[str],
    evidence_layer: str,
    table_path: str,
    metric_definition: str | None = None,
    case_definition: str | None = None,
) -> dict[str, object]:
    payload = {
        "figure_id": figure_id,
        "path": str(path),
        "exists": path.exists(),
        "source_path": source_paths,
        "evidence_layer": evidence_layer,
        "table_path": table_path,
    }
    if metric_definition is not None:
        payload["metric_definition"] = metric_definition
    if case_definition is not None:
        payload["case_definition"] = case_definition
    return payload


def _detect_plot_font() -> PlotFontSelection:
    try:
        from matplotlib import font_manager
    except Exception:
        return PlotFontSelection(None, True, "matplotlib unavailable; used ASCII-safe labels")
    available_names = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in DEFAULT_CJK_FONT_CANDIDATES:
        if candidate in available_names:
            return PlotFontSelection(candidate, False, f"using CJK font {candidate}")
    return PlotFontSelection(None, True, "CJK font missing; used ASCII-safe labels")


def _import_matplotlib(font: PlotFontSelection):
    from matplotlib import pyplot as plt
    from matplotlib import rcParams

    if font.family:
        rcParams["font.family"] = [font.family]
    rcParams["axes.unicode_minus"] = False
    return plt, rcParams


def _pick_label(font: PlotFontSelection, cn_label: str, ascii_label: str) -> str:
    return ascii_label if font.ascii_only else cn_label


def _infer_view_id(sample_id: str) -> str:
    return sample_id.split("::", 1)[0] if "::" in sample_id else sample_id


def render_stage_i_thesis_materials_report(
    *,
    run_id: str,
    sources: Mapping[str, Mapping[str, object]],
    table_entries: list[Mapping[str, object]],
    figure_entries: list[Mapping[str, object]],
) -> str:
    weak_label_rows = pd.DataFrame(_build_weak_label_rows(sources))
    best_rows = (
        weak_label_rows.sort_values("test_total")
        .groupby("sample_source", as_index=False)
        .first()[["sample_source", "sample_count", "task_entry_count", "best_test_total", "best_child_run_id"]]
    )
    summary_lines = [
        f"# Stage I Thesis Materials - {run_id}",
        "",
        "## Key Findings",
        "",
    ]
    for row in best_rows.to_dict(orient="records"):
        summary_lines.append(
            f"- `{row['sample_source']}`: sample_count=`{int(row['sample_count'])}`, "
            f"task_entry_count=`{int(row['task_entry_count'])}`, best_test_total=`{row['best_test_total']:.6f}`, "
            f"best_child_run_id=`{row['best_child_run_id']}`"
        )
    summary_lines.extend(
        [
            f"- rigid_body rotation audit: `{sources['rotation_audit']['payload']['rotation_status']}`",
            "",
            "## Tables",
            "",
        ]
    )
    summary_lines.extend(f"- `{entry['table_id']}`: `{entry['path']}`" for entry in table_entries)
    summary_lines.extend(["", "## Figures", ""])
    summary_lines.extend(f"- `{entry['figure_id']}`: `{entry['path']}`" for entry in figure_entries)
    return "\n".join(summary_lines)
