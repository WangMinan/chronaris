"""task evaluation runtime/service smoke facade and deployment-boundary artifacts."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix
from chronaris.modeling.common.run_observer import (
    StageIRunProgress,
    open_task_eval_run_observer,
)
from chronaris.schema.models import StreamKind
from chronaris.serving.runtime_inference import (
    StageIRuntimeInferenceConfig,
    StageIRuntimeInferenceRunResult,
    run_task_eval_runtime_inference,
)
from chronaris.serving.runtime_schema_contract import (
    build_runtime_schema_contract,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronaris-matplotlib")

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/runs"
DEFAULT_REPORT_ROOT = "docs/artifacts/runs"
DEFAULT_CJK_FONT_CANDIDATES = (
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans SC",
    "Source Han Sans SC",
    "AR PL UMing CN",
)


@dataclass(frozen=True, slots=True)
class StageIRuntimeSmokeConfig:
    run_id: str
    checkpoint_path: str
    sample_jsonl_path: str
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    device: str = "cpu"
    replay_mode: str = "both"
    batch_size: int | None = None
    max_windows: int | None = None
    strict_feature_schema: bool = False
    export_canonical_payload: bool = True


@dataclass(frozen=True, slots=True)
class StageIRuntimeSmokeRunResult:
    run_id: str
    artifact_root: str
    summary_path: str
    report_path: str
    figure_manifest_path: str
    predictions_jsonl_path: str
    error_cases_path: str
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PlotFontSelection:
    family: str | None
    ascii_only: bool
    note: str


def run_task_eval_runtime_smoke(
    config: StageIRuntimeSmokeConfig,
) -> StageIRuntimeSmokeRunResult:
    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="task_eval_runtime_service_smoke",
        logger=LOGGER,
        initial_progress={
            "artifact_root": str(run_root),
            "strict_feature_schema": config.strict_feature_schema,
            "export_canonical_payload": config.export_canonical_payload,
        },
    ) as progress:
        return _run_task_eval_runtime_smoke_observed(
            config=config,
            run_root=run_root,
            progress=progress,
        )


def _run_task_eval_runtime_smoke_observed(
    *,
    config: StageIRuntimeSmokeConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageIRuntimeSmokeRunResult:
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    raw_rows = _load_runtime_jsonl_rows(config.sample_jsonl_path)
    samples = tuple(_deserialize_runtime_sample_payload(row) for row in raw_rows)
    progress.update("input_loaded", sample_count=len(samples))

    schema_contract_result = build_runtime_schema_contract(
        checkpoint_path=config.checkpoint_path,
        samples=samples,
        output_root=run_root,
        export_canonical_payload=config.export_canonical_payload,
    )
    progress.update("schema_contract_written", schema_contract_path=schema_contract_result.schema_contract_path)

    runtime_result = _run_runtime_inference(
        config=config,
        run_root=run_root,
        samples=samples,
        strict_feature_schema=False,
        run_id_suffix="runtime",
        replay_mode=config.replay_mode,
    )
    progress.update("runtime_inference_finished", runtime_summary_path=runtime_result.summary_path)

    canonical_result = _run_runtime_inference(
        config=config,
        run_root=run_root,
        samples=schema_contract_result.canonical_samples,
        strict_feature_schema=True,
        run_id_suffix="canonical",
        replay_mode="batch",
    )
    progress.update("canonical_inference_finished", canonical_runtime_summary_path=canonical_result.summary_path)

    error_cases = _collect_error_cases(
        config=config,
        run_root=run_root,
        raw_rows=raw_rows,
        samples=samples,
        schema_contract=schema_contract_result.contract,
    )
    error_cases_path = run_root / "runtime_error_cases.json"
    error_cases_path.write_text(
        json.dumps(error_cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress.update("error_cases_written", error_case_count=len(error_cases))

    font = _detect_plot_font()
    native_runtime_summary = json.loads(Path(runtime_result.summary_path).read_text(encoding="utf-8"))
    canonical_runtime_summary = json.loads(Path(canonical_result.summary_path).read_text(encoding="utf-8"))
    figure_entries = _write_figures(
        run_root=run_root,
        font=font,
        config=config,
        runtime_result=runtime_result,
        error_cases=error_cases,
        schema_contract=schema_contract_result.contract,
        canonical_result=canonical_result,
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

    native_contract = schema_contract_result.contract["native_input"]
    canonical_contract = schema_contract_result.contract["canonical_payload"]
    summary = {
        "run_id": config.run_id,
        "status": "success",
        "artifact_root": str(run_root),
        "checkpoint_path": str(Path(config.checkpoint_path)),
        "sample_jsonl_path": str(Path(config.sample_jsonl_path)),
        "input_sample_count": len(samples),
        "view_count": len({_infer_view_id(sample) for sample in samples}),
        "view_ids": sorted({_infer_view_id(sample) for sample in samples}),
        "runtime_summary_path": runtime_result.summary_path,
        "predictions_jsonl_path": runtime_result.predictions_jsonl_path,
        "predictions_csv_path": runtime_result.predictions_csv_path,
        "canonical_runtime_summary_path": canonical_result.summary_path,
        "canonical_predictions_jsonl_path": canonical_result.predictions_jsonl_path,
        "canonical_payload_path": schema_contract_result.canonical_payload_path,
        "schema_contract_path": schema_contract_result.schema_contract_path,
        "error_cases_path": str(error_cases_path),
        "figure_manifest_path": str(figure_manifest_path),
        "diagnostics": native_runtime_summary.get("diagnostics"),
        "canonical_diagnostics": canonical_runtime_summary.get("diagnostics"),
        "incremental_consistency": native_runtime_summary.get("incremental_consistency"),
        "task_heads": native_runtime_summary.get("task_heads"),
        "native_feature_schema_status": native_contract["comparison"]["status"],
        "canonical_feature_schema_status": canonical_contract["comparison"]["status"],
        "expected_vehicle_feature_count": schema_contract_result.contract["expected_schema"]["vehicle"]["feature_count"],
        "input_vehicle_feature_count": native_contract["vehicle"]["feature_count"],
        "canonical_vehicle_feature_count": canonical_contract["vehicle"]["feature_count"],
        "missing_vehicle_feature_count": native_contract["comparison"]["vehicle"]["missing_feature_count"],
        "missing_vehicle_measurement_group_counts": native_contract["comparison"]["vehicle"]["missing_measurement_group_counts"],
        "strict_native_feature_schema_probe": _find_error_case(error_cases, "native_strict_feature_schema"),
    }
    summary_path = run_root / "runtime_service_smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_path = report_root / f"task-eval-runtime-service-smoke-{config.run_id}.md"
    report_path.write_text(
        render_task_eval_runtime_smoke_report(
            summary=summary,
            error_cases=error_cases,
            figure_entries=figure_entries,
        )
        + "\n",
        encoding="utf-8",
    )
    progress.finish(
        summary_path=str(summary_path),
        report_path=str(report_path),
        figure_manifest_path=str(figure_manifest_path),
    )
    return StageIRuntimeSmokeRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        report_path=str(report_path),
        figure_manifest_path=str(figure_manifest_path),
        predictions_jsonl_path=runtime_result.predictions_jsonl_path or "",
        error_cases_path=str(error_cases_path),
        summary=summary,
    )


def _run_runtime_inference(
    *,
    config: StageIRuntimeSmokeConfig,
    run_root: Path,
    samples: Sequence[E0ExperimentSample],
    strict_feature_schema: bool,
    run_id_suffix: str,
    replay_mode: str,
) -> StageIRuntimeInferenceRunResult:
    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return run_task_eval_runtime_inference(
        StageIRuntimeInferenceConfig(
            run_id=f"{config.run_id}-{run_id_suffix}",
            checkpoint_path=str(checkpoint_path),
            artifact_root=str(run_root / "runtime_inference"),
            report_root=config.report_root,
            device=config.device,
            export_predictions_csv=True,
            emit_predictions_jsonl=True,
            batch_size=config.batch_size,
            max_windows=config.max_windows,
            strict_feature_schema=strict_feature_schema,
            replay_mode=replay_mode,
        ),
        samples=samples,
    )


def _collect_error_cases(
    *,
    config: StageIRuntimeSmokeConfig,
    run_root: Path,
    raw_rows: Sequence[Mapping[str, object]],
    samples: Sequence[E0ExperimentSample],
    schema_contract: Mapping[str, object],
) -> list[dict[str, object]]:
    cases = []
    bad_checkpoint = str(Path(config.checkpoint_path).with_name("missing-checkpoint.pt"))
    cases.append(_capture_error_case("missing_checkpoint", "checkpoint path does not exist", lambda: _ensure_checkpoint_exists(bad_checkpoint)))
    broken_fields = dict(raw_rows[0])
    broken_fields.pop("vehicle", None)
    cases.append(_capture_error_case("missing_fields", "drop vehicle from the first payload", lambda: _validate_runtime_sample_payload(broken_fields)))
    empty_window = _build_empty_window_payload(raw_rows[0])
    cases.append(_capture_error_case("empty_window", "replace stream values with zero-length windows", lambda: _validate_runtime_sample_payload(empty_window)))
    mismatched_samples = list(samples)
    mismatched_samples[0] = _add_schema_mismatch_feature(samples[0])
    cases.append(
        _capture_error_case(
            "schema_mismatch",
            "add one extra vehicle feature and compare against checkpoint schema",
            lambda: _probe_schema_mismatch(
                expected_schema=schema_contract["expected_schema"],
                sample=mismatched_samples[0],
            ),
        )
    )
    if config.strict_feature_schema:
        cases.append(
            _capture_error_case(
                "native_strict_feature_schema",
                "run native input with strict_feature_schema=True",
                lambda: _probe_native_strict_feature_schema(
                    config=config,
                    run_root=run_root,
                    samples=samples,
                    schema_contract=schema_contract,
                ),
            )
        )
    return cases


def _capture_error_case(case_id: str, trigger: str, fn) -> dict[str, object]:
    try:
        fn()
    except Exception as exc:  # pragma: no cover - exercised in tests through outputs
        return {
            "case_id": case_id,
            "trigger": trigger,
            "status": "expected_failure",
            "error_type": exc.__class__.__name__,
            "message": str(exc),
        }
    return {
        "case_id": case_id,
        "trigger": trigger,
        "status": "unexpected_success",
        "error_type": None,
        "message": "expected an error but the probe succeeded",
    }


def _load_runtime_jsonl_rows(path_like: str) -> list[dict[str, object]]:
    path = Path(path_like)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"runtime sample jsonl is empty: {path}")
    for row in rows:
        _validate_runtime_sample_payload(row)
    return rows


def _validate_runtime_sample_payload(payload: Mapping[str, object]) -> None:
    required_fields = ("sample_id", "sortie_id", "start_offset_ms", "end_offset_ms", "physiology", "vehicle")
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"runtime payload missing required fields: {missing}")
    _validate_runtime_stream_payload(payload["physiology"], field_name="physiology")
    _validate_runtime_stream_payload(payload["vehicle"], field_name="vehicle")


def _validate_runtime_stream_payload(payload: Mapping[str, object], *, field_name: str) -> None:
    required_fields = ("stream_kind", "point_count", "feature_names", "point_offsets_ms", "point_measurements", "values")
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"runtime stream payload `{field_name}` missing fields: {missing}")
    point_count = int(payload["point_count"])
    values = payload.get("values", [])
    feature_names = payload.get("feature_names", [])
    if point_count <= 0 or not values:
        raise ValueError(f"empty window detected in `{field_name}` stream")
    if len(values) != point_count:
        raise ValueError(f"runtime stream payload `{field_name}` has inconsistent point_count")
    if not feature_names:
        raise ValueError(f"runtime stream payload `{field_name}` has no feature_names")


def _deserialize_runtime_sample_payload(payload: Mapping[str, object]) -> E0ExperimentSample:
    return E0ExperimentSample(
        sample_id=str(payload["sample_id"]),
        sortie_id=str(payload["sortie_id"]),
        start_offset_ms=int(payload["start_offset_ms"]),
        end_offset_ms=int(payload["end_offset_ms"]),
        physiology=_deserialize_runtime_stream(payload["physiology"]),
        vehicle=_deserialize_runtime_stream(payload["vehicle"]),
        notes=tuple(str(value) for value in payload.get("notes", [])),
    )


def _deserialize_runtime_stream(payload: Mapping[str, object]) -> NumericStreamMatrix:
    return NumericStreamMatrix(
        stream_kind=StreamKind(str(payload["stream_kind"])),
        point_count=int(payload["point_count"]),
        feature_names=tuple(str(value) for value in payload.get("feature_names", [])),
        point_offsets_ms=tuple(int(value) for value in payload.get("point_offsets_ms", [])),
        point_measurements=tuple(str(value) for value in payload.get("point_measurements", [])),
        values=tuple(tuple(float(value) for value in row) for row in payload.get("values", [])),
        dropped_fields=tuple(str(value) for value in payload.get("dropped_fields", [])),
    )


def _ensure_checkpoint_exists(path_like: str) -> None:
    if not Path(path_like).exists():
        raise FileNotFoundError(f"checkpoint not found: {path_like}")


def _build_empty_window_payload(payload: Mapping[str, object]) -> dict[str, object]:
    row = json.loads(json.dumps(payload))
    for field_name in ("physiology", "vehicle"):
        row[field_name]["point_count"] = 0
        row[field_name]["point_offsets_ms"] = []
        row[field_name]["point_measurements"] = []
        row[field_name]["values"] = []
    return row


def _add_schema_mismatch_feature(sample: E0ExperimentSample) -> E0ExperimentSample:
    return E0ExperimentSample(
        sample_id=sample.sample_id,
        sortie_id=sample.sortie_id,
        start_offset_ms=sample.start_offset_ms,
        end_offset_ms=sample.end_offset_ms,
        physiology=sample.physiology,
        vehicle=NumericStreamMatrix(
            stream_kind=sample.vehicle.stream_kind,
            point_count=sample.vehicle.point_count,
            feature_names=sample.vehicle.feature_names + ("schema.extra",),
            point_offsets_ms=sample.vehicle.point_offsets_ms,
            point_measurements=sample.vehicle.point_measurements,
            values=tuple(tuple(list(row) + [0.0]) for row in sample.vehicle.values),
            dropped_fields=sample.vehicle.dropped_fields,
        ),
        notes=sample.notes,
    )


def _probe_schema_mismatch(
    *,
    expected_schema: Mapping[str, object],
    sample: E0ExperimentSample,
) -> None:
    physiology_expected = tuple(expected_schema["physiology"]["feature_names"])
    vehicle_expected = tuple(expected_schema["vehicle"]["feature_names"])
    missing_physiology = len([name for name in physiology_expected if name not in sample.physiology.feature_names])
    extra_physiology = len([name for name in sample.physiology.feature_names if name not in physiology_expected])
    missing_vehicle = [name for name in vehicle_expected if name not in sample.vehicle.feature_names]
    extra_vehicle = [name for name in sample.vehicle.feature_names if name not in vehicle_expected]
    if missing_physiology or extra_physiology or missing_vehicle or extra_vehicle:
        raise ValueError(
            "runtime feature schema mismatch: "
            f"missing_physiology_count={missing_physiology}, "
            f"extra_physiology_count={extra_physiology}, "
            f"missing_vehicle_count={len(missing_vehicle)}, "
            f"extra_vehicle_count={len(extra_vehicle)}, "
            f"extra_vehicle_names={extra_vehicle[:5]}"
        )


def _probe_native_strict_feature_schema(
    *,
    config: StageIRuntimeSmokeConfig,
    run_root: Path,
    samples: Sequence[E0ExperimentSample],
    schema_contract: Mapping[str, object],
) -> None:
    try:
        _run_runtime_inference(
            config=config,
            run_root=run_root,
            samples=samples,
            strict_feature_schema=True,
            run_id_suffix="native-strict-probe",
            replay_mode="batch",
        )
    except ValueError as error:
        vehicle_comparison = schema_contract["native_input"]["comparison"]["vehicle"]
        grouped = dict(vehicle_comparison.get("missing_measurement_group_counts") or {})
        top_groups = dict(list(grouped.items())[:6])
        raise ValueError(
            "native strict feature schema mismatch: "
            f"missing_vehicle_count={vehicle_comparison.get('missing_feature_count')}, "
            f"extra_vehicle_count={vehicle_comparison.get('extra_feature_count')}, "
            f"missing_measurement_groups={top_groups}"
        ) from error


def _write_figures(
    *,
    run_root: Path,
    font: PlotFontSelection,
    config: StageIRuntimeSmokeConfig,
    runtime_result: StageIRuntimeInferenceRunResult,
    error_cases: Sequence[Mapping[str, object]],
    schema_contract: Mapping[str, object],
    canonical_result: StageIRuntimeInferenceRunResult,
) -> list[dict[str, object]]:
    runtime_summary = json.loads(Path(runtime_result.summary_path).read_text(encoding="utf-8"))
    return [
        _plot_runtime_service_flow(run_root / "runtime_service_flow.png", font, config, runtime_result, schema_contract),
        _plot_runtime_payload_schema(run_root / "runtime_payload_schema.png", font, config, runtime_summary, schema_contract, canonical_result),
        _plot_runtime_error_cases(run_root / "runtime_error_cases.png", font, error_cases, config, runtime_result, schema_contract),
    ]


def _plot_runtime_service_flow(path: Path, font: PlotFontSelection, config: StageIRuntimeSmokeConfig, runtime_result: StageIRuntimeInferenceRunResult, schema_contract: Mapping[str, object]) -> dict[str, object]:
    plt, _ = _import_matplotlib(font)
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.4, 2.45))
    box_w = 1.55
    box_h = 0.48
    box_y = 0.86
    arrow_y = box_y + box_h / 2
    stages = [
        (0.55, "checkpoint", "model weights"),
        (3.55, "native jsonl", "runtime samples"),
        (6.55, "schema contract", "native + canonical"),
        (9.55, "service replay", "prediction JSONL"),
    ]
    for x0, title, subtitle in stages:
        ax.add_patch(
            FancyBboxPatch(
                (x0, box_y),
                box_w,
                box_h,
                boxstyle="round,pad=0.05,rounding_size=0.04",
                facecolor="#dcebdc",
                edgecolor="#355b3e",
                linewidth=1.25,
                zorder=2,
            )
        )
        ax.text(x0 + box_w / 2, box_y + 0.31, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#1f3d2b", zorder=3)
        ax.text(x0 + box_w / 2, box_y + 0.14, subtitle, ha="center", va="center", fontsize=6.8, color="#40564a", zorder=3)
    for idx, (x0, _, _) in enumerate(stages[:-1]):
        next_x = stages[idx + 1][0]
        ax.annotate(
            "",
            xy=(next_x - 0.22, arrow_y),
            xytext=(x0 + box_w + 0.22, arrow_y),
            arrowprops={"arrowstyle": "->", "lw": 1.55, "color": "#355b3e", "shrinkA": 0, "shrinkB": 0},
            zorder=1,
        )
    ax.set_xlim(0, 11.65)
    ax.set_ylim(0.68, 1.54)
    ax.set_axis_off()
    ax.set_title(_pick_label(font, "runtime/service 流程", "Runtime Service Flow"), fontsize=12, pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="runtime_service_flow",
        path=path,
        source_paths=[config.sample_jsonl_path, config.checkpoint_path, runtime_result.summary_path],
        evidence_layer="runtime_service",
        metric_definition="Checkpoint cold-start, native payload validation, schema contract, and service replay outputs.",
    )


def _plot_runtime_payload_schema(path: Path, font: PlotFontSelection, config: StageIRuntimeSmokeConfig, runtime_summary: Mapping[str, object], schema_contract: Mapping[str, object], canonical_result: StageIRuntimeInferenceRunResult) -> dict[str, object]:
    plt, _ = _import_matplotlib(font)
    native = schema_contract["native_input"]
    canonical = schema_contract["canonical_payload"]
    expected_vehicle = schema_contract["expected_schema"]["vehicle"]["feature_count"]
    native_vehicle = native["vehicle"]["feature_count"]
    canonical_vehicle = canonical["vehicle"]["feature_count"]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_axis_off()
    left = [
        "native JSONL",
        f"vehicle feature count = {native_vehicle}",
        f"native status = {native['comparison']['status']}",
        f"missing vehicle features = {native['comparison']['vehicle']['missing_feature_count']}",
        f"feature_schema_status = {runtime_summary['diagnostics']['feature_schema_status']}",
    ]
    right = [
        "canonical JSONL",
        f"vehicle feature count = {canonical_vehicle}",
        f"canonical status = {canonical['comparison']['status']}",
        f"expected vehicle feature count = {expected_vehicle}",
        f"strict runtime report = {Path(canonical_result.summary_path).name}",
    ]
    ax.text(0.18, 0.92, "Native Payload", fontsize=12, fontweight="bold", ha="center")
    ax.text(0.78, 0.92, "Canonical Payload", fontsize=12, fontweight="bold", ha="center")
    for index, text in enumerate(left):
        ax.text(0.05, 0.8 - index * 0.14, f"- {text}", fontsize=9, ha="left")
    for index, text in enumerate(right):
        ax.text(0.58, 0.8 - index * 0.14, f"- {text}", fontsize=9, ha="left")
    ax.annotate("", xy=(0.55, 0.55), xytext=(0.43, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#355b3e"})
    ax.text(0.5, 0.18, f"vehicle schema gap: {native_vehicle} -> {expected_vehicle}", fontsize=11, ha="center")
    ax.set_title(_pick_label(font, "runtime payload schema", "Runtime Payload Schema"), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="runtime_payload_schema",
        path=path,
        source_paths=[config.sample_jsonl_path, str(Path(canonical_result.summary_path).with_name("runtime_inference_predictions.jsonl")), str(Path(config.artifact_root) / config.run_id / "runtime_schema_contract.json")],
        evidence_layer="runtime_service",
        metric_definition="Native payload stays aligned while canonical payload proves exact service-layer schema.",
    )


def _plot_runtime_error_cases(path: Path, font: PlotFontSelection, error_cases: Sequence[Mapping[str, object]], config: StageIRuntimeSmokeConfig, runtime_result: StageIRuntimeInferenceRunResult, schema_contract: Mapping[str, object]) -> dict[str, object]:
    plt, _ = _import_matplotlib(font)
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_axis_off()
    ax.text(0.02, 0.95, _pick_label(font, "错误样例与诊断字段", "Runtime Error Cases"), fontsize=12, fontweight="bold", ha="left")
    for index, row in enumerate(error_cases):
        y0 = 0.84 - index * 0.16
        ax.text(0.03, y0, f"{row['case_id']} ({row['status']})", fontsize=10, fontweight="bold", ha="left")
        ax.text(0.03, y0 - 0.05, f"trigger: {row['trigger']}", fontsize=8, ha="left")
        ax.text(0.03, y0 - 0.10, f"{row['error_type']}: {row['message']}", fontsize=8, ha="left")
    ax.text(
        0.03,
        0.06,
        f"native={schema_contract['native_input']['comparison']['status']} | canonical={schema_contract['canonical_payload']['comparison']['status']}",
        fontsize=9,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _figure_entry(
        figure_id="runtime_error_cases",
        path=path,
        source_paths=[config.sample_jsonl_path, runtime_result.summary_path, str(Path(config.artifact_root) / config.run_id / "runtime_schema_contract.json")],
        evidence_layer="runtime_service",
        case_definition="Negative probes include missing checkpoint, missing field, empty window, schema mismatch, and native strict schema failure when requested.",
    )


def _figure_entry(
    *,
    figure_id: str,
    path: Path,
    source_paths: list[str],
    evidence_layer: str,
    metric_definition: str | None = None,
    case_definition: str | None = None,
) -> dict[str, object]:
    row = {
        "figure_id": figure_id,
        "path": str(path),
        "exists": path.exists(),
        "source_path": source_paths,
        "evidence_layer": evidence_layer,
    }
    if metric_definition is not None:
        row["metric_definition"] = metric_definition
    if case_definition is not None:
        row["case_definition"] = case_definition
    return row


def _infer_view_id(sample: E0ExperimentSample) -> str:
    return sample.sample_id.split("::", 1)[0] if "::" in sample.sample_id else sample.sortie_id


def _find_error_case(error_cases: Sequence[Mapping[str, object]], case_id: str) -> Mapping[str, object] | None:
    for row in error_cases:
        if row.get("case_id") == case_id:
            return row
    return None


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


def render_task_eval_runtime_smoke_report(
    *,
    summary: Mapping[str, object],
    error_cases: Sequence[Mapping[str, object]],
    figure_entries: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        f"# task evaluation Runtime Service Smoke - {summary['run_id']}",
        "",
        f"- checkpoint_path: `{summary['checkpoint_path']}`",
        f"- sample_jsonl_path: `{summary['sample_jsonl_path']}`",
        f"- input_sample_count: `{summary['input_sample_count']}`",
        f"- view_ids: `{summary['view_ids']}`",
        f"- runtime_summary_path: `{summary['runtime_summary_path']}`",
        f"- canonical_runtime_summary_path: `{summary['canonical_runtime_summary_path']}`",
        f"- schema_contract_path: `{summary['schema_contract_path']}`",
        f"- native_feature_schema_status: `{summary['native_feature_schema_status']}`",
        f"- canonical_feature_schema_status: `{summary['canonical_feature_schema_status']}`",
        f"- vehicle_feature_gap: `{summary['input_vehicle_feature_count']} -> {summary['expected_vehicle_feature_count']}`",
        "",
        "## Error Cases",
        "",
    ]
    for row in error_cases:
        lines.append(f"- `{row['case_id']}`: `{row['status']}` | `{row['error_type']}` | `{row['message']}`")
    lines.extend(["", "## Figures", ""])
    lines.extend(f"- `{entry['figure_id']}`: `{entry['path']}`" for entry in figure_entries)
    return "\n".join(lines)
