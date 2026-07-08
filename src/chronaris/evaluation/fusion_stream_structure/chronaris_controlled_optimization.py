"""Controlled Chronaris candidate optimization for E3 follow-up."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from chronaris.evaluation.dingxin.pipelines.benchmark_data import (
    CLASS_LABEL_TO_ID,
    TASK_MANEUVER,
    TASK_RESPONSE,
    derive_private_proxy_task_entries,
)
from chronaris.evaluation.dingxin.pipelines.thirdparty_comparison import (
    StageIPrivateThirdPartyComparisonConfig,
    _sequence_arrays,
)
from chronaris.evaluation.fusion_stream_structure.contracts import validate_input_frame
from chronaris.evaluation.fusion_stream_structure.deep_baseline_representation_export import (
    build_deep_baseline_sequence_frame,
    train_fold_and_export_embeddings,
)
from chronaris.evaluation.fusion_stream_structure.deep_baseline_representation_io import (
    fold_status_row,
    json_default,
    repo_rel,
    validate_deep_baseline_representation_frame,
)
from chronaris.evaluation.fusion_stream_structure.deep_baseline_representation_types import (
    DeepBaselineRepresentationExportConfig,
    REPO_ROOT,
)
from chronaris.evaluation.fusion_stream_structure.chronaris_controlled_specs import (
    E3_TARGET_METRICS,
    LOWER_IS_BETTER,
    REPRESENTATION_FAMILY,
    ChronarisCandidateSpec,
    ControlledOptimizationConfig,
    default_candidate_specs,
)
from chronaris.evaluation.fusion_stream_structure.chronaris_controlled_outputs import (
    candidate_summary_base,
    write_blocked_outputs,
    write_candidate_metric_tables,
    write_candidate_registry,
    write_failure_cases,
    write_locked_outputs,
    write_no_locked_candidate,
    write_paper_boundary,
    write_pareto_selection,
    write_representation_outputs,
    write_dev_manifest,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name, seed_torch


def run_controlled_optimization(config: ControlledOptimizationConfig) -> dict[str, object]:
    output_root = _resolve_path(config.output_root)
    dev_root = output_root / config.dev_run_id
    confirm_root = output_root / config.confirm_run_id
    representation_root = output_root / config.representation_run_id
    for root in (dev_root, confirm_root, representation_root):
        root.mkdir(parents=True, exist_ok=True)

    specs = default_candidate_specs()
    if config.max_candidates is not None:
        specs = specs[: max(0, int(config.max_candidates))]
    write_candidate_registry(dev_root / "candidate_registry.csv", specs)

    runtime_device = resolve_torch_device_name(config.device)
    if config.require_cuda and runtime_device != "cuda":
        return write_blocked_outputs(config, dev_root, confirm_root, representation_root, specs, runtime_device)

    base_export_config = _base_export_config(config, epochs=config.dev_epochs)
    sequence_frame, build_context = build_deep_baseline_sequence_frame(base_export_config)
    arrays, masks, time_axis, labels = _sequence_arrays(sequence_frame)
    groups = tuple(sorted(sequence_frame["view_id"].astype(str).unique()))
    label_maps = _label_maps(build_context["records"])
    old_refs = _old_reference_metrics(config)

    all_embedding_rows: list[dict[str, object]] = []
    all_checkpoint_rows: list[dict[str, object]] = []
    all_fold_status_rows: list[dict[str, object]] = []
    all_curve_rows: list[dict[str, object]] = []
    t1_t2_rows: list[dict[str, object]] = []
    e3_rows: list[dict[str, object]] = []
    candidate_summaries: list[dict[str, object]] = []

    smoke_passed: list[ChronarisCandidateSpec] = []
    for spec in specs:
        result = _run_candidate_phase(
            spec=spec,
            phase="smoke",
            config=config,
            base_export_config=base_export_config,
            sequence_frame=sequence_frame,
            arrays=arrays,
            masks=masks,
            time_axis=time_axis,
            labels=labels,
            groups=groups[: max(1, int(config.smoke_max_folds))],
            runtime_device=runtime_device,
            epochs=int(config.smoke_epochs),
            phase_root=dev_root / "smoke_runs" / spec.candidate_id,
            label_maps=label_maps,
            run_e3=False,
            old_refs=old_refs,
        )
        t1_t2_rows.extend(result["metric_rows"])
        all_checkpoint_rows.extend(result["checkpoint_rows"])
        all_fold_status_rows.extend(result["fold_status_rows"])
        all_curve_rows.extend(result["curve_rows"])
        smoke_ok = bool(result["status"] == "completed" and result["completed_fold_count"] > 0)
        candidate_summaries.append({**candidate_summary_base(spec), "smoke_status": result["status"], "smoke_passed": smoke_ok})
        if smoke_ok:
            smoke_passed.append(spec)

    dev_results: dict[str, Mapping[str, object]] = {}
    for spec in smoke_passed:
        result = _run_candidate_phase(
            spec=spec,
            phase="dev",
            config=config,
            base_export_config=base_export_config,
            sequence_frame=sequence_frame,
            arrays=arrays,
            masks=masks,
            time_axis=time_axis,
            labels=labels,
            groups=groups if config.dev_max_folds is None else groups[: max(0, int(config.dev_max_folds))],
            runtime_device=runtime_device,
            epochs=int(config.dev_epochs),
            phase_root=representation_root / "candidate_runs" / spec.candidate_id / "dev",
            label_maps=label_maps,
            run_e3=True,
            old_refs=old_refs,
            e3_output_root=dev_root / "e3_runs",
        )
        dev_results[spec.candidate_id] = result
        all_embedding_rows.extend(result["embedding_rows"])
        all_checkpoint_rows.extend(result["checkpoint_rows"])
        all_fold_status_rows.extend(result["fold_status_rows"])
        all_curve_rows.extend(result["curve_rows"])
        t1_t2_rows.extend(result["metric_rows"])
        e3_rows.extend(result["e3_metric_rows"])

    selection = _select_pareto_candidate(dev_results, old_refs)
    write_candidate_metric_tables(dev_root, t1_t2_rows, e3_rows)
    write_pareto_selection(dev_root / "pareto_selection.md", selection, old_refs)
    write_failure_cases(dev_root / "failure_cases.md", specs, smoke_passed, selection)
    write_paper_boundary(dev_root / "paper_boundary.md", selection)

    locked_result: Mapping[str, object] | None = None
    if selection["selected_candidate_id"] and config.run_confirmation:
        selected_spec = next(spec for spec in specs if spec.candidate_id == selection["selected_candidate_id"])
        locked_result = _run_candidate_phase(
            spec=selected_spec,
            phase="locked_confirmation",
            config=config,
            base_export_config=base_export_config,
            sequence_frame=sequence_frame,
            arrays=arrays,
            masks=masks,
            time_axis=time_axis,
            labels=labels,
            groups=groups,
            runtime_device=runtime_device,
            epochs=int(config.confirm_epochs),
            phase_root=representation_root / "candidate_runs" / selected_spec.candidate_id / "locked_confirmation",
            label_maps=label_maps,
            run_e3=True,
            old_refs=old_refs,
            e3_output_root=confirm_root / "e3_runs",
        )
        all_embedding_rows.extend(locked_result["embedding_rows"])
        all_checkpoint_rows.extend(locked_result["checkpoint_rows"])
        all_fold_status_rows.extend(locked_result["fold_status_rows"])
        all_curve_rows.extend(locked_result["curve_rows"])
        write_locked_outputs(confirm_root, selected_spec, locked_result, selection, config, old_refs)
    else:
        write_no_locked_candidate(confirm_root / "no_locked_candidate.md", selection, dev_results)

    write_representation_outputs(
        representation_root=representation_root,
        config=config,
        specs=specs,
        embedding_rows=all_embedding_rows,
        checkpoint_rows=all_checkpoint_rows,
        fold_status_rows=all_fold_status_rows,
        curve_rows=all_curve_rows,
        sequence_frame=sequence_frame,
        runtime_device=runtime_device,
    )
    summary = {
        "run_id": config.dev_run_id,
        "status": "completed",
        "training_invoked": True,
        "candidate_count": len(specs),
        "smoke_pass_count": len(smoke_passed),
        "dev_completed_count": sum(1 for result in dev_results.values() if result.get("status") == "completed"),
        "pareto_selected_candidate": selection["selected_candidate_id"],
        "locked_candidate_id": locked_result.get("candidate_id") if locked_result else None,
        "confirmed_metrics_changed": False,
        "thesis_protocol_snapshot_modified": False,
        "old_deep_baseline_modified": False,
        "old_e3_validation_modified": False,
        "representation_root": repo_rel(representation_root),
        "dev_root": repo_rel(dev_root),
        "confirm_root": repo_rel(confirm_root),
        "selection": selection,
        "old_reference_metrics": old_refs,
    }
    (dev_root / "candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    write_dev_manifest(dev_root, config, specs, summary)
    return summary


def _run_candidate_phase(
    *,
    spec: ChronarisCandidateSpec,
    phase: str,
    config: ControlledOptimizationConfig,
    base_export_config: DeepBaselineRepresentationExportConfig,
    sequence_frame: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    time_axis: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
    runtime_device: str,
    epochs: int,
    phase_root: Path,
    label_maps: Mapping[str, Mapping[str, object]],
    run_e3: bool,
    old_refs: Mapping[str, object],
    e3_output_root: Path | None = None,
) -> dict[str, object]:
    phase_root.mkdir(parents=True, exist_ok=True)
    export_config = replace(
        base_export_config,
        run_id=f"{config.representation_run_id}-{phase}-{spec.candidate_id}",
        models=(spec.model_name,),
        epochs=int(epochs),
        hidden_dim=int(spec.hidden_dim),
        dropout=float(spec.dropout),
        weight_decay=float(spec.weight_decay),
        learning_rate=float(spec.learning_rate),
        layers=int(spec.layers),
        num_heads=int(spec.num_heads),
        representation_family=REPRESENTATION_FAMILY,
        skip_completed=True,
    )
    thirdparty_config = _thirdparty_config_from_export(export_config)
    seed_torch(int(config.seed), device=runtime_device)
    embedding_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    fold_status_rows: list[dict[str, object]] = []
    for fold_index, fold_group in enumerate(groups, start=1):
        train_idx = sequence_frame.index[sequence_frame["view_id"].astype(str) != fold_group].to_numpy(dtype=int)
        test_idx = sequence_frame.index[sequence_frame["view_id"].astype(str) == fold_group].to_numpy(dtype=int)
        if len(train_idx) == 0 or len(test_idx) == 0:
            fold_status_rows.append(fold_status_row(export_config, spec.model_name, fold_index, fold_group, train_idx, test_idx, "skipped", reason="empty_train_or_test"))
            continue
        try:
            rows, curves, checkpoint_row, status_rows = train_fold_and_export_embeddings(
                config=export_config,
                thirdparty_config=thirdparty_config,
                frame=sequence_frame,
                arrays=arrays,
                masks=masks,
                time_axis=time_axis,
                labels=labels,
                model_name=spec.model_name,
                fold_index=fold_index,
                fold_group=fold_group,
                train_idx=train_idx,
                test_idx=test_idx,
                runtime_device=runtime_device,
                run_root=phase_root,
            )
            rows = _normalize_candidate_embedding_rows(rows, spec=spec, phase=phase)
            embedding_rows.extend(rows)
            curve_rows.extend(_annotate_rows(curves, spec=spec, phase=phase))
            if checkpoint_row:
                checkpoint_rows.append({**checkpoint_row, "candidate_id": spec.candidate_id, "candidate_phase": phase})
            fold_status_rows.extend(_annotate_rows(status_rows, spec=spec, phase=phase))
        except Exception as exc:  # pragma: no cover - host/runtime dependent.
            fold_status_rows.append(
                {
                    **fold_status_row(export_config, spec.model_name, fold_index, fold_group, train_idx, test_idx, "failed", reason=type(exc).__name__ + ":" + str(exc), traceback_text=traceback.format_exc()),
                    "candidate_id": spec.candidate_id,
                    "candidate_phase": phase,
                }
            )
    processed_rows = _postprocess_embedding_rows(embedding_rows, spec)
    candidate_frame = pd.DataFrame(processed_rows)
    if not candidate_frame.empty:
        validate_deep_baseline_representation_frame(candidate_frame)
        candidate_frame.to_csv(phase_root / "chronaris_oof_embeddings_long.csv", index=False)
    pd.DataFrame(checkpoint_rows).to_csv(phase_root / "checkpoint_manifest.csv", index=False)
    pd.DataFrame(fold_status_rows).to_csv(phase_root / "fold_status.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(phase_root / "training_curves.csv", index=False)
    metric_rows = _candidate_t1_t2_metrics(
        candidate_frame,
        fold_status_rows,
        labels=label_maps,
        spec=spec,
        phase=phase,
        old_refs=old_refs,
    )
    e3_metric_rows: list[dict[str, object]] = []
    e3_run_root = None
    if run_e3 and not candidate_frame.empty and e3_output_root is not None:
        e3_metric_rows, e3_run_root = _run_candidate_e3(
            candidate_frame=candidate_frame,
            spec=spec,
            phase=phase,
            config=config,
            e3_output_root=e3_output_root,
            phase_root=phase_root,
            old_refs=old_refs,
        )
    t2_rmse = _metric_value(metric_rows, "rmse")
    t1_macro = _metric_value(metric_rows, "macro_f1")
    result = {
        "candidate_id": spec.candidate_id,
        "candidate_phase": phase,
        "status": "completed" if any(row.get("status") in {"completed", "completed_skipped"} for row in fold_status_rows) else "failed",
        "completed_fold_count": sum(1 for row in fold_status_rows if row.get("status") in {"completed", "completed_skipped"}),
        "expected_fold_count": len(groups),
        "embedding_rows": processed_rows,
        "checkpoint_rows": checkpoint_rows,
        "fold_status_rows": fold_status_rows,
        "curve_rows": curve_rows,
        "metric_rows": metric_rows,
        "e3_metric_rows": e3_metric_rows,
        "e3_run_root": repo_rel(e3_run_root) if e3_run_root else None,
        "t2_rmse": t2_rmse,
        "t1_macro_f1": t1_macro,
        "e3_positive_signal_count": sum(1 for row in e3_metric_rows if row.get("e3_target_improved_vs_old_chronaris") is True),
    }
    (phase_root / "candidate_phase_summary.json").write_text(json.dumps({k: v for k, v in result.items() if k not in {"embedding_rows", "checkpoint_rows", "fold_status_rows", "curve_rows"}}, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    return result


def _thirdparty_config_from_export(config: DeepBaselineRepresentationExportConfig) -> StageIPrivateThirdPartyComparisonConfig:
    return StageIPrivateThirdPartyComparisonConfig(
        run_id=config.run_id,
        e_run_manifest_path=config.e_run_manifest_path,
        f_run_manifest_path=config.f_run_manifest_path,
        output_root=config.output_root,
        report_root=config.output_root,
        models=tuple(config.models),
        seeds=(int(config.seed),),
        split_strategy=(config.split_strategy,),
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        learning_rate=float(config.learning_rate),
        hidden_dim=int(config.hidden_dim),
        num_heads=int(config.num_heads),
        layers=int(config.layers),
        dropout=float(config.dropout),
        weight_decay=float(config.weight_decay),
        grad_clip_norm=float(config.grad_clip_norm),
        device=config.device,
        require_cuda=bool(config.require_cuda),
        resume=bool(config.resume),
        skip_completed=bool(config.skip_completed),
        allow_partial=bool(config.allow_partial),
        heartbeat_seconds=float(config.heartbeat_seconds),
        batch_log_interval=int(config.batch_log_interval),
        tensor_cache=config.tensor_cache,
        max_cache_gb=float(config.max_cache_gb),
        pin_memory=bool(config.pin_memory),
        non_blocking_copy=bool(config.non_blocking_copy),
        auto_batch_size=bool(config.auto_batch_size),
        batch_size_candidates=tuple(int(value) for value in config.batch_size_candidates),
        amp=config.amp,
        grad_scaler=bool(config.grad_scaler),
        amp_eval=bool(config.amp_eval),
        torch_compile=config.torch_compile,
        profile_gpu=bool(config.profile_gpu),
        eval_batch_size=config.eval_batch_size,
        num_workers=int(config.num_workers),
        checkpoint_policy=config.checkpoint_policy,
    )


def _base_export_config(config: ControlledOptimizationConfig, *, epochs: int) -> DeepBaselineRepresentationExportConfig:
    return DeepBaselineRepresentationExportConfig(
        run_id=config.representation_run_id,
        output_root=config.output_root,
        e_run_manifest_path=config.e_run_manifest_path,
        f_run_manifest_path=config.f_run_manifest_path,
        models=("chronaris_v2_task_heads",),
        task_name=TASK_RESPONSE,
        task_type="regression",
        split_strategy="leave_one_view_out",
        seed=int(config.seed),
        epochs=int(epochs),
        batch_size=int(config.batch_size),
        eval_batch_size=config.eval_batch_size,
        device=config.device,
        require_cuda=bool(config.require_cuda),
        tensor_cache=config.tensor_cache,
        max_cache_gb=float(config.max_cache_gb),
        pin_memory=bool(config.pin_memory),
        non_blocking_copy=bool(config.non_blocking_copy),
        auto_batch_size=bool(config.auto_batch_size),
        amp=config.amp,
        torch_compile="off",
        representation_family=REPRESENTATION_FAMILY,
        skip_completed=True,
        allow_partial=True,
    )


def _normalize_candidate_embedding_rows(rows: Sequence[Mapping[str, object]], *, spec: ChronarisCandidateSpec, phase: str) -> list[dict[str, object]]:
    normalized = []
    for row in rows:
        item = dict(row)
        item["method_name"] = "chronaris"
        item["source_model_name"] = spec.model_name
        item["candidate_id"] = spec.candidate_id
        item["candidate_phase"] = phase
        item["representation_postprocess"] = spec.representation_postprocess
        item["representation_family"] = REPRESENTATION_FAMILY
        normalized.append(item)
    return normalized


def _postprocess_embedding_rows(rows: Sequence[Mapping[str, object]], spec: ChronarisCandidateSpec) -> list[dict[str, object]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows).sort_values(["sortie_id", "view_id", "time", "window_id"], kind="mergesort").reset_index(drop=True)
    feature_cols = _feature_columns(frame)
    values = frame.loc[:, feature_cols].to_numpy(dtype=np.float32)
    if spec.representation_postprocess in {"l2_normalized", "normalized_temporal_delta_append"}:
        denom = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.maximum(denom, 1e-8)
        frame.loc[:, feature_cols] = values
    if spec.representation_postprocess in {"temporal_delta_append", "normalized_temporal_delta_append"}:
        delta_blocks = []
        for _key, group in frame.groupby(["sortie_id", "view_id"], sort=False):
            group_values = group.loc[:, feature_cols].to_numpy(dtype=np.float32)
            deltas = np.zeros_like(group_values)
            deltas[1:] = group_values[1:] - group_values[:-1]
            delta_blocks.append(pd.DataFrame(deltas, index=group.index))
        delta_frame = pd.concat(delta_blocks).sort_index()
        start = len(feature_cols) + 1
        delta_frame.columns = [f"fusion_feature_{start + offset}" for offset in range(delta_frame.shape[1])]
        frame = pd.concat([frame, delta_frame.astype(float)], axis=1)
    return frame.to_dict(orient="records")


def _candidate_t1_t2_metrics(
    candidate_frame: pd.DataFrame,
    fold_status_rows: Sequence[Mapping[str, object]],
    *,
    labels: Mapping[str, Mapping[str, object]],
    spec: ChronarisCandidateSpec,
    phase: str,
    old_refs: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fold_frame = pd.DataFrame(list(fold_status_rows))
    for metric in ("rmse", "mae", "nrmse"):
        column = f"metric_{metric}"
        values = pd.to_numeric(fold_frame.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        value = float(values.mean()) if not values.empty else None
        rows.append(_metric_row(spec, phase, TASK_RESPONSE, metric, value, "lower_is_better", "direct_t2_head_fold_mean", old_refs))
    t1 = _evaluate_t1_probe(candidate_frame, labels.get(TASK_MANEUVER, {}), seed=17)
    for metric, value in t1.items():
        rows.append(_metric_row(spec, phase, TASK_MANEUVER, metric, value, "higher_is_better", "linear_probe_on_held_out_t2_embeddings", old_refs))
    return rows


def _evaluate_t1_probe(candidate_frame: pd.DataFrame, label_by_sample: Mapping[str, object], *, seed: int) -> dict[str, float | None]:
    if candidate_frame.empty:
        return {"macro_f1": None, "balanced_accuracy": None}
    frame = candidate_frame.copy()
    frame["t1_label"] = [label_by_sample.get(str(window_id)) for window_id in frame["window_id"].astype(str)]
    frame = frame.dropna(subset=["t1_label"]).copy()
    if frame.empty:
        return {"macro_f1": None, "balanced_accuracy": None}
    frame["y"] = [CLASS_LABEL_TO_ID[str(value)] for value in frame["t1_label"]]
    feature_cols = _feature_columns(frame)
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    for group in sorted(frame["view_id"].astype(str).unique()):
        train = frame[frame["view_id"].astype(str) != group]
        test = frame[frame["view_id"].astype(str) == group]
        if train.empty or test.empty or train["y"].nunique() < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed),
        )
        model.fit(train[feature_cols].to_numpy(dtype=float), train["y"].to_numpy(dtype=int))
        pred = model.predict(test[feature_cols].to_numpy(dtype=float))
        y_true_all.extend(test["y"].to_numpy(dtype=int).tolist())
        y_pred_all.extend(pred.astype(int).tolist())
    if not y_true_all:
        return {"macro_f1": None, "balanced_accuracy": None}
    return {
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
    }


def _run_candidate_e3(
    *,
    candidate_frame: pd.DataFrame,
    spec: ChronarisCandidateSpec,
    phase: str,
    config: ControlledOptimizationConfig,
    e3_output_root: Path,
    phase_root: Path,
    old_refs: Mapping[str, object],
) -> tuple[list[dict[str, object]], Path | None]:
    e3_output_root.mkdir(parents=True, exist_ok=True)
    input_path = phase_root / "candidate_four_method_e3_input_long.csv"
    baseline = pd.read_csv(_resolve_path(config.old_baseline_input_path), low_memory=False)
    baseline = baseline[baseline["method_name"].astype(str).isin(["naive_time_sync", "mult", "contiformer"])].copy()
    baseline = baseline[baseline["window_id"].astype(str).isin(set(candidate_frame["window_id"].astype(str)))].copy()
    combined = pd.concat([candidate_frame, baseline], ignore_index=True, sort=False)
    validate_input_frame(combined)
    combined.to_csv(input_path, index=False)
    run_id = f"{phase}_{spec.candidate_id}_e3"
    command = [
        sys.executable,
        "scripts/evaluation/fusion_stream_structure/run_fusion_stream_structure_benchmark.py",
        "--fusion-stream-table",
        str(input_path),
        "--methods",
        "chronaris",
        "naive_time_sync",
        "mult",
        "contiformer",
        "--output-root",
        str(e3_output_root),
        "--run-id",
        run_id,
        "--min-T",
        str(config.min_T),
        "--source-training-run",
        str(phase_root),
        "--representation-family",
        REPRESENTATION_FAMILY,
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    (phase_root / "e3_command.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return [
            {
                "candidate_id": spec.candidate_id,
                "candidate_phase": phase,
                "metric": "e3_run",
                "status": "failed",
                "reason": f"returncode={completed.returncode}",
            }
        ], None
    e3_root = e3_output_root / run_id
    metrics = pd.read_csv(e3_root / "e3_metrics_long.csv")
    return _summarize_candidate_e3(metrics, spec=spec, phase=phase, old_refs=old_refs, e3_root=e3_root), e3_root


def _summarize_candidate_e3(
    metrics: pd.DataFrame,
    *,
    spec: ChronarisCandidateSpec,
    phase: str,
    old_refs: Mapping[str, object],
    e3_root: Path,
) -> list[dict[str, object]]:
    rows = []
    completed = metrics[metrics["status"] == "completed"].copy()
    for (method_name, metric), group in completed.groupby(["method_name", "metric"], sort=True):
        value = float(pd.to_numeric(group["value"], errors="coerce").dropna().mean())
        old_chronaris = old_refs.get("e3", {}).get(str(metric), {}).get("chronaris")
        mult = old_refs.get("e3", {}).get(str(metric), {}).get("mult")
        contiformer = old_refs.get("e3", {}).get(str(metric), {}).get("contiformer")
        naive = old_refs.get("e3", {}).get(str(metric), {}).get("naive_time_sync")
        improved = None
        if method_name == "chronaris" and metric in E3_TARGET_METRICS and old_chronaris is not None:
            improved = value > float(old_chronaris) + 1e-9
        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "candidate_phase": phase,
                "method_name": method_name,
                "metric": metric,
                "mean_value": value,
                "status": "completed",
                "old_chronaris_value": old_chronaris,
                "delta_vs_old_chronaris": None if old_chronaris is None else value - float(old_chronaris),
                "old_mult_value": mult,
                "old_contiformer_value": contiformer,
                "old_naive_time_sync_value": naive,
                "e3_target_metric": metric in E3_TARGET_METRICS,
                "e3_target_improved_vs_old_chronaris": improved,
                "e3_run_root": repo_rel(e3_root),
            }
        )
    unavailable = metrics[metrics["status"] != "completed"].groupby(["method_name", "metric"], sort=True).size()
    for (method_name, metric), count in unavailable.items():
        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "candidate_phase": phase,
                "method_name": method_name,
                "metric": metric,
                "mean_value": None,
                "status": "unavailable",
                "unavailable_count": int(count),
                "e3_run_root": repo_rel(e3_root),
            }
        )
    return rows


def _select_pareto_candidate(dev_results: Mapping[str, Mapping[str, object]], old_refs: Mapping[str, object]) -> dict[str, object]:
    old_t2 = float(old_refs["t2"]["rmse"]["chronaris"])
    old_t1_macro = float(old_refs["t1"]["macro_f1"]["chronaris"])
    old_t1_bal = float(old_refs["t1"]["balanced_accuracy"]["chronaris"])
    candidates = []
    for candidate_id, result in dev_results.items():
        rows = result.get("metric_rows", [])
        metric_by_name = {(row["task_name"], row["metric"]): row.get("value") for row in rows}
        t2_rmse = metric_by_name.get((TASK_RESPONSE, "rmse"))
        t1_macro = metric_by_name.get((TASK_MANEUVER, "macro_f1"))
        t1_bal = metric_by_name.get((TASK_MANEUVER, "balanced_accuracy"))
        e3_positive = int(result.get("e3_positive_signal_count") or 0)
        pass_row = {
            "candidate_id": candidate_id,
            "t2_rmse": t2_rmse,
            "t2_improved": t2_rmse is not None and float(t2_rmse) < old_t2,
            "t1_macro_f1": t1_macro,
            "t1_balanced_accuracy": t1_bal,
            "t1_not_degraded": (
                t1_macro is not None
                and t1_bal is not None
                and float(t1_macro) >= old_t1_macro - 0.02
                and float(t1_bal) >= old_t1_bal - 0.02
            ),
            "e3_positive_signal_count": e3_positive,
            "e3_positive_signal": e3_positive > 0,
            "completed_fold_count": result.get("completed_fold_count"),
            "expected_fold_count": result.get("expected_fold_count"),
        }
        pass_row["pareto_pass"] = bool(pass_row["t2_improved"] and pass_row["t1_not_degraded"] and pass_row["e3_positive_signal"] and pass_row["completed_fold_count"] == pass_row["expected_fold_count"])
        candidates.append(pass_row)
    passed = [row for row in candidates if row["pareto_pass"]]
    selected = None
    if passed:
        selected = sorted(passed, key=lambda row: (float(row["t2_rmse"]), -int(row["e3_positive_signal_count"]), str(row["candidate_id"])))[0]["candidate_id"]
    return {
        "selected_candidate_id": selected,
        "pareto_rows": candidates,
        "selection_rule": "T2 RMSE improvement; T1 macro-F1/balanced accuracy within 0.02 of old Chronaris; at least one target E3 metric improved; all dev folds complete.",
    }


def _old_reference_metrics(config: ControlledOptimizationConfig) -> dict[str, object]:
    consistency = pd.read_csv(_resolve_path(config.old_consistency_table_path))
    e3 = pd.read_csv(_resolve_path(config.old_e3_method_summary_path))
    refs: dict[str, object] = {"t1": {}, "t2": {}, "e3": {}}
    for row in consistency.itertuples(index=False):
        if row.source != "dingxin_thirdparty_comparison":
            continue
        task_key = "t1" if row.task == TASK_MANEUVER else "t2" if row.task == TASK_RESPONSE else None
        if task_key is None:
            continue
        refs[task_key][row.metric] = {
            "chronaris": float(row.chronaris_value),
            "mult": _finite_or_none(row.mult_value),
            "contiformer": _finite_or_none(row.contiformer_value),
            "naive_or_classical": _finite_or_none(row.naive_or_classical_value),
        }
    for row in e3.itertuples(index=False):
        refs["e3"].setdefault(row.metric, {})[row.method_name] = _finite_or_none(row.mean)
    return refs


def _label_maps(records: pd.DataFrame) -> dict[str, dict[str, object]]:
    payload = derive_private_proxy_task_entries(records)
    return {
        TASK_MANEUVER: {entry.sample_id: entry.label_value for entry in payload["by_task"][TASK_MANEUVER] if entry.label_value is not None},
        TASK_RESPONSE: {entry.sample_id: entry.label_value for entry in payload["by_task"][TASK_RESPONSE] if entry.label_value is not None},
    }


def _metric_row(spec: ChronarisCandidateSpec, phase: str, task_name: str, metric: str, value: float | None, direction: str, source: str, old_refs: Mapping[str, object]) -> dict[str, object]:
    ref_group = "t1" if task_name == TASK_MANEUVER else "t2"
    ref = old_refs.get(ref_group, {}).get(metric, {})
    old = ref.get("chronaris") if isinstance(ref, Mapping) else None
    delta = None
    improved = None
    if value is not None and old is not None:
        delta = float(value) - float(old)
        improved = delta < 0 if metric in LOWER_IS_BETTER else delta > 0
    return {
        "candidate_id": spec.candidate_id,
        "candidate_phase": phase,
        "task_name": task_name,
        "metric": metric,
        "value": value,
        "direction": direction,
        "source": source,
        "old_chronaris_value": old,
        "delta_vs_old_chronaris": delta,
        "improved_vs_old_chronaris": improved,
        "mult_value": ref.get("mult") if isinstance(ref, Mapping) else None,
        "contiformer_value": ref.get("contiformer") if isinstance(ref, Mapping) else None,
        "naive_or_classical_value": ref.get("naive_or_classical") if isinstance(ref, Mapping) else None,
    }


def _annotate_rows(rows: Sequence[Mapping[str, object]], *, spec: ChronarisCandidateSpec, phase: str) -> list[dict[str, object]]:
    return [{**dict(row), "candidate_id": spec.candidate_id, "candidate_phase": phase} for row in rows]


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


def _metric_value(rows: Sequence[Mapping[str, object]], metric: str) -> float | None:
    for row in rows:
        if row.get("metric") == metric and row.get("value") is not None:
            return float(row["value"])
    return None


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


__all__ = [
    "ChronarisCandidateSpec",
    "ControlledOptimizationConfig",
    "default_candidate_specs",
    "run_controlled_optimization",
]
