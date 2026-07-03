"""P28 public fusion GPU optimization profiling workflow."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from chronaris.evaluation import (
    evaluate_classification_predictions,
    evaluate_regression_predictions,
)
from chronaris.pipelines.stage_i.common.baseline_models import build_loso_splits
from chronaris.pipelines.stage_i.common.deep_models import build_stage_i_deep_model
from chronaris.pipelines.stage_i.common.run_observer import open_stage_i_run_observer
from chronaris.pipelines.stage_i.public.deep_baseline_runtime import (
    _classification_loss,
    _extract_target_values,
    _fit_target_transform,
    _iter_batches,
    _load_prepared_sequence_dataset,
    _regression_loss,
    _transform_targets,
)
from chronaris.pipelines.stage_i.public.fusion_refresh import (
    PublicFusionRefreshCandidate,
)
from chronaris.pipelines.stage_i.common.gpu_runtime import (
    AmpRuntime,
    completed_profile_key,
    gpu_runtime_snapshot,
    json_dumps,
    make_grad_scaler,
    prepare_fold_tensors,
    resolve_amp_runtime,
    validate_optimization_summary_schema,
)
from chronaris.pipelines.stage_i.public.fusion_gpuopt_reporting import (
    build_optimization_summary,
    build_perf_summary,
    render_plots,
    render_report,
)
from chronaris.pipelines.stage_i.public.fusion_gpuopt_training import (
    evaluate_workload,
    maybe_compile_model,
    run_profile_batch,
    select_batch_size,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name, seed_torch

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class StageIPublicFusionGPUOptConfig:
    run_id: str = "20260701T-stage-i-public-fusion-refresh-r1-gpuopt-r1"
    base_p28_run_id: str = "20260701T-stage-i-public-fusion-refresh-r1"
    base_p28_root: str = (
        "docs/artifacts/assets/stage_i_public_fusion_refresh/"
        "20260701T-stage-i-public-fusion-refresh-r1"
    )
    artifact_root: str | None = None
    report_root: str = "docs/artifacts/stage_i"
    device: str = "cuda"
    require_cuda: bool = True
    profile_epochs: int = 1
    profile_batches: int = 20
    baseline_tensor_cache: str = "off"
    tensor_cache: str = "auto"
    cache_full_dataset: bool = True
    max_cache_gb: float = 18.0
    pin_memory: bool = True
    non_blocking_copy: bool = True
    auto_batch_size: bool = True
    batch_size_candidates: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64)
    amp: str = "bf16"
    grad_scaler: bool = True
    amp_eval: bool = True
    torch_compile: str = "off"
    compile_warmup_steps: int = 5
    eval_batch_size: int | None = None
    heartbeat_seconds: float = 60.0
    batch_log_interval: int = 20
    skip_completed: bool = True


@dataclass(frozen=True, slots=True)
class StageIPublicFusionGPUOptResult:
    run_id: str
    artifact_root: str
    summary_path: str
    perf_summary_path: str
    report_path: str
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Workload:
    dataset_id: str
    track: str
    task: str
    evaluation_group: str
    split_group: str
    fold_index: int
    fold_count: int
    group_indices: np.ndarray
    train_global_indices: np.ndarray
    test_global_indices: np.ndarray
    target_full: np.ndarray
    train_targets: np.ndarray
    truth_values: np.ndarray
    label_order: tuple[int, ...] | None
    target_transform: Mapping[str, float | str] | None


def run_stage_i_public_fusion_gpuopt(
    config: StageIPublicFusionGPUOptConfig,
) -> StageIPublicFusionGPUOptResult:
    base_root = _resolve_path(config.base_p28_root)
    artifact_root = (
        _resolve_path(config.artifact_root)
        if config.artifact_root is not None
        else base_root / "gpu_optimization"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=artifact_root,
        run_id=config.run_id,
        stage_name="stage_i_public_fusion_gpuopt",
        logger=LOGGER,
        initial_progress={
            "base_p28_run_id": config.base_p28_run_id,
            "base_p28_root": str(base_root),
            "artifact_root": str(artifact_root),
            "profile_epochs": config.profile_epochs,
            "profile_batches": config.profile_batches,
            "tensor_cache": config.tensor_cache,
            "amp": config.amp,
            "torch_compile": config.torch_compile,
        },
    ) as progress:
        result = _run_gpuopt_core(config=config, base_root=base_root, artifact_root=artifact_root, progress=progress)
        progress.finish(
            status=result.summary.get("status", "completed"),
            optimization_summary_path=result.summary_path,
            gpu_perf_summary_path=result.perf_summary_path,
            report_path=result.report_path,
        )
    _copy_observer_files(artifact_root)
    return result


def _run_gpuopt_core(
    *,
    config: StageIPublicFusionGPUOptConfig,
    base_root: Path,
    artifact_root: Path,
    progress,
) -> StageIPublicFusionGPUOptResult:
    runtime_device = resolve_torch_device_name(config.device)
    if runtime_device != "cuda" and config.require_cuda:
        raise RuntimeError(f"P28-GPUOPT requires CUDA; resolved device={runtime_device}.")
    plots_root = artifact_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    base_summary = _load_json(base_root / "fusion_refresh_summary.json")
    base_config = _load_json(base_root / "fusion_refresh_config.json")
    candidate = _load_best_candidate(base_root, base_summary)
    prepared_roots = base_config["config"]["dataset_prepared_roots"]
    datasets = {
        dataset_id: _load_prepared_sequence_dataset(path)
        for dataset_id, path in prepared_roots.items()
    }
    workloads = _select_workloads(datasets=datasets)
    snapshot = gpu_runtime_snapshot()
    optimization_config = {
        **asdict(config),
        "base_p28_status": base_summary.get("status"),
        "base_p28_artifact_root": str(base_root),
        "candidate": asdict(candidate),
        "workloads": [_workload_summary(workload) for workload in workloads],
        "runtime_snapshot": snapshot,
    }
    (artifact_root / "optimization_config.json").write_text(
        json_dumps(optimization_config),
        encoding="utf-8",
    )
    progress.update("config_written", optimization_config=str(artifact_root / "optimization_config.json"))

    batch_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    fallbacks: list[dict[str, object]] = []
    completed_keys = _read_completed_keys(artifact_root)
    for workload in workloads:
        key = completed_profile_key(
            dataset_id=workload.dataset_id,
            candidate_id=candidate.candidate_id,
            seed=42,
            track=workload.track,
            evaluation_group=workload.evaluation_group,
            split_group=workload.split_group,
        )
        if config.skip_completed and key in completed_keys:
            progress.update("workload_skip_completed", profile_key=key)
            continue
        for profile_stage in ("baseline", "optimized"):
            progress.update(
                "profile_start",
                profile_stage=profile_stage,
                dataset_id=workload.dataset_id,
                track=workload.track,
                evaluation_group=workload.evaluation_group,
                split_group=workload.split_group,
            )
            profile = _profile_workload(
                config=config,
                candidate=candidate,
                dataset=datasets[workload.dataset_id],
                workload=workload,
                runtime_device=runtime_device,
                profile_stage=profile_stage,
            )
            batch_rows.extend(profile["batch_rows"])
            fold_rows.append(profile["fold_summary"])
            curve_rows.extend(profile["curve_rows"])
            prediction_rows.append(profile["predictions"])
            fallbacks.extend(profile["fallbacks"])
            progress.update(
                "profile_done",
                profile_stage=profile_stage,
                dataset_id=workload.dataset_id,
                evaluation_group=workload.evaluation_group,
                samples_per_sec=profile["fold_summary"].get("samples_per_sec"),
                selected_batch_size=profile["fold_summary"].get("batch_size"),
            )
        completed_keys.add(key)
        _write_completed_keys(artifact_root, completed_keys)

    batch_frame = pd.DataFrame(batch_rows)
    fold_frame = pd.DataFrame(fold_rows)
    curve_frame = pd.DataFrame(curve_rows)
    predictions = pd.concat(prediction_rows, axis=0, ignore_index=True) if prediction_rows else pd.DataFrame()
    batch_csv = artifact_root / "gpu_perf_batches.csv"
    fold_csv = artifact_root / "gpu_perf_fold_summary.csv"
    curves_csv = artifact_root / "training_curves.gpuopt.csv"
    fold_metrics_csv = artifact_root / "fold_metrics.gpuopt.csv"
    predictions_csv = artifact_root / "predictions.gpuopt.csv"
    batch_frame.to_csv(batch_csv, index=False)
    fold_frame.to_csv(fold_csv, index=False)
    fold_frame.to_csv(fold_metrics_csv, index=False)
    curve_frame.to_csv(curves_csv, index=False)
    predictions.to_csv(predictions_csv, index=False)

    perf_summary = build_perf_summary(config, candidate, fold_frame, batch_frame, snapshot, fallbacks)
    perf_summary_path = artifact_root / "gpu_perf_summary.json"
    perf_summary_path.write_text(json_dumps(perf_summary), encoding="utf-8")
    resume_command = _resume_command(config)
    (artifact_root / "resume_command.txt").write_text(resume_command + "\n", encoding="utf-8")
    candidate_summary = {
        "run_id": config.run_id,
        "base_p28_run_id": config.base_p28_run_id,
        "candidate": asdict(candidate),
        "representative_fold_count": int(len(fold_frame)),
        "profile_epochs": int(config.profile_epochs),
        "profile_batches": int(config.profile_batches),
    }
    (artifact_root / "candidate_summary.gpuopt.json").write_text(
        json_dumps(candidate_summary),
        encoding="utf-8",
    )
    figure_paths = render_plots(artifact_root=artifact_root, batch_frame=batch_frame, fold_frame=fold_frame)
    report_path = _resolve_path(config.report_root) / f"stage-i-public-fusion-gpu-optimization-{config.run_id}.md"
    optimization_summary = build_optimization_summary(
        config=config,
        candidate=candidate,
        perf_summary=perf_summary,
        snapshot=snapshot,
        fallbacks=fallbacks,
        artifact_root=artifact_root,
        figure_paths=figure_paths,
        resume_command=resume_command,
    )
    ok, missing = validate_optimization_summary_schema(optimization_summary)
    optimization_summary["schema_status"] = "ok" if ok else "missing_fields"
    optimization_summary["schema_missing_fields"] = missing
    optimization_summary_path = artifact_root / "optimization_summary.json"
    optimization_summary_path.write_text(json_dumps(optimization_summary), encoding="utf-8")
    report_path.write_text(
        render_report(
            summary=optimization_summary,
            perf_summary=perf_summary,
            figure_paths=figure_paths,
            artifact_root=artifact_root,
            batch_csv=batch_csv,
            fold_csv=fold_csv,
            resume_command=resume_command,
        )
        + "\n",
        encoding="utf-8",
    )
    progress.update(
        "artifacts_written",
        optimization_summary_path=str(optimization_summary_path),
        gpu_perf_summary_path=str(perf_summary_path),
        report_path=str(report_path),
    )
    return StageIPublicFusionGPUOptResult(
        run_id=config.run_id,
        artifact_root=str(artifact_root),
        summary_path=str(optimization_summary_path),
        perf_summary_path=str(perf_summary_path),
        report_path=str(report_path),
        summary=optimization_summary,
    )


def _profile_workload(
    *,
    config: StageIPublicFusionGPUOptConfig,
    candidate: PublicFusionRefreshCandidate,
    dataset: Mapping[str, object],
    workload: _Workload,
    runtime_device: str,
    profile_stage: str,
) -> dict[str, object]:
    bundle = dataset["bundle"]
    ordered_modalities = tuple(dataset["entries"][0].modality_schema)
    seed_torch(42, device=runtime_device)
    model = build_stage_i_deep_model(
        model_name="chronaris_public_fusion",
        ordered_modalities=ordered_modalities,
        modality_input_dims={name: bundle.modality_arrays[name].shape[-1] for name in ordered_modalities},
        output_dim=len(workload.label_order or (0,)) if workload.task == "classification" else 1,
        hidden_dim=candidate.hidden_dim,
        num_heads=candidate.num_heads,
        layers=candidate.layers,
        dropout=candidate.dropout,
        fusion_event_bias_weight=candidate.fusion_event_bias_weight,
        fusion_lag_window_points=candidate.fusion_lag_window_points,
        fusion_normalize_states=candidate.fusion_normalize_states,
    ).to(runtime_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate.learning_rate, weight_decay=float(candidate.weight_decay))
    criterion = (
        _classification_loss(workload.train_targets, device=runtime_device, output_dim=len(workload.label_order or ()), sampling_policy=_sampling_policy(workload.dataset_id))
        if workload.task == "classification"
        else _regression_loss(candidate.regression_loss, huber_delta=candidate.huber_delta)
    )
    cache_mode = config.baseline_tensor_cache if profile_stage == "baseline" else config.tensor_cache
    amp = resolve_amp_runtime(requested_mode="off" if profile_stage == "baseline" else config.amp, device=runtime_device, grad_scaler=config.grad_scaler)
    scaler = make_grad_scaler(amp, device=runtime_device)
    prepared = prepare_fold_tensors(
        modality_arrays=bundle.modality_arrays,
        modality_masks=bundle.modality_masks,
        time_axis=bundle.time_axis,
        ordered_modalities=ordered_modalities,
        train_indices=workload.train_global_indices,
        targets=workload.target_full,
        requested_mode=cache_mode,
        device=runtime_device,
        max_cache_gb=config.max_cache_gb,
        pin_memory=config.pin_memory,
        non_blocking_copy=config.non_blocking_copy,
    )
    fallbacks = []
    if prepared.fallback_reason:
        fallbacks.append({"profile_stage": profile_stage, "fallback_reason": prepared.fallback_reason})
    batch_size, batch_attempts = select_batch_size(
        config=config,
        candidate=candidate,
        model=model,
        criterion=criterion,
        prepared=prepared,
        workload=workload,
        runtime_device=runtime_device,
        profile_stage=profile_stage,
        amp=amp,
    )
    if profile_stage == "optimized":
        model, compile_info = maybe_compile_model(
            config=config,
            model=model,
            prepared=prepared,
            workload=workload,
            runtime_device=runtime_device,
        )
    else:
        compile_info = {"compile_mode": "off", "compile_status": "off", "compile_warmup_time_s": 0.0, "compile_fallback_reason": None}
    if compile_info.get("compile_fallback_reason"):
        fallbacks.append({"profile_stage": profile_stage, "fallback_reason": compile_info["compile_fallback_reason"]})

    if runtime_device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    batch_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    train_started = time.monotonic()
    last_heartbeat = train_started
    batches = _iter_batches(
        len(workload.train_global_indices),
        batch_size=batch_size,
        seed=42,
        sampling_policy=_sampling_policy(workload.dataset_id),
        labels=workload.train_targets if workload.task == "classification" else None,
    )[: config.profile_batches]
    for epoch in range(config.profile_epochs):
        losses: list[float] = []
        for batch_number, relative_indices in enumerate(batches, start=1):
            row, loss_value = run_profile_batch(
                config=config,
                candidate=candidate,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                amp=amp,
                prepared=prepared,
                workload=workload,
                ordered_modalities=ordered_modalities,
                relative_indices=relative_indices,
                runtime_device=runtime_device,
                profile_stage=profile_stage,
                epoch=epoch + 1,
                batch_number=batch_number,
                batch_count=len(batches),
                batch_size=batch_size,
                compile_info=compile_info,
                batch_attempts=batch_attempts,
            )
            losses.append(loss_value)
            batch_rows.append(row)
            now = time.monotonic()
            should_log = batch_number == 1 or batch_number == len(batches) or batch_number % config.batch_log_interval == 0
            should_heartbeat = now - last_heartbeat >= config.heartbeat_seconds
            if should_log or should_heartbeat:
                LOGGER.info(
                    "[heartbeat] run_id=%s stage=%s dataset=%s candidate=%s fold=%s/%s epoch=%s/%s batch=%s/%s elapsed_s=%.1f last_loss=%.6f samples_per_sec=%.2f gpu_mem_gb=%.3f",
                    config.run_id,
                    profile_stage,
                    workload.dataset_id,
                    candidate.candidate_id,
                    workload.fold_index,
                    workload.fold_count,
                    epoch + 1,
                    config.profile_epochs,
                    batch_number,
                    len(batches),
                    now - train_started,
                    loss_value,
                    row["samples_per_sec"],
                    row["gpu_memory_allocated_gb"] or 0.0,
                )
                last_heartbeat = now
        curve_rows.append(
            {
                "profile_stage": profile_stage,
                "dataset_id": workload.dataset_id,
                "track": workload.track,
                "evaluation_group": workload.evaluation_group,
                "fold_index": workload.fold_index,
                "split_group": workload.split_group,
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "batch_count": len(losses),
                "batch_size": batch_size,
                "tensor_cache_mode": prepared.tensor_cache_mode,
                "amp_mode": amp.resolved_mode,
            }
        )
    evaluation_started = time.monotonic()
    predictions = evaluate_workload(
        model=model,
        prepared=prepared,
        workload=workload,
        runtime_device=runtime_device,
        batch_size=config.eval_batch_size or batch_size,
        amp=amp if config.amp_eval else AmpRuntime("off", "off", False, None, False),
    )
    evaluation_time = time.monotonic() - evaluation_started
    metrics = (
        evaluate_classification_predictions(predictions, label_order=workload.label_order)
        if workload.task == "classification"
        else evaluate_regression_predictions(predictions)
    )
    for row in batch_rows:
        row["evaluation_time_s"] = evaluation_time if row["batch_index"] == len(batches) else 0.0
    total_samples = sum(int(row["sample_count"]) for row in batch_rows)
    total_time = sum(float(row["batch_total_time_s"]) for row in batch_rows)
    fold_summary = {
        "run_id": config.run_id,
        "base_p28_run_id": config.base_p28_run_id,
        "profile_stage": profile_stage,
        "dataset_id": workload.dataset_id,
        "candidate_id": candidate.candidate_id,
        "seed": 42,
        "track": workload.track,
        "evaluation_group": workload.evaluation_group,
        "split_group": workload.split_group,
        "fold_index": workload.fold_index,
        "fold_count": workload.fold_count,
        "train_sample_count": int(len(workload.train_global_indices)),
        "test_sample_count": int(len(workload.test_global_indices)),
        "profile_batch_count": int(len(batch_rows)),
        "batch_size": int(batch_size),
        "samples_per_sec": float(total_samples / total_time) if total_time > 0 else None,
        "evaluation_time_s": evaluation_time,
        "tensor_cache_mode": prepared.tensor_cache_mode,
        "estimated_cache_gb": prepared.estimated_cache_gb,
        "cache_build_time_s": prepared.cache_build_time_s,
        "cache_hit_count": int(prepared.cache_hit_count),
        "cache_fallback_reason": prepared.fallback_reason,
        "amp_mode": amp.resolved_mode,
        **compile_info,
        **_primary_metric_fields(workload.task, metrics),
    }
    predictions["profile_stage"] = profile_stage
    return {
        "batch_rows": batch_rows,
        "fold_summary": fold_summary,
        "curve_rows": curve_rows,
        "predictions": predictions,
        "fallbacks": fallbacks,
    }


def _select_workloads(*, datasets: Mapping[str, Mapping[str, object]]) -> list[_Workload]:
    workloads: list[_Workload] = []
    if "nasa_csm" in datasets:
        workloads.append(_build_workload(datasets["nasa_csm"], dataset_id="nasa_csm", track="objective", group_name="combined"))
    if "uab_workload_dataset" in datasets:
        workloads.append(_build_workload(datasets["uab_workload_dataset"], dataset_id="uab_workload_dataset", track="subjective", group_name="n_back"))
        workloads.append(_build_workload(datasets["uab_workload_dataset"], dataset_id="uab_workload_dataset", track="subjective", group_name="heat_the_chair"))
    return workloads


def _build_workload(dataset: Mapping[str, object], *, dataset_id: str, track: str, group_name: str) -> _Workload:
    entries = dataset["entries"]
    if dataset_id == "nasa_csm":
        group_indices = np.asarray([i for i, entry in enumerate(entries) if entry.training_role == "primary" and entry.subset_id in {"benchmark", "loft"}], dtype=int)
        label_order = (1, 2, 5)
    else:
        group_indices = np.asarray([i for i, entry in enumerate(entries) if entry.training_role == "primary" and entry.subset_id == group_name], dtype=int)
        label_order = None
    group_entries = [entries[index] for index in group_indices]
    split_groups = np.asarray([entry.split_group for entry in group_entries], dtype=object)
    splits = build_loso_splits(split_groups)
    split, fold_index = _representative_split(splits)
    target_values = np.asarray(_extract_target_values(group_entries, track=track), dtype=np.float32)
    target_full = np.zeros(len(entries), dtype=np.float32)
    if track == "objective":
        label_to_index = {int(label): position for position, label in enumerate(label_order or ())}
        mapped = np.asarray([label_to_index[int(value)] for value in target_values], dtype=np.float32)
        target_full[group_indices] = mapped
        train_targets = mapped[split.train_indices].astype(int)
        truth = target_values[split.test_indices].astype(int)
        transform = None
    else:
        transform = _fit_target_transform(target_values[split.train_indices], transform_name="zscore_train")
        transformed = _transform_targets(target_values, transform)
        target_full[group_indices] = transformed
        train_targets = transformed[split.train_indices]
        truth = target_values[split.test_indices]
    return _Workload(
        dataset_id=dataset_id,
        track=track,
        task="classification" if track == "objective" else "regression",
        evaluation_group=group_name,
        split_group=str(split.split_group),
        fold_index=fold_index,
        fold_count=len(splits),
        group_indices=group_indices,
        train_global_indices=group_indices[split.train_indices],
        test_global_indices=group_indices[split.test_indices],
        target_full=target_full,
        train_targets=train_targets,
        truth_values=truth,
        label_order=label_order,
        target_transform=transform,
    )


def _representative_split(splits):
    ranked = sorted(enumerate(splits, start=1), key=lambda item: len(item[1].test_indices))
    return ranked[len(ranked) // 2][1], ranked[len(ranked) // 2][0]


def _load_best_candidate(base_root: Path, base_summary: Mapping[str, object]) -> PublicFusionRefreshCandidate:
    candidate_id = base_summary["best_by_dataset_task"]["nasa_csm"]["combined"]["candidate_id"]
    for row in _load_json(base_root / "candidate_grid.json")["screen_candidates"]:
        if row["candidate_id"] == candidate_id:
            return PublicFusionRefreshCandidate(**row)
    raise KeyError(f"candidate not found in candidate_grid.json: {candidate_id}")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else Path.cwd() / path


def _sampling_policy(dataset_id: str) -> str:
    return "balanced_class" if dataset_id == "nasa_csm" else "none"


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _primary_metric_fields(task: str, metrics: Mapping[str, object]) -> dict[str, float | None]:
    if task == "classification":
        return {"macro_f1": float(metrics["macro_f1"]), "balanced_accuracy": float(metrics["balanced_accuracy"]), "rmse": None, "mae": None}
    return {"macro_f1": None, "balanced_accuracy": None, "rmse": float(metrics["rmse"]), "mae": float(metrics["mae"])}


def _workload_summary(workload: _Workload) -> dict[str, object]:
    return {
        "dataset_id": workload.dataset_id,
        "track": workload.track,
        "evaluation_group": workload.evaluation_group,
        "split_group": workload.split_group,
        "fold_index": workload.fold_index,
        "fold_count": workload.fold_count,
        "train_sample_count": int(len(workload.train_global_indices)),
        "test_sample_count": int(len(workload.test_global_indices)),
    }


def _resume_command(config: StageIPublicFusionGPUOptConfig) -> str:
    return (
        "python scripts/stage_i/public/run_public_fusion_gpuopt.py "
        f"--run-id {config.run_id} "
        f"--base-p28-run-id {config.base_p28_run_id} "
        f"--base-p28-root {config.base_p28_root} "
        f"--profile-batches {config.profile_batches} "
        f"--profile-epochs {config.profile_epochs} "
        "--device cuda --require-cuda "
        f"--tensor-cache {config.tensor_cache} "
        f"--max-cache-gb {config.max_cache_gb:g} "
        "--auto-batch-size "
        "--batch-size-candidates "
        + " ".join(str(value) for value in config.batch_size_candidates)
        + f" --amp {config.amp} --torch-compile {config.torch_compile} --skip-completed"
    )


def _read_completed_keys(root: Path) -> set[str]:
    path = root / "optimization_completed_folds.csv"
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["profile_key"] for row in csv.DictReader(handle)}


def _write_completed_keys(root: Path, keys: set[str]) -> None:
    path = root / "optimization_completed_folds.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["profile_key"])
        writer.writeheader()
        for key in sorted(keys):
            writer.writerow({"profile_key": key})


def _copy_observer_files(root: Path) -> None:
    if (root / "run.log").exists():
        shutil.copyfile(root / "run.log", root / "optimization_run.log")
    if (root / "progress.json").exists():
        shutil.copyfile(root / "progress.json", root / "optimization_progress.json")
