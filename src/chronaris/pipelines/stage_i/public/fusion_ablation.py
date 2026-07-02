"""P31 public chronaris fusion ablation over NASA/UAB prepared sequences."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronaris-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from chronaris.evaluation import (
    evaluate_classification_predictions,
    evaluate_regression_predictions,
)
from chronaris.pipelines.stage_i.common.run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)
from chronaris.pipelines.stage_i.common.plot_labels import (
    label_horizontal_bars,
    label_stack_totals,
    label_vertical_bars,
)
from chronaris.pipelines.stage_i.public.deep_baseline import (
    StageIDeepBaselineConfig,
    StageIDeepBaselineRunResult,
    run_stage_i_deep_baseline,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name

REPO_ROOT = Path(__file__).resolve().parents[5]
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

P28_SOURCE_RUN_ID = "20260701T-stage-i-public-fusion-refresh-r1"
P28_SOURCE_ROOT = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_i_public_fusion_refresh"
    / P28_SOURCE_RUN_ID
)
DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_public_fusion_ablation"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"
EVIDENCE_ROLE = "public_adapter_context_proxy_evidence"


@dataclass(frozen=True, slots=True)
class PublicFusionAblationVariant:
    variant_id: str
    description: str
    model_name: str = "chronaris_public_fusion"
    hidden_dim: int = 64
    layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    fusion_event_bias_weight: float = 0.25
    fusion_lag_window_points: int | None = 16
    fusion_normalize_states: bool = True
    learning_rate: float = 1e-3
    batch_size: int = 128
    regression_loss: str = "smooth_l1"
    huber_delta: float = 1.0
    target_transform: str = "zscore_train"
    weight_decay: float = 1e-5
    gradient_clip_max_norm: float | None = 1.0
    ablation_family: str = "full"
    train_sampling_policy: str | None = None


@dataclass(frozen=True, slots=True)
class StageIPublicFusionAblationConfig:
    run_id: str
    dataset_prepared_roots: Mapping[str, str]
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    datasets: tuple[str, ...] = ("nasa_csm", "uab_workload_dataset")
    variants: tuple[str, ...] = ()
    screen_epochs: int = 5
    confirm_epochs: int = 20
    screen_max_folds: int | None = 2
    confirm_max_folds: int | None = None
    seed: int = 42
    extra_confirm_seeds: tuple[int, ...] = ()
    device: str = "auto"
    require_cuda: bool = True
    resume: bool = False
    skip_completed: bool = True
    allow_partial: bool = False
    screen_only: bool = False
    confirm_only: bool = False
    heartbeat_seconds: float = 60.0
    batch_log_interval: int = 20
    tensor_cache: str = "auto"
    max_cache_gb: float = 18.0
    pin_memory: bool = True
    non_blocking_copy: bool = True
    auto_batch_size: bool = True
    batch_size_candidates: tuple[int, ...] = (24576, 16384, 8192, 4096, 2048, 1024, 512, 256, 128)
    amp: str = "bf16"
    grad_scaler: bool = True
    amp_eval: bool = True
    torch_compile: str = "default"
    profile_gpu: bool = True
    eval_batch_size: int | None = None
    num_workers: int = 24
    parallel_fold_prep: int = 8
    parallel_candidates: int = 1
    checkpoint_policy: str = "last"
    base_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class StageIPublicFusionAblationRunResult:
    run_id: str
    artifact_root: str
    summary_path: str
    evidence_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


def default_public_fusion_ablation_variants() -> tuple[PublicFusionAblationVariant, ...]:
    full = PublicFusionAblationVariant(
        variant_id="full",
        description="P28 best chronaris_public_fusion refresh configuration.",
        ablation_family="full",
    )
    return (
        full,
        replace(
            full,
            variant_id="v3_stream_role",
            description="P35 stream-role-aware adaptive route for public context-proxy streams.",
            model_name="chronaris_v3_stream_role",
            ablation_family="stream_role",
        ),
        replace(
            full,
            variant_id="v3_no_role_gate",
            description="P35 stream-role-aware wrapper without a forced role route.",
            model_name="v3_no_role_gate",
            ablation_family="stream_role",
        ),
        replace(
            full,
            variant_id="v3_force_private_causal",
            description="Force the private causal vehicle route on public context-proxy streams.",
            model_name="v3_force_private_causal",
            ablation_family="stream_role",
        ),
        replace(
            full,
            variant_id="v3_context_adapter_only",
            description="Force the public context-adapter route for context-proxy streams.",
            model_name="v3_context_adapter_only",
            ablation_family="stream_role",
        ),
        replace(
            full,
            variant_id="no_lag_window",
            description="Disable lag-aware causal fusion window.",
            fusion_lag_window_points=None,
            ablation_family="lag_window",
        ),
        replace(
            full,
            variant_id="lag_window_4",
            description="Use 4-point fusion lag window.",
            fusion_lag_window_points=4,
            ablation_family="lag_window",
        ),
        replace(
            full,
            variant_id="lag_window_8",
            description="Use 8-point fusion lag window.",
            fusion_lag_window_points=8,
            ablation_family="lag_window",
        ),
        replace(
            full,
            variant_id="no_event_bias",
            description="Set event-bias weight to zero.",
            fusion_event_bias_weight=0.0,
            ablation_family="event_bias",
        ),
        replace(
            full,
            variant_id="event_bias_0p5",
            description="Use event-bias weight 0.5.",
            fusion_event_bias_weight=0.5,
            ablation_family="event_bias",
        ),
        replace(
            full,
            variant_id="event_bias_0p75",
            description="Use event-bias weight 0.75.",
            fusion_event_bias_weight=0.75,
            ablation_family="event_bias",
        ),
        replace(
            full,
            variant_id="no_fusion_normalization",
            description="Disable causal fusion state normalization.",
            fusion_normalize_states=False,
            ablation_family="normalization",
        ),
        replace(
            full,
            variant_id="no_normalize_states",
            description="Disable causal fusion state normalization.",
            fusion_normalize_states=False,
            ablation_family="normalization",
        ),
        replace(
            full,
            variant_id="physiology_only",
            description="Use only the physiology stream.",
            model_name="chronaris_public_fusion_physiology_only",
            ablation_family="stream",
        ),
        replace(
            full,
            variant_id="single_stream_physio_only",
            description="Use only the physiology stream.",
            model_name="chronaris_public_fusion_physiology_only",
            ablation_family="context_stream",
        ),
        replace(
            full,
            variant_id="no_context_proxy_stream",
            description="Remove the public context-proxy stream.",
            model_name="chronaris_public_fusion_physiology_only",
            ablation_family="context_stream",
        ),
        replace(
            full,
            variant_id="context_only",
            description="Use only the public context-proxy stream.",
            model_name="chronaris_public_fusion_context_only",
            ablation_family="stream",
        ),
        replace(
            full,
            variant_id="simple_dual_stream_concat",
            description="Replace causal fusion with late dual-stream concat.",
            model_name="chronaris_public_fusion_simple_concat",
            ablation_family="fusion_head",
        ),
        replace(
            full,
            variant_id="no_causal_fusion",
            description="Replace causal masked fusion with late dual-stream concat.",
            model_name="late_concat_no_causal_fusion",
            ablation_family="causal_fusion",
        ),
        replace(
            full,
            variant_id="no_target_transform",
            description="Use no UAB regression target transform.",
            target_transform="none",
            ablation_family="target_transform",
        ),
        replace(
            full,
            variant_id="regression_loss_mse",
            description="Use MSE for UAB regression.",
            regression_loss="mse",
            ablation_family="regression_loss",
        ),
        replace(
            full,
            variant_id="mse_loss",
            description="Use MSE for UAB regression.",
            regression_loss="mse",
            ablation_family="regression_loss",
        ),
        replace(
            full,
            variant_id="regression_loss_huber",
            description="Use Huber loss for UAB regression.",
            regression_loss="huber",
            ablation_family="regression_loss",
        ),
        replace(
            full,
            variant_id="no_balanced_class",
            description="Disable balanced-class sampling for NASA classification.",
            train_sampling_policy="none",
            ablation_family="class_balance",
        ),
        replace(
            full,
            variant_id="smaller_capacity_h64_l1",
            description="Use one fusion/model layer with hidden_dim=64.",
            layers=1,
            ablation_family="capacity",
        ),
        replace(
            full,
            variant_id="best_full_reproduce",
            description="Reproduce the full P28 public fusion configuration.",
            ablation_family="reproduce",
        ),
    )


def run_stage_i_public_fusion_ablation(
    config: StageIPublicFusionAblationConfig,
) -> StageIPublicFusionAblationRunResult:
    run_root = _resolve_path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_public_fusion_ablation",
        logger=LOGGER,
        initial_progress={
            "artifact_root": str(run_root),
            "datasets": list(config.datasets),
            "variants": list(config.variants),
            "screen_epochs": config.screen_epochs,
            "confirm_epochs": config.confirm_epochs,
            "resume": config.resume,
            "skip_completed": config.skip_completed,
            "evidence_role": EVIDENCE_ROLE,
            "gpuopt_enabled": True,
            "base_run_id": config.base_run_id,
            "tensor_cache": config.tensor_cache,
            "auto_batch_size": config.auto_batch_size,
            "amp": config.amp,
            "torch_compile": config.torch_compile,
            "num_workers": config.num_workers,
            "parallel_fold_prep": config.parallel_fold_prep,
        },
    ) as progress:
        try:
            result = _run_observed(config=config, run_root=run_root, progress=progress)
            progress.finish(
                status=result.summary.get("status", "completed"),
                summary_path=result.summary_path,
                evidence_manifest_path=result.evidence_manifest_path,
                report_path=result.report_path,
            )
            return result
        except BaseException as exc:
            payload = _blocked_payload(config, run_root, exc)
            blocked_path = run_root / "blocked_reason.json"
            partial_path = run_root / "partial_summary.json"
            blocked_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            partial_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            progress.update(
                "blocked",
                status="blocked",
                blocked_reason=str(exc),
                blocked_reason_path=str(blocked_path),
                partial_summary_path=str(partial_path),
            )
            if not config.allow_partial:
                raise
            manifest_path = run_root / "evidence_manifest.json"
            report_path = _resolve_path(config.report_root) / f"stage-i-public-fusion-ablation-{config.run_id}.md"
            payload["evidence_manifest_path"] = str(manifest_path)
            payload["report_path"] = str(report_path)
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return StageIPublicFusionAblationRunResult(
                run_id=config.run_id,
                artifact_root=str(run_root),
                summary_path=str(partial_path),
                evidence_manifest_path=str(manifest_path),
                report_path=str(report_path),
                summary=payload,
            )


def _run_observed(
    *,
    config: StageIPublicFusionAblationConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageIPublicFusionAblationRunResult:
    runtime_device = resolve_torch_device_name(config.device)
    if config.require_cuda and runtime_device != "cuda":
        raise RuntimeError(f"P31 requires CUDA for completed runs; resolved {runtime_device}.")
    variants = _select_variants(config.variants)
    datasets = tuple(dataset for dataset in config.datasets if dataset in config.dataset_prepared_roots)
    if not datasets:
        raise ValueError("no configured P31 datasets have prepared roots.")
    variant_manifest_path = run_root / "ablation_variant_manifest.json"
    variant_manifest = {
        "run_id": config.run_id,
        "generated_at_utc": _utc_now(),
        "evidence_role": EVIDENCE_ROLE,
        "second_stream_role": "context_proxy",
        "second_stream_is_real_vehicle": False,
        "p28_source_run_id": P28_SOURCE_RUN_ID,
        "base_run_id": config.base_run_id,
        "config_delta": _gpuopt_config_delta(config),
        "gpuopt_enabled": True,
        "variants": [asdict(variant) for variant in variants],
        "target_transform_policy": "UAB target transforms are fit on train folds only in deep_baseline_runtime.",
    }
    variant_manifest_path.write_text(
        json.dumps(variant_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    variants_alias_path = run_root / "public_fusion_ablation_variants.json"
    variants_alias_path.write_text(
        json.dumps(variant_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_path = run_root / "public_ablation_config.json"
    config_payload = {
        "run_id": config.run_id,
        "runtime_device": runtime_device,
        "base_run_id": config.base_run_id,
        "gpuopt_enabled": True,
        "config_delta": _gpuopt_config_delta(config),
        "config": _config_dict(config),
    }
    config_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_alias_path = run_root / "public_fusion_ablation_config.json"
    config_alias_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    screen_rows: list[dict[str, object]] = []
    confirm_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    if not config.confirm_only:
        for dataset_id in datasets:
            for variant in variants:
                progress.update("screen_start", dataset_id=dataset_id, variant_id=variant.variant_id)
                result = _run_variant(
                    stage="screen",
                    config=config,
                    run_root=run_root,
                    dataset_id=dataset_id,
                    variant=variant,
                    epochs=config.screen_epochs,
                    max_folds=config.screen_max_folds,
                    seed=config.seed,
                )
                row = _leaderboard_row(
                    stage="screen",
                    dataset_id=dataset_id,
                    variant=variant,
                    seed=config.seed,
                    result=result,
                )
                screen_rows.append(row)
                fold_rows.extend(_candidate_fold_rows(result, "screen", dataset_id, variant.variant_id, config.seed))
                curve_rows.extend(_training_curve_rows(result, "screen", dataset_id, variant.variant_id, config.seed))
                _write_partial_tables(run_root, screen_rows, confirm_rows, fold_rows, curve_rows)
                progress.update("screen_done", dataset_id=dataset_id, variant_id=variant.variant_id)
    elif (run_root / "screen_leaderboard.csv").exists():
        screen_rows = pd.read_csv(run_root / "screen_leaderboard.csv").to_dict(orient="records")

    if not config.screen_only:
        confirm_seeds = (config.seed, *config.extra_confirm_seeds)
        confirm_tasks = [
            (dataset_id, variant, seed)
            for dataset_id in datasets
            for variant in variants
            for seed in confirm_seeds
        ]
        variant_by_id = {variant.variant_id: variant for variant in variants}

        def _append_confirm_result(
            *,
            dataset_id: str,
            variant: PublicFusionAblationVariant,
            seed: int,
            result: StageIDeepBaselineRunResult,
        ) -> None:
            confirm_rows.append(
                _leaderboard_row(
                    stage="confirm",
                    dataset_id=dataset_id,
                    variant=variant,
                    seed=seed,
                    result=result,
                )
            )
            fold_rows.extend(_candidate_fold_rows(result, "confirm", dataset_id, variant.variant_id, seed))
            curve_rows.extend(_training_curve_rows(result, "confirm", dataset_id, variant.variant_id, seed))
            predictions = pd.read_csv(result.predictions_path)
            predictions["stage"] = "confirm"
            predictions["variant_id"] = variant.variant_id
            predictions["seed"] = int(seed)
            prediction_frames.append(predictions)
            _write_partial_tables(run_root, screen_rows, confirm_rows, fold_rows, curve_rows)
            progress.update(
                "confirm_done",
                dataset_id=dataset_id,
                variant_id=variant.variant_id,
                seed=seed,
            )

        worker_count = max(int(config.parallel_candidates), 1)
        if worker_count == 1:
            for dataset_id, variant, seed in confirm_tasks:
                progress.update(
                    "confirm_start",
                    dataset_id=dataset_id,
                    variant_id=variant.variant_id,
                    seed=seed,
                )
                result = _run_variant(
                    stage="confirm",
                    config=config,
                    run_root=run_root,
                    dataset_id=dataset_id,
                    variant=variant,
                    epochs=config.confirm_epochs,
                    max_folds=config.confirm_max_folds,
                    seed=seed,
                )
                _append_confirm_result(
                    dataset_id=dataset_id,
                    variant=variant,
                    seed=seed,
                    result=result,
                )
        else:
            LOGGER.info(
                "P31 confirm_parallel_start candidates=%d workers=%d",
                len(confirm_tasks),
                worker_count,
            )
            mp_context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as pool:
                futures = {}
                for dataset_id, variant, seed in confirm_tasks:
                    progress.update(
                        "confirm_start",
                        dataset_id=dataset_id,
                        variant_id=variant.variant_id,
                        seed=seed,
                    )
                    future = pool.submit(
                        _run_confirm_task,
                        config,
                        str(run_root),
                        dataset_id,
                        variant,
                        int(seed),
                    )
                    futures[future] = (dataset_id, variant.variant_id, int(seed))
                for future in as_completed(futures):
                    dataset_id, variant_id, seed = futures[future]
                    result = future.result()
                    _append_confirm_result(
                        dataset_id=dataset_id,
                        variant=variant_by_id[variant_id],
                        seed=seed,
                        result=result,
                    )

    screen_frame = _sort_leaderboard(pd.DataFrame(screen_rows))
    confirm_frame = _sort_leaderboard(pd.DataFrame(confirm_rows))
    fold_frame = pd.DataFrame(fold_rows)
    curve_frame = pd.DataFrame(curve_rows)
    prediction_frame = (
        pd.concat(prediction_frames, axis=0, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    screen_path = run_root / "screen_leaderboard.csv"
    confirm_path = run_root / "confirm_leaderboard.csv"
    fold_path = run_root / "fold_metrics.csv"
    curve_path = run_root / "training_curves.csv"
    prediction_path = run_root / "fold_predictions.csv"
    prediction_alias_path = run_root / "predictions.csv"
    screen_frame.to_csv(screen_path, index=False)
    confirm_frame.to_csv(confirm_path, index=False)
    fold_frame.to_csv(fold_path, index=False)
    curve_frame.to_csv(curve_path, index=False)
    prediction_frame.to_csv(prediction_path, index=False)
    prediction_frame.to_csv(prediction_alias_path, index=False)

    long_frame = _build_long_metrics(confirm_frame)
    ablation_summary = _build_ablation_summary(long_frame)
    long_path = run_root / "model_comparison_long.csv"
    ablation_long_path = run_root / "ablation_long.csv"
    ablation_wide_path = run_root / "ablation_wide.csv"
    component_contribution_path = run_root / "component_contribution.csv"
    ablation_csv_path = run_root / "ablation_summary.csv"
    ablation_json_path = run_root / "ablation_summary.json"
    long_frame.to_csv(long_path, index=False)
    long_frame.to_csv(ablation_long_path, index=False)
    _build_ablation_wide(long_frame).to_csv(ablation_wide_path, index=False)
    ablation_summary.to_csv(ablation_csv_path, index=False)
    component_contribution = _build_component_contribution(ablation_summary)
    component_contribution.to_csv(component_contribution_path, index=False)
    ablation_json = {
        "run_id": config.run_id,
        "evidence_role": EVIDENCE_ROLE,
        "rows": ablation_summary.to_dict(orient="records"),
    }
    ablation_json_path.write_text(
        json.dumps(ablation_json, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    figure_paths = _render_figures(run_root, confirm_frame, ablation_summary, fold_frame, curve_frame)
    gpu_perf_path = run_root / "gpu_perf_summary.json"
    gpu_perf_batches_path = run_root / "gpu_perf_batches.csv"
    gpu_perf_fold_path = run_root / "gpu_perf_fold_summary.csv"
    gpu_perf_summary = _gpu_perf_summary(runtime_device, config=config, curve_frame=curve_frame)
    gpu_perf_path.write_text(
        json.dumps(gpu_perf_summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _gpu_perf_batches(curve_frame).to_csv(gpu_perf_batches_path, index=False)
    _gpu_perf_fold_summary(curve_frame).to_csv(gpu_perf_fold_path, index=False)
    status = "completed" if not confirm_frame.empty and _is_full_confirm(confirm_frame, config) else "partial"
    confirm_fold_frame = _stage_fold_frame(fold_frame, "confirm")
    screen_fold_frame = _stage_fold_frame(fold_frame, "screen")
    completed_fold_count = _completed_fold_count(confirm_fold_frame)
    summary_path = run_root / "public_fusion_ablation_summary.json"
    manifest_path = run_root / "evidence_manifest.json"
    report_path = _resolve_path(config.report_root) / f"stage-i-public-fusion-ablation-{config.run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    resume_command_path = run_root / "resume_command.txt"
    resume_command_path.write_text(_resume_command(config) + "\n", encoding="utf-8")
    summary = {
        "run_id": config.run_id,
        "status": status,
        "generated_at_utc": _utc_now(),
        "runtime_device": runtime_device,
        "artifact_root": str(run_root),
        "evidence_role": EVIDENCE_ROLE,
        "dataset_role": "public_context_proxy",
        "screen_leaderboard_csv": str(screen_path),
        "confirm_leaderboard_csv": str(confirm_path),
        "model_comparison_long_csv": str(long_path),
        "ablation_long_csv": str(ablation_long_path),
        "ablation_wide_csv": str(ablation_wide_path),
        "component_contribution_csv": str(component_contribution_path),
        "ablation_summary_csv": str(ablation_csv_path),
        "ablation_summary_json": str(ablation_json_path),
        "fold_metrics_csv": str(fold_path),
        "training_curves_csv": str(curve_path),
        "fold_predictions_csv": str(prediction_path),
        "predictions_csv": str(prediction_alias_path),
        "public_ablation_config_json": str(config_path),
        "public_fusion_ablation_config_json": str(config_alias_path),
        "ablation_variant_manifest_json": str(variant_manifest_path),
        "public_fusion_ablation_variants_json": str(variants_alias_path),
        "gpu_perf_summary_json": str(gpu_perf_path),
        "gpu_perf_batches_csv": str(gpu_perf_batches_path),
        "gpu_perf_fold_summary_csv": str(gpu_perf_fold_path),
        "figure_paths": figure_paths,
        "evidence_manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "run_log_path": str(run_root / "run.log"),
        "progress_path": str(run_root / "progress.json"),
        "resume_command_txt": str(resume_command_path),
        "completed_fold_count": completed_fold_count,
        "expected_fold_count": _expected_fold_count(confirm_frame),
        "screen_completed_fold_count": _completed_fold_count(screen_fold_frame),
        "source_prepared_roots": dict(config.dataset_prepared_roots),
        "p28_source_run_id": P28_SOURCE_RUN_ID,
        "base_run_id": config.base_run_id,
        "gpuopt_enabled": True,
        "config_delta": _gpuopt_config_delta(config),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    if status != "completed":
        (run_root / "partial_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
    manifest = {
        **summary,
        "summary_path": str(summary_path),
        "protocol": {
            "screen": {
                "epochs": config.screen_epochs,
                "max_folds": config.screen_max_folds,
                "seed": config.seed,
            },
            "confirm": {
                "epochs": config.confirm_epochs,
                "max_folds": config.confirm_max_folds,
                "seeds": [config.seed, *config.extra_confirm_seeds],
            },
            "evidence_role": EVIDENCE_ROLE,
            "data_role": "public_context_proxy",
            "target_transform_policy": "fit on train fold only",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(summary, ablation_summary, confirm_frame) + "\n", encoding="utf-8")
    return StageIPublicFusionAblationRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        evidence_manifest_path=str(manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def load_p28_prepared_roots(source_root: str | Path = P28_SOURCE_ROOT) -> dict[str, str]:
    manifest_path = Path(source_root) / "evidence_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"P28 evidence manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roots = manifest.get("source_prepared_roots")
    if not isinstance(roots, Mapping):
        raise ValueError(f"P28 manifest does not contain source_prepared_roots: {manifest_path}")
    return {str(key): str(value) for key, value in roots.items()}


def _select_variants(filter_ids: Sequence[str]) -> tuple[PublicFusionAblationVariant, ...]:
    variants = default_public_fusion_ablation_variants()
    if not filter_ids:
        return variants
    selected = set(filter_ids)
    filtered = tuple(variant for variant in variants if variant.variant_id in selected)
    missing = selected.difference({variant.variant_id for variant in variants})
    if missing:
        raise ValueError(f"unknown P31 ablation variants: {sorted(missing)}")
    return filtered


def _run_variant(
    *,
    stage: str,
    config: StageIPublicFusionAblationConfig,
    run_root: Path,
    dataset_id: str,
    variant: PublicFusionAblationVariant,
    epochs: int,
    max_folds: int | None,
    seed: int,
) -> StageIDeepBaselineRunResult:
    candidate_root = run_root / stage / dataset_id / f"{variant.variant_id}__seed{seed}"
    candidate_root.mkdir(parents=True, exist_ok=True)
    (candidate_root / "config.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "dataset_id": dataset_id,
                "variant": asdict(variant),
                "seed": int(seed),
                "epochs": int(epochs),
                "max_folds": max_folds,
                "prepared_root": config.dataset_prepared_roots[dataset_id],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    cached = _load_completed_candidate(candidate_root, dataset_id=dataset_id, variant=variant)
    if config.resume and config.skip_completed and cached is not None:
        LOGGER.info(
            "P31 candidate_skip_completed stage=%s dataset=%s variant=%s seed=%s",
            stage,
            dataset_id,
            variant.variant_id,
            seed,
        )
        return cached
    return run_stage_i_deep_baseline(
        StageIDeepBaselineConfig(
            model_name=variant.model_name,
            dataset_id=dataset_id,
            profile="window_v2",
            prepared_artifact_root=config.dataset_prepared_roots[dataset_id],
            artifact_root=str(candidate_root),
            epochs=epochs,
            learning_rate=variant.learning_rate,
            batch_size=variant.batch_size,
            hidden_dim=variant.hidden_dim,
            num_heads=variant.num_heads,
            layers=variant.layers,
            dropout=variant.dropout,
            fusion_event_bias_weight=variant.fusion_event_bias_weight,
            fusion_lag_window_points=variant.fusion_lag_window_points,
            fusion_normalize_states=variant.fusion_normalize_states,
            max_folds=max_folds,
            seed=seed,
            device=config.device,
            train_sampling_policy=(
                variant.train_sampling_policy
                if variant.train_sampling_policy is not None
                else ("balanced_class" if dataset_id == "nasa_csm" else "none")
            ),
            regression_loss=variant.regression_loss,
            huber_delta=variant.huber_delta,
            target_transform=variant.target_transform if dataset_id == "uab_workload_dataset" else "none",
            gradient_clip_max_norm=variant.gradient_clip_max_norm,
            weight_decay=variant.weight_decay,
            heartbeat_seconds=config.heartbeat_seconds,
            batch_log_interval=config.batch_log_interval,
            tensor_cache=config.tensor_cache,
            max_cache_gb=config.max_cache_gb,
            pin_memory=config.pin_memory,
            non_blocking_copy=config.non_blocking_copy,
            auto_batch_size=config.auto_batch_size,
            batch_size_candidates=config.batch_size_candidates,
            amp=config.amp,
            grad_scaler=config.grad_scaler,
            amp_eval=config.amp_eval,
            torch_compile=config.torch_compile,
            profile_gpu=config.profile_gpu,
            eval_batch_size=config.eval_batch_size,
            checkpoint_policy=config.checkpoint_policy,
        )
    )


def _run_confirm_task(
    config: StageIPublicFusionAblationConfig,
    run_root: str,
    dataset_id: str,
    variant: PublicFusionAblationVariant,
    seed: int,
) -> StageIDeepBaselineRunResult:
    return _run_variant(
        stage="confirm",
        config=config,
        run_root=Path(run_root),
        dataset_id=dataset_id,
        variant=variant,
        epochs=config.confirm_epochs,
        max_folds=config.confirm_max_folds,
        seed=int(seed),
    )


def _load_completed_candidate(
    candidate_root: Path,
    *,
    dataset_id: str,
    variant: PublicFusionAblationVariant,
) -> StageIDeepBaselineRunResult | None:
    summary_path = candidate_root / "deep_baseline_summary.json"
    predictions_path = candidate_root / "fold_predictions.csv"
    report_path = candidate_root / "deep_baseline_report.md"
    if not summary_path.exists() or not predictions_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return StageIDeepBaselineRunResult(
        dataset_id=dataset_id,
        model_name=variant.model_name,
        artifact_root=str(candidate_root),
        summary_path=str(summary_path),
        report_path=str(report_path),
        predictions_path=str(predictions_path),
        summary=summary,
    )


def _leaderboard_row(
    *,
    stage: str,
    dataset_id: str,
    variant: PublicFusionAblationVariant,
    seed: int,
    result: StageIDeepBaselineRunResult,
) -> dict[str, object]:
    row = {
        "stage": stage,
        "dataset_id": dataset_id,
        "variant_id": variant.variant_id,
        "seed": int(seed),
        "summary_path": result.summary_path,
        "artifact_root": result.artifact_root,
        **asdict(variant),
    }
    summary = result.summary
    if dataset_id == "nasa_csm":
        combined = summary["objective"]["groups"]["combined"]
        row.update(
            {
                "primary_metric": "combined_macro_f1",
                "selection_score": float(combined["macro_f1"]),
                "secondary_score": float(combined["balanced_accuracy"]),
                "combined_macro_f1": float(combined["macro_f1"]),
                "combined_balanced_accuracy": float(combined["balanced_accuracy"]),
                "benchmark_only_macro_f1": float(summary["objective"]["groups"]["benchmark_only"]["macro_f1"]),
                "loft_only_macro_f1": float(summary["objective"]["groups"]["loft_only"]["macro_f1"]),
                "sample_count": int(combined["sample_count"]),
                "fold_count": int(combined["fold_count"]),
            }
        )
        return row
    subjective = summary["subjective"]["groups"]
    objective = summary["objective"]["groups"]
    mean_rmse = (
        float(subjective["n_back"]["rmse"])
        + float(subjective["heat_the_chair"]["rmse"])
    ) / 2.0
    row.update(
        {
            "primary_metric": "mean_rmse",
            "selection_score": mean_rmse,
            "secondary_score": (
                float(subjective["n_back"]["mae"])
                + float(subjective["heat_the_chair"]["mae"])
            )
            / 2.0,
            "mean_rmse": mean_rmse,
            "n_back_rmse": float(subjective["n_back"]["rmse"]),
            "n_back_mae": float(subjective["n_back"]["mae"]),
            "heat_the_chair_rmse": float(subjective["heat_the_chair"]["rmse"]),
            "heat_the_chair_mae": float(subjective["heat_the_chair"]["mae"]),
            "n_back_macro_f1": float(objective["n_back"]["macro_f1"]),
            "n_back_balanced_accuracy": float(objective["n_back"]["balanced_accuracy"]),
            "heat_the_chair_macro_f1": float(objective["heat_the_chair"]["macro_f1"]),
            "heat_the_chair_balanced_accuracy": float(objective["heat_the_chair"]["balanced_accuracy"]),
            "sample_count": int(subjective["n_back"]["sample_count"] + subjective["heat_the_chair"]["sample_count"]),
            "fold_count": int(max(subjective["n_back"]["fold_count"], subjective["heat_the_chair"]["fold_count"])),
        }
    )
    return row


def _candidate_fold_rows(
    result: StageIDeepBaselineRunResult,
    stage: str,
    dataset_id: str,
    variant_id: str,
    seed: int,
) -> list[dict[str, object]]:
    frame = pd.read_csv(result.predictions_path)
    rows: list[dict[str, object]] = []
    for (track, group, split_group), fold in frame.groupby(
        ["track", "evaluation_group", "split_group"],
        sort=False,
    ):
        if track == "objective":
            labels = sorted(set(fold["y_true"].astype(int)) | set(fold["y_pred"].astype(int)))
            metrics = evaluate_classification_predictions(fold, label_order=labels)
            payload = {
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        else:
            metrics = evaluate_regression_predictions(fold)
            payload = {"rmse": metrics["rmse"], "mae": metrics["mae"]}
        rows.append(
            {
                "stage": stage,
                "dataset_id": dataset_id,
                "variant_id": variant_id,
                "seed": int(seed),
                "track": track,
                "evaluation_group": group,
                "split_group": split_group,
                "sample_count": int(metrics["sample_count"]),
                **payload,
            }
        )
    pd.DataFrame(rows).to_csv(Path(result.artifact_root) / "fold_metrics.csv", index=False)
    return rows


def _training_curve_rows(
    result: StageIDeepBaselineRunResult,
    stage: str,
    dataset_id: str,
    variant_id: str,
    seed: int,
) -> list[dict[str, object]]:
    path = result.summary.get("training_curves_path")
    if not path or not Path(path).exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    frame["stage"] = stage
    frame["dataset_id"] = dataset_id
    frame["variant_id"] = variant_id
    frame["seed"] = int(seed)
    return frame.to_dict(orient="records")


def _write_partial_tables(
    run_root: Path,
    screen_rows: Sequence[Mapping[str, object]],
    confirm_rows: Sequence[Mapping[str, object]],
    fold_rows: Sequence[Mapping[str, object]],
    curve_rows: Sequence[Mapping[str, object]],
) -> None:
    if screen_rows:
        _sort_leaderboard(pd.DataFrame(screen_rows)).to_csv(run_root / "screen_leaderboard.partial.csv", index=False)
    if confirm_rows:
        _sort_leaderboard(pd.DataFrame(confirm_rows)).to_csv(run_root / "confirm_leaderboard.partial.csv", index=False)
    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(run_root / "fold_metrics.partial.csv", index=False)
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(run_root / "training_curves.partial.csv", index=False)


def _sort_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    pieces = []
    for dataset_id, subset in frame.groupby("dataset_id", sort=False):
        ascending = dataset_id == "uab_workload_dataset"
        pieces.append(
            subset.sort_values(
                ["selection_score", "secondary_score", "variant_id"],
                ascending=[ascending, ascending, True],
            )
        )
    return pd.concat(pieces, axis=0, ignore_index=True)


def _build_long_metrics(confirm_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in confirm_frame.to_dict(orient="records"):
        common = {
            "stage": "confirm",
            "dataset_id": row["dataset_id"],
            "variant_id": row["variant_id"],
            "ablation_family": row["ablation_family"],
            "seed": int(row["seed"]),
            "summary_path": row["summary_path"],
            "artifact_root": row["artifact_root"],
            "evidence_role": EVIDENCE_ROLE,
            "data_role": "public_context_proxy",
        }
        if row["dataset_id"] == "nasa_csm":
            for metric_name in ("combined_macro_f1", "combined_balanced_accuracy", "benchmark_only_macro_f1", "loft_only_macro_f1"):
                if metric_name in row and pd.notna(row[metric_name]):
                    rows.append({**common, "task_group": metric_name.rsplit("_", 2)[0], "metric": metric_name, "value": float(row[metric_name])})
            continue
        for metric_name in (
            "n_back_rmse",
            "n_back_mae",
            "heat_the_chair_rmse",
            "heat_the_chair_mae",
            "n_back_macro_f1",
            "n_back_balanced_accuracy",
            "heat_the_chair_macro_f1",
            "heat_the_chair_balanced_accuracy",
            "mean_rmse",
        ):
            if metric_name in row and pd.notna(row[metric_name]):
                task_group = "uab_mean" if metric_name == "mean_rmse" else metric_name.rsplit("_", 1)[0]
                rows.append({**common, "task_group": task_group, "metric": metric_name, "value": float(row[metric_name])})
    return pd.DataFrame(rows)


def _build_ablation_summary(long_frame: pd.DataFrame) -> pd.DataFrame:
    if long_frame.empty:
        return pd.DataFrame()
    full = long_frame[long_frame["variant_id"] == "full"]
    full_lookup = {
        (row["dataset_id"], row["task_group"], row["metric"], int(row["seed"])): float(row["value"])
        for row in full.to_dict(orient="records")
    }
    rows = []
    for row in long_frame.to_dict(orient="records"):
        key = (row["dataset_id"], row["task_group"], row["metric"], int(row["seed"]))
        full_value = full_lookup.get(key)
        metric = str(row["metric"])
        value = float(row["value"])
        if full_value is None:
            delta = float("nan")
            rel = float("nan")
        elif _lower_is_better(metric):
            delta = value - full_value
            rel = delta / abs(full_value) * 100.0 if abs(full_value) > 1e-12 else float("nan")
        else:
            delta = full_value - value
            rel = delta / abs(value) * 100.0 if abs(value) > 1e-12 else float("nan")
        rows.append(
            {
                **row,
                "full_value": full_value,
                "full_minus_ablation_delta": delta,
                "delta_abs": delta,
                "delta_rel_pct": rel,
                "positive_delta_means": "full_better",
            }
        )
    frame = pd.DataFrame(rows)
    group_cols = ["dataset_id", "task_group", "metric", "variant_id", "ablation_family"]
    numeric_cols = ["value", "full_value", "delta_abs", "delta_rel_pct"]
    grouped = frame.groupby(group_cols, dropna=False, sort=False)[numeric_cols].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column).rstrip("_") for column in grouped.columns]
    return grouped.reset_index()


def _build_ablation_wide(long_frame: pd.DataFrame) -> pd.DataFrame:
    if long_frame.empty:
        return pd.DataFrame()
    return long_frame.pivot_table(
        index=["dataset_id", "task_group", "metric", "seed"],
        columns="variant_id",
        values="value",
        aggfunc="first",
    ).reset_index()


def _build_component_contribution(ablation_summary: pd.DataFrame) -> pd.DataFrame:
    if ablation_summary.empty:
        return pd.DataFrame()
    subset = ablation_summary[ablation_summary["variant_id"] != "full"].copy()
    if subset.empty or "delta_abs_mean" not in subset:
        return pd.DataFrame()
    grouped = (
        subset.groupby(["dataset_id", "metric", "ablation_family"], dropna=False, sort=False)
        .agg(
            component_delta_abs_mean=("delta_abs_mean", "mean"),
            component_delta_abs_max=("delta_abs_mean", "max"),
            component_delta_rel_pct_mean=("delta_rel_pct_mean", "mean"),
            row_count=("delta_abs_mean", "count"),
        )
        .reset_index()
    )
    grouped["positive_delta_means"] = "full_better"
    return grouped


def _render_figures(
    run_root: Path,
    confirm_frame: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    fold_frame: pd.DataFrame,
    curve_frame: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "fig_public_ablation_nasa_macro_f1": str(run_root / "fig_public_ablation_nasa_macro_f1.png"),
        "fig_public_ablation_nasa_balanced_accuracy": str(run_root / "fig_public_ablation_nasa_balanced_accuracy.png"),
        "fig_public_ablation_uab_rmse": str(run_root / "fig_public_ablation_uab_rmse.png"),
        "fig_public_ablation_delta_heatmap": str(run_root / "fig_public_ablation_delta_heatmap.png"),
        "fig_public_ablation_win_summary": str(run_root / "fig_public_ablation_win_summary.png"),
        "fig_public_ablation_config_sensitivity": str(run_root / "fig_public_ablation_config_sensitivity.png"),
        "fig_public_ablation_context_contribution": str(run_root / "fig_public_ablation_context_contribution.png"),
        "fig_public_ablation_component_contribution": str(run_root / "fig_public_ablation_component_contribution.png"),
        "fig_public_ablation_fold_stability": str(run_root / "fig_public_ablation_fold_stability.png"),
        "fig_public_ablation_gpu_throughput": str(run_root / "fig_public_ablation_gpu_throughput.png"),
    }
    _bar_metric(confirm_frame, paths["fig_public_ablation_nasa_macro_f1"], "nasa_csm", "combined_macro_f1", "NASA combined macro-F1 by ablation", higher=True)
    _bar_metric(confirm_frame, paths["fig_public_ablation_nasa_balanced_accuracy"], "nasa_csm", "combined_balanced_accuracy", "NASA combined balanced accuracy by ablation", higher=True)
    _uab_rmse_plot(confirm_frame, paths["fig_public_ablation_uab_rmse"])
    _delta_heatmap(ablation_summary, paths["fig_public_ablation_delta_heatmap"])
    _win_summary(ablation_summary, paths["fig_public_ablation_win_summary"])
    _config_sensitivity(confirm_frame, paths["fig_public_ablation_config_sensitivity"])
    _context_contribution(confirm_frame, paths["fig_public_ablation_context_contribution"])
    _component_contribution_plot(ablation_summary, paths["fig_public_ablation_component_contribution"])
    _fold_stability(fold_frame, paths["fig_public_ablation_fold_stability"])
    _gpu_throughput_plot(curve_frame, paths["fig_public_ablation_gpu_throughput"])
    return paths


def _bar_metric(frame: pd.DataFrame, path: str, dataset_id: str, metric: str, title: str, *, higher: bool) -> None:
    subset = frame[(frame["dataset_id"] == dataset_id) & frame[metric].notna()].copy() if metric in frame else pd.DataFrame()
    fig, axis = plt.subplots(figsize=(10, 4.8))
    if subset.empty:
        axis.text(0.5, 0.5, "no rows", ha="center", va="center")
        axis.axis("off")
    else:
        subset = subset.sort_values(metric, ascending=not higher)
        values = subset[metric].astype(float)
        bars = axis.barh(np.arange(len(subset)), values, color="#2f6f9f")
        label_horizontal_bars(axis, bars, values)
        axis.set_yticks(np.arange(len(subset)))
        axis.set_yticklabels(subset["variant_id"], fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel(metric)
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _uab_rmse_plot(frame: pd.DataFrame, path: str) -> None:
    subset = (
        frame[frame["dataset_id"] == "uab_workload_dataset"].copy()
        if not frame.empty and "dataset_id" in frame
        else pd.DataFrame()
    )
    fig, axis = plt.subplots(figsize=(10, 5))
    if subset.empty:
        axis.text(0.5, 0.5, "no UAB rows", ha="center", va="center")
        axis.axis("off")
    else:
        subset = subset.sort_values("mean_rmse")
        x = np.arange(len(subset))
        width = 0.38
        n_back = subset["n_back_rmse"].astype(float)
        heat = subset["heat_the_chair_rmse"].astype(float)
        bars_n = axis.bar(x - width / 2, n_back, width, label="n_back", color="#38761d")
        bars_h = axis.bar(x + width / 2, heat, width, label="heat_the_chair", color="#b45f06")
        label_vertical_bars(axis, bars_n, n_back)
        label_vertical_bars(axis, bars_h, heat)
        axis.set_xticks(x)
        axis.set_xticklabels(subset["variant_id"], rotation=35, ha="right", fontsize=8)
        axis.set_ylabel("RMSE")
        axis.set_title("UAB subjective RMSE by ablation")
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _delta_heatmap(frame: pd.DataFrame, path: str) -> None:
    subset = frame[frame["variant_id"] != "full"].copy() if not frame.empty else pd.DataFrame()
    metric_order = [
        "combined_macro_f1",
        "combined_balanced_accuracy",
        "n_back_rmse",
        "heat_the_chair_rmse",
        "mean_rmse",
    ]
    subset = subset[subset["metric"].isin(metric_order)] if not subset.empty else subset
    variants = list(dict.fromkeys(subset["variant_id"])) if not subset.empty else ["none"]
    metrics = [metric for metric in metric_order if metric in set(subset["metric"])] or ["none"]
    data = np.zeros((len(metrics), len(variants)), dtype=float)
    for i, metric in enumerate(metrics):
        for j, variant in enumerate(variants):
            rows = subset[(subset["metric"] == metric) & (subset["variant_id"] == variant)]
            data[i, j] = float(rows["delta_abs_mean"].iloc[0]) if not rows.empty else np.nan
    fig, axis = plt.subplots(figsize=(max(7, len(variants) * 0.7), max(3.5, len(metrics) * 0.7)))
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    image = axis.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    axis.set_title("Full minus ablation delta (positive means full better)")
    axis.set_xticks(np.arange(len(variants)))
    axis.set_xticklabels(variants, rotation=35, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(metrics)))
    axis.set_yticklabels(metrics)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            axis.text(j, i, "" if np.isnan(value) else f"{value:+.4f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _win_summary(frame: pd.DataFrame, path: str) -> None:
    subset = frame[frame["variant_id"] != "full"].copy() if not frame.empty else pd.DataFrame()
    records = []
    for row in subset.to_dict(orient="records"):
        delta = float(row.get("delta_abs_mean", 0.0))
        records.append({"variant_id": row["variant_id"], "status": "W" if delta > 0 else ("T" if abs(delta) < 1e-9 else "L")})
    counts = pd.DataFrame(records)
    fig, axis = plt.subplots(figsize=(9, 4.5))
    if counts.empty:
        axis.text(0.5, 0.5, "no ablation rows", ha="center", va="center")
        axis.axis("off")
    else:
        table = counts.groupby(["variant_id", "status"]).size().unstack(fill_value=0)
        for status in ("W", "T", "L"):
            if status not in table:
                table[status] = 0
        table = table[["W", "T", "L"]]
        bottom = np.zeros(len(table))
        colors = {"W": "#3c8d3c", "T": "#8c8c8c", "L": "#c0504d"}
        for status in ("W", "T", "L"):
            axis.bar(table.index, table[status], bottom=bottom, label=status, color=colors[status])
            bottom += table[status].to_numpy()
        label_stack_totals(axis, table.index, bottom)
        axis.set_xticks(np.arange(len(table)), table.index, rotation=35, ha="right", fontsize=8)
        axis.set_ylabel("metric count")
        axis.set_title("Full vs ablation W/T/L summary")
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _config_sensitivity(frame: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fields = ["fusion_lag_window_points", "fusion_event_bias_weight", "fusion_normalize_states", "regression_loss"]
    for axis, field in zip(axes.ravel(), fields, strict=True):
        if frame.empty or field not in frame:
            axis.text(0.5, 0.5, "no rows", ha="center", va="center")
            axis.axis("off")
            continue
        subset = frame.copy()
        x = pd.factorize(subset[field].astype(str))[0] if field == "regression_loss" else subset[field].fillna(-1).astype(float)
        axis.scatter(x, subset["selection_score"].astype(float), alpha=0.75)
        axis.set_xlabel(field)
        axis.set_ylabel("selection_score")
    fig.suptitle("Ablation config sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _context_contribution(frame: pd.DataFrame, path: str) -> None:
    if frame.empty or "variant_id" not in frame or "dataset_id" not in frame:
        subset = pd.DataFrame()
    else:
        subset = frame[
            frame["variant_id"].isin(
                [
                    "full",
                    "physiology_only",
                    "single_stream_physio_only",
                    "no_context_proxy_stream",
                    "context_only",
                    "simple_dual_stream_concat",
                    "no_causal_fusion",
                ]
            )
        ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _plot_small_axis(axes[0], subset[subset["dataset_id"] == "nasa_csm"], "combined_macro_f1", "NASA macro-F1")
    _plot_small_axis(axes[1], subset[subset["dataset_id"] == "uab_workload_dataset"], "mean_rmse", "UAB mean RMSE")
    fig.suptitle("Context stream and causal fusion contribution")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _component_contribution_plot(frame: pd.DataFrame, path: str) -> None:
    component = _build_component_contribution(frame)
    fig, axis = plt.subplots(figsize=(10, 5))
    if component.empty:
        axis.text(0.5, 0.5, "no component rows", ha="center", va="center")
        axis.axis("off")
    else:
        grouped = (
            component.groupby("ablation_family", sort=False)["component_delta_abs_mean"]
            .mean()
            .sort_values(ascending=False)
        )
        colors = ["#3c8d3c" if value >= 0 else "#c0504d" for value in grouped.to_numpy(dtype=float)]
        values = grouped.to_numpy(dtype=float)
        bars = axis.barh(np.arange(len(grouped)), values, color=colors)
        label_horizontal_bars(axis, bars, values)
        axis.set_yticks(np.arange(len(grouped)))
        axis.set_yticklabels(grouped.index, fontsize=8)
        axis.invert_yaxis()
        axis.axvline(0.0, color="#555555", linewidth=0.8)
        axis.set_xlabel("mean delta; positive means full better")
        axis.set_title("Component contribution summary")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_small_axis(axis, frame: pd.DataFrame, metric: str, title: str) -> None:
    if frame.empty or metric not in frame:
        axis.text(0.5, 0.5, "no rows", ha="center", va="center")
        axis.axis("off")
        return
    values = frame[metric].astype(float)
    bars = axis.bar(frame["variant_id"], values, color="#4f81bd")
    label_vertical_bars(axis, bars, values)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=35, labelsize=8)


def _fold_stability(frame: pd.DataFrame, path: str) -> None:
    if frame.empty:
        subset = pd.DataFrame()
    else:
        macro_f1 = frame["macro_f1"] if "macro_f1" in frame else pd.Series(np.nan, index=frame.index)
        rmse = frame["rmse"] if "rmse" in frame else pd.Series(np.nan, index=frame.index)
        subset = frame[
            ((frame["dataset_id"] == "nasa_csm") & (frame["evaluation_group"] == "combined") & macro_f1.notna())
            | ((frame["dataset_id"] == "uab_workload_dataset") & rmse.notna())
        ].copy()
    fig, axis = plt.subplots(figsize=(11, 5))
    if subset.empty:
        axis.text(0.5, 0.5, "no fold rows", ha="center", va="center")
        axis.axis("off")
    else:
        values = subset["macro_f1"] if "macro_f1" in subset else pd.Series(np.nan, index=subset.index)
        if "rmse" in subset:
            values = values.fillna(subset["rmse"])
        values = values.astype(float)
        subset["metric_value"] = values
        labels = list(dict.fromkeys(subset["variant_id"]))
        data = [subset[subset["variant_id"] == label]["metric_value"].to_numpy() for label in labels]
        axis.boxplot(data, labels=labels, showfliers=False)
        axis.tick_params(axis="x", rotation=35, labelsize=8)
        axis.set_title("Fold-level metric stability")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _gpu_throughput_plot(curve_frame: pd.DataFrame, path: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 4.8))
    if curve_frame.empty or "batch_size" not in curve_frame:
        axis.text(0.5, 0.5, "no GPU runtime rows", ha="center", va="center")
        axis.axis("off")
    else:
        group_cols = [column for column in ("dataset_id", "variant_id") if column in curve_frame]
        grouped = (
            curve_frame.groupby(group_cols, dropna=False, sort=False)
            .agg(
                selected_batch_size=("batch_size", "max"),
                cache_gb=("actual_cache_gb", "max") if "actual_cache_gb" in curve_frame else ("batch_size", "count"),
            )
            .reset_index()
        )
        grouped["label"] = grouped[group_cols].astype(str).agg(" / ".join, axis=1)
        grouped = grouped.sort_values("selected_batch_size", ascending=False).head(20)
        y = np.arange(len(grouped))
        values = grouped["selected_batch_size"].astype(float)
        bars = axis.barh(y, values, color="#2f6f9f")
        label_horizontal_bars(axis, bars, values)
        axis.set_yticks(y)
        axis.set_yticklabels(grouped["label"], fontsize=7)
        axis.invert_yaxis()
        axis.set_xlabel("selected batch size")
        axis.set_title("GPUOPT selected batch size by variant")
        for index, row in enumerate(grouped.to_dict(orient="records")):
            axis.text(
                float(row["selected_batch_size"]),
                index,
                f" cache {float(row['cache_gb']):.2f} GB",
                va="center",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _lower_is_better(metric: str) -> bool:
    return metric.endswith("_rmse") or metric.endswith("_mae") or metric == "mean_rmse"


def _is_full_confirm(confirm_frame: pd.DataFrame, config: StageIPublicFusionAblationConfig) -> bool:
    if confirm_frame.empty:
        return False
    required_columns = {"dataset_id", "variant_id", "seed"}
    if not required_columns.issubset(confirm_frame.columns):
        return False
    expected_datasets = set(config.datasets)
    expected_variants = {variant.variant_id for variant in _select_variants(config.variants)}
    expected_seeds = {int(config.seed), *[int(seed) for seed in config.extra_confirm_seeds]}
    observed = {
        (str(row["dataset_id"]), str(row["variant_id"]), int(row["seed"]))
        for row in confirm_frame[list(required_columns)].dropna().to_dict(orient="records")
    }
    expected = {
        (dataset_id, variant_id, seed)
        for dataset_id in expected_datasets
        for variant_id in expected_variants
        for seed in expected_seeds
    }
    if not expected.issubset(observed):
        return False
    nasa = confirm_frame[confirm_frame["dataset_id"] == "nasa_csm"]
    if not nasa.empty and int(nasa["fold_count"].max()) < 16:
        return False
    return True


def _completed_fold_count(fold_frame: pd.DataFrame) -> int:
    if fold_frame.empty:
        return 0
    return int(
        fold_frame[
            ["stage", "dataset_id", "variant_id", "seed", "track", "evaluation_group", "split_group"]
        ]
        .drop_duplicates()
        .shape[0]
    )


def _stage_fold_frame(fold_frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    if fold_frame.empty or "stage" not in fold_frame:
        return pd.DataFrame()
    return fold_frame[fold_frame["stage"] == stage].copy()


def _expected_fold_count(confirm_frame: pd.DataFrame) -> int:
    if confirm_frame.empty:
        return 0
    total = 0
    for row in confirm_frame.to_dict(orient="records"):
        if row["dataset_id"] == "nasa_csm":
            total += int(row.get("fold_count", 0)) * 3
        else:
            total += int(row.get("fold_count", 0)) * 4
    return total


def _gpu_perf_summary(
    runtime_device: str,
    *,
    config: StageIPublicFusionAblationConfig,
    curve_frame: pd.DataFrame,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "runtime_device": runtime_device,
        "tensor_cache_mode": config.tensor_cache,
        "auto_batch_size": bool(config.auto_batch_size),
        "batch_size_candidates": [int(value) for value in config.batch_size_candidates],
        "amp_mode": config.amp,
        "amp_eval": bool(config.amp_eval),
        "torch_compile_mode": config.torch_compile,
        "num_workers": int(config.num_workers),
        "parallel_fold_prep": int(config.parallel_fold_prep),
        "parallel_candidates": int(config.parallel_candidates),
        "oom_fallback_count": 0,
    }
    if not curve_frame.empty:
        if "batch_size" in curve_frame:
            payload["best_batch_size"] = int(pd.to_numeric(curve_frame["batch_size"], errors="coerce").max())
        for field in ("tensor_cache_mode", "amp_mode", "compile_status"):
            if field in curve_frame:
                payload[field + "_observed"] = sorted(set(curve_frame[field].dropna().astype(str)))
        if "cache_fallback_reason" in curve_frame:
            fallback = curve_frame["cache_fallback_reason"].dropna().astype(str)
            payload["cache_fallback_count"] = int(fallback.shape[0])
            payload["cache_fallback_reasons"] = sorted(set(fallback))
    try:
        payload["torch_version"] = torch.__version__
        payload["cuda_version"] = torch.version.cuda
        payload["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            payload["gpu_name"] = torch.cuda.get_device_name(0)
            payload["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
            payload["max_memory_reserved_gb"] = torch.cuda.max_memory_reserved(0) / (1024 ** 3)
        payload["runtime_snapshot"] = _gpu_runtime_snapshot_safe()
    except Exception as exc:  # pragma: no cover - diagnostics only
        payload["gpu_probe_error"] = str(exc)
    return payload


def _gpu_perf_batches(curve_frame: pd.DataFrame) -> pd.DataFrame:
    if curve_frame.empty:
        return pd.DataFrame()
    keep = [
        column
        for column in (
            "stage",
            "dataset_id",
            "variant_id",
            "seed",
            "track",
            "evaluation_group",
            "fold_index",
            "split_group",
            "epoch",
            "batch_count",
            "batch_size",
            "requested_batch_size",
            "tensor_cache_mode",
            "amp_mode",
            "compile_status",
            "cache_build_time_s",
            "estimated_cache_gb",
            "actual_cache_gb",
        )
        if column in curve_frame
    ]
    return curve_frame[keep].copy()


def _gpu_perf_fold_summary(curve_frame: pd.DataFrame) -> pd.DataFrame:
    if curve_frame.empty or "fold_index" not in curve_frame:
        return pd.DataFrame()
    keys = [
        column
        for column in ("stage", "dataset_id", "variant_id", "seed", "track", "evaluation_group", "fold_index", "split_group")
        if column in curve_frame
    ]
    agg = {
        "batch_size": "max",
        "batch_count": "sum",
        "train_loss": "last",
    }
    agg = {key: value for key, value in agg.items() if key in curve_frame}
    summary = curve_frame.groupby(keys, dropna=False, sort=False).agg(agg).reset_index()
    for field in ("tensor_cache_mode", "amp_mode", "compile_status"):
        if field in curve_frame:
            lookup = curve_frame.groupby(keys, dropna=False, sort=False)[field].agg(lambda s: ",".join(sorted(set(s.dropna().astype(str)))))
            summary = summary.merge(lookup.reset_index(), on=keys, how="left")
    return summary


def _gpu_runtime_snapshot_safe() -> dict[str, object]:
    try:
        from chronaris.pipelines.stage_i.common.gpu_runtime import gpu_runtime_snapshot

        return gpu_runtime_snapshot()
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"snapshot_error": type(exc).__name__ + ":" + str(exc)}


def _render_report(summary: Mapping[str, object], ablation_summary: pd.DataFrame, confirm_frame: pd.DataFrame) -> str:
    lines = [
        f"# Stage I Public Fusion Ablation - {summary['run_id']}",
        "",
        "## Executive Summary",
        "",
        "P31 decomposes the P28 chronaris_public_fusion refresh result on NASA/UAB. "
        "The public branch is evaluated as public_adapter_context_proxy_evidence, with the second stream recorded as context_proxy rather than private vehicle telemetry. "
        "The ablation table reports absolute and relative deltas for lag window, event bias, fusion normalization, stream contribution, fusion head, target transform and regression-loss settings.",
        "",
        "## Dataset and public evidence role",
        "",
        f"- evidence_role: `{summary['evidence_role']}`",
        f"- source_prepared_roots: `{summary['source_prepared_roots']}`",
        f"- P28 source run: `{summary['p28_source_run_id']}`",
        "",
        "## Ablation design",
        "",
        f"- variant_manifest: `{summary['ablation_variant_manifest_json']}`",
        f"- config: `{summary['public_ablation_config_json']}`",
        "",
        "## Main leaderboard",
        "",
        _markdown_table(
            confirm_frame.head(30),
            ["dataset_id", "variant_id", "seed", "primary_metric", "selection_score", "secondary_score", "fold_count"],
        ),
        "",
        "## Ablation deltas",
        "",
        _markdown_table(
            ablation_summary.head(40),
            ["dataset_id", "task_group", "metric", "variant_id", "value_mean", "delta_abs_mean", "delta_rel_pct_mean"],
        ),
        "",
        "## Figure index",
        "",
    ]
    for name, path in (summary.get("figure_paths") or {}).items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- screen_leaderboard: `{summary['screen_leaderboard_csv']}`",
            f"- confirm_leaderboard: `{summary['confirm_leaderboard_csv']}`",
            f"- fold_metrics: `{summary['fold_metrics_csv']}`",
            f"- training_curves: `{summary['training_curves_csv']}`",
            f"- predictions: `{summary['fold_predictions_csv']}`",
            f"- evidence_manifest: `{summary['evidence_manifest_path']}`",
            f"- run_log: `{summary['run_log_path']}`",
            f"- progress: `{summary['progress_path']}`",
            "",
            "## Midterm-ready wording",
            "",
            "P31 public ablation decomposes the P28 chronaris_public_fusion refresh result on NASA/UAB. "
            "The table reports how lag-aware fusion, event bias, public context stream use, causal fusion head, target transform and regression loss contribute to NASA attention-state classification and UAB workload regression under the same prepared public split protocol.",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    existing = [column for column in columns if column in frame.columns]
    rows = ["| " + " | ".join(existing) + " |", "| " + " | ".join("---" for _ in existing) + " |"]
    for _, row in frame[existing].iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, float):
                cells.append(f"{value:.4f}" if math.isfinite(value) else "")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _blocked_payload(
    config: StageIPublicFusionAblationConfig,
    run_root: Path,
    exc: BaseException,
) -> dict[str, object]:
    return {
        "run_id": config.run_id,
        "status": "blocked",
        "generated_at_utc": _utc_now(),
        "artifact_root": str(run_root),
        "blocked_reason": str(exc),
        "blocked_type": type(exc).__name__,
        "resume_command": _resume_command(config),
    }


def _resume_command(config: StageIPublicFusionAblationConfig) -> str:
    parts = [
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python",
        "scripts/stage_i/public/run_public_fusion_ablation.py",
        "--run-id",
        config.run_id,
        "--datasets",
        *config.datasets,
        "--screen-epochs",
        str(config.screen_epochs),
        "--confirm-epochs",
        str(config.confirm_epochs),
        "--screen-max-folds",
        _max_folds_arg(config.screen_max_folds),
        "--confirm-max-folds",
        _max_folds_arg(config.confirm_max_folds),
        "--seed",
        str(config.seed),
        "--device",
        config.device,
        "--require-cuda",
        "--resume",
        "--skip-completed",
        "--allow-partial",
        "--tensor-cache",
        config.tensor_cache,
        "--max-cache-gb",
        str(config.max_cache_gb),
        "--amp",
        config.amp,
        "--torch-compile",
        config.torch_compile,
        "--num-workers",
        str(config.num_workers),
        "--parallel-fold-prep",
        str(config.parallel_fold_prep),
        "--parallel-candidates",
        str(config.parallel_candidates),
        "--checkpoint-policy",
        str(config.checkpoint_policy),
    ]
    if config.auto_batch_size:
        parts.append("--auto-batch-size")
    parts.extend(["--batch-size-candidates", *[str(value) for value in config.batch_size_candidates]])
    if config.variants:
        parts.extend(["--variants", *config.variants])
    if config.extra_confirm_seeds:
        parts.extend(["--extra-confirm-seeds", *[str(value) for value in config.extra_confirm_seeds]])
    if config.base_run_id:
        parts.extend(["--base-run-id", config.base_run_id])
    return " ".join(parts)


def _max_folds_arg(value: int | None) -> str:
    return "none" if value is None else str(value)


def _gpuopt_config_delta(config: StageIPublicFusionAblationConfig) -> dict[str, object]:
    return {
        "tensor_cache": config.tensor_cache,
        "max_cache_gb": float(config.max_cache_gb),
        "auto_batch_size": bool(config.auto_batch_size),
        "batch_size_candidates": [int(value) for value in config.batch_size_candidates],
        "amp": config.amp,
        "amp_eval": bool(config.amp_eval),
        "torch_compile": config.torch_compile,
        "profile_gpu": bool(config.profile_gpu),
        "num_workers": int(config.num_workers),
        "parallel_fold_prep": int(config.parallel_fold_prep),
        "parallel_candidates": int(config.parallel_candidates),
        "checkpoint_policy": str(config.checkpoint_policy),
        "config_lineage": "gpuopt rerun uses same prepared roots, labels and split construction; it must not be merged into base run leaderboards without protocol/config columns.",
    }


def _config_dict(config: StageIPublicFusionAblationConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["dataset_prepared_roots"] = dict(config.dataset_prepared_roots)
    return payload


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
