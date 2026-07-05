"""Dingxin / feature export real-data third-party comparison."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronaris-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chronaris.dataset import dump_task_eval_private_task_entries
from chronaris.models.alignment.contrastive import (
    HardNegativeSamplerConfig,
    info_nce_loss,
    stratified_hard_negative_samples,
    supervised_contrastive_margin_loss,
)
from chronaris.models.alignment.task_heads_v2 import (
    class_balanced_focal_loss,
    gate_regularization_loss,
)
from chronaris.modeling.common.deep_models import build_task_eval_deep_model
from chronaris.modeling.common.gpu_runtime import (
    choose_auto_batch_size,
    get_train_batch,
    gpu_runtime_snapshot,
    iter_eval_batches,
    make_grad_scaler,
    prepare_fold_tensors,
    resolve_amp_runtime,
)
from chronaris.modeling.common.run_observer import (
    StageIRunProgress,
    open_task_eval_run_observer,
)
from chronaris.modeling.common.plot_labels import (
    label_horizontal_bars,
    label_stack_totals,
    label_vertical_bars,
)
from chronaris.evaluation.dingxin.pipelines.benchmark_data import (
    CLASS_LABEL_TO_ID,
    TASK_MANEUVER,
    TASK_RESPONSE,
    TASK_RETRIEVAL,
    build_private_sequence_frame,
    build_variant_feature_frames,
    derive_private_proxy_task_entries,
    load_aligned_private_records,
    merge_task_features,
)
from chronaris.evaluation.dingxin.pipelines.leakage_audit import (
    audit_label_feature_overlap,
    write_label_feature_overlap_audit,
)
from chronaris.evaluation.dingxin.pipelines.leakage_safe_ablation import (
    _derived_features,
    _filter_feature_frame,
    _forbidden_families,
    _label_source_fields,
)
from chronaris.evaluation.dingxin.pipelines.feature_utils import cosine_similarity_numpy
from chronaris.pipelines.torch_runtime import resolve_torch_device_name, seed_torch

REPO_ROOT = Path(__file__).resolve().parents[5]
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/runs"
DEFAULT_REPORT_ROOT = "docs/artifacts/runs"
EVIDENCE_ROLE = "private_real_dual_stream"
DATASET_ID = "private_feature_export"

MODEL_ORDER = (
    "chronaris_full",
    "mult",
    "contiformer",
    "naive_time_sync",
    "classical_baseline",
)
DEEP_MODEL_NAMES = {
    "mult",
    "contiformer",
    "chronaris_v2_task_heads",
    "v2_no_vehicle_aux_head",
    "v2_no_vehicle_aux",
    "v2_no_residual_t2_head",
    "v2_no_residual_t2",
    "v2_no_contrastive_t3_loss",
    "chronaris_v3_stream_role",
    "chronaris_v3_stream_role_fusion",
    "v3_stream_role",
    "v3_stream_role_fusion",
    "v3_stream_role_adaptive",
    "v3_no_role_gate",
    "v3_fixed_causal_lag",
    "v3_force_private_causal",
    "v3_context_adapter_only",
}
FEATURE_MODEL_SOURCES = {
    "chronaris_full": ("chronaris_opt", "full_safe"),
    "naive_time_sync": ("naive_sync", "dual_projection"),
    "classical_baseline": ("f_full", "dual_projection"),
}

TASK_DISPLAY_NAMES = {
    TASK_MANEUVER: "分类任务",
    TASK_RESPONSE: "回归任务",
    TASK_RETRIEVAL: "检索任务",
}

METRIC_DISPLAY_NAMES = {
    "macro_f1": "macro-F1",
    "balanced_accuracy": "balanced accuracy",
    "rmse": "RMSE",
    "mae": "MAE",
    "nrmse": "NRMSE",
    "top1": "Top-1",
    "top3": "Top-3",
    "top5": "Top-5",
    "mrr": "MRR",
}

MODEL_DISPLAY_NAMES = {
    "chronaris_full": "Chronaris完整模型",
    "mult": "MulT",
    "contiformer": "ContiFormer",
    "naive_time_sync": "朴素时间同步基线",
    "classical_baseline": "传统特征基线",
    "p37_t3_info_nce_temp0p05_hardw2": "检索任务：InfoNCE低温候选",
    "p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10": "分类任务：焦点损失+门控候选",
    "p37_t1_focal_gamma1_gate0p65_ls0p05": "分类任务：轻门控候选",
}


def _is_p37_private_model(model_name: str) -> bool:
    normalized = str(model_name).strip().lower()
    return normalized.startswith("p37_t1_") or normalized.startswith("p37_t3_")


def _private_deep_model_applies_to_task(model_name: str, task_type: str) -> bool:
    normalized = str(model_name).strip().lower()
    if normalized.startswith("p37_t1_"):
        return task_type == "classification"
    if normalized.startswith("p37_t3_"):
        return task_type == "retrieval"
    return True


def _p37_t1_config(model_name: str) -> dict[str, float | bool]:
    normalized = str(model_name).strip().lower()
    if not normalized.startswith("p37_t1_"):
        return {}
    gamma = 2.0 if "gamma2" in normalized else 1.0 if "gamma1" in normalized else 0.0
    label_smoothing = 0.1 if "ls0p10" in normalized else 0.05 if "ls0p05" in normalized else 0.0
    target_gate = 0.85 if "gate0p85" in normalized else 0.75 if "gate0p75" in normalized else 0.65 if "gate0p65" in normalized else 0.55 if "gate0p55" in normalized else 0.65
    collapse_margin = 0.10 if "collapse0p10" in normalized else 0.05
    return {
        "focal_gamma": gamma,
        "label_smoothing": label_smoothing,
        "target_vehicle_contribution": target_gate,
        "collapse_margin": collapse_margin,
        "gate_regularization_weight": 0.05,
        "class_balanced": True,
    }


def _p37_t3_config(model_name: str) -> dict[str, float | str]:
    normalized = str(model_name).strip().lower()
    if not normalized.startswith("p37_t3_"):
        return {"temperature": 0.1, "hard_negative_weight": 1.0, "margin": 0.0, "loss_variant": "in_batch_info_nce"}
    temperature = 0.05 if "temp0p05" in normalized else 0.2 if "temp0p20" in normalized else 0.1
    hard_weight = 2.0 if "hardw2" in normalized else 1.0
    margin = 0.2 if "margin0p20" in normalized else 0.1 if "margin0p10" in normalized else 0.0
    loss_variant = "supervised_contrastive_margin" if "supcon" in normalized or margin > 0.0 else "info_nce"
    return {
        "temperature": temperature,
        "hard_negative_weight": hard_weight,
        "margin": margin,
        "loss_variant": loss_variant,
    }


def _class_balanced_weights(labels: np.ndarray, *, device: torch.device) -> torch.Tensor:
    if labels.size == 0:
        raise ValueError("cannot compute class weights from an empty train fold.")
    class_count = max(int(np.max(labels)) + 1, len(CLASS_LABEL_TO_ID))
    counts = np.bincount(labels.astype(int), minlength=class_count).astype(np.float32)
    weights = np.zeros_like(counts, dtype=np.float32)
    present = counts > 0
    weights[present] = float(counts[present].sum()) / (float(present.sum()) * counts[present])
    if present.any():
        weights[~present] = 0.0
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


@dataclass(frozen=True, slots=True)
class StageIPrivateThirdPartyComparisonConfig:
    run_id: str
    e_run_manifest_path: str
    f_run_manifest_path: str
    output_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    models: tuple[str, ...] = MODEL_ORDER
    seeds: tuple[int, ...] = (17, 29, 43, 71, 97)
    split_strategy: tuple[str, ...] = ("leave_one_view_out", "leave_one_sortie_out")
    epochs: int = 20
    screen_epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    num_heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
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
    checkpoint_policy: str = "last"


@dataclass(frozen=True, slots=True)
class StageIPrivateThirdPartyComparisonRunResult:
    run_id: str
    artifact_root: str
    summary_path: str
    evidence_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


def run_task_eval_private_thirdparty_comparison(
    config: StageIPrivateThirdPartyComparisonConfig,
) -> StageIPrivateThirdPartyComparisonRunResult:
    run_root = _resolve_path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="task_eval_private_thirdparty_comparison",
        logger=LOGGER,
        initial_progress={
            "artifact_root": str(run_root),
            "models": list(config.models),
            "seeds": list(config.seeds),
            "split_strategy": list(config.split_strategy),
            "evidence_role": EVIDENCE_ROLE,
            "resume": config.resume,
            "skip_completed": config.skip_completed,
            "gpuopt_enabled": True,
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
            report_path = _resolve_path(config.report_root) / f"task-eval-private-thirdparty-comparison-{config.run_id}.md"
            payload["evidence_manifest_path"] = str(manifest_path)
            payload["report_path"] = str(report_path)
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return StageIPrivateThirdPartyComparisonRunResult(
                run_id=config.run_id,
                artifact_root=str(run_root),
                summary_path=str(partial_path),
                evidence_manifest_path=str(manifest_path),
                report_path=str(report_path),
                summary=payload,
            )


def _run_observed(
    *,
    config: StageIPrivateThirdPartyComparisonConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageIPrivateThirdPartyComparisonRunResult:
    runtime_device = resolve_torch_device_name(config.device)
    if config.require_cuda and runtime_device != "cuda":
        raise RuntimeError(f"P30 requires CUDA for completed runs; resolved {runtime_device}.")
    records = load_aligned_private_records(
        e_run_manifest_path=config.e_run_manifest_path,
        f_run_manifest_path=config.f_run_manifest_path,
    )
    task_payload = derive_private_proxy_task_entries(records)
    variant_frames, diagnostics = build_variant_feature_frames(
        records,
        enable_optimized_chronaris=True,
        target_variant_name="chronaris_opt",
        lag_window_points=3,
        residual_mode="raw_window_stats",
    )
    progress.update(
        "sources_loaded",
        sample_count=int(len(records)),
        task_entry_count=int(len(task_payload["entries"])),
        view_count=int(records["view_id"].nunique()),
        sortie_count=int(records["sortie_id"].nunique()),
    )
    task_manifest_path = run_root / "task_manifest.jsonl"
    dump_task_eval_private_task_entries(task_payload["entries"], path=task_manifest_path)
    schema_path, dataset_summary_path = _write_private_sequence_contracts(
        run_root=run_root,
        records=records,
        task_payload=task_payload,
        config=config,
    )
    split_manifest = _build_split_manifest(records, task_payload["by_task"], config.split_strategy)
    split_manifest_path = run_root / "split_manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_json_path, audit_csv_path = _write_audit(
        run_root=run_root,
        task_payload=task_payload,
        variant_frames=variant_frames,
    )
    config_path = run_root / "private_thirdparty_config.json"
    config_alias_path = run_root / "private_third_party_config.json"
    config_payload = {
        "run_id": config.run_id,
        "runtime_device": runtime_device,
        "gpuopt_enabled": True,
        "gpuopt_config": _gpuopt_config(config),
        "config": asdict(config),
    }
    config_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_alias_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fold_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for task_name, entries in task_payload["by_task"].items():
        task_type = entries[0].task_type if entries else None
        for model_name in config.models:
            progress.update("model_task_start", task_name=task_name, model_name=model_name)
            if model_name in DEEP_MODEL_NAMES or _is_p37_private_model(model_name):
                task_fold_rows, task_curve_rows, task_predictions = _run_deep_task(
                    config=config,
                    run_root=run_root,
                    records=records,
                    task_entries=entries,
                    task_name=str(task_name),
                    task_type=str(task_type),
                    model_name=model_name,
                    runtime_device=runtime_device,
                )
            else:
                task_fold_rows, task_curve_rows, task_predictions = _run_feature_task(
                    config=config,
                    records=records,
                    task_entries=entries,
                    task_name=str(task_name),
                    task_type=str(task_type),
                    model_name=model_name,
                    variant_frames=variant_frames,
                )
            fold_rows.extend(task_fold_rows)
            training_rows.extend(task_curve_rows)
            prediction_rows.extend(task_predictions)
            _write_private_partial_tables(run_root, fold_rows, training_rows)
            progress.update("model_task_done", task_name=task_name, model_name=model_name)

    fold_frame = pd.DataFrame(fold_rows)
    training_frame = pd.DataFrame(training_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    seed_metrics = _aggregate_seed_metrics(fold_frame)
    long_frame = _aggregate_long_metrics(seed_metrics)
    wide_frame = _build_wide(long_frame)
    improvement = _build_improvement_summary(long_frame)
    leaderboard = _build_candidate_leaderboard(seed_metrics)

    fold_path = run_root / "fold_metrics.csv"
    seed_path = run_root / "seed_metrics.csv"
    curves_path = run_root / "training_curves.csv"
    predictions_path = run_root / "fold_predictions.csv"
    long_path = run_root / "model_comparison_long.csv"
    wide_path = run_root / "model_comparison_wide.csv"
    private_long_path = run_root / "private_third_party_long.csv"
    private_wide_path = run_root / "private_third_party_wide.csv"
    private_leaderboard_path = run_root / "private_third_party_leaderboard.csv"
    improvement_path = run_root / "improvement_summary.csv"
    leaderboard_path = run_root / "candidate_leaderboard.csv"
    fold_frame.to_csv(fold_path, index=False)
    seed_metrics.to_csv(seed_path, index=False)
    training_frame.to_csv(curves_path, index=False)
    predictions_frame.to_csv(predictions_path, index=False)
    long_frame.to_csv(long_path, index=False)
    wide_frame.to_csv(wide_path, index=False)
    long_frame.to_csv(private_long_path, index=False)
    wide_frame.to_csv(private_wide_path, index=False)
    improvement.to_csv(improvement_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)
    leaderboard.to_csv(private_leaderboard_path, index=False)

    figure_paths = _render_figures(run_root, long_frame, improvement, fold_frame, predictions_frame, training_frame)
    gpu_perf_summary = _gpu_perf_summary(runtime_device, config, training_frame=training_frame)
    gpu_perf_path = run_root / "gpu_perf_summary.json"
    gpu_perf_batches_path = run_root / "gpu_perf_batches.csv"
    gpu_perf_fold_path = run_root / "gpu_perf_fold_summary.csv"
    optimization_summary_path = run_root / "optimization_summary.json"
    gpu_perf_path.write_text(json.dumps(gpu_perf_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(gpu_perf_summary.get("batch_rows", [])).to_csv(gpu_perf_batches_path, index=False)
    pd.DataFrame(gpu_perf_summary.get("fold_rows", [])).to_csv(gpu_perf_fold_path, index=False)
    optimization_summary_path.write_text(
        json.dumps(
            {
                "run_id": config.run_id,
                "device": runtime_device,
                "batch_size": config.batch_size,
                "amp_mode": gpu_perf_summary.get("amp_mode"),
                "tensor_cache_mode": gpu_perf_summary.get("tensor_cache_mode"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status = "completed" if not long_frame.empty else "partial"
    summary_path = run_root / "private_thirdparty_summary.json"
    summary_alias_path = run_root / "private_third_party_summary.json"
    manifest_path = run_root / "evidence_manifest.json"
    report_path = _resolve_path(config.report_root) / f"task-eval-private-thirdparty-comparison-{config.run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    resume_command_path = run_root / "resume_command.txt"
    resume_command_path.write_text(_resume_command(config) + "\n", encoding="utf-8")
    summary = {
        "run_id": config.run_id,
        "status": status,
        "generated_at_utc": _utc_now(),
        "runtime_device": runtime_device,
        "artifact_root": str(run_root),
        "dataset_id": DATASET_ID,
        "evidence_role": EVIDENCE_ROLE,
        "data_role": "private_real_dual_stream",
        "sample_facts": {
            "sample_count": int(len(records)),
            "view_count": int(records["view_id"].nunique()),
            "sortie_count": int(records["sortie_id"].nunique()),
        },
        "private_thirdparty_config_json": str(config_path),
        "private_sequence_schema_json": str(schema_path),
        "private_sequence_dataset_summary_json": str(dataset_summary_path),
        "task_manifest_jsonl": str(task_manifest_path),
        "split_manifest_json": str(split_manifest_path),
        "label_feature_overlap_audit_json": str(audit_json_path),
        "label_feature_overlap_audit_csv": str(audit_csv_path),
        "model_comparison_long_csv": str(long_path),
        "model_comparison_wide_csv": str(wide_path),
        "improvement_summary_csv": str(improvement_path),
        "fold_metrics_csv": str(fold_path),
        "seed_metrics_csv": str(seed_path),
        "training_curves_csv": str(curves_path),
        "candidate_leaderboard_csv": str(leaderboard_path),
        "fold_predictions_csv": str(predictions_path),
        "gpu_perf_summary_json": str(gpu_perf_path),
        "gpu_perf_batches_csv": str(gpu_perf_batches_path),
        "gpu_perf_fold_summary_csv": str(gpu_perf_fold_path),
        "optimization_summary_json": str(optimization_summary_path),
        "figure_paths": figure_paths,
        "evidence_manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "run_log_path": str(run_root / "run.log"),
        "progress_path": str(run_root / "progress.json"),
        "completed_fold_count": _completed_fold_count(fold_frame),
        "expected_fold_count": _expected_fold_count(fold_frame),
        "diagnostics": diagnostics,
        "private_third_party_config_json": str(config_alias_path),
        "private_third_party_long_csv": str(private_long_path),
        "private_third_party_wide_csv": str(private_wide_path),
        "private_third_party_leaderboard_csv": str(private_leaderboard_path),
        "predictions_csv": str(predictions_path),
        "resume_command_txt": str(resume_command_path),
        "gpuopt_enabled": True,
        "gpuopt_config": _gpuopt_config(config),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    summary_alias_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    manifest = {
        **summary,
        "summary_path": str(summary_path),
        "protocol": {
            "splits": list(config.split_strategy),
            "seeds": list(config.seeds),
            "label_policy": "labels unchanged; test fold statistics excluded from normalization and target transforms",
            "candidate_policy": "retrieval task: same sortie cross-pilot candidates",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    if status != "completed":
        (run_root / "partial_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
    report_path.write_text(_render_report(summary, long_frame, improvement) + "\n", encoding="utf-8")
    return StageIPrivateThirdPartyComparisonRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        evidence_manifest_path=str(manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def _write_private_sequence_contracts(
    *,
    run_root: Path,
    records: pd.DataFrame,
    task_payload: Mapping[str, object],
    config: StageIPrivateThirdPartyComparisonConfig,
) -> tuple[Path, Path]:
    first = records.iloc[0]
    physiology = np.asarray(first["f_view"].physiology_reference_projection[int(first["f_index"])], dtype=np.float32)
    vehicle = np.asarray(first["f_view"].vehicle_reference_projection[int(first["f_index"])], dtype=np.float32)
    schema = {
        "dataset_id": DATASET_ID,
        "evidence_role": EVIDENCE_ROLE,
        "first_stream_role": "physiology",
        "second_stream_role": "real_vehicle_timeseries",
        "second_stream_is_real_vehicle": True,
        "split_strategy": list(config.split_strategy),
        "sample_count": int(len(records)),
        "view_count": int(records["view_id"].nunique()),
        "sortie_count": int(records["sortie_id"].nunique()),
        "modalities": {
            "physiology": {
                "feature_dim": int(physiology.shape[-1]),
                "sequence_length": int(physiology.shape[0]),
            },
            "vehicle": {
                "feature_dim": int(vehicle.shape[-1]),
                "sequence_length": int(vehicle.shape[0]),
                "role": "real_vehicle_timeseries",
            },
        },
    }
    summary = {
        "dataset_id": DATASET_ID,
        "evidence_role": EVIDENCE_ROLE,
        "sample_count": int(len(records)),
        "view_count": int(records["view_id"].nunique()),
        "sortie_count": int(records["sortie_id"].nunique()),
        "task_summary": task_payload["summary"],
    }
    schema_path = run_root / "private_sequence_schema.json"
    summary_path = run_root / "private_sequence_dataset_summary.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return schema_path, summary_path


def _write_audit(
    *,
    run_root: Path,
    task_payload: Mapping[str, object],
    variant_frames: Mapping[str, pd.DataFrame],
) -> tuple[Path, Path]:
    audits = []
    deep_features = tuple(
        [f"physiology_projection_{index:02d}" for index in range(16)]
        + [f"real_vehicle_timeseries_projection_{index:02d}" for index in range(16)]
    )
    for task_name, entries in task_payload["by_task"].items():
        audits.append(
            audit_label_feature_overlap(
                task_name=str(task_name),
                label_source_fields=_label_source_fields(str(task_name), task_payload["summary"]),
                input_feature_fields=deep_features,
                derived_input_features=_derived_features(str(task_name)),
                forbidden_feature_families=_forbidden_families(str(task_name)),
                leakage_safe=True,
            )
        )
        for model_name, (source_variant, feature_family) in FEATURE_MODEL_SOURCES.items():
            del model_name
            safe_frame = _filter_feature_frame(
                variant_frames[source_variant],
                feature_family=feature_family,
            )
            feature_fields = _feature_names(safe_frame)
            audits.append(
                audit_label_feature_overlap(
                    task_name=str(task_name),
                    label_source_fields=_label_source_fields(str(task_name), task_payload["summary"]),
                    input_feature_fields=feature_fields,
                    derived_input_features=_derived_features(str(task_name)),
                    forbidden_feature_families=_forbidden_families(str(task_name)),
                    leakage_safe=True,
                )
            )
    return write_label_feature_overlap_audit(audits, output_root=run_root)


def _feature_names(frame: pd.DataFrame) -> tuple[str, ...]:
    if frame.empty or "feature_values" not in frame:
        return ()
    names: set[str] = set()
    for values in frame["feature_values"].head(20):
        if isinstance(values, Mapping):
            names.update(str(key) for key in values)
    return tuple(sorted(names))


def _build_split_manifest(
    records: pd.DataFrame,
    by_task: Mapping[str, Sequence[object]],
    strategies: Sequence[str],
) -> dict[str, object]:
    tasks = {}
    for task_name, entries in by_task.items():
        valid = [entry for entry in entries if entry.label_value is not None]
        tasks[str(task_name)] = {
            "valid_entry_count": len(valid),
            "strategies": {
                strategy: _split_rows(valid, records, strategy)
                for strategy in strategies
            },
        }
    return {
        "dataset_id": DATASET_ID,
        "evidence_role": EVIDENCE_ROLE,
        "strategies": list(strategies),
        "tasks": tasks,
    }


def _split_rows(entries: Sequence[object], records: pd.DataFrame, strategy: str) -> list[dict[str, object]]:
    record_by_sample = records.set_index("sample_id", drop=False)
    groups = {}
    for entry in entries:
        record = record_by_sample.loc[entry.sample_id]
        group = record["view_id"] if strategy == "leave_one_view_out" else record["sortie_id"]
        groups.setdefault(str(group), []).append(entry.sample_id)
    rows = []
    all_count = len(entries)
    for fold_index, (group, sample_ids) in enumerate(sorted(groups.items()), start=1):
        rows.append(
            {
                "fold_index": fold_index,
                "split_strategy": strategy,
                "test_group": group,
                "train_count": all_count - len(sample_ids),
                "test_count": len(sample_ids),
                "fold_skipped_reason": None if all_count > len(sample_ids) else "no_train_samples",
            }
        )
    return rows


def _run_feature_task(
    *,
    config: StageIPrivateThirdPartyComparisonConfig,
    records: pd.DataFrame,
    task_entries: Sequence[object],
    task_name: str,
    task_type: str,
    model_name: str,
    variant_frames: Mapping[str, pd.DataFrame],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    source_variant, feature_family = FEATURE_MODEL_SOURCES[model_name]
    safe_frame = _filter_feature_frame(variant_frames[source_variant], feature_family=feature_family)
    merged = merge_task_features(task_entries, safe_frame, task_type=task_type)
    if merged.empty:
        return [], [], []
    if task_type == "retrieval":
        return _run_feature_retrieval(config, merged, task_name, model_name)
    return _run_feature_supervised(config, records, merged, task_name, task_type, model_name)


def _run_feature_supervised(
    config: StageIPrivateThirdPartyComparisonConfig,
    records: pd.DataFrame,
    frame: pd.DataFrame,
    task_name: str,
    task_type: str,
    model_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    record_by_sample = records.set_index("sample_id", drop=False)
    frame = frame.copy()
    frame["sortie_group"] = [record_by_sample.loc[sample_id]["sortie_id"] for sample_id in frame["sample_id"]]
    feature_columns = [column for column in frame.columns if column.startswith("feat__")]
    if not feature_columns:
        return _skipped_feature_rows(config, frame, task_name, model_name, "no_safe_feature_columns"), [], []
    fold_rows = []
    predictions = []
    for strategy in config.split_strategy:
        group_column = "split_group" if strategy == "leave_one_view_out" else "sortie_group"
        for seed in config.seeds:
            for fold_index, group in enumerate(sorted(frame[group_column].astype(str).unique()), start=1):
                train = frame[frame[group_column].astype(str) != group]
                test = frame[frame[group_column].astype(str) == group]
                if train.empty or test.empty:
                    fold_rows.append(_skipped_fold(task_name, model_name, strategy, seed, fold_index, group, "empty_train_or_test"))
                    continue
                if task_type == "classification" and train["y_label"].nunique() < 2:
                    fold_rows.append(_skipped_fold(task_name, model_name, strategy, seed, fold_index, group, "single_train_class"))
                    continue
                estimator = (
                    LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
                    if task_type == "classification"
                    else Ridge(alpha=1.0, random_state=seed)
                )
                pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
                pipeline.fit(train[feature_columns].to_numpy(dtype=float), train["y_label"].to_numpy())
                pred = pipeline.predict(test[feature_columns].to_numpy(dtype=float))
                metrics = _supervised_metrics(test["y_label"].to_numpy(), pred, task_type)
                fold_rows.append(
                    _fold_metric_row(task_name, model_name, strategy, seed, fold_index, group, len(test), metrics)
                )
                predictions.extend(
                    _prediction_rows(
                        test,
                        task_name=task_name,
                        model_name=model_name,
                        split_strategy=strategy,
                        seed=seed,
                        fold_index=fold_index,
                        y_pred=pred,
                    )
                )
    return fold_rows, [], predictions


def _skipped_feature_rows(
    config: StageIPrivateThirdPartyComparisonConfig,
    frame: pd.DataFrame,
    task_name: str,
    model_name: str,
    reason: str,
) -> list[dict[str, object]]:
    rows = []
    for strategy in config.split_strategy:
        group_column = "split_group"
        if strategy == "leave_one_sortie_out" and "sortie_id" in frame:
            group_column = "sortie_id"
        for seed in config.seeds:
            groups = sorted(frame[group_column].astype(str).unique()) if group_column in frame else ["all"]
            for fold_index, group in enumerate(groups, start=1):
                rows.append(_skipped_fold(task_name, model_name, strategy, seed, fold_index, group, reason))
    return rows


def _run_feature_retrieval(
    config: StageIPrivateThirdPartyComparisonConfig,
    frame: pd.DataFrame,
    task_name: str,
    model_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    fold_rows = []
    prediction_rows = []
    frame = frame.dropna(subset=["paired_sample_id"]).copy()
    for strategy in config.split_strategy:
        group_column = "split_group" if strategy == "leave_one_view_out" else "sortie_id"
        for seed in config.seeds:
            for fold_index, group in enumerate(sorted(frame[group_column].astype(str).unique()), start=1):
                test = frame[frame[group_column].astype(str) == group]
                metrics, preds = _retrieval_metrics(test, frame)
                fold_rows.append(
                    _fold_metric_row(task_name, model_name, strategy, seed, fold_index, group, len(preds), metrics)
                )
                for row in preds:
                    row.update({"model_name": model_name, "split_strategy": strategy, "seed": seed, "fold_index": fold_index})
                prediction_rows.extend(preds)
    return fold_rows, [], prediction_rows


def _run_deep_task(
    *,
    config: StageIPrivateThirdPartyComparisonConfig,
    run_root: Path,
    records: pd.DataFrame,
    task_entries: Sequence[object],
    task_name: str,
    task_type: str,
    model_name: str,
    runtime_device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if not _private_deep_model_applies_to_task(model_name, task_type):
        return [], [], []
    frame = _private_sequence_frame(task_entries, records, task_type=task_type)
    if frame.empty:
        return [], [], []
    if task_type == "retrieval":
        return _run_deep_retrieval(config, run_root, frame, task_name, model_name, runtime_device)
    return _run_deep_supervised(config, run_root, frame, task_name, task_type, model_name, runtime_device)


def _private_sequence_frame(task_entries: Sequence[object], records: pd.DataFrame, *, task_type: str) -> pd.DataFrame:
    if task_type == "retrieval":
        return _private_retrieval_sequence_frame(task_entries, records)
    base = build_private_sequence_frame(task_entries, records, task_type=task_type)
    if base.empty:
        return base
    extra = records.set_index("sample_id", drop=False)
    base = base.copy()
    base["sortie_id"] = [extra.loc[row.sample_id]["sortie_id"] for row in base.itertuples(index=False)]
    base["pilot_id"] = [extra.loc[row.sample_id]["pilot_id"] for row in base.itertuples(index=False)]
    base["view_id"] = [extra.loc[row.sample_id]["view_id"] for row in base.itertuples(index=False)]
    base["paired_sample_id"] = [getattr(row.task_entry, "paired_sample_id", None) for row in base.itertuples(index=False)]
    return base


def _private_retrieval_sequence_frame(task_entries: Sequence[object], records: pd.DataFrame) -> pd.DataFrame:
    record_by_sample = records.set_index("sample_id", drop=False)
    rows = []
    for entry in task_entries:
        if not getattr(entry, "paired_sample_id", None):
            continue
        if entry.sample_id not in record_by_sample.index:
            continue
        record = record_by_sample.loc[entry.sample_id]
        if record["f_index"] is None:
            continue
        f_view = record["f_view"]
        sample_index = int(record["f_index"])
        physiology = np.asarray(f_view.physiology_reference_projection[sample_index], dtype=np.float32)
        vehicle = np.asarray(f_view.vehicle_reference_projection[sample_index], dtype=np.float32)
        time_axis = np.asarray(f_view.reference_offsets_s[sample_index], dtype=np.float32)
        rows.append(
            {
                "sample_id": entry.sample_id,
                "split_group": entry.view_id,
                "task_entry": entry,
                "physiology_sequence": physiology,
                "vehicle_sequence": vehicle,
                "physiology_mask": np.isfinite(physiology).any(axis=1).astype(np.uint8),
                "vehicle_mask": np.isfinite(vehicle).any(axis=1).astype(np.uint8),
                "time_axis": time_axis,
                "y_label": 0.0,
                "sortie_id": record["sortie_id"],
                "pilot_id": record["pilot_id"],
                "view_id": record["view_id"],
                "window_index": getattr(entry, "window_index", None),
                "paired_sample_id": entry.paired_sample_id,
            }
        )
    return pd.DataFrame(rows)


def _run_deep_supervised(
    config: StageIPrivateThirdPartyComparisonConfig,
    run_root: Path,
    frame: pd.DataFrame,
    task_name: str,
    task_type: str,
    model_name: str,
    runtime_device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    arrays, masks, time_axis, labels = _sequence_arrays(frame)
    fold_rows = []
    curve_rows = []
    prediction_rows = []
    for strategy in config.split_strategy:
        group_column = "view_id" if strategy == "leave_one_view_out" else "sortie_id"
        for seed in config.seeds:
            for fold_index, group in enumerate(sorted(frame[group_column].astype(str).unique()), start=1):
                train_idx = frame.index[frame[group_column].astype(str) != group].to_numpy(dtype=int)
                test_idx = frame.index[frame[group_column].astype(str) == group].to_numpy(dtype=int)
                if len(train_idx) == 0 or len(test_idx) == 0:
                    fold_rows.append(_skipped_fold(task_name, model_name, strategy, seed, fold_index, group, "empty_train_or_test"))
                    continue
                if task_type == "classification" and len(set(labels[train_idx].astype(int))) < 2:
                    fold_rows.append(_skipped_fold(task_name, model_name, strategy, seed, fold_index, group, "single_train_class"))
                    continue
                model = _build_private_model(model_name, task_type, config, runtime_device, arrays)
                prepared = prepare_fold_tensors(
                    modality_arrays=arrays,
                    modality_masks=masks,
                    time_axis=time_axis,
                    ordered_modalities=("physiology", "vehicle"),
                    train_indices=train_idx,
                    targets=labels.astype(np.float32),
                    requested_mode=config.tensor_cache,
                    device=runtime_device,
                    max_cache_gb=config.max_cache_gb,
                    pin_memory=config.pin_memory,
                    non_blocking_copy=config.non_blocking_copy,
                )
                fold_curve = _train_supervised_model(
                    model=model,
                    prepared=prepared,
                    labels=labels,
                    train_indices=train_idx,
                    task_type=task_type,
                    config=config,
                    seed=seed + fold_index,
                    context={
                        "task_name": task_name,
                        "model_name": model_name,
                        "split_strategy": strategy,
                        "fold_index": fold_index,
                        "fold_group": group,
                    },
                )
                curve_rows.extend(fold_curve)
                pred = _predict_supervised(model, prepared, test_idx, labels, task_type, config=config)
                metrics = _supervised_metrics(labels[test_idx], pred, task_type)
                fold_rows.append(_fold_metric_row(task_name, model_name, strategy, seed, fold_index, group, len(test_idx), metrics))
                prediction_rows.extend(
                    _prediction_rows(
                        frame.iloc[test_idx],
                        task_name=task_name,
                        model_name=model_name,
                        split_strategy=strategy,
                        seed=seed,
                        fold_index=fold_index,
                        y_pred=pred,
                    )
                )
                _save_checkpoint(config, run_root, model, task_name, model_name, strategy, seed, fold_index)
    return fold_rows, curve_rows, prediction_rows


def _run_deep_retrieval(
    config: StageIPrivateThirdPartyComparisonConfig,
    run_root: Path,
    frame: pd.DataFrame,
    task_name: str,
    model_name: str,
    runtime_device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    frame = frame.dropna(subset=["paired_sample_id"]).reset_index(drop=True)
    arrays, masks, time_axis, _labels = _sequence_arrays(frame)
    fold_rows = []
    curve_rows = []
    prediction_rows = []
    for strategy in config.split_strategy:
        group_column = "view_id" if strategy == "leave_one_view_out" else "sortie_id"
        for seed in config.seeds:
            for fold_index, group in enumerate(sorted(frame[group_column].astype(str).unique()), start=1):
                train_idx = frame.index[frame[group_column].astype(str) != group].to_numpy(dtype=int)
                test_idx = frame.index[frame[group_column].astype(str) == group].to_numpy(dtype=int)
                if len(train_idx) == 0 or len(test_idx) == 0:
                    fold_rows.append(_skipped_fold(task_name, model_name, strategy, seed, fold_index, group, "empty_train_or_test"))
                    continue
                model = _build_private_model(model_name, "retrieval", config, runtime_device, arrays)
                prepared = prepare_fold_tensors(
                    modality_arrays=arrays,
                    modality_masks=masks,
                    time_axis=time_axis,
                    ordered_modalities=("physiology", "vehicle"),
                    train_indices=train_idx,
                    targets=np.zeros(len(frame), dtype=np.float32),
                    requested_mode=config.tensor_cache,
                    device=runtime_device,
                    max_cache_gb=config.max_cache_gb,
                    pin_memory=config.pin_memory,
                    non_blocking_copy=config.non_blocking_copy,
                )
                fold_curve = _train_retrieval_model(
                    model=model,
                    frame=frame,
                    prepared=prepared,
                    train_indices=train_idx,
                    config=config,
                    seed=seed + fold_index,
                    context={
                        "task_name": task_name,
                        "model_name": model_name,
                        "split_strategy": strategy,
                        "fold_index": fold_index,
                        "fold_group": group,
                    },
                )
                curve_rows.extend(fold_curve)
                embeddings = _embed_all(model, prepared, config=config)
                metrics, preds = _retrieval_metrics(frame.iloc[test_idx], frame, embeddings=embeddings)
                fold_rows.append(_fold_metric_row(task_name, model_name, strategy, seed, fold_index, group, len(preds), metrics))
                for row in preds:
                    row.update({"model_name": model_name, "split_strategy": strategy, "seed": seed, "fold_index": fold_index})
                prediction_rows.extend(preds)
                _save_checkpoint(config, run_root, model, task_name, model_name, strategy, seed, fold_index)
    return fold_rows, curve_rows, prediction_rows


def _sequence_arrays(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    arrays = {
        "physiology": np.stack(frame["physiology_sequence"].to_list(), axis=0).astype(np.float32),
        "vehicle": np.stack(frame["vehicle_sequence"].to_list(), axis=0).astype(np.float32),
    }
    masks = {
        "physiology": np.stack(frame["physiology_mask"].to_list(), axis=0).astype(np.uint8),
        "vehicle": np.stack(frame["vehicle_mask"].to_list(), axis=0).astype(np.uint8),
    }
    time_axis = np.stack(frame["time_axis"].to_list(), axis=0).astype(np.float32)
    labels = frame["y_label"].to_numpy(dtype=np.float32)
    return arrays, masks, time_axis, labels


def _build_private_model(
    model_name: str,
    task_type: str,
    config: StageIPrivateThirdPartyComparisonConfig,
    runtime_device: str,
    arrays: Mapping[str, np.ndarray],
):
    output_dim = 3 if task_type == "classification" else 1 if task_type == "regression" else None
    return build_task_eval_deep_model(
        model_name=model_name,
        ordered_modalities=("physiology", "vehicle"),
        modality_input_dims={
            "physiology": int(arrays["physiology"].shape[-1]),
            "vehicle": int(arrays["vehicle"].shape[-1]),
        },
        output_dim=output_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        layers=config.layers,
        dropout=config.dropout,
        dataset_id=DATASET_ID,
    ).to(runtime_device)


def _train_supervised_model(
    *,
    model,
    prepared,
    labels: np.ndarray,
    train_indices: np.ndarray,
    task_type: str,
    config: StageIPrivateThirdPartyComparisonConfig,
    seed: int,
    context: Mapping[str, object],
) -> list[dict[str, object]]:
    runtime_device = next(model.parameters()).device
    device_name = runtime_device.type
    seed_torch(seed, device=device_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model_name = str(context.get("model_name", ""))
    t1_config = _p37_t1_config(model_name)
    class_weights = (
        _class_balanced_weights(labels[train_indices].astype(int), device=runtime_device)
        if task_type == "classification" and t1_config.get("class_balanced")
        else None
    )
    criterion = (
        torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(t1_config.get("label_smoothing", 0.0)),
        )
        if task_type == "classification"
        else torch.nn.SmoothL1Loss()
    )
    amp = resolve_amp_runtime(requested_mode=config.amp, device=device_name, grad_scaler=config.grad_scaler)
    scaler = make_grad_scaler(amp, device=device_name)
    selected_batch_size, batch_attempts = _select_private_batch_size(
        config=config,
        model=model,
        criterion=criterion,
        prepared=prepared,
        train_indices=train_indices,
        task_type=task_type,
        amp=amp,
        device_name=device_name,
    )
    model, compile_info = _maybe_compile_private_model(
        config=config,
        model=model,
        prepared=prepared,
        indices=train_indices,
        device_name=device_name,
    )
    curve_rows = []
    started = time.monotonic()
    for epoch in range(config.epochs):
        losses = []
        gate_means = []
        batch_count = 0
        for batch_number, batch in enumerate(_iter_batches(train_indices, selected_batch_size, seed + epoch), start=1):
            batch_count = batch_number
            optimizer.zero_grad(set_to_none=True)
            modality_batch, mask_batch, time_batch, target_tensor = get_train_batch(
                prepared,
                batch,
                device=device_name,
            )
            with amp.autocast(device=device_name):
                output = model(modality_batch, time_axis=time_batch, modality_masks=mask_batch)
                if output.logits is None:
                    raise ValueError("private deep model returned no logits")
                if task_type == "classification":
                    target_long = target_tensor.to(dtype=torch.long)
                    if t1_config and float(t1_config.get("focal_gamma", 0.0)) > 0.0:
                        loss = class_balanced_focal_loss(
                            output.logits,
                            target_long,
                            class_weights=class_weights,
                            gamma=float(t1_config.get("focal_gamma", 2.0)),
                            label_smoothing=float(t1_config.get("label_smoothing", 0.0)),
                        )
                    else:
                        loss = criterion(output.logits, target_long)
                    aux = getattr(output, "auxiliary_outputs", None) or {}
                    gate = aux.get("gate") if isinstance(aux, Mapping) else None
                    if gate is not None and t1_config:
                        gate_means.append(float(gate.detach().float().mean().cpu().item()))
                        loss = loss + float(t1_config["gate_regularization_weight"]) * gate_regularization_loss(
                            gate,
                            target_vehicle_contribution=float(t1_config["target_vehicle_contribution"]),
                            collapse_margin=float(t1_config["collapse_margin"]),
                        )
                else:
                    loss = criterion(output.logits, target_tensor.to(dtype=torch.float32).view(-1, 1))
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            _sync_private(device_name)
            losses.append(float(loss.detach().cpu().item()))
            if batch_number == 1 or batch_number % max(config.batch_log_interval, 1) == 0:
                snapshot = gpu_runtime_snapshot() if config.profile_gpu else {}
                LOGGER.info(
                    "[heartbeat] run_id=%s stage=P30 task=%s model=%s split=%s fold=%s epoch=%d/%d batch=%d loss=%.6f gpu_mem_gb=%.3f elapsed_s=%.1f device=%s",
                    config.run_id,
                    context.get("task_name"),
                    context.get("model_name"),
                    context.get("split_strategy"),
                    context.get("fold_group"),
                    epoch + 1,
                    config.epochs,
                    batch_number,
                    losses[-1],
                    float(snapshot.get("gpu_memory_allocated_gb") or 0.0),
                    time.monotonic() - started,
                    device_name,
                )
        curve_rows.append(
            {
                **context,
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "batch_count": int(batch_count),
                "batch_size": int(selected_batch_size),
                "requested_batch_size": int(config.batch_size),
                "tensor_cache_mode": prepared.tensor_cache_mode,
                "requested_tensor_cache": config.tensor_cache,
                "cache_build_time_s": float(prepared.cache_build_time_s),
                "estimated_cache_gb": float(prepared.estimated_cache_gb),
                "actual_cache_gb": float(prepared.actual_cache_gb),
                "cache_fallback_reason": prepared.fallback_reason,
                "amp_mode": amp.resolved_mode,
                "amp_fallback_reason": amp.fallback_reason,
                "auto_batch_size": bool(config.auto_batch_size),
                "batch_size_attempts": json.dumps(batch_attempts),
                "torch_compile": config.torch_compile,
                **compile_info,
                "p37_focal_gamma": t1_config.get("focal_gamma") if t1_config else None,
                "p37_label_smoothing": t1_config.get("label_smoothing") if t1_config else None,
                "p37_target_vehicle_contribution": t1_config.get("target_vehicle_contribution") if t1_config else None,
                "p37_gate_mean": float(np.mean(gate_means)) if gate_means else None,
                "class_weight_source": "train_fold_only" if class_weights is not None else None,
            }
        )
    return curve_rows


def _train_retrieval_model(
    *,
    model,
    frame: pd.DataFrame,
    prepared,
    train_indices: np.ndarray,
    config: StageIPrivateThirdPartyComparisonConfig,
    seed: int,
    context: Mapping[str, object],
) -> list[dict[str, object]]:
    runtime_device = next(model.parameters()).device
    device_name = runtime_device.type
    seed_torch(seed, device=device_name)
    sample_to_index = {sample_id: index for index, sample_id in enumerate(frame["sample_id"].tolist())}
    pairs = []
    train_set = set(int(index) for index in train_indices.tolist())
    for index in train_indices:
        paired = frame.iloc[int(index)]["paired_sample_id"]
        paired_index = sample_to_index.get(paired)
        if paired_index is not None and paired_index in train_set:
            pairs.append((int(index), int(paired_index)))
    if not pairs:
        return [{**context, "epoch": 0, "train_loss": float("nan"), "skipped_reason": "no_train_pairs"}]
    model_name = str(context.get("model_name", ""))
    t3_config = _p37_t3_config(model_name)
    frame_rows = frame.to_dict(orient="records")
    train_pool = tuple(int(index) for index in train_indices.tolist())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp = resolve_amp_runtime(requested_mode=config.amp, device=device_name, grad_scaler=config.grad_scaler)
    scaler = make_grad_scaler(amp, device=device_name)
    selected_batch_size = int(config.batch_size)
    if config.auto_batch_size:
        selected_batch_size, _attempts = choose_auto_batch_size(
            candidates=config.batch_size_candidates,
            try_batch=lambda batch_size: None,
            fallback=config.batch_size,
        )
    model, compile_info = _maybe_compile_private_model(
        config=config,
        model=model,
        prepared=prepared,
        indices=train_indices,
        device_name=device_name,
    )
    curve_rows = []
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    for epoch in range(config.epochs):
        rng.shuffle(pairs)
        losses = []
        hard_negative_counts = []
        for start in range(0, len(pairs), max(selected_batch_size, 1)):
            batch_pairs = pairs[start : start + max(selected_batch_size, 1)]
            query_idx = np.asarray([left for left, _ in batch_pairs], dtype=int)
            positive_idx = np.asarray([right for _, right in batch_pairs], dtype=int)
            candidate_indices: list[int] = [int(index) for index in positive_idx.tolist()]
            hard_samples_by_anchor: dict[int, list[int]] = {}
            if _is_p37_private_model(model_name):
                for row_number, (anchor, positive) in enumerate(batch_pairs):
                    samples = stratified_hard_negative_samples(
                        frame_rows,
                        anchor_index=int(anchor),
                        positive_index=int(positive),
                        pool_indices=train_pool,
                        config=HardNegativeSamplerConfig(near_window_radius=2, max_per_kind=2),
                    )
                    hard_samples_by_anchor[row_number] = [sample.candidate_index for sample in samples]
                    candidate_indices.extend(sample.candidate_index for sample in samples)
            candidate_idx = np.asarray(list(dict.fromkeys(candidate_indices)), dtype=int)
            positive_positions = torch.as_tensor(
                [int(np.where(candidate_idx == positive)[0][0]) for positive in positive_idx],
                dtype=torch.long,
                device=runtime_device,
            )
            candidate_weights = torch.ones((len(query_idx), len(candidate_idx)), device=runtime_device)
            if _is_p37_private_model(model_name):
                hard_weight = float(t3_config["hard_negative_weight"])
                for row_number, hard_indices in hard_samples_by_anchor.items():
                    for candidate_index in hard_indices:
                        matches = np.where(candidate_idx == int(candidate_index))[0]
                        if len(matches):
                            candidate_weights[row_number, int(matches[0])] = hard_weight
                hard_negative_counts.append(sum(len(indices) for indices in hard_samples_by_anchor.values()))
            optimizer.zero_grad(set_to_none=True)
            q_modalities, q_masks, q_time, _ = get_train_batch(prepared, query_idx, device=device_name)
            p_modalities, p_masks, p_time, _ = get_train_batch(prepared, candidate_idx, device=device_name)
            with amp.autocast(device=device_name):
                query = model(q_modalities, time_axis=q_time, modality_masks=q_masks).pooled_embedding
                candidates = model(p_modalities, time_axis=p_time, modality_masks=p_masks).pooled_embedding
                query = torch.nn.functional.normalize(query, dim=-1)
                candidates = torch.nn.functional.normalize(candidates, dim=-1)
                loss = info_nce_loss(
                    query,
                    candidates,
                    positive_positions,
                    temperature=float(t3_config["temperature"]),
                    candidate_weights=candidate_weights,
                )
                if float(t3_config.get("margin", 0.0)) > 0.0:
                    scores = query @ candidates.T
                    loss = loss + supervised_contrastive_margin_loss(
                        scores,
                        positive_positions,
                        margin=float(t3_config["margin"]),
                        negative_weights=candidate_weights,
                    )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            _sync_private(device_name)
            losses.append(float(loss.detach().cpu().item()))
        LOGGER.info(
            "[heartbeat] stage=P30 task=%s model=%s split=%s fold=%s epoch=%d/%d loss=%.6f elapsed_s=%.1f",
            context.get("task_name"),
            context.get("model_name"),
            context.get("split_strategy"),
            context.get("fold_group"),
            epoch + 1,
            config.epochs,
            float(np.mean(losses)) if losses else float("nan"),
            time.monotonic() - started,
        )
        curve_rows.append(
            {
                **context,
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "batch_count": int(math.ceil(len(pairs) / max(selected_batch_size, 1))),
                "batch_size": int(selected_batch_size),
                "requested_batch_size": int(config.batch_size),
                "tensor_cache_mode": prepared.tensor_cache_mode,
                "requested_tensor_cache": config.tensor_cache,
                "cache_build_time_s": float(prepared.cache_build_time_s),
                "estimated_cache_gb": float(prepared.estimated_cache_gb),
                "actual_cache_gb": float(prepared.actual_cache_gb),
                "cache_fallback_reason": prepared.fallback_reason,
                "amp_mode": amp.resolved_mode,
                "amp_fallback_reason": amp.fallback_reason,
                "auto_batch_size": bool(config.auto_batch_size),
                "torch_compile": config.torch_compile,
                "p37_loss_variant": t3_config.get("loss_variant"),
                "p37_temperature": t3_config.get("temperature"),
                "p37_hard_negative_weight": t3_config.get("hard_negative_weight"),
                "p37_margin": t3_config.get("margin"),
                "p37_hard_negative_count": float(np.mean(hard_negative_counts)) if hard_negative_counts else 0.0,
                "candidate_pool_policy": "same_sortie_cross_pilot_eval; train uses metadata-only hard negatives",
                **compile_info,
            }
        )
    return curve_rows


def _predict_supervised(model, prepared, indices: np.ndarray, labels: np.ndarray, task_type: str, *, config) -> np.ndarray:
    device_name = next(model.parameters()).device.type
    amp = resolve_amp_runtime(
        requested_mode=config.amp if config.amp_eval else "off",
        device=device_name,
        grad_scaler=False,
    )
    logits = []
    model.eval()
    with torch.inference_mode():
        for modality_batch, mask_batch, time_batch, _target_tensor in iter_eval_batches(
            prepared,
            indices,
            batch_size=config.eval_batch_size or config.batch_size,
            device=device_name,
        ):
            with amp.autocast(device=device_name):
                output = model(modality_batch, time_axis=time_batch, modality_masks=mask_batch)
            logits.append(output.logits.detach().float().cpu())
    values = torch.cat(logits, dim=0).numpy() if logits else np.empty((0, 1), dtype=np.float32)
    if task_type == "classification":
        return np.nan_to_num(values, nan=0.0).argmax(axis=1)
    pred = np.nan_to_num(values.reshape(-1), nan=float(np.nanmean(labels)))
    return pred


def _embed_all(model, prepared, *, config) -> np.ndarray:
    indices = np.arange(prepared.time_tensor.shape[0], dtype=int)
    device_name = next(model.parameters()).device.type
    amp = resolve_amp_runtime(
        requested_mode=config.amp if config.amp_eval else "off",
        device=device_name,
        grad_scaler=False,
    )
    chunks = []
    model.eval()
    with torch.inference_mode():
        for modality_batch, mask_batch, time_batch, _target_tensor in iter_eval_batches(
            prepared,
            indices,
            batch_size=config.eval_batch_size or config.batch_size,
            device=device_name,
        ):
            with amp.autocast(device=device_name):
                output = model(modality_batch, time_axis=time_batch, modality_masks=mask_batch)
            chunks.append(output.pooled_embedding.detach().float().cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32) if chunks else np.empty((0, 1), dtype=np.float32)


def _select_private_batch_size(
    *,
    config: StageIPrivateThirdPartyComparisonConfig,
    model,
    criterion,
    prepared,
    train_indices: np.ndarray,
    task_type: str,
    amp,
    device_name: str,
) -> tuple[int, list[dict[str, object]]]:
    if not config.auto_batch_size or device_name != "cuda":
        return int(config.batch_size), [{"batch_size": int(config.batch_size), "status": "fixed"}]

    def _try(batch_size: int) -> None:
        batch = train_indices[: min(int(batch_size), len(train_indices))]
        if len(batch) == 0:
            return
        modalities, masks, time_values, target = get_train_batch(
            prepared,
            batch,
            device=device_name,
        )
        model.zero_grad(set_to_none=True)
        with amp.autocast(device=device_name):
            logits = model(modalities, time_axis=time_values, modality_masks=masks).logits
            if logits is None:
                raise ValueError("private deep model returned no logits during auto batch probing")
            loss = (
                criterion(logits, target.to(dtype=torch.long))
                if task_type == "classification"
                else criterion(logits, target.to(dtype=torch.float32).view(-1, 1))
            )
        loss.backward()
        model.zero_grad(set_to_none=True)
        _sync_private(device_name)

    return choose_auto_batch_size(
        candidates=config.batch_size_candidates,
        try_batch=_try,
        fallback=config.batch_size,
    )


def _maybe_compile_private_model(
    *,
    config: StageIPrivateThirdPartyComparisonConfig,
    model,
    prepared,
    indices: np.ndarray,
    device_name: str,
) -> tuple[object, dict[str, object]]:
    if (
        config.torch_compile == "off"
        or device_name != "cuda"
        or not hasattr(torch, "compile")
    ):
        return model, {
            "compile_mode": config.torch_compile,
            "compile_status": "off",
            "compile_warmup_time_s": 0.0,
            "compile_fallback_reason": None,
        }
    started = time.monotonic()
    try:
        compiled = torch.compile(
            model,
            mode=None if config.torch_compile == "default" else config.torch_compile,
        )
        sample = indices[: min(8, len(indices))]
        modalities, masks, time_values, _target = get_train_batch(
            prepared,
            sample,
            device=device_name,
        )
        model.eval()
        compiled.eval()
        with torch.inference_mode():
            eager = model(modalities, time_axis=time_values, modality_masks=masks).pooled_embedding
            compiled_out = compiled(modalities, time_axis=time_values, modality_masks=masks).pooled_embedding
        max_diff = float((eager - compiled_out).abs().max().detach().cpu().item())
        if not np.isfinite(max_diff) or max_diff > 1e-3:
            return model, {
                "compile_mode": config.torch_compile,
                "compile_status": "fallback",
                "compile_warmup_time_s": time.monotonic() - started,
                "compile_fallback_reason": f"compile_sanity_diff={max_diff}",
            }
        return compiled, {
            "compile_mode": config.torch_compile,
            "compile_status": "enabled",
            "compile_warmup_time_s": time.monotonic() - started,
            "compile_fallback_reason": None,
        }
    except Exception as exc:  # pragma: no cover - host/runtime dependent.
        return model, {
            "compile_mode": config.torch_compile,
            "compile_status": "fallback",
            "compile_warmup_time_s": time.monotonic() - started,
            "compile_fallback_reason": type(exc).__name__ + ":" + str(exc),
        }


def _sync_private(device_name: str) -> None:
    if device_name == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _retrieval_metrics(test_frame: pd.DataFrame, all_frame: pd.DataFrame, *, embeddings: np.ndarray | None = None) -> tuple[dict[str, float], list[dict[str, object]]]:
    sample_to_pos = {sample_id: index for index, sample_id in enumerate(all_frame["sample_id"].tolist())}
    rows = []
    for query in test_frame.itertuples(index=False):
        candidates = all_frame[
            (all_frame["sortie_id"] == query.sortie_id)
            & (all_frame["pilot_id"] != query.pilot_id)
        ].copy()
        if candidates.empty or query.paired_sample_id not in set(candidates["sample_id"]):
            continue
        if embeddings is None:
            query_vec = query.feature_vector
            candidate_matrix = np.stack(candidates["feature_vector"].to_list(), axis=0)
        else:
            query_vec = embeddings[sample_to_pos[query.sample_id]]
            candidate_matrix = np.stack([embeddings[sample_to_pos[sample_id]] for sample_id in candidates["sample_id"]], axis=0)
        similarities = cosine_similarity_numpy(np.asarray(query_vec, dtype=np.float32), np.asarray(candidate_matrix, dtype=np.float32))
        order = np.argsort(-similarities)
        ranked = candidates.iloc[order].reset_index(drop=True)
        rank = int(ranked.index[ranked["sample_id"] == query.paired_sample_id][0]) + 1
        rows.append(
            {
                "task_name": TASK_RETRIEVAL,
                "sample_id": query.sample_id,
                "paired_sample_id": query.paired_sample_id,
                "rank": rank,
                "top1_hit": int(rank <= 1),
                "top3_hit": int(rank <= 3),
                "top5_hit": int(rank <= 5),
                "reciprocal_rank": 1.0 / rank,
                "candidate_count": int(len(candidates)),
                "candidate_pool_policy": "same_sortie_cross_pilot",
            }
        )
    if not rows:
        return {"top1": float("nan"), "top3": float("nan"), "top5": float("nan"), "mrr": float("nan")}, []
    frame = pd.DataFrame(rows)
    return {
        "top1": float(frame["top1_hit"].mean()),
        "top3": float(frame["top3_hit"].mean()),
        "top5": float(frame["top5_hit"].mean()),
        "mrr": float(frame["reciprocal_rank"].mean()),
    }, rows


def _iter_batches(indices: np.ndarray, batch_size: int, seed: int) -> Sequence[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(indices)
    return tuple(order[start : start + max(batch_size, 1)] for start in range(0, len(order), max(batch_size, 1)))


def _supervised_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> dict[str, float]:
    if task_type == "classification":
        labels = sorted(set(np.asarray(y_true, dtype=int).tolist()) | set(np.asarray(y_pred, dtype=int).tolist()))
        return {
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nrmse": float(_rmse(y_true, y_pred) / max(np.nanstd(y_true), 1e-8)),
    }


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(float(mean_squared_error(y_true, y_pred))))


def _fold_metric_row(
    task_name: str,
    model_name: str,
    split_strategy: str,
    seed: int,
    fold_index: int,
    group: str,
    sample_count: int,
    metrics: Mapping[str, float],
) -> dict[str, object]:
    return {
        "task_name": task_name,
        "model_name": model_name,
        "split_strategy": split_strategy,
        "seed": int(seed),
        "fold_index": int(fold_index),
        "test_group": str(group),
        "sample_count": int(sample_count),
        "status": "completed",
        **metrics,
    }


def _skipped_fold(
    task_name: str,
    model_name: str,
    split_strategy: str,
    seed: int,
    fold_index: int,
    group: str,
    reason: str,
) -> dict[str, object]:
    return {
        "task_name": task_name,
        "model_name": model_name,
        "split_strategy": split_strategy,
        "seed": int(seed),
        "fold_index": int(fold_index),
        "test_group": str(group),
        "sample_count": 0,
        "status": "skipped",
        "fold_skipped_reason": reason,
    }


def _prediction_rows(
    frame: pd.DataFrame,
    *,
    task_name: str,
    model_name: str,
    split_strategy: str,
    seed: int,
    fold_index: int,
    y_pred: Sequence[float | int],
) -> list[dict[str, object]]:
    rows = []
    for item, pred in zip(frame.itertuples(index=False), y_pred, strict=True):
        rows.append(
            {
                "task_name": task_name,
                "model_name": model_name,
                "split_strategy": split_strategy,
                "seed": int(seed),
                "fold_index": int(fold_index),
                "sample_id": item.sample_id,
                "y_true": item.y_label,
                "y_pred": float(pred),
            }
        )
    return rows


def _aggregate_seed_metrics(fold_frame: pd.DataFrame) -> pd.DataFrame:
    if fold_frame.empty:
        return pd.DataFrame()
    metric_cols = ["macro_f1", "balanced_accuracy", "rmse", "mae", "nrmse", "top1", "top3", "top5", "mrr"]
    rows = []
    completed = fold_frame[fold_frame["status"] == "completed"].copy()
    for keys, subset in completed.groupby(["task_name", "model_name", "split_strategy", "seed"], sort=False):
        row = {
            "task_name": keys[0],
            "model_name": keys[1],
            "split_strategy": keys[2],
            "seed": int(keys[3]),
            "completed_fold_count": int(subset.shape[0]),
            "sample_count": int(subset["sample_count"].sum()),
        }
        for metric in metric_cols:
            if metric in subset:
                values = subset[metric].dropna().astype(float)
                if not values.empty:
                    row[metric] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_long_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    if seed_metrics.empty:
        return pd.DataFrame()
    rows = []
    metric_by_task = {
        TASK_MANEUVER: ("macro_f1", "balanced_accuracy"),
        TASK_RESPONSE: ("rmse", "mae", "nrmse"),
        TASK_RETRIEVAL: ("top1", "top3", "top5", "mrr"),
    }
    for keys, subset in seed_metrics.groupby(["task_name", "model_name", "split_strategy"], sort=False):
        for metric in metric_by_task.get(keys[0], ()):
            if metric not in subset:
                continue
            values = subset[metric].dropna().astype(float)
            if values.empty:
                continue
            rows.append(
                {
                    "task_name": keys[0],
                    "model_name": keys[1],
                    "split_strategy": keys[2],
                    "metric": metric,
                    "value_mean": float(values.mean()),
                    "value_std": float(values.std(ddof=0)),
                    "seed_count": int(values.shape[0]),
                    "completed_fold_count": int(subset["completed_fold_count"].sum()),
                    "sample_count": int(subset["sample_count"].sum()),
                    "higher_is_better": not _lower_is_better(metric),
                }
            )
    return pd.DataFrame(rows)


def _build_wide(long_frame: pd.DataFrame) -> pd.DataFrame:
    if long_frame.empty:
        return pd.DataFrame()
    return long_frame.pivot_table(
        index=["task_name", "split_strategy", "metric"],
        columns="model_name",
        values="value_mean",
        aggfunc="first",
    ).reset_index()


def _build_improvement_summary(long_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, subset in long_frame.groupby(["task_name", "split_strategy", "metric"], sort=False):
        chronaris = subset[subset["model_name"] == "chronaris_full"]
        if chronaris.empty:
            continue
        chronaris_value = float(chronaris["value_mean"].iloc[0])
        for row in subset.to_dict(orient="records"):
            if row["model_name"] == "chronaris_full":
                continue
            baseline = float(row["value_mean"])
            if _lower_is_better(str(keys[2])):
                delta = baseline - chronaris_value
            else:
                delta = chronaris_value - baseline
            rel = delta / abs(baseline) * 100.0 if abs(baseline) > 1e-12 else float("nan")
            rows.append(
                {
                    "task_name": keys[0],
                    "split_strategy": keys[1],
                    "metric": keys[2],
                    "baseline_model": row["model_name"],
                    "chronaris_value": chronaris_value,
                    "baseline_value": baseline,
                    "delta_abs": delta,
                    "delta_rel_pct": rel,
                    "positive_delta_means": "chronaris_better",
                }
            )
    return pd.DataFrame(rows)


def _build_candidate_leaderboard(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    return seed_metrics.sort_values(["task_name", "split_strategy", "model_name", "seed"]) if not seed_metrics.empty else seed_metrics


def _display_task_name(value: object) -> str:
    text = str(value)
    return TASK_DISPLAY_NAMES.get(text, text.replace("_", " "))


def _display_metric_name(value: object) -> str:
    text = str(value)
    return METRIC_DISPLAY_NAMES.get(text, text)


def _display_task_metric(task_name: object, metric: object) -> str:
    return f"{_display_task_name(task_name)} / {_display_metric_name(metric)}"


def _display_model_label(value: object) -> str:
    text = str(value)
    return MODEL_DISPLAY_NAMES.get(text, text.replace("_", " "))


def _with_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    if "task_name" in display:
        display["task"] = display["task_name"].map(_display_task_name)
    if "metric" in display:
        display["metric_display"] = display["metric"].map(_display_metric_name)
    if "model_name" in display:
        display["model"] = display["model_name"].map(_display_model_label)
    if "baseline_model" in display:
        display["baseline"] = display["baseline_model"].map(_display_model_label)
    return display


def _configure_plot_font() -> None:
    try:
        from matplotlib import font_manager
    except Exception:
        return
    for family in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.sans-serif"] = [family, *plt.rcParams.get("font.sans-serif", [])]
        plt.rcParams["axes.unicode_minus"] = False
        return


def _render_figures(
    run_root: Path,
    long_frame: pd.DataFrame,
    improvement: pd.DataFrame,
    fold_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    training_frame: pd.DataFrame,
) -> dict[str, str]:
    _configure_plot_font()
    paths = {
        "fig_private_third_party_task_leaderboard": str(run_root / "fig_private_third_party_task_leaderboard.png"),
        "fig_private_third_party_delta_heatmap": str(run_root / "fig_private_third_party_delta_heatmap.png"),
        "fig_private_third_party_fold_variance": str(run_root / "fig_private_third_party_fold_variance.png"),
        "fig_private_third_party_training_curves": str(run_root / "fig_private_third_party_training_curves.png"),
        "fig_private_third_party_retrieval_topk": str(run_root / "fig_private_third_party_retrieval_topk.png"),
        "fig_private_third_party_gpu_throughput": str(run_root / "fig_private_third_party_gpu_throughput.png"),
        "fig_private_thirdparty_t1_macro_f1": str(run_root / "fig_private_thirdparty_t1_macro_f1.png"),
        "fig_private_thirdparty_t2_rmse": str(run_root / "fig_private_thirdparty_t2_rmse.png"),
        "fig_private_thirdparty_t3_retrieval": str(run_root / "fig_private_thirdparty_t3_retrieval.png"),
        "fig_private_thirdparty_delta_heatmap": str(run_root / "fig_private_thirdparty_delta_heatmap.png"),
        "fig_private_thirdparty_fold_stability": str(run_root / "fig_private_thirdparty_fold_stability.png"),
        "fig_private_thirdparty_confusion_t1": str(run_root / "fig_private_thirdparty_confusion_t1.png"),
        "fig_private_thirdparty_t2_error_distribution": str(run_root / "fig_private_thirdparty_t2_error_distribution.png"),
        "fig_private_thirdparty_t3_retrieval_curve": str(run_root / "fig_private_thirdparty_t3_retrieval_curve.png"),
    }
    _private_task_leaderboard(long_frame, paths["fig_private_third_party_task_leaderboard"])
    _private_delta_heatmap(improvement, paths["fig_private_third_party_delta_heatmap"])
    _private_fold_stability(fold_frame, paths["fig_private_third_party_fold_variance"])
    _private_training_curves(training_frame, paths["fig_private_third_party_training_curves"])
    _retrieval_grouped(long_frame, paths["fig_private_third_party_retrieval_topk"])
    _private_gpu_throughput(training_frame, paths["fig_private_third_party_gpu_throughput"])
    _metric_bar(long_frame, TASK_MANEUVER, "macro_f1", paths["fig_private_thirdparty_t1_macro_f1"], "分类任务 macro-F1 模型榜", higher=True)
    _metric_bar(long_frame, TASK_RESPONSE, "rmse", paths["fig_private_thirdparty_t2_rmse"], "回归任务 RMSE 模型榜", higher=False)
    _retrieval_grouped(long_frame, paths["fig_private_thirdparty_t3_retrieval"])
    _private_delta_heatmap(improvement, paths["fig_private_thirdparty_delta_heatmap"])
    _private_fold_stability(fold_frame, paths["fig_private_thirdparty_fold_stability"])
    _t1_confusion(predictions, paths["fig_private_thirdparty_confusion_t1"])
    _t2_errors(predictions, paths["fig_private_thirdparty_t2_error_distribution"])
    _t3_curve(long_frame, paths["fig_private_thirdparty_t3_retrieval_curve"])
    return paths


def _private_task_leaderboard(frame: pd.DataFrame, path: str) -> None:
    subset = frame.copy() if not frame.empty else pd.DataFrame()
    priority = {
        (TASK_MANEUVER, "macro_f1"): "分类任务 macro-F1",
        (TASK_RESPONSE, "rmse"): "回归任务 RMSE",
        (TASK_RETRIEVAL, "top1"): "检索任务 Top-1",
        (TASK_RETRIEVAL, "mrr"): "检索任务 MRR",
    }
    subset = subset[
        [(row.task_name, row.metric) in priority for row in subset.itertuples(index=False)]
    ] if not subset.empty else subset
    fig, axis = plt.subplots(figsize=(11, 5.5))
    if subset.empty:
        axis.text(0.5, 0.5, "no leaderboard rows", ha="center", va="center")
        axis.axis("off")
    else:
        subset = subset.copy()
        subset["task_metric"] = [priority[(row.task_name, row.metric)] for row in subset.itertuples(index=False)]
        subset["model_label"] = subset["model_name"].map(_display_model_label)
        pivot = subset.pivot_table(index="model_label", columns="task_metric", values="value_mean", aggfunc="mean")
        normalized = pivot.copy()
        for column in normalized:
            values = normalized[column].astype(float)
            if "RMSE" in column:
                normalized[column] = values.min() / values.replace(0, np.nan)
            else:
                normalized[column] = values / max(float(values.max()), 1e-8)
        normalized = normalized.fillna(0.0)
        bottom = np.zeros(len(normalized))
        x = np.arange(len(normalized))
        for column in normalized.columns:
            axis.bar(x, normalized[column], bottom=bottom, label=column)
            bottom += normalized[column].to_numpy(dtype=float)
        label_stack_totals(axis, x, bottom)
        axis.set_xticks(x)
        axis.set_xticklabels(normalized.index, rotation=20, ha="right")
        axis.set_ylabel("归一化分数；越高越好")
        axis.set_title("鼎新真实数据第三方模型任务榜")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _private_training_curves(frame: pd.DataFrame, path: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    if frame.empty or "train_loss" not in frame:
        axis.text(0.5, 0.5, "no training curve rows", ha="center", va="center")
        axis.axis("off")
    else:
        subset = frame.copy()
        for model_name, model_rows in subset.groupby("model_name", sort=False):
            curve = model_rows.groupby("epoch", sort=True)["train_loss"].mean()
            axis.plot(curve.index, curve.values, marker="o", linewidth=1.4, label=_display_model_label(model_name))
        axis.set_xlabel("epoch")
        axis.set_ylabel("mean train loss")
        axis.set_title("鼎新真实数据第三方模型训练曲线")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _private_gpu_throughput(frame: pd.DataFrame, path: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 4.8))
    if frame.empty or "batch_size" not in frame:
        axis.text(0.5, 0.5, "no GPU runtime rows", ha="center", va="center")
        axis.axis("off")
    else:
        grouped = (
            frame.groupby("model_name", sort=False)
            .agg(
                selected_batch_size=("batch_size", "max"),
                cache_gb=("actual_cache_gb", "max") if "actual_cache_gb" in frame else ("batch_size", "count"),
            )
            .sort_values("selected_batch_size", ascending=False)
        )
        values = grouped["selected_batch_size"].astype(float)
        bars = axis.barh(np.arange(len(grouped)), values, color="#2f6f9f")
        label_horizontal_bars(axis, bars, values)
        axis.set_yticks(np.arange(len(grouped)))
        axis.set_yticklabels([_display_model_label(value) for value in grouped.index])
        axis.invert_yaxis()
        axis.set_xlabel("selected batch size")
        axis.set_title("鼎新真实数据训练批量选择")
        for index, row in enumerate(grouped.to_dict(orient="records")):
            axis.text(float(row["selected_batch_size"]), index, f" cache {float(row['cache_gb']):.2f} GB", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _metric_bar(frame: pd.DataFrame, task: str, metric: str, path: str, title: str, *, higher: bool) -> None:
    subset = frame[(frame["task_name"] == task) & (frame["metric"] == metric)].copy() if not frame.empty else pd.DataFrame()
    fig, axis = plt.subplots(figsize=(9, 4.8))
    if subset.empty:
        axis.text(0.5, 0.5, "no rows", ha="center", va="center")
        axis.axis("off")
    else:
        grouped = subset.groupby("model_name", sort=False)["value_mean"].mean().sort_values(ascending=not higher)
        values = grouped.values
        bars = axis.barh(np.arange(len(grouped)), values, color="#2f6f9f")
        label_horizontal_bars(axis, bars, values)
        axis.set_yticks(np.arange(len(grouped)))
        axis.set_yticklabels([_display_model_label(value) for value in grouped.index])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel(metric)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _retrieval_grouped(frame: pd.DataFrame, path: str) -> None:
    subset = frame[(frame["task_name"] == TASK_RETRIEVAL) & frame["metric"].isin(["top1", "top3", "top5", "mrr"])].copy() if not frame.empty else pd.DataFrame()
    fig, axis = plt.subplots(figsize=(10, 5))
    if subset.empty:
        axis.text(0.5, 0.5, "no retrieval rows", ha="center", va="center")
        axis.axis("off")
    else:
        pivot = subset.pivot_table(index="model_name", columns="metric", values="value_mean", aggfunc="mean").fillna(0.0)
        metrics = [metric for metric in ("top1", "top3", "top5", "mrr") if metric in pivot]
        x = np.arange(len(pivot))
        width = 0.8 / max(len(metrics), 1)
        for offset, metric in enumerate(metrics):
            values = pivot[metric].astype(float)
            bars = axis.bar(x + offset * width, values, width, label=metric)
            label_vertical_bars(axis, bars, values)
        axis.set_xticks(x + width * (len(metrics) - 1) / 2)
        axis.set_xticklabels([_display_model_label(value) for value in pivot.index], rotation=20, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_title("检索任务 Top-k / MRR 对比")
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _private_delta_heatmap(frame: pd.DataFrame, path: str) -> None:
    subset = frame.copy() if not frame.empty else pd.DataFrame()
    rows = (
        list(dict.fromkeys((_display_task_metric(row.task_name, row.metric) for row in subset.itertuples(index=False))))
        if not subset.empty
        else ["none"]
    )
    cols = list(dict.fromkeys((_display_model_label(value) for value in subset["baseline_model"].tolist()))) if not subset.empty else ["none"]
    data = np.zeros((len(rows), len(cols)), dtype=float)
    lookup = {}
    for row in subset.to_dict(orient="records"):
        lookup[(_display_task_metric(row["task_name"], row["metric"]), _display_model_label(row["baseline_model"]))] = row["delta_abs"]
    for i, row_label in enumerate(rows):
        for j, col in enumerate(cols):
            data[i, j] = float(lookup.get((row_label, col), np.nan))
    fig, axis = plt.subplots(figsize=(max(7, len(cols) * 1.2), max(4, len(rows) * 0.45)))
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    image = axis.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    axis.set_title("Chronaris 相对鼎新基线模型的提升")
    axis.set_xticks(np.arange(len(cols)))
    axis.set_xticklabels(cols, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(rows, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            axis.text(j, i, "" if np.isnan(data[i, j]) else f"{data[i, j]:+.4f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _private_fold_stability(frame: pd.DataFrame, path: str) -> None:
    subset = frame[frame["status"] == "completed"].copy() if not frame.empty else pd.DataFrame()
    fig, axis = plt.subplots(figsize=(11, 5))
    if subset.empty:
        axis.text(0.5, 0.5, "no fold rows", ha="center", va="center")
        axis.axis("off")
    else:
        metric = []
        for row in subset.to_dict(orient="records"):
            metric.append(row.get("macro_f1", row.get("rmse", row.get("top1", np.nan))))
        subset["metric_value"] = metric
        labels = list(dict.fromkeys(subset["model_name"]))
        data = [subset[subset["model_name"] == label]["metric_value"].dropna().astype(float).to_numpy() for label in labels]
        axis.boxplot(data, labels=[_display_model_label(label) for label in labels], showfliers=False)
        axis.tick_params(axis="x", rotation=20)
        axis.set_title("鼎新真实数据第三方模型折间稳定性")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _t1_confusion(predictions: pd.DataFrame, path: str) -> None:
    subset = predictions[(predictions["task_name"] == TASK_MANEUVER) & (predictions["model_name"] == "chronaris_full")].copy() if not predictions.empty else pd.DataFrame()
    fig, axis = plt.subplots(figsize=(4.5, 4))
    if subset.empty:
        axis.text(0.5, 0.5, "no classification-task predictions", ha="center", va="center")
        axis.axis("off")
    else:
        labels = [0, 1, 2]
        matrix = np.zeros((3, 3), dtype=int)
        for truth, pred in zip(subset["y_true"].astype(int), subset["y_pred"].astype(int), strict=False):
            if truth in labels and pred in labels:
                matrix[truth, pred] += 1
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(np.arange(3))
        axis.set_yticks(np.arange(3))
        axis.set_xticklabels(["low", "medium", "high"])
        axis.set_yticklabels(["low", "medium", "high"])
        axis.set_xlabel("predicted")
        axis.set_ylabel("true")
        axis.set_title("分类任务 Chronaris 混淆矩阵")
        for i in range(3):
            for j in range(3):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _t2_errors(predictions: pd.DataFrame, path: str) -> None:
    subset = predictions[predictions["task_name"] == TASK_RESPONSE].copy() if not predictions.empty else pd.DataFrame()
    fig, axis = plt.subplots(figsize=(9, 4.8))
    if subset.empty:
        axis.text(0.5, 0.5, "no regression-task predictions", ha="center", va="center")
        axis.axis("off")
    else:
        for model_name, group in subset.groupby("model_name", sort=False):
            errors = group["y_pred"].astype(float) - group["y_true"].astype(float)
            axis.hist(errors, bins=20, alpha=0.45, label=model_name)
        axis.set_title("回归任务预测误差分布")
        axis.set_xlabel("prediction - truth")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _t3_curve(frame: pd.DataFrame, path: str) -> None:
    subset = frame[(frame["task_name"] == TASK_RETRIEVAL) & frame["metric"].isin(["top1", "top3", "top5"])].copy() if not frame.empty else pd.DataFrame()
    fig, axis = plt.subplots(figsize=(8, 4.5))
    if subset.empty:
        axis.text(0.5, 0.5, "no retrieval-task rows", ha="center", va="center")
        axis.axis("off")
    else:
        order = {"top1": 1, "top3": 3, "top5": 5}
        for model_name, group in subset.groupby("model_name", sort=False):
            grouped = group.groupby("metric")["value_mean"].mean()
            xs = [order[metric] for metric in grouped.index]
            axis.plot(xs, grouped.values, marker="o", label=_display_model_label(model_name))
        axis.set_xticks([1, 3, 5])
        axis.set_ylim(0, 1.05)
        axis.set_xlabel("k")
        axis.set_ylabel("top-k accuracy")
        axis.set_title("检索任务 Top-k 曲线")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_checkpoint(config: StageIPrivateThirdPartyComparisonConfig, run_root: Path, model, task_name: str, model_name: str, split_strategy: str, seed: int, fold_index: int) -> None:
    policy = _checkpoint_policy(config.checkpoint_policy)
    if policy == "off":
        return
    checkpoint_root = run_root / "checkpoints"
    if policy == "epoch_and_fold":
        checkpoint_root = checkpoint_root / task_name / model_name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_name = (
        f"{split_strategy}_seed{seed}_fold{fold_index:03d}.pt"
        if policy == "epoch_and_fold"
        else "checkpoint_last.pt"
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "task_name": task_name, "model_name": model_name, "split_strategy": split_strategy, "seed": seed, "fold_index": fold_index},
        checkpoint_root / checkpoint_name,
    )


def _checkpoint_policy(value: str) -> str:
    normalized = str(value).strip().lower()
    choices = {"off", "last", "epoch_and_fold"}
    if normalized not in choices:
        raise ValueError(f"unsupported checkpoint_policy '{value}'; expected one of {tuple(sorted(choices))}")
    return normalized


def _gpu_perf_summary(
    runtime_device: str,
    config: StageIPrivateThirdPartyComparisonConfig,
    *,
    training_frame: pd.DataFrame | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "runtime_device": runtime_device,
        "best_batch_size": config.batch_size,
        "amp_mode": config.amp,
        "tensor_cache_mode": config.tensor_cache,
        "auto_batch_size": bool(config.auto_batch_size),
        "batch_size_candidates": [int(value) for value in config.batch_size_candidates],
        "torch_compile_mode": config.torch_compile,
        "num_workers": int(config.num_workers),
        "parallel_fold_prep": int(config.parallel_fold_prep),
        "oom_fallback_count": 0,
        "batch_rows": [],
        "fold_rows": [],
    }
    if training_frame is not None and not training_frame.empty:
        if "batch_size" in training_frame:
            payload["best_batch_size"] = int(pd.to_numeric(training_frame["batch_size"], errors="coerce").max())
        for field in ("tensor_cache_mode", "amp_mode", "compile_status"):
            if field in training_frame:
                payload[f"{field}_observed"] = sorted(set(training_frame[field].dropna().astype(str)))
        if "cache_fallback_reason" in training_frame:
            fallback = training_frame["cache_fallback_reason"].dropna().astype(str)
            payload["cache_fallback_count"] = int(fallback.shape[0])
            payload["cache_fallback_reasons"] = sorted(set(fallback))
        payload["batch_rows"] = _private_gpu_perf_rows(training_frame)
        payload["fold_rows"] = _private_gpu_perf_fold_rows(training_frame)
    payload["torch_version"] = torch.__version__
    payload["cuda_version"] = torch.version.cuda
    payload["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(0)
        payload["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        payload["max_memory_reserved_gb"] = torch.cuda.max_memory_reserved(0) / (1024 ** 3)
    return payload


def _private_gpu_perf_rows(training_frame: pd.DataFrame) -> list[dict[str, object]]:
    keep = [
        column
        for column in (
            "task_name",
            "model_name",
            "split_strategy",
            "fold_index",
            "fold_group",
            "epoch",
            "batch_count",
            "batch_size",
            "tensor_cache_mode",
            "actual_cache_gb",
            "amp_mode",
            "compile_status",
            "train_loss",
        )
        if column in training_frame
    ]
    return training_frame[keep].to_dict(orient="records") if keep else []


def _private_gpu_perf_fold_rows(training_frame: pd.DataFrame) -> list[dict[str, object]]:
    keys = [
        column
        for column in ("task_name", "model_name", "split_strategy", "fold_index", "fold_group")
        if column in training_frame
    ]
    if not keys or "batch_size" not in training_frame:
        return []
    agg: dict[str, tuple[str, str]] = {
        "batch_size": ("batch_size", "max"),
    }
    if "actual_cache_gb" in training_frame:
        agg["actual_cache_gb"] = ("actual_cache_gb", "max")
    if "batch_count" in training_frame:
        agg["batch_count"] = ("batch_count", "sum")
    return training_frame.groupby(keys, dropna=False).agg(**agg).reset_index().to_dict(orient="records")


def _gpuopt_config(config: StageIPrivateThirdPartyComparisonConfig) -> dict[str, object]:
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
        "checkpoint_policy": str(config.checkpoint_policy),
    }


def _resume_command(config: StageIPrivateThirdPartyComparisonConfig) -> str:
    parts = [
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python",
        "scripts/task_eval/private/run_private_thirdparty_comparison.py",
        "--run-id",
        config.run_id,
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
        "--checkpoint-policy",
        config.checkpoint_policy,
    ]
    if config.auto_batch_size:
        parts.append("--auto-batch-size")
    parts.extend(["--batch-size-candidates", *[str(value) for value in config.batch_size_candidates]])
    return " ".join(parts)


def _write_private_partial_tables(run_root: Path, fold_rows: Sequence[Mapping[str, object]], training_rows: Sequence[Mapping[str, object]]) -> None:
    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(run_root / "fold_metrics.partial.csv", index=False)
    if training_rows:
        pd.DataFrame(training_rows).to_csv(run_root / "training_curves.partial.csv", index=False)


def _completed_fold_count(fold_frame: pd.DataFrame) -> int:
    if fold_frame.empty:
        return 0
    return int(fold_frame[fold_frame["status"] == "completed"].shape[0])


def _expected_fold_count(fold_frame: pd.DataFrame) -> int:
    return int(fold_frame.shape[0]) if not fold_frame.empty else 0


def _lower_is_better(metric: str) -> bool:
    return metric in {"rmse", "mae", "nrmse"}


def _render_report(summary: Mapping[str, object], long_frame: pd.DataFrame, improvement: pd.DataFrame) -> str:
    display_long = _with_display_columns(long_frame)
    display_improvement = _with_display_columns(improvement)
    lines = [
        f"# task evaluation Dingxin Real-Data Third-party Comparison - {summary['run_id']}",
        "",
        "## Executive Summary",
        "",
        "On the Dingxin / feature export real dual-stream dataset, Chronaris is compared with MulT and ContiFormer under the same leakage-safe split manifest. "
        "The comparison uses real physiology and real vehicle time-series streams, with label-source fields and identity/time-position features excluded from model inputs. "
        "Across the classification, regression and retrieval component-diagnostic tasks, the report provides model-level leaderboard, fold-level stability and Chronaris-vs-third-party deltas.",
        "",
        "## Dataset and protocol",
        "",
        f"- evidence_role: `{summary['evidence_role']}`",
        f"- sample_facts: `{summary['sample_facts']}`",
        f"- split_manifest: `{summary['split_manifest_json']}`",
        "",
        "## Leakage-safe audit",
        "",
        f"- audit_json: `{summary['label_feature_overlap_audit_json']}`",
        f"- audit_csv: `{summary['label_feature_overlap_audit_csv']}`",
        "",
        "## Main leaderboard",
        "",
        _markdown_table(display_long.head(60), ["task", "split_strategy", "model", "metric_display", "value_mean", "value_std", "seed_count"]),
        "",
        "## Improvement over third-party baselines",
        "",
        _markdown_table(display_improvement.head(60), ["task", "split_strategy", "metric_display", "baseline", "chronaris_value", "baseline_value", "delta_abs", "delta_rel_pct"]),
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
            f"- config: `{summary['private_thirdparty_config_json']}`",
            f"- evidence_manifest: `{summary['evidence_manifest_path']}`",
            f"- run_log: `{summary['run_log_path']}`",
            f"- progress: `{summary['progress_path']}`",
            "",
            "## Midterm-ready wording",
            "",
            "鼎新 / feature export 分支在同一 leakage-safe 任务协议下比较 Chronaris、MulT 与 ContiFormer，量化真实生理流和真实航电流连续对齐场景中的模型适配性。分类任务、回归任务和检索任务均属于从现有鼎新数据派生的组件诊断任务，结果以 fold-level stability、mean/std 和 Chronaris-vs-baseline delta 展示。",
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
    config: StageIPrivateThirdPartyComparisonConfig,
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
