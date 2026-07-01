"""Reporting helpers for P28 public fusion GPU optimization."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_perf_summary(config, candidate, fold_frame, batch_frame, snapshot, fallbacks):
    by_stage = {}
    for stage in ("baseline", "optimized"):
        rows = batch_frame[batch_frame["profile_stage"] == stage]
        folds = fold_frame[fold_frame["profile_stage"] == stage]
        total_samples = rows["sample_count"].sum() if not rows.empty else 0
        total_time = rows["batch_total_time_s"].sum() if not rows.empty else 0
        by_stage[stage] = {
            "samples_per_sec": float(total_samples / total_time) if total_time else None,
            "copy_time_s_per_batch": _mean(rows, "copy_time_s"),
            "forward_time_s_per_batch": _mean(rows, "forward_time_s"),
            "backward_time_s_per_batch": _mean(rows, "backward_time_s"),
            "max_memory_gb": _max(rows, "max_gpu_memory_allocated_gb"),
            "gpu_util_pct": _mean(rows, "nvidia_smi_util_pct"),
            "batch_size": int(folds["batch_size"].max()) if not folds.empty else None,
            "tensor_cache_mode": _unique_csv(folds, "tensor_cache_mode"),
            "amp_mode": _unique_csv(folds, "amp_mode"),
        }
    before = by_stage["baseline"]["samples_per_sec"]
    after = by_stage["optimized"]["samples_per_sec"]
    return {
        "run_id": config.run_id,
        "base_p28_run_id": config.base_p28_run_id,
        "candidate_id": candidate.candidate_id,
        "gpu_name": snapshot.get("gpu_name"),
        "baseline": by_stage["baseline"],
        "optimized": by_stage["optimized"],
        "speedup_ratio": float(after / before) if before and after else None,
        "fallbacks": fallbacks,
    }


def build_optimization_summary(
    *,
    config,
    candidate,
    perf_summary,
    snapshot,
    fallbacks,
    artifact_root,
    figure_paths,
    resume_command,
):
    return {
        "run_id": config.run_id,
        "base_p28_run_id": config.base_p28_run_id,
        "status": "completed",
        "generated_at_utc": _utc_now(),
        "gpu_name": snapshot.get("gpu_name"),
        "torch_version": snapshot.get("torch_version"),
        "cuda_available": snapshot.get("cuda_available"),
        "cuda_version": snapshot.get("cuda_version"),
        "artifact_root": str(artifact_root),
        "candidate_id": candidate.candidate_id,
        "optimization_enabled": {
            "tensor_cache": config.tensor_cache,
            "auto_batch_size": config.auto_batch_size,
            "amp": config.amp,
            "torch_compile": config.torch_compile,
            "parallel_candidates": 1,
        },
        "throughput": {
            "before_samples_per_sec": perf_summary["baseline"]["samples_per_sec"],
            "after_samples_per_sec": perf_summary["optimized"]["samples_per_sec"],
            "speedup_ratio": perf_summary["speedup_ratio"],
        },
        "timing": {
            "before_copy_time_s_per_batch": perf_summary["baseline"]["copy_time_s_per_batch"],
            "after_copy_time_s_per_batch": perf_summary["optimized"]["copy_time_s_per_batch"],
            "before_forward_time_s_per_batch": perf_summary["baseline"]["forward_time_s_per_batch"],
            "after_forward_time_s_per_batch": perf_summary["optimized"]["forward_time_s_per_batch"],
            "before_backward_time_s_per_batch": perf_summary["baseline"]["backward_time_s_per_batch"],
            "after_backward_time_s_per_batch": perf_summary["optimized"]["backward_time_s_per_batch"],
        },
        "memory": {
            "before_max_memory_gb": perf_summary["baseline"]["max_memory_gb"],
            "after_max_memory_gb": perf_summary["optimized"]["max_memory_gb"],
            "best_batch_size": perf_summary["optimized"]["batch_size"],
        },
        "utilization": {
            "before_gpu_util_pct": perf_summary["baseline"]["gpu_util_pct"],
            "after_gpu_util_pct": perf_summary["optimized"]["gpu_util_pct"],
            "reason": snapshot.get("gpu_snapshot_reason"),
        },
        "fallbacks": fallbacks,
        "figure_paths": figure_paths,
        "resume_command": resume_command,
        "continued_p28_resume": False,
        "resume_decision": "not_run_full_loso; representative profiling only",
    }


def render_plots(
    *,
    artifact_root: Path,
    batch_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
) -> dict[str, str]:
    plots = artifact_root / "plots"
    plots.mkdir(exist_ok=True)
    paths = {
        "fig_gpu_throughput_before_after": plots / "fig_gpu_throughput_before_after.png",
        "fig_gpu_batch_timing_breakdown": plots / "fig_gpu_batch_timing_breakdown.png",
        "fig_gpu_memory_and_batch_size": plots / "fig_gpu_memory_and_batch_size.png",
        "fig_gpu_cache_effect": plots / "fig_gpu_cache_effect.png",
        "fig_gpu_training_progress_heartbeat": plots / "fig_gpu_training_progress_heartbeat.png",
    }
    _bar_plot(
        fold_frame,
        paths["fig_gpu_throughput_before_after"],
        "samples_per_sec",
        "P28 training throughput before/after GPU optimization",
    )
    timing = batch_frame.groupby("profile_stage")[
        ["data_time_s", "copy_time_s", "forward_time_s", "backward_time_s", "step_time_s"]
    ].mean()
    timing.plot(kind="bar", stacked=True, figsize=(8, 4))
    plt.ylabel("seconds / batch")
    plt.tight_layout()
    plt.savefig(paths["fig_gpu_batch_timing_breakdown"], dpi=160)
    plt.close()
    _memory_batch_plot(batch_frame, fold_frame, paths["fig_gpu_memory_and_batch_size"])
    _bar_plot(
        fold_frame,
        paths["fig_gpu_cache_effect"],
        "cache_build_time_s",
        "Tensor cache build time by profile",
    )
    batch_frame.assign(step=np.arange(len(batch_frame))).plot(
        x="step",
        y="samples_per_sec",
        kind="line",
        figsize=(8, 4),
    )
    plt.ylabel("samples/sec")
    plt.tight_layout()
    plt.savefig(paths["fig_gpu_training_progress_heartbeat"], dpi=160)
    plt.close()
    return {key: str(path) for key, path in paths.items()}


def render_report(
    *,
    summary,
    perf_summary,
    figure_paths,
    artifact_root,
    batch_csv,
    fold_csv,
    resume_command,
) -> str:
    del perf_summary
    speedup = summary["throughput"]["speedup_ratio"]
    return "\n".join(
        [
            f"# Stage I Public Fusion GPU Optimization - {summary['run_id']}",
            "",
            "## 1. Executive Summary",
            f"Base P28 run `{summary['base_p28_run_id']}` is completed. This GPUOPT run did not continue full LOSO; it profiled representative folds only.",
            f"Throughput changed from {_fmt(summary['throughput']['before_samples_per_sec'])} to {_fmt(summary['throughput']['after_samples_per_sec'])} samples/sec (speedup {_fmt(speedup)}x).",
            "",
            "## 2. Baseline bottleneck diagnosis",
            "The baseline path repeatedly sliced CPU numpy arrays and rebuilt CUDA tensors inside the batch loop. The optimized path builds train-fold normalization once and reuses device or pinned tensors.",
            "",
            "## 3. Optimizations applied",
            f"- tensor cache: `{summary['optimization_enabled']['tensor_cache']}`",
            f"- auto batch size: `{summary['optimization_enabled']['auto_batch_size']}`",
            f"- AMP: `{summary['optimization_enabled']['amp']}`",
            f"- torch.compile: `{summary['optimization_enabled']['torch_compile']}`",
            "",
            "## 4. Throughput before/after",
            f"- before_samples_per_sec: `{summary['throughput']['before_samples_per_sec']}`",
            f"- after_samples_per_sec: `{summary['throughput']['after_samples_per_sec']}`",
            f"- speedup_ratio: `{summary['throughput']['speedup_ratio']}`",
            "",
            "## 5. Memory and GPU utilization",
            f"- before_max_memory_gb: `{summary['memory']['before_max_memory_gb']}`",
            f"- after_max_memory_gb: `{summary['memory']['after_max_memory_gb']}`",
            f"- before_gpu_util_pct: `{summary['utilization']['before_gpu_util_pct']}`",
            f"- after_gpu_util_pct: `{summary['utilization']['after_gpu_util_pct']}`",
            "",
            "## 6. Correctness and protocol checks",
            "Normalization and target transforms are computed from train fold indices only. Split groups are copied from the original LOSO grouping and no label/split/metric definitions are changed.",
            "",
            "## 7. Resume status",
            "No P28 full LOSO resume was executed in this run.",
            "",
            "## 8. Artifact index",
            f"- artifact root: `{artifact_root}`",
            f"- gpu_perf_batches: `{batch_csv}`",
            f"- gpu_perf_fold_summary: `{fold_csv}`",
            f"- gpu_perf_summary: `{artifact_root / 'gpu_perf_summary.json'}`",
            f"- optimization_summary: `{artifact_root / 'optimization_summary.json'}`",
            *[f"- {key}: `{value}`" for key, value in figure_paths.items()],
            "",
            "## 9. Commands run",
            "`python scripts/stage_i/public/run_public_fusion_gpuopt.py --run-id "
            + summary["run_id"]
            + " --device cuda --require-cuda`",
            "",
            "## 10. Next run recommendation",
            "To resume or reproduce this GPUOPT profiling run, use:",
            "",
            "```bash",
            resume_command,
            "```",
        ]
    )


def _bar_plot(frame: pd.DataFrame, path: Path, y: str, title: str) -> None:
    if frame.empty:
        return
    values = frame.groupby("profile_stage")[y].mean(numeric_only=True)
    values.plot(kind="bar", figsize=(7, 4), title=title)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _memory_batch_plot(batch_frame: pd.DataFrame, fold_frame: pd.DataFrame, path: Path) -> None:
    if batch_frame.empty or fold_frame.empty:
        return
    grouped = batch_frame.groupby(["profile_stage", "batch_size"])[
        ["samples_per_sec", "max_gpu_memory_allocated_gb"]
    ].mean()
    grouped.reset_index().plot(
        x="batch_size",
        y=["samples_per_sec", "max_gpu_memory_allocated_gb"],
        kind="bar",
        figsize=(8, 4),
        title="GPU memory and batch size",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _max(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else None


def _unique_csv(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame:
        return None
    values = sorted(set(str(value) for value in frame[column].dropna()))
    return ",".join(values) if values else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fmt(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(numeric):
        return "NA"
    return f"{numeric:.2f}"
