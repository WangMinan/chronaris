"""P34 task-aware head optimization runner."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from chronaris.models.alignment.contrastive import info_nce_loss, retrieval_metrics_from_scores
from chronaris.models.alignment.task_heads_v2 import (
    ContrastiveRetrievalProjectionHead,
    PhysiologyResponseResidualRegressionHead,
    VehicleDominantAuxiliaryClassificationHead,
)
from chronaris.modeling.common.gpu_runtime import gpu_runtime_snapshot
from chronaris.modeling.common.plot_labels import (
    label_vertical_bars,
)
from chronaris.evaluation.dingxin.pipelines.thirdparty_comparison import (
    StageIPrivateThirdPartyComparisonConfig,
    StageIPrivateThirdPartyComparisonRunResult,
    run_task_eval_private_thirdparty_comparison,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_P30_ROOT = (
    REPO_ROOT
    / "docs/artifacts/runs"
    / "20260702T-task-eval-private-thirdparty-comparison-gpuopt-r1"
)
DEFAULT_ARTIFACT_ROOT = "docs/artifacts/runs"
DEFAULT_REPORT_ROOT = "docs/artifacts/runs"
P34_MODELS = (
    "chronaris_v2_task_heads",
    "v2_no_vehicle_aux_head",
    "v2_no_residual_t2_head",
    "v2_no_contrastive_t3_loss",
)


@dataclass(frozen=True, slots=True)
class StageITaskHeadOptimizationConfig:
    run_id: str
    e_run_manifest_path: str
    f_run_manifest_path: str
    p30_root: str = str(DEFAULT_P30_ROOT)
    output_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    models: tuple[str, ...] = P34_MODELS
    seeds: tuple[int, ...] = (42,)
    split_strategies: tuple[str, ...] = ("leave_one_view_out",)
    epochs: int = 2
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    num_heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    device: str = "cuda"
    require_cuda: bool = True
    tensor_cache: str = "auto"
    max_cache_gb: float = 18.0
    auto_batch_size: bool = True
    batch_size_candidates: tuple[int, ...] = (2048, 1024, 512, 256, 128)
    amp: str = "bf16"
    torch_compile: str = "off"
    cpu_workers: int = 24
    parallel_fold_prep: int = 8
    heartbeat_seconds: float = 60.0
    batch_log_interval: int = 20
    profile_gpu: bool = True
    allow_partial: bool = True
    skip_completed: bool = True
    screen_only: bool = True
    checkpoint_policy: str = "last"


@dataclass(frozen=True, slots=True)
class StageITaskHeadOptimizationResult:
    run_id: str
    artifact_root: str
    summary_path: str
    evidence_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


def run_task_eval_task_head_optimization(
    config: StageITaskHeadOptimizationConfig,
) -> StageITaskHeadOptimizationResult:
    device = resolve_torch_device_name(config.device)
    if config.require_cuda and device != "cuda":
        raise RuntimeError(f"P34 requires CUDA for paper-facing runs; resolved {device}.")
    inner = _run_private_v2(config, device)
    run_root = Path(inner.artifact_root)
    p30_root = _resolve_path(config.p30_root)
    long_path = _alias(run_root / "model_comparison_long.csv", run_root / "task_head_metrics_long.csv")
    wide_path = _alias(run_root / "model_comparison_wide.csv", run_root / "task_head_metrics_wide.csv")
    _alias(run_root / "private_thirdparty_config.json", run_root / "task_heads_config.json")
    _write_json(run_root / "task_head_candidate_grid.json", {"models": list(config.models)})
    improvement_path = _write_improvement_vs_p30(
        p30_root=p30_root,
        p34_wide_path=wide_path,
        output_path=run_root / "improvement_vs_p30.csv",
    )
    _alias(run_root / "split_manifest.json", run_root / "split_manifest.json")
    t3_path = _write_t3_metrics(long_path, run_root / "t3_retrieval_metrics.csv")
    gate_path, residual_path, contrastive_path, cuda_diag = _write_cuda_head_diagnostics(run_root, device)
    figure_paths = _render_p34_figures(
        run_root=run_root,
        long_path=long_path,
        improvement_path=improvement_path,
        gate_path=gate_path,
        residual_path=residual_path,
        contrastive_path=contrastive_path,
    )
    gpu_perf_path = _ensure_gpu_perf_summary(run_root, config, cuda_diag)
    resume_path = run_root / "resume_command.txt"
    resume_path.write_text(_resume_command(config) + "\n", encoding="utf-8")
    status = _status_from_config(config, inner.summary)
    summary_path = run_root / "task_head_optimization_summary.json"
    report_path = _resolve_path(config.report_root) / f"task-eval-task-aware-heads-{config.run_id}.md"
    manifest_path = run_root / "evidence_manifest.json"
    summary = {
        "run_id": config.run_id,
        "status": status,
        "generated_at_utc": _utc_now(),
        "runtime_device": device,
        "artifact_root": str(run_root),
        "p30_reference_root": str(p30_root),
        "models": list(config.models),
        "seeds": list(config.seeds),
        "split_strategies": list(config.split_strategies),
        "task_head_metrics_long_csv": str(long_path),
        "task_head_metrics_wide_csv": str(wide_path),
        "improvement_vs_p30_csv": str(improvement_path),
        "fold_metrics_csv": str(run_root / "fold_metrics.csv"),
        "seed_metrics_csv": str(run_root / "seed_metrics.csv"),
        "training_curves_csv": str(run_root / "training_curves.csv"),
        "t3_retrieval_metrics_csv": str(t3_path),
        "gate_contribution_summary_csv": str(gate_path),
        "t2_residual_decomposition_csv": str(residual_path),
        "contrastive_diagnostics_csv": str(contrastive_path),
        "gpu_perf_summary_json": str(gpu_perf_path),
        "figure_paths": figure_paths,
        "evidence_manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "run_log_path": str(run_root / "run.log"),
        "progress_path": str(run_root / "progress.json"),
        "resume_command_txt": str(resume_path),
        "inner_private_comparison_summary": inner.summary_path,
        "protocol_boundary": (
            "This run writes new optimized Chronaris candidates only; confirmed baselines are read-only "
            "references. Labels, splits and same_sortie_cross_pilot retrieval policy are unchanged."
        ),
    }
    _write_json(summary_path, summary)
    manifest = {**summary, "summary_path": str(summary_path), "stage": "P34"}
    _write_json(manifest_path, manifest)
    if status != "completed":
        _write_json(run_root / "partial_summary.json", summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(summary) + "\n", encoding="utf-8")
    return StageITaskHeadOptimizationResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        evidence_manifest_path=str(manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def _run_private_v2(
    config: StageITaskHeadOptimizationConfig,
    device: str,
) -> StageIPrivateThirdPartyComparisonRunResult:
    return run_task_eval_private_thirdparty_comparison(
        StageIPrivateThirdPartyComparisonConfig(
            run_id=config.run_id,
            e_run_manifest_path=config.e_run_manifest_path,
            f_run_manifest_path=config.f_run_manifest_path,
            output_root=str(_resolve_path(config.output_root)),
            report_root=str(_resolve_path(config.report_root)),
            models=config.models,
            seeds=config.seeds,
            split_strategy=config.split_strategies,
            epochs=config.epochs,
            screen_epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            layers=config.layers,
            dropout=config.dropout,
            device=device,
            require_cuda=config.require_cuda,
            skip_completed=config.skip_completed,
            allow_partial=config.allow_partial,
            screen_only=config.screen_only,
            heartbeat_seconds=config.heartbeat_seconds,
            batch_log_interval=config.batch_log_interval,
            tensor_cache=config.tensor_cache,
            max_cache_gb=config.max_cache_gb,
            auto_batch_size=config.auto_batch_size,
            batch_size_candidates=config.batch_size_candidates,
            amp=config.amp,
            torch_compile=config.torch_compile,
            profile_gpu=config.profile_gpu,
            num_workers=config.cpu_workers,
            parallel_fold_prep=config.parallel_fold_prep,
            checkpoint_policy=config.checkpoint_policy,
        )
    )


def _write_improvement_vs_p30(*, p30_root: Path, p34_wide_path: Path, output_path: Path) -> Path:
    p30 = pd.read_csv(p30_root / "model_comparison_wide.csv")
    p34 = pd.read_csv(p34_wide_path)
    merged = p30.merge(p34, on=["task_name", "split_strategy", "metric"], how="outer", suffixes=("_p30", ""))
    rows = []
    for row in merged.to_dict(orient="records"):
        baseline = row.get("chronaris_full")
        optimized = row.get("chronaris_v2_task_heads")
        if pd.isna(baseline) or pd.isna(optimized):
            continue
        metric = str(row["metric"])
        delta = float(baseline) - float(optimized) if metric in {"rmse", "mae", "nrmse"} else float(optimized) - float(baseline)
        rows.append(
            {
                "task_name": row["task_name"],
                "split_strategy": row["split_strategy"],
                "metric": metric,
                "p30_chronaris_v1": float(baseline),
                "p34_chronaris_v2": float(optimized),
                "delta_abs_positive_is_better": delta,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _write_cuda_head_diagnostics(run_root: Path, device: str) -> tuple[Path, Path, Path, dict[str, object]]:
    torch.manual_seed(42)
    batch, steps, hidden = 8, 5, 16
    vehicle = torch.randn(batch, steps, hidden, device=device)
    physiology = torch.randn(batch, steps, hidden, device=device)
    fused = torch.randn(batch, steps, hidden * 3, device=device)
    t1 = VehicleDominantAuxiliaryClassificationHead(vehicle_dim=hidden, fused_dim=hidden * 3, output_dim=3).to(device)
    t2 = PhysiologyResponseResidualRegressionHead(
        physiology_dim=hidden,
        vehicle_dim=hidden,
        fused_dim=hidden * 3,
        output_dim=1,
    ).to(device)
    t3 = ContrastiveRetrievalProjectionHead(input_dim=hidden * 3, embedding_dim=hidden).to(device)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    t1_out = t1(vehicle_states=vehicle, fused_states=fused)
    t2_out = t2(physiology_states=physiology, vehicle_states=vehicle, fused_states=fused)
    embeddings = t3(fused)
    scores = embeddings @ embeddings.T
    positives = torch.arange(batch, device=device)
    loss = info_nce_loss(embeddings, embeddings, positives)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = max(time.perf_counter() - start, 1e-9)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("P34 contrastive CUDA smoke produced non-finite loss.")
    gate_path = run_root / "gate_contribution_summary.csv"
    residual_path = run_root / "t2_residual_decomposition.csv"
    contrastive_path = run_root / "contrastive_diagnostics.csv"
    pd.DataFrame([t1_out.contribution_summary().to_jsonable()]).to_csv(gate_path, index=False)
    pd.DataFrame([t2_out.decomposition_summary()]).to_csv(residual_path, index=False)
    metrics = retrieval_metrics_from_scores(scores.detach(), positives)
    metrics.update({"info_nce_loss": float(loss.detach().item()), "temperature": 0.07})
    pd.DataFrame([metrics]).to_csv(contrastive_path, index=False)
    cuda_diag = gpu_runtime_snapshot()
    cuda_diag.update(
        {
            "samples_per_sec": float(batch / elapsed_s),
            "batch_time_sec": float(elapsed_s),
            "copy_time_sec": None,
            "forward_time_sec": float(elapsed_s),
            "backward_time_sec": None,
            "step_time_sec": None,
            "timing_basis": "P34 CUDA task-head diagnostic forward smoke; not full training throughput",
        }
    )
    return gate_path, residual_path, contrastive_path, cuda_diag


def _write_t3_metrics(long_path: Path, output_path: Path) -> Path:
    frame = pd.read_csv(long_path)
    frame = frame[frame["task_name"].str.contains("T3", na=False)].copy()
    frame.to_csv(output_path, index=False)
    return output_path


def _render_p34_figures(
    *,
    run_root: Path,
    long_path: Path,
    improvement_path: Path,
    gate_path: Path,
    residual_path: Path,
    contrastive_path: Path,
) -> dict[str, str]:
    _configure_plot_font()
    long_frame = pd.read_csv(long_path)
    improvement = pd.read_csv(improvement_path) if improvement_path.exists() else pd.DataFrame()
    paths = {
        "fig_p34_t1_macro_f1_leaderboard": str(run_root / "fig_p34_t1_macro_f1_leaderboard.png"),
        "fig_p34_t2_rmse_leaderboard": str(run_root / "fig_p34_t2_rmse_leaderboard.png"),
        "fig_p34_t3_retrieval_leaderboard": str(run_root / "fig_p34_t3_retrieval_leaderboard.png"),
        "fig_p34_delta_vs_p30_heatmap": str(run_root / "fig_p34_delta_vs_p30_heatmap.png"),
        "fig_p34_t1_gate_contribution": str(run_root / "fig_p34_t1_gate_contribution.png"),
        "fig_p34_t2_residual_decomposition": str(run_root / "fig_p34_t2_residual_decomposition.png"),
        "fig_p34_t3_positive_negative_margin": str(run_root / "fig_p34_t3_positive_negative_margin.png"),
        "fig_p34_training_curves": str(run_root / "fig_p34_training_curves.png"),
    }
    _bar_metric(long_frame, "T1", "macro_f1", paths["fig_p34_t1_macro_f1_leaderboard"])
    _bar_metric(long_frame, "T2", "rmse", paths["fig_p34_t2_rmse_leaderboard"])
    _bar_metric(long_frame, "T3", "mrr", paths["fig_p34_t3_retrieval_leaderboard"])
    _delta_plot(improvement, paths["fig_p34_delta_vs_p30_heatmap"])
    _single_row_bar(pd.read_csv(gate_path), paths["fig_p34_t1_gate_contribution"])
    _single_row_bar(pd.read_csv(residual_path), paths["fig_p34_t2_residual_decomposition"])
    _single_row_bar(pd.read_csv(contrastive_path), paths["fig_p34_t3_positive_negative_margin"])
    curves = pd.read_csv(run_root / "training_curves.csv") if (run_root / "training_curves.csv").exists() else pd.DataFrame()
    _curve_plot(curves, paths["fig_p34_training_curves"])
    return paths


def _bar_metric(frame: pd.DataFrame, task_prefix: str, metric: str, path: str) -> None:
    subset = frame[(frame["task_name"].str.startswith(task_prefix)) & (frame["metric"] == metric)]
    fig, ax = plt.subplots(figsize=(8, 4))
    if subset.empty:
        ax.text(0.5, 0.5, "no rows", ha="center", va="center")
    else:
        labels = subset["model_name"] + "\n" + subset["split_strategy"]
        values = subset["value_mean"].astype(float)
        bars = ax.bar(labels, values)
        label_vertical_bars(ax, bars, values)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.set_ylabel(metric)
        ax.set_title(_task_prefix_label(task_prefix))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _delta_plot(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if frame.empty:
        ax.text(0.5, 0.5, "no overlap with confirmed reference", ha="center", va="center")
    else:
        labels = frame.apply(lambda row: f"{_task_name_label(row['task_name'])}\n{row['metric']}", axis=1)
        values = frame["delta_abs_positive_is_better"].astype(float)
        bars = ax.bar(labels, values)
        label_vertical_bars(ax, bars, values)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.set_ylabel("delta; positive means current candidate is better")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _single_row_bar(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    row = frame.iloc[0].to_dict() if not frame.empty else {}
    values = {key: float(value) for key, value in row.items() if isinstance(value, (int, float, np.floating))}
    if not values:
        ax.text(0.5, 0.5, "no diagnostics", ha="center", va="center")
    else:
        bars = ax.bar(list(values), list(values.values()))
        label_vertical_bars(ax, bars, list(values.values()))
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _curve_plot(frame: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if frame.empty or "train_loss" not in frame:
        ax.text(0.5, 0.5, "no curves", ha="center", va="center")
    else:
        for name, subset in frame.groupby("model_name", sort=False):
            ax.plot(subset["epoch"], subset["train_loss"], marker="o", label=name)
        ax.legend(fontsize=7)
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _task_prefix_label(task_prefix: str) -> str:
    return {
        "T1": "分类任务",
        "T2": "回归任务",
        "T3": "检索任务",
    }.get(task_prefix, task_prefix)


def _task_name_label(task_name: object) -> str:
    text = str(task_name)
    if text.startswith("T1"):
        return "分类任务"
    if text.startswith("T2"):
        return "回归任务"
    if text.startswith("T3"):
        return "检索任务"
    return text.replace("_", " ")


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


def _ensure_gpu_perf_summary(
    run_root: Path,
    config: StageITaskHeadOptimizationConfig,
    cuda_diag: Mapping[str, object],
) -> Path:
    path = run_root / "gpu_perf_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(
        {
            "run_id": config.run_id,
            "stage": "P34",
            "runtime_device": config.device,
            "gpu_name": cuda_diag.get("gpu_name"),
            "torch_version": cuda_diag.get("torch_version"),
            "cuda_version": cuda_diag.get("cuda_version"),
            "tensor_cache_mode": config.tensor_cache,
            "amp_mode": config.amp,
            "torch_compile_mode": config.torch_compile,
            "checkpoint_policy": config.checkpoint_policy,
            "best_batch_size": config.batch_size_candidates[0] if config.auto_batch_size else config.batch_size,
            "samples_per_sec": cuda_diag.get("samples_per_sec"),
            "batch_time_sec": cuda_diag.get("batch_time_sec"),
            "copy_time_sec": cuda_diag.get("copy_time_sec"),
            "forward_time_sec": cuda_diag.get("forward_time_sec"),
            "backward_time_sec": cuda_diag.get("backward_time_sec"),
            "step_time_sec": cuda_diag.get("step_time_sec"),
            "timing_basis": cuda_diag.get("timing_basis"),
        }
    )
    _write_json(path, payload)
    return path


def _render_report(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            f"# task evaluation Task-aware Heads - {summary['run_id']}",
            "",
            f"- status: `{summary['status']}`",
            f"- runtime_device: `{summary['runtime_device']}`",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- confirmed comparison reference: `{summary['p30_reference_root']}`",
            "- boundary: labels, splits and historical artifacts are unchanged.",
            "",
            "## Outputs",
            f"- metrics long: `{summary['task_head_metrics_long_csv']}`",
            f"- metrics wide: `{summary['task_head_metrics_wide_csv']}`",
            f"- improvement vs confirmed comparison reference: `{summary['improvement_vs_p30_csv']}`",
            f"- gate summary: `{summary['gate_contribution_summary_csv']}`",
            f"- residual decomposition: `{summary['t2_residual_decomposition_csv']}`",
            f"- contrastive diagnostics: `{summary['contrastive_diagnostics_csv']}`",
            f"- GPU summary: `{summary['gpu_perf_summary_json']}`",
            f"- resume command: `{summary['resume_command_txt']}`",
        ]
    )


def _status_from_config(
    config: StageITaskHeadOptimizationConfig,
    inner_summary: Mapping[str, object],
) -> str:
    if inner_summary.get("status") != "completed":
        return "partial"
    full_protocol = (
        set(config.split_strategies) == {"leave_one_view_out", "leave_one_sortie_out"}
        and set(config.seeds) >= {42, 17, 29}
        and config.epochs >= 20
        and not config.screen_only
    )
    return "completed" if full_protocol else "partial"


def _resume_command(config: StageITaskHeadOptimizationConfig) -> str:
    return (
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/task_eval/private/run_task_head_optimization.py "
        f"--resume --resume-run-id {config.run_id} --device cuda --require-cuda "
        f"--tensor-cache auto --auto-batch-size --amp bf16 --skip-completed "
        f"--checkpoint-policy {config.checkpoint_policy}"
    )


def _alias(source: Path, target: Path) -> Path:
    if source.resolve() == target.resolve():
        return target
    if source.exists():
        shutil.copy2(source, target)
    return target


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _resolve_path(path_like: str | Path) -> Path:
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
