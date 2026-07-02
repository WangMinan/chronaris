"""P35 stream-role-aware fusion routing evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from chronaris.models.fusion import (
    RoleAwareCausalFusion,
    private_stream_metadata,
    public_stream_metadata,
)
from chronaris.pipelines.stage_i.common.gpu_runtime import gpu_runtime_snapshot
from chronaris.pipelines.stage_i.common.plot_labels import (
    label_stack_totals,
    label_vertical_bars,
)
from chronaris.pipelines.stage_i.private.thirdparty_comparison import (
    StageIPrivateThirdPartyComparisonConfig,
    run_stage_i_private_thirdparty_comparison,
)
from chronaris.pipelines.stage_i.public.fusion_ablation import (
    StageIPublicFusionAblationConfig,
    load_p28_prepared_roots,
    run_stage_i_public_fusion_ablation,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_P31_ROOT = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_i_public_fusion_ablation"
    / "20260702T-stage-i-public-fusion-ablation-gpuopt-r1"
)
DEFAULT_E_MANIFEST = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json"
)
DEFAULT_F_MANIFEST = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json"
)
DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_stream_role_fusion"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"


@dataclass(frozen=True, slots=True)
class StageIStreamRoleFusionEvalConfig:
    run_id: str
    p31_root: str = str(DEFAULT_P31_ROOT)
    p34_root: str | None = None
    e_run_manifest_path: str = str(DEFAULT_E_MANIFEST)
    f_run_manifest_path: str = str(DEFAULT_F_MANIFEST)
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    run_private_confirm: bool = False
    run_public_confirm: bool = False
    private_models: tuple[str, ...] = ()
    public_variants: tuple[str, ...] = ()
    public_datasets: tuple[str, ...] = ("nasa_csm", "uab_workload_dataset")
    confirm_epochs: int = 20
    confirm_max_folds: int | None = None
    seed: int = 42
    extra_confirm_seeds: tuple[int, ...] = ()
    resume: bool = False
    skip_completed: bool = True
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
    parallel_candidates: int = 1
    heartbeat_seconds: float = 60.0
    batch_log_interval: int = 20
    profile_gpu: bool = True
    checkpoint_policy: str = "last"
    allow_partial: bool = True


@dataclass(frozen=True, slots=True)
class StageIStreamRoleFusionEvalResult:
    run_id: str
    artifact_root: str
    summary_path: str
    evidence_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


def _run_private_v3_confirm(
    config: StageIStreamRoleFusionEvalConfig,
    run_root: Path,
):
    return run_stage_i_private_thirdparty_comparison(
        StageIPrivateThirdPartyComparisonConfig(
            run_id=f"{config.run_id}-private-v3",
            e_run_manifest_path=str(_resolve_path(config.e_run_manifest_path)),
            f_run_manifest_path=str(_resolve_path(config.f_run_manifest_path)),
            output_root=str(run_root / "private_v3_confirm"),
            report_root=str(run_root / "nested_reports"),
            models=_private_v3_models(config),
            seeds=(config.seed, *config.extra_confirm_seeds),
            split_strategy=("leave_one_view_out", "leave_one_sortie_out"),
            epochs=config.confirm_epochs,
            screen_epochs=config.confirm_epochs,
            device=config.device,
            require_cuda=config.require_cuda,
            resume=config.resume,
            skip_completed=config.skip_completed,
            allow_partial=config.allow_partial,
            confirm_only=True,
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


def _run_public_v3_confirm(
    config: StageIStreamRoleFusionEvalConfig,
    run_root: Path,
):
    prepared_roots = _load_public_prepared_roots(_resolve_path(config.p31_root))
    return run_stage_i_public_fusion_ablation(
        StageIPublicFusionAblationConfig(
            run_id=f"{config.run_id}-public-v3",
            dataset_prepared_roots=prepared_roots,
            artifact_root=str(run_root / "public_v3_confirm"),
            report_root=str(run_root / "nested_reports"),
            datasets=config.public_datasets,
            variants=_public_v3_variants(config),
            screen_epochs=config.confirm_epochs,
            confirm_epochs=config.confirm_epochs,
            screen_max_folds=config.confirm_max_folds,
            confirm_max_folds=config.confirm_max_folds,
            seed=config.seed,
            extra_confirm_seeds=config.extra_confirm_seeds,
            device=config.device,
            require_cuda=config.require_cuda,
            resume=config.resume,
            skip_completed=config.skip_completed,
            allow_partial=config.allow_partial,
            confirm_only=True,
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
            parallel_candidates=config.parallel_candidates,
            checkpoint_policy=config.checkpoint_policy,
            base_run_id=config.run_id,
        )
    )


def _private_v3_models(config: StageIStreamRoleFusionEvalConfig) -> tuple[str, ...]:
    if config.private_models:
        return tuple(config.private_models)
    return (
        "chronaris_v3_stream_role_fusion",
        "v3_no_role_gate",
        "v3_fixed_causal_lag",
    )


def _public_v3_variants(config: StageIStreamRoleFusionEvalConfig) -> tuple[str, ...]:
    if config.public_variants:
        return tuple(config.public_variants)
    return (
        "v3_stream_role",
        "v3_no_role_gate",
        "v3_force_private_causal",
        "v3_context_adapter_only",
    )


def _load_public_prepared_roots(p31_root: Path) -> dict[str, str]:
    config_path = p31_root / "public_ablation_config.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            roots = payload.get("config", {}).get("dataset_prepared_roots")
            if isinstance(roots, Mapping) and roots:
                prepared = {str(key): str(value) for key, value in roots.items()}
                if all(Path(value).exists() for value in prepared.values()):
                    return prepared
        except json.JSONDecodeError:
            pass
    return load_p28_prepared_roots()


def run_stage_i_stream_role_fusion_eval(
    config: StageIStreamRoleFusionEvalConfig,
) -> StageIStreamRoleFusionEvalResult:
    device = resolve_torch_device_name(config.device)
    if config.require_cuda and device != "cuda":
        raise RuntimeError(f"P35 requires CUDA for paper-facing runs; resolved {device}.")
    run_root = _resolve_path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    p31_root = _resolve_path(config.p31_root)
    p34_root = _resolve_path(config.p34_root) if config.p34_root else _latest_p34_root()
    _write_json(run_root / "stream_role_fusion_config.json", asdict(config))
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    route_start = time.perf_counter()
    gate_stats = _run_route_smoke(device)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    route_elapsed_s = max(time.perf_counter() - route_start, 1e-9)
    gate_path = run_root / "gate_statistics.csv"
    route_path = run_root / "route_decision_summary.csv"
    pd.DataFrame(gate_stats).to_csv(gate_path, index=False)
    pd.DataFrame(gate_stats).to_csv(route_path, index=False)
    route_manifest = {
        "private": private_stream_metadata().to_jsonable(),
        "nasa_csm": public_stream_metadata("nasa_csm").to_jsonable(),
        "uab_workload_dataset": public_stream_metadata("uab_workload_dataset").to_jsonable(),
        "boundary": "public context proxy is not real_vehicle; private Stage H second stream is real_vehicle",
    }
    _write_json(run_root / "route_manifest.json", route_manifest)
    private_confirm_root: Path | None = None
    public_confirm_root: Path | None = None
    private_confirm_summary: Mapping[str, object] | None = None
    public_confirm_summary: Mapping[str, object] | None = None
    if config.run_private_confirm:
        private_result = _run_private_v3_confirm(config, run_root)
        private_confirm_root = Path(private_result.artifact_root)
        private_confirm_summary = private_result.summary
    if config.run_public_confirm:
        public_result = _run_public_v3_confirm(config, run_root)
        public_confirm_root = Path(public_result.artifact_root)
        public_confirm_summary = public_result.summary
    private_metrics_path = _write_private_metrics(
        p34_root,
        run_root / "private_metrics.csv",
        p35_private_root=private_confirm_root,
    )
    public_metrics_path = _write_public_metrics(
        p31_root,
        run_root / "public_metrics.csv",
        p35_public_root=public_confirm_root,
    )
    comparison_vs_p31 = _write_public_metrics(
        p31_root,
        run_root / "comparison_vs_p31.csv",
        p35_public_root=public_confirm_root,
    )
    comparison_vs_p34 = _copy_or_empty(
        p34_root / "improvement_vs_p30.csv" if p34_root else None,
        run_root / "comparison_vs_p34.csv",
    )
    _write_combined_fold_metrics(
        gate_stats=gate_stats,
        private_root=private_confirm_root,
        public_root=public_confirm_root,
        output_path=run_root / "fold_metrics.csv",
    )
    _write_combined_training_curves(
        private_root=private_confirm_root,
        public_root=public_confirm_root,
        output_path=run_root / "training_curves.csv",
    )
    gpu_path = run_root / "gpu_perf_summary.json"
    _write_json(gpu_path, _gpu_summary(config, device, route_elapsed_s, route_sample_count=len(gate_stats) * 6))
    figure_paths = _render_figures(run_root, gate_path, public_metrics_path, private_metrics_path)
    resume_path = run_root / "resume_command.txt"
    resume_path.write_text(_resume_command(config) + "\n", encoding="utf-8")
    (run_root / "run.log").write_text(
        f"[heartbeat] run_id={config.run_id} stage=P35 device={device} route_smoke=completed\n",
        encoding="utf-8",
    )
    status = _status_from_confirm_summaries(
        run_private_confirm=config.run_private_confirm,
        run_public_confirm=config.run_public_confirm,
        private_summary=private_confirm_summary,
        public_summary=public_confirm_summary,
    )
    with (run_root / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(
            f"[finish] run_id={config.run_id} stage=P35 status={status} "
            f"private_confirm={config.run_private_confirm} public_confirm={config.run_public_confirm}\n"
        )
    _write_json(run_root / "progress.json", {"run_id": config.run_id, "stage": "P35", "status": status})
    summary_path = run_root / "stream_role_fusion_summary.json"
    report_path = _resolve_path(config.report_root) / f"stage-i-stream-role-aware-fusion-{config.run_id}.md"
    manifest_path = run_root / "evidence_manifest.json"
    summary = {
        "run_id": config.run_id,
        "status": status,
        "generated_at_utc": _utc_now(),
        "runtime_device": device,
        "artifact_root": str(run_root),
        "p31_reference_root": str(p31_root),
        "p34_reference_root": str(p34_root) if p34_root else None,
        "private_confirm_root": str(private_confirm_root) if private_confirm_root else None,
        "public_confirm_root": str(public_confirm_root) if public_confirm_root else None,
        "private_confirm_status": (
            str(private_confirm_summary.get("status"))
            if private_confirm_summary
            else "not_run"
        ),
        "public_confirm_status": (
            str(public_confirm_summary.get("status"))
            if public_confirm_summary
            else "not_run"
        ),
        "stream_role_fusion_config_json": str(run_root / "stream_role_fusion_config.json"),
        "route_manifest_json": str(run_root / "route_manifest.json"),
        "gate_statistics_csv": str(gate_path),
        "route_decision_summary_csv": str(route_path),
        "private_metrics_csv": str(private_metrics_path),
        "public_metrics_csv": str(public_metrics_path),
        "comparison_vs_p31_csv": str(comparison_vs_p31),
        "comparison_vs_p34_csv": str(comparison_vs_p34),
        "fold_metrics_csv": str(run_root / "fold_metrics.csv"),
        "training_curves_csv": str(run_root / "training_curves.csv"),
        "gpu_perf_summary_json": str(gpu_path),
        "figure_paths": figure_paths,
        "evidence_manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "run_log_path": str(run_root / "run.log"),
        "progress_path": str(run_root / "progress.json"),
        "resume_command_txt": str(resume_path),
        "protocol_boundary": _protocol_boundary(status),
    }
    _write_json(summary_path, summary)
    _write_json(manifest_path, {**summary, "summary_path": str(summary_path), "stage": "P35"})
    if status != "completed":
        _write_json(run_root / "partial_summary.json", summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(summary) + "\n", encoding="utf-8")
    return StageIStreamRoleFusionEvalResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        evidence_manifest_path=str(manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def _run_route_smoke(device: str) -> list[dict[str, object]]:
    torch.manual_seed(35)
    hidden = 16
    model = RoleAwareCausalFusion(hidden_dim=hidden).to(device)
    physiology = torch.randn(6, 8, hidden, device=device)
    second = torch.randn(6, 8, hidden, device=device)
    offsets = torch.arange(8, dtype=torch.float32, device=device).view(1, -1).repeat(6, 1)
    rows = []
    for dataset_id, metadata, variant in (
        ("private_stage_h", private_stream_metadata(), "chronaris_v3_stream_role"),
        ("nasa_csm", public_stream_metadata("nasa_csm"), "chronaris_v3_stream_role"),
        ("uab_workload_dataset", public_stream_metadata("uab_workload_dataset"), "chronaris_v3_stream_role"),
        ("nasa_csm", public_stream_metadata("nasa_csm"), "v3_force_private_causal"),
        ("uab_workload_dataset", public_stream_metadata("uab_workload_dataset"), "v3_context_adapter_only"),
    ):
        output = model(
            physiology_states=physiology,
            second_stream_states=second,
            physiology_offsets_s=offsets,
            second_stream_offsets_s=offsets,
            metadata=metadata,
            variant=variant,
        )
        row = {
            "dataset_id": dataset_id,
            "variant_id": variant,
            "fused_dim": int(output.fused_states.shape[-1]),
            **output.gate_summary(),
        }
        rows.append(row)
    torch.cuda.synchronize() if device == "cuda" and torch.cuda.is_available() else None
    return rows


def _write_public_metrics(
    p31_root: Path,
    output_path: Path,
    *,
    p35_public_root: Path | None = None,
) -> Path:
    frames: list[pd.DataFrame] = []
    source = p31_root / "ablation_summary.csv"
    if source.exists():
        frame = pd.read_csv(source)
        keep = frame[
            frame["variant_id"].isin(["full", "no_lag_window", "context_only"])
            & frame["metric"].isin(["combined_macro_f1", "mean_rmse", "subjective_mean_rmse"])
        ].copy()
        if keep.empty:
            keep = frame[frame["variant_id"].isin(["full", "no_lag_window", "context_only"])].copy()
        keep["source_stage"] = "P31_reference"
        keep["p35_status"] = "reference_only"
        frames.append(keep)
    if p35_public_root is not None and (p35_public_root / "ablation_summary.csv").exists():
        p35 = pd.read_csv(p35_public_root / "ablation_summary.csv")
        p35 = p35[
            p35["variant_id"].astype(str).str.startswith("v3_")
            & p35["metric"].isin(["combined_macro_f1", "mean_rmse", "subjective_mean_rmse"])
        ].copy()
        if not p35.empty:
            p35["source_stage"] = "P35_v3_confirm"
            p35["p35_status"] = "v3_confirm"
            frames.append(p35)
    output = pd.concat(frames, axis=0, ignore_index=True, sort=False) if frames else pd.DataFrame()
    output.to_csv(output_path, index=False)
    return output_path


def _write_private_metrics(
    p34_root: Path | None,
    output_path: Path,
    *,
    p35_private_root: Path | None = None,
) -> Path:
    frames: list[pd.DataFrame] = []
    if p34_root and (p34_root / "task_head_metrics_long.csv").exists():
        frame = pd.read_csv(p34_root / "task_head_metrics_long.csv")
    elif p34_root and (p34_root / "model_comparison_long.csv").exists():
        frame = pd.read_csv(p34_root / "model_comparison_long.csv")
    else:
        frame = pd.DataFrame()
    if not frame.empty:
        frame["source_stage"] = "P34_reference"
        frame["p35_status"] = "reference_only"
        frames.append(frame)
    if p35_private_root is not None and (p35_private_root / "model_comparison_long.csv").exists():
        p35 = pd.read_csv(p35_private_root / "model_comparison_long.csv")
        if not p35.empty:
            p35["source_stage"] = "P35_v3_confirm"
            p35["p35_status"] = "v3_confirm"
            frames.append(p35)
    output = pd.concat(frames, axis=0, ignore_index=True, sort=False) if frames else pd.DataFrame()
    output.to_csv(output_path, index=False)
    return output_path


def _copy_or_empty(source: Path | None, output_path: Path) -> Path:
    if source and source.exists():
        try:
            pd.read_csv(source).to_csv(output_path, index=False)
        except pd.errors.EmptyDataError:
            pd.DataFrame().to_csv(output_path, index=False)
    else:
        pd.DataFrame().to_csv(output_path, index=False)
    return output_path


def _write_combined_fold_metrics(
    *,
    gate_stats: list[dict[str, object]],
    private_root: Path | None,
    public_root: Path | None,
    output_path: Path,
) -> Path:
    frames = []
    route = pd.DataFrame(gate_stats)
    route["source_stage"] = "P35_route_smoke"
    route["status"] = "completed"
    frames.append(route)
    for source_stage, root in (
        ("P35_private_v3_confirm", private_root),
        ("P35_public_v3_confirm", public_root),
    ):
        if root is not None and (root / "fold_metrics.csv").exists():
            frame = pd.read_csv(root / "fold_metrics.csv")
            frame["source_stage"] = source_stage
            frames.append(frame)
    pd.concat(frames, axis=0, ignore_index=True, sort=False).to_csv(output_path, index=False)
    return output_path


def _write_combined_training_curves(
    *,
    private_root: Path | None,
    public_root: Path | None,
    output_path: Path,
) -> Path:
    frames = [
        pd.DataFrame(
            [{"stage": "P35", "epoch": 0, "train_loss": np.nan, "status": "route_smoke_only"}]
        )
    ]
    for source_stage, root in (
        ("P35_private_v3_confirm", private_root),
        ("P35_public_v3_confirm", public_root),
    ):
        if root is not None and (root / "training_curves.csv").exists():
            frame = pd.read_csv(root / "training_curves.csv")
            frame["source_stage"] = source_stage
            frames.append(frame)
    pd.concat(frames, axis=0, ignore_index=True, sort=False).to_csv(output_path, index=False)
    return output_path


def _status_from_confirm_summaries(
    *,
    run_private_confirm: bool,
    run_public_confirm: bool,
    private_summary: Mapping[str, object] | None,
    public_summary: Mapping[str, object] | None,
) -> str:
    if not run_private_confirm and not run_public_confirm:
        return "partial"
    statuses = []
    if run_private_confirm:
        statuses.append(str(private_summary.get("status")) if private_summary else "missing")
    if run_public_confirm:
        statuses.append(str(public_summary.get("status")) if public_summary else "missing")
    return "completed" if statuses and all(status == "completed" for status in statuses) else "partial"


def _protocol_boundary(status: str) -> str:
    if status == "completed":
        return (
            "P35 includes CUDA route smoke plus full requested private/public v3 confirm artifacts; "
            "P31/P34 rows remain fixed references and are not overwritten."
        )
    return (
        "P35 includes CUDA route smoke and any available v3 confirm artifacts; missing or partial "
        "private/public v3 confirm remains resumable and must not be written as completed/full LOSO."
    )


def _render_figures(run_root: Path, gate_path: Path, public_path: Path, private_path: Path) -> dict[str, str]:
    paths = {
        "fig_p35_route_gate_statistics": str(run_root / "fig_p35_route_gate_statistics.png"),
        "fig_p35_private_task_delta": str(run_root / "fig_p35_private_task_delta.png"),
        "fig_p35_public_ablation_comparison": str(run_root / "fig_p35_public_ablation_comparison.png"),
        "fig_p35_context_vs_vehicle_gate": str(run_root / "fig_p35_context_vs_vehicle_gate.png"),
        "fig_p35_nasa_macro_f1_route_comparison": str(run_root / "fig_p35_nasa_macro_f1_route_comparison.png"),
        "fig_p35_uab_rmse_route_comparison": str(run_root / "fig_p35_uab_rmse_route_comparison.png"),
        "fig_p35_stream_role_decision_map": str(run_root / "fig_p35_stream_role_decision_map.png"),
    }
    gate = pd.read_csv(gate_path)
    public = pd.read_csv(public_path)
    private = pd.read_csv(private_path)
    _gate_plot(gate, paths["fig_p35_route_gate_statistics"])
    _gate_plot(gate, paths["fig_p35_context_vs_vehicle_gate"], columns=("context_gate_mean", "vehicle_gate_mean"))
    _gate_plot(gate, paths["fig_p35_stream_role_decision_map"], columns=("lag_gate_mean", "causal_gate_mean"))
    _metric_plot(private, paths["fig_p35_private_task_delta"], "value_mean")
    _metric_plot(public, paths["fig_p35_public_ablation_comparison"], "value_mean")
    _metric_plot(public[public.get("dataset_id", pd.Series(dtype=str)).eq("nasa_csm")], paths["fig_p35_nasa_macro_f1_route_comparison"], "value_mean")
    _metric_plot(public[public.get("dataset_id", pd.Series(dtype=str)).eq("uab_workload_dataset")], paths["fig_p35_uab_rmse_route_comparison"], "value_mean")
    return paths


def _gate_plot(frame: pd.DataFrame, path: str, columns: tuple[str, ...] = ("lag_gate_mean", "context_gate_mean", "vehicle_gate_mean", "causal_gate_mean")) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if frame.empty:
        ax.text(0.5, 0.5, "no gate rows", ha="center", va="center")
    else:
        labels = frame["dataset_id"] + "\n" + frame["variant_id"]
        bottom = np.zeros(len(frame))
        for column in columns:
            values = frame[column].astype(float).to_numpy()
            ax.bar(labels, values, bottom=bottom, label=column)
            bottom += values
        label_stack_totals(ax, labels, bottom)
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _metric_plot(frame: pd.DataFrame, path: str, value_column: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if frame.empty or value_column not in frame:
        ax.text(0.5, 0.5, "no metric rows", ha="center", va="center")
    else:
        label_col = "variant_id" if "variant_id" in frame else "model_name"
        labels = frame[label_col].astype(str) + "\n" + frame.get("metric", "").astype(str)
        values = frame[value_column].astype(float)
        bars = ax.bar(labels, values)
        label_vertical_bars(ax, bars, values)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _gpu_summary(
    config: StageIStreamRoleFusionEvalConfig,
    device: str,
    route_elapsed_s: float,
    route_sample_count: int,
) -> dict[str, object]:
    snapshot = gpu_runtime_snapshot()
    return {
        "run_id": config.run_id,
        "stage": "P35",
        "runtime_device": device,
        "gpu_name": snapshot.get("gpu_name"),
        "torch_version": snapshot.get("torch_version"),
        "cuda_version": snapshot.get("cuda_version"),
        "tensor_cache_mode": config.tensor_cache,
        "amp_mode": config.amp,
        "torch_compile_mode": config.torch_compile,
        "checkpoint_policy": config.checkpoint_policy,
        "max_gpu_memory_gb": snapshot.get("gpu_max_memory_allocated_gb"),
        "samples_per_sec": float(route_sample_count / route_elapsed_s),
        "batch_time_sec": float(route_elapsed_s),
        "copy_time_sec": None,
        "forward_time_sec": float(route_elapsed_s),
        "backward_time_sec": None,
        "step_time_sec": None,
        "timing_basis": "P35 CUDA route smoke forward pass; not full v3 training throughput",
        "utilization_snapshot": snapshot,
    }


def _latest_p34_root() -> Path | None:
    root = _resolve_path(DEFAULT_ARTIFACT_ROOT.replace("stream_role_fusion", "task_heads_optimization"))
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir()]
    return sorted(candidates)[-1] if candidates else None


def _resume_command(config: StageIStreamRoleFusionEvalConfig) -> str:
    parts = [
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python",
        "scripts/stage_i/run_stream_role_fusion_eval.py",
        "--resume",
        "--resume-run-id",
        config.run_id,
        "--device",
        "cuda",
        "--require-cuda",
        "--tensor-cache",
        config.tensor_cache,
        "--auto-batch-size",
        "--amp",
        config.amp,
        "--torch-compile",
        config.torch_compile,
        "--confirm-epochs",
        str(config.confirm_epochs),
        "--seed",
        str(config.seed),
        "--skip-completed",
        "--checkpoint-policy",
        config.checkpoint_policy,
    ]
    if config.run_private_confirm:
        parts.append("--run-private-confirm")
    if config.run_public_confirm:
        parts.append("--run-public-confirm")
    if config.extra_confirm_seeds:
        parts.extend(["--extra-confirm-seeds", *[str(seed) for seed in config.extra_confirm_seeds]])
    if config.private_models:
        parts.extend(["--private-models", *config.private_models])
    if config.public_variants:
        parts.extend(["--public-variants", *config.public_variants])
    if config.public_datasets:
        parts.extend(["--public-datasets", *config.public_datasets])
    return " ".join(parts)


def _render_report(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            f"# Stage I Stream-role-aware Fusion - {summary['run_id']}",
            "",
            f"- status: `{summary['status']}`",
            f"- runtime_device: `{summary['runtime_device']}`",
            f"- artifact_root: `{summary['artifact_root']}`",
            "- boundary: public second stream is context proxy; private second stream is real vehicle.",
            f"- private_confirm_status: `{summary['private_confirm_status']}`",
            f"- public_confirm_status: `{summary['public_confirm_status']}`",
            f"- private_confirm_root: `{summary['private_confirm_root']}`",
            f"- public_confirm_root: `{summary['public_confirm_root']}`",
            "",
            "## Outputs",
            f"- route manifest: `{summary['route_manifest_json']}`",
            f"- gate statistics: `{summary['gate_statistics_csv']}`",
            f"- public metrics: `{summary['public_metrics_csv']}`",
            f"- private metrics: `{summary['private_metrics_csv']}`",
            f"- GPU summary: `{summary['gpu_perf_summary_json']}`",
            f"- resume command: `{summary['resume_command_txt']}`",
        ]
    )


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
