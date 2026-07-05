"""P37 optimized Chronaris metric-calibration runner and report builder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from chronaris.evidence.metric_calibration_render import (
    render_figures,
    render_report,
)
from chronaris.evaluation.dingxin.pipelines.thirdparty_comparison import (
    StageIPrivateThirdPartyComparisonConfig,
    run_task_eval_private_thirdparty_comparison,
)
from chronaris.evaluation.public_datasets.pipelines.fusion_ablation import (
    StageIPublicFusionAblationConfig,
    load_p28_prepared_roots,
    run_task_eval_public_fusion_ablation,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_P30_ROOT = REPO_ROOT / "docs/artifacts/runs/2026-07-02_dingxin-thirdparty-comparison"
DEFAULT_P31_ROOT = REPO_ROOT / "docs/artifacts/runs/2026-07-02_public-fusion-ablation"
DEFAULT_P34_ROOT = REPO_ROOT / "docs/artifacts/runs/2026-07-02_task-head-calibration"
DEFAULT_P35_ROOT = REPO_ROOT / "docs/artifacts/runs/2026-07-02_stream-role-fusion"
DEFAULT_P36_ROOT = REPO_ROOT / "docs/artifacts/runs/2026-07-02_selected-model-reevaluation"
DEFAULT_E_MANIFEST = REPO_ROOT / "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json"
DEFAULT_F_MANIFEST = REPO_ROOT / "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json"
DEFAULT_ARTIFACT_ROOT = "docs/artifacts/runs"
DEFAULT_REPORT_ROOT = "docs/artifacts/runs"

T1_CANDIDATES = (
    "p37_t1_focal_gamma1_gate0p65_ls0p05",
    "p37_t1_focal_gamma2_gate0p75_vehicle_skip_collapse0p05",
    "p37_t1_focal_gamma2_gate0p85_ls0p10_collapse0p10",
)
T3_CANDIDATES = (
    "p37_t3_info_nce_temp0p05_hardw2",
    "p37_t3_info_nce_temp0p10_hardw2",
    "p37_t3_info_nce_temp0p20_hardw1",
    "p37_t3_supcon_margin0p10_hardw2",
)
PUBLIC_CANDIDATES = (
    "p37_public_no_lag_prior_context075",
    "p37_public_no_lag_prior_context085",
    "p37_public_context_adapter_only_cap2x_do0p2",
    "p37_public_force_adaptive_context_gate",
)


def infer_existing_nested_roots(run_root: Path, run_id: str) -> dict[str, str | None]:
    candidates = {
        "t3_screen_root": run_root / "nested_private" / f"{run_id}-t3-screen",
        "t1_screen_root": run_root / "nested_private" / f"{run_id}-t1-screen",
        "private_confirm_root": run_root / "nested_private" / f"{run_id}-private-confirm",
        "public_screen_root": run_root / "nested_public" / f"{run_id}-public-screen",
        "public_confirm_root": run_root / "nested_public" / f"{run_id}-public-confirm",
    }
    return {key: str(path) for key, path in candidates.items() if path.exists()}


def status_from_outputs(config: object, roots: Mapping[str, str | None]) -> str:
    if roots.get("private_confirm_root") and roots.get("public_confirm_root"):
        return "completed"
    if getattr(config, "run_private_confirm") and "private_confirm_root" not in roots:
        return "partial"
    if getattr(config, "run_public_confirm") and "public_confirm_root" not in roots:
        return "partial"
    return "completed" if getattr(config, "run_private_confirm") and getattr(config, "run_public_confirm") else "partial"


_status_from_outputs = status_from_outputs


@dataclass(frozen=True, slots=True)
class StageIMetricCalibrationConfig:
    run_id: str
    p30_root: str = str(DEFAULT_P30_ROOT)
    p31_root: str = str(DEFAULT_P31_ROOT)
    p34_root: str = str(DEFAULT_P34_ROOT)
    p35_root: str = str(DEFAULT_P35_ROOT)
    p36_root: str = str(DEFAULT_P36_ROOT)
    e_run_manifest_path: str = str(DEFAULT_E_MANIFEST)
    f_run_manifest_path: str = str(DEFAULT_F_MANIFEST)
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    run_private_screen: bool = True
    run_private_confirm: bool = True
    run_public_screen: bool = True
    run_public_confirm: bool = True
    private_confirm_top_t1: int = 2
    private_confirm_top_t3: int = 1
    public_confirm_top: int = 2
    public_confirm_extra_seed17: bool = False
    device: str = "cuda"
    require_cuda: bool = True
    tensor_cache: str = "auto"
    max_cache_gb: float = 18.0
    auto_batch_size: bool = True
    batch_size_candidates: tuple[int, ...] = (2048, 1024, 512, 256, 128)
    amp: str = "bf16"
    torch_compile: str = "default"
    cpu_workers: int = 24
    parallel_fold_prep: int = 8
    parallel_candidates: int = 1
    heartbeat_seconds: float = 60.0
    batch_log_interval: int = 20
    profile_gpu: bool = True
    allow_partial: bool = True
    skip_completed: bool = True
    checkpoint_policy: str = "last"


@dataclass(frozen=True, slots=True)
class StageIMetricCalibrationResult:
    run_id: str
    artifact_root: str
    summary_path: str
    evidence_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


def run_task_eval_metric_calibration(
    config: StageIMetricCalibrationConfig,
) -> StageIMetricCalibrationResult:
    device = resolve_torch_device_name(config.device)
    if config.require_cuda and device != "cuda":
        raise RuntimeError(f"P37 requires CUDA for training runs; resolved {device}.")
    run_root = _resolve_path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "metric_calibration_config.json", asdict(config))

    p30_root = _resolve_path(config.p30_root)
    p31_root = _resolve_path(config.p31_root)
    p34_root = _resolve_path(config.p34_root)
    p35_root = _resolve_path(config.p35_root)
    p36_root = _resolve_path(config.p36_root)
    nested_roots: dict[str, str | None] = infer_existing_nested_roots(run_root, config.run_id)

    if config.run_private_screen:
        t3_screen = _run_private_candidates(
            config,
            run_root,
            run_id=f"{config.run_id}-t3-screen",
            models=T3_CANDIDATES,
            seeds=(42,),
            epochs=5,
            screen_only=True,
        )
        nested_roots["t3_screen_root"] = t3_screen.artifact_root
        t1_screen = _run_private_candidates(
            config,
            run_root,
            run_id=f"{config.run_id}-t1-screen",
            models=T1_CANDIDATES,
            seeds=(42,),
            epochs=5,
            screen_only=True,
        )
        nested_roots["t1_screen_root"] = t1_screen.artifact_root

    t3_models = _select_private_models(
        Path(nested_roots.get("t3_screen_root") or ""),
        task_prefix="T3",
        primary_metric="mrr",
        top_n=config.private_confirm_top_t3,
        fallback=T3_CANDIDATES[: config.private_confirm_top_t3],
    )
    t1_models = _select_private_models(
        Path(nested_roots.get("t1_screen_root") or ""),
        task_prefix="T1",
        primary_metric="macro_f1",
        top_n=config.private_confirm_top_t1,
        fallback=T1_CANDIDATES[: config.private_confirm_top_t1],
    )
    if config.run_private_confirm:
        private_confirm = _run_private_candidates(
            config,
            run_root,
            run_id=f"{config.run_id}-private-confirm",
            models=tuple(dict.fromkeys((*t3_models, *t1_models))),
            seeds=(42, 17, 29),
            epochs=20,
            screen_only=False,
        )
        nested_roots["private_confirm_root"] = private_confirm.artifact_root

    if config.run_public_screen:
        public_screen = _run_public_candidates(
            config,
            run_root,
            run_id=f"{config.run_id}-public-screen",
            variants=PUBLIC_CANDIDATES,
            screen_only=True,
            confirm_only=False,
            screen_epochs=5,
            confirm_epochs=5,
            screen_max_folds=2,
            confirm_max_folds=2,
            extra_seeds=(),
        )
        nested_roots["public_screen_root"] = public_screen.artifact_root

    public_variants = _select_public_variants(
        Path(nested_roots.get("public_screen_root") or ""),
        top_n=config.public_confirm_top,
        fallback=PUBLIC_CANDIDATES[: config.public_confirm_top],
    )
    if config.run_public_confirm:
        public_confirm = _run_public_candidates(
            config,
            run_root,
            run_id=f"{config.run_id}-public-confirm",
            variants=public_variants,
            screen_only=False,
            confirm_only=True,
            screen_epochs=5,
            confirm_epochs=20,
            screen_max_folds=2,
            confirm_max_folds=None,
            extra_seeds=(17,) if config.public_confirm_extra_seed17 else (),
        )
        nested_roots["public_confirm_root"] = public_confirm.artifact_root

    tables = _write_terminal_tables(
        run_root=run_root,
        p30_root=p30_root,
        p31_root=p31_root,
        p34_root=p34_root,
        p35_root=p35_root,
        nested_roots=nested_roots,
    )
    status = status_from_outputs(config, nested_roots)
    accepted, rejected = _candidate_summaries(tables)
    accepted_path = run_root / "accepted_candidate_summary.json"
    rejected_path = run_root / "rejected_candidate_summary.json"
    _write_json(accepted_path, accepted)
    _write_json(rejected_path, rejected)
    gpu_path = _write_gpu_summary(run_root, nested_roots)
    figure_paths = render_figures(run_root, {**tables, "gpu_perf_summary_json": gpu_path})
    manifest_path = run_root / "evidence_manifest.json"
    report_path = _resolve_path(config.report_root) / f"task-eval-metric-calibration-{config.run_id}.md"
    resume_path = run_root / "resume_command.txt"
    resume_path.write_text(_resume_command(config) + "\n", encoding="utf-8")
    generated_at = _utc_now()
    summary = {
        "run_id": config.run_id,
        "status": status,
        "generated_at_utc": generated_at,
        "runtime_device": device,
        "artifact_root": str(run_root),
        "p30_root": str(p30_root),
        "p31_root": str(p31_root),
        "p34_root": str(p34_root),
        "p35_root": str(p35_root),
        "p36_root": str(p36_root),
        **nested_roots,
        **{key: str(value) for key, value in tables.items()},
        "accepted_candidate_summary_json": str(accepted_path),
        "rejected_candidate_summary_json": str(rejected_path),
        "gpu_perf_summary_json": str(gpu_path),
        "figure_paths": figure_paths,
        "report_path": str(report_path),
        "evidence_manifest_path": str(manifest_path),
        "resume_command_txt": str(resume_path),
        "protocol_boundary": (
            "P37 uses P30/P31/P34/P35/P36 as fixed references and writes only new "
            "metric-calibration artifacts. Public rows remain context-proxy evidence."
        ),
    }
    summary_path = run_root / "metric_calibration_summary.json"
    _write_json(summary_path, summary)
    _write_json(manifest_path, {**summary, "summary_path": str(summary_path), "stage": "P37"})
    _write_json(run_root / "progress.json", {"run_id": config.run_id, "stage": "P37", "status": status})
    _write_run_log(run_root, config, status, nested_roots)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, accepted, rejected) + "\n", encoding="utf-8")
    return StageIMetricCalibrationResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        evidence_manifest_path=str(manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def _run_private_candidates(
    config: StageIMetricCalibrationConfig,
    run_root: Path,
    *,
    run_id: str,
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    epochs: int,
    screen_only: bool,
):
    return run_task_eval_private_thirdparty_comparison(
        StageIPrivateThirdPartyComparisonConfig(
            run_id=run_id,
            e_run_manifest_path=str(_resolve_path(config.e_run_manifest_path)),
            f_run_manifest_path=str(_resolve_path(config.f_run_manifest_path)),
            output_root=str(run_root / "nested_private"),
            report_root=str(run_root / "nested_reports"),
            models=models,
            seeds=seeds,
            split_strategy=("leave_one_view_out", "leave_one_sortie_out"),
            epochs=epochs,
            screen_epochs=epochs,
            device=config.device,
            require_cuda=config.require_cuda,
            resume=config.skip_completed,
            skip_completed=config.skip_completed,
            allow_partial=config.allow_partial,
            screen_only=screen_only,
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


def _run_public_candidates(
    config: StageIMetricCalibrationConfig,
    run_root: Path,
    *,
    run_id: str,
    variants: tuple[str, ...],
    screen_only: bool,
    confirm_only: bool,
    screen_epochs: int,
    confirm_epochs: int,
    screen_max_folds: int | None,
    confirm_max_folds: int | None,
    extra_seeds: tuple[int, ...],
):
    return run_task_eval_public_fusion_ablation(
        StageIPublicFusionAblationConfig(
            run_id=run_id,
            dataset_prepared_roots=load_p28_prepared_roots(),
            artifact_root=str(run_root / "nested_public"),
            report_root=str(run_root / "nested_reports"),
            datasets=("nasa_csm", "uab_workload_dataset"),
            variants=variants,
            screen_epochs=screen_epochs,
            confirm_epochs=confirm_epochs,
            screen_max_folds=screen_max_folds,
            confirm_max_folds=confirm_max_folds,
            seed=42,
            extra_confirm_seeds=extra_seeds,
            device=config.device,
            require_cuda=config.require_cuda,
            resume=config.skip_completed,
            skip_completed=config.skip_completed,
            allow_partial=config.allow_partial,
            screen_only=screen_only,
            confirm_only=confirm_only,
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


def _select_private_models(
    root: Path,
    *,
    task_prefix: str,
    primary_metric: str,
    top_n: int,
    fallback: Sequence[str],
) -> tuple[str, ...]:
    path = root / "model_comparison_long.csv"
    if not path.exists():
        return tuple(fallback)
    frame = pd.read_csv(path)
    subset = frame[
        frame["task_name"].astype(str).str.startswith(task_prefix)
        & frame["metric"].astype(str).eq(primary_metric)
    ].copy()
    if subset.empty:
        return tuple(fallback)
    ranking = (
        subset.groupby("model_name", as_index=False)["value_mean"]
        .mean()
        .sort_values(["value_mean", "model_name"], ascending=[False, True])
    )
    return tuple(ranking["model_name"].head(top_n).astype(str).tolist()) or tuple(fallback)


def _select_public_variants(root: Path, *, top_n: int, fallback: Sequence[str]) -> tuple[str, ...]:
    path = root / "screen_leaderboard.csv"
    if not path.exists():
        return tuple(fallback)
    frame = pd.read_csv(path)
    if frame.empty:
        return tuple(fallback)
    rows = []
    for variant, subset in frame.groupby("variant_id"):
        nasa = subset[(subset["dataset_id"] == "nasa_csm") & subset["primary_metric"].eq("combined_macro_f1")]
        uab = subset[(subset["dataset_id"] == "uab_workload_dataset") & subset["primary_metric"].eq("mean_rmse")]
        score = 0.0
        if not nasa.empty:
            score += float(nasa["selection_score"].mean())
        if not uab.empty:
            score -= float(uab["selection_score"].mean()) / 10.0
        rows.append({"variant_id": variant, "score": score})
    ranking = pd.DataFrame(rows).sort_values(["score", "variant_id"], ascending=[False, True])
    return tuple(ranking["variant_id"].head(top_n).astype(str).tolist()) or tuple(fallback)


def _write_terminal_tables(
    *,
    run_root: Path,
    p30_root: Path,
    p31_root: Path,
    p34_root: Path,
    p35_root: Path,
    nested_roots: Mapping[str, str | None],
) -> dict[str, Path]:
    private_confirm_root = Path(nested_roots.get("private_confirm_root") or "")
    t3_root = private_confirm_root if private_confirm_root.exists() else Path(nested_roots.get("t3_screen_root") or "")
    t1_root = private_confirm_root if private_confirm_root.exists() else Path(nested_roots.get("t1_screen_root") or "")
    t3_metrics = _private_task_metrics(t3_root, "T3")
    t1_metrics = _private_task_metrics(t1_root, "T1")
    p34_t3 = _private_task_metrics(p34_root, "T3")
    p34_t1 = _private_task_metrics(p34_root, "T1")
    t3_path = run_root / "t3_metric_calibration_metrics.csv"
    t1_path = run_root / "t1_calibration_metrics.csv"
    t3_metrics.to_csv(t3_path, index=False)
    t1_metrics.to_csv(t1_path, index=False)
    _copy_if_exists(t3_root / "fold_predictions.csv", run_root / "t3_retrieval_predictions.csv")
    _copy_if_exists(private_confirm_root / "training_curves.csv", run_root / "training_curves.csv")
    _copy_if_exists(private_confirm_root / "fold_metrics.csv", run_root / "fold_metrics.csv")
    _write_t1_gate_sweep(run_root, t1_root)
    _write_private_delta(t3_metrics, p34_t3, run_root / "t3_delta_vs_p34.csv").to_csv(run_root / "t3_delta_vs_p34.csv", index=False)
    _write_private_delta(t1_metrics, p34_t1, run_root / "t1_delta_vs_p34.csv").to_csv(run_root / "t1_delta_vs_p34.csv", index=False)
    public_root = Path(nested_roots.get("public_confirm_root") or nested_roots.get("public_screen_root") or "")
    public_metrics = _public_metrics(public_root)
    public_path = run_root / "public_route_calibration_metrics.csv"
    public_metrics.to_csv(public_path, index=False)
    route_gate = _route_gate_calibration(p35_root)
    gate_path = run_root / "route_gate_calibration.csv"
    route_gate.to_csv(gate_path, index=False)
    _public_delta(public_metrics, p31_root, p35_root).to_csv(run_root / "comparison_vs_p31_best.csv", index=False)
    _public_delta(public_metrics, p35_root, p35_root).to_csv(run_root / "comparison_vs_p35.csv", index=False)
    _write_private_delta(t3_metrics, p34_t3, run_root / "p37_delta_vs_p34.csv").to_csv(run_root / "p37_delta_vs_p34.csv", index=False)
    _public_delta(public_metrics, p35_root, p35_root).to_csv(run_root / "p37_delta_vs_p35.csv", index=False)
    _delta_vs_p30_p31(t1_metrics, t3_metrics, public_metrics, p30_root, p31_root).to_csv(
        run_root / "p37_delta_vs_p30_p31.csv",
        index=False,
    )
    _write_t3_diagnostics(run_root, t3_root)
    return {
        "t3_metric_calibration_metrics_csv": t3_path,
        "t1_calibration_metrics_csv": t1_path,
        "public_route_calibration_metrics_csv": public_path,
        "route_gate_calibration_csv": gate_path,
        "p37_delta_vs_p34_csv": run_root / "p37_delta_vs_p34.csv",
        "p37_delta_vs_p35_csv": run_root / "p37_delta_vs_p35.csv",
        "p37_delta_vs_p30_p31_csv": run_root / "p37_delta_vs_p30_p31.csv",
    }


def _private_task_metrics(root: Path, task_prefix: str) -> pd.DataFrame:
    path = root / "model_comparison_long.csv"
    if not path.exists():
        path = root / "task_head_metrics_long.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame[frame["task_name"].astype(str).str.startswith(task_prefix)].copy()


def _public_metrics(root: Path) -> pd.DataFrame:
    confirm = root / "confirm_leaderboard.csv"
    screen = root / "screen_leaderboard.csv"
    if confirm.exists() and not pd.read_csv(confirm).empty:
        return pd.read_csv(confirm)
    if screen.exists():
        return pd.read_csv(screen)
    return pd.DataFrame()


def _write_private_delta(p37: pd.DataFrame, baseline: pd.DataFrame, _path: Path) -> pd.DataFrame:
    if p37.empty or baseline.empty:
        return pd.DataFrame()
    baseline_best = baseline[baseline["model_name"].astype(str).str.contains("chronaris_v2|task_heads", regex=True)]
    rows = []
    for keys, p37_subset in p37.groupby(["task_name", "split_strategy", "metric"], sort=False):
        base = baseline_best[
            (baseline_best["task_name"] == keys[0])
            & (baseline_best["split_strategy"] == keys[1])
            & (baseline_best["metric"] == keys[2])
        ]
        if base.empty:
            continue
        best = _best_private_row(p37_subset)
        baseline_value = float(base["value_mean"].iloc[0])
        p37_value = float(best["value_mean"])
        metric = str(keys[2])
        delta = baseline_value - p37_value if metric in {"rmse", "mae", "nrmse"} else p37_value - baseline_value
        rows.append(
            {
                "task_name": keys[0],
                "split_strategy": keys[1],
                "metric": metric,
                "p34_value": baseline_value,
                "p37_model": best["model_name"],
                "p37_value": p37_value,
                "delta_positive_is_better": delta,
            }
        )
    return pd.DataFrame(rows)


def _best_private_row(frame: pd.DataFrame) -> pd.Series:
    metric = str(frame["metric"].iloc[0])
    ascending = metric in {"rmse", "mae", "nrmse"}
    return frame.sort_values(["value_mean", "model_name"], ascending=[ascending, True]).iloc[0]


def _route_gate_calibration(p35_root: Path) -> pd.DataFrame:
    source = p35_root / "gate_statistics.csv"
    if not source.exists():
        return pd.DataFrame()
    frame = pd.read_csv(source)
    rows = frame.to_dict(orient="records")
    for record in rows:
        record["p37_private_preservation_check"] = (
            "preserved" if record.get("second_stream_is_real_vehicle") in {True, "True"} else "public_context_proxy"
        )
    return pd.DataFrame(rows)


def _public_delta(public_metrics: pd.DataFrame, reference_root: Path, _p35_root: Path) -> pd.DataFrame:
    if public_metrics.empty:
        return pd.DataFrame()
    rows = []
    p35 = _safe_read(reference_root / "public_metrics.csv")
    p31 = _safe_read(reference_root / "ablation_summary.csv")
    for record in public_metrics.to_dict(orient="records"):
        dataset = record.get("dataset_id")
        metric = record.get("primary_metric")
        value = float(record.get("selection_score"))
        ref_value = np.nan
        ref_name = "reference"
        if dataset == "nasa_csm":
            ref_rows = _filter_public_reference(p35, dataset, "v3_stream_role")
            if not ref_rows.empty:
                ref_value = float(ref_rows["value_mean"].iloc[0])
                ref_name = "P35 v3_stream_role"
            elif not p31.empty:
                ref_rows = p31[(p31["dataset_id"] == dataset) & (p31["variant_id"] == "no_lag_window")]
                if not ref_rows.empty:
                    ref_value = float(ref_rows["value_mean"].iloc[0])
                    ref_name = "P31 no_lag_window"
            delta = value - ref_value if not np.isnan(ref_value) else np.nan
        else:
            ref_rows = _filter_public_reference(p35, dataset, "v3_stream_role")
            if not ref_rows.empty:
                ref_value = float(ref_rows["value_mean"].iloc[0])
                ref_name = "P35 v3_stream_role"
            elif not p31.empty:
                ref_rows = p31[(p31["dataset_id"] == dataset) & (p31["variant_id"] == "context_only")]
                if not ref_rows.empty:
                    ref_value = float(ref_rows["value_mean"].iloc[0])
                    ref_name = "P31 context_only"
            delta = ref_value - value if not np.isnan(ref_value) else np.nan
        rows.append(
            {
                "dataset_id": dataset,
                "metric": metric,
                "p37_variant": record.get("variant_id"),
                "p37_value": value,
                "reference": ref_name,
                "reference_value": ref_value,
                "delta_positive_is_better": delta,
            }
        )
    return pd.DataFrame(rows)


def _filter_public_reference(frame: pd.DataFrame, dataset_id: object, variant_id: str) -> pd.DataFrame:
    if frame.empty or not {"dataset_id", "variant_id"}.issubset(frame.columns):
        return pd.DataFrame()
    return frame[(frame["dataset_id"] == dataset_id) & (frame["variant_id"] == variant_id)]


def _delta_vs_p30_p31(
    t1: pd.DataFrame,
    t3: pd.DataFrame,
    public: pd.DataFrame,
    p30_root: Path,
    p31_root: Path,
) -> pd.DataFrame:
    rows = []
    p30 = _safe_read(p30_root / "model_comparison_long.csv")
    for frame in (t1, t3):
        if frame.empty or p30.empty:
            continue
        for keys, subset in frame.groupby(["task_name", "split_strategy", "metric"], sort=False):
            base = p30[
                (p30["task_name"] == keys[0])
                & (p30["split_strategy"] == keys[1])
                & (p30["metric"] == keys[2])
                & (p30["model_name"] == "chronaris_full")
            ]
            if base.empty:
                continue
            best = _best_private_row(subset)
            metric = str(keys[2])
            ref = float(base["value_mean"].iloc[0])
            val = float(best["value_mean"])
            delta = ref - val if metric in {"rmse", "mae", "nrmse"} else val - ref
            rows.append({"scope": "private", "metric": metric, "reference": "P30 chronaris_full", "p37_value": val, "reference_value": ref, "delta_positive_is_better": delta})
    rows.extend(_public_delta(public, p31_root, p31_root).to_dict(orient="records"))
    return pd.DataFrame(rows)


def _write_t3_diagnostics(run_root: Path, private_root: Path) -> None:
    curves = _safe_read(private_root / "training_curves.csv")
    if not curves.empty:
        keep = [col for col in curves.columns if col.startswith("p37_") or col in {"model_name", "epoch", "train_loss", "candidate_pool_policy"}]
        curves[keep].to_csv(run_root / "t3_contrastive_loss_sweep.csv", index=False)
        curves[keep].to_csv(run_root / "t3_hard_negative_diagnostics.csv", index=False)
    else:
        pd.DataFrame().to_csv(run_root / "t3_contrastive_loss_sweep.csv", index=False)
        pd.DataFrame().to_csv(run_root / "t3_hard_negative_diagnostics.csv", index=False)
    preds = _safe_read(private_root / "fold_predictions.csv")
    if not preds.empty and "rank" in preds:
        preds[["rank", "candidate_count", "top1_hit", "top3_hit", "top5_hit", "reciprocal_rank"]].describe().to_csv(
            run_root / "t3_similarity_distribution.csv"
        )
    else:
        pd.DataFrame().to_csv(run_root / "t3_similarity_distribution.csv", index=False)


def _write_t1_gate_sweep(run_root: Path, private_root: Path) -> None:
    curves = _safe_read(private_root / "training_curves.csv")
    if curves.empty:
        pd.DataFrame().to_csv(run_root / "t1_gate_sweep.csv", index=False)
        (run_root / "t1_confusion_matrices").mkdir(exist_ok=True)
        return
    keep = [
        col
        for col in curves.columns
        if col
        in {
            "model_name",
            "split_strategy",
            "fold_group",
            "epoch",
            "train_loss",
            "p37_focal_gamma",
            "p37_label_smoothing",
            "p37_target_vehicle_contribution",
            "p37_gate_mean",
            "class_weight_source",
        }
    ]
    curves[keep].to_csv(run_root / "t1_gate_sweep.csv", index=False)
    matrix_dir = run_root / "t1_confusion_matrices"
    matrix_dir.mkdir(exist_ok=True)
    preds = _safe_read(private_root / "fold_predictions.csv")
    if not preds.empty and {"y_true", "y_pred", "model_name"}.issubset(preds.columns):
        for model_name, subset in preds.groupby("model_name"):
            valid = subset.copy()
            valid["y_true"] = pd.to_numeric(valid["y_true"], errors="coerce")
            valid["y_pred"] = pd.to_numeric(valid["y_pred"], errors="coerce")
            valid = valid[np.isfinite(valid["y_true"]) & np.isfinite(valid["y_pred"])]
            if valid.empty:
                continue
            y_true = valid["y_true"].astype(int)
            y_pred = valid["y_pred"].astype(int)
            labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
            matrix = pd.crosstab(
                y_true,
                y_pred,
                rownames=["true"],
                colnames=["pred"],
                dropna=False,
            ).reindex(index=labels, columns=labels, fill_value=0)
            matrix.to_csv(matrix_dir / f"{model_name}_confusion.csv")


def _candidate_summaries(tables: Mapping[str, Path]) -> tuple[dict[str, object], dict[str, object]]:
    accepted: dict[str, object] = {"accepted": []}
    rejected: dict[str, object] = {"rejected": []}
    t3_delta = _safe_read(tables["p37_delta_vs_p34_csv"])
    if not t3_delta.empty and (t3_delta["delta_positive_is_better"] > 0).any():
        accepted["accepted"].append({"scope": "T3", "reason": "P37 improves at least one P34 retrieval ranking metric."})
    else:
        rejected["rejected"].append({"scope": "T3", "reason": "No P37 retrieval metric exceeded P34 in available rows."})
    t1_delta = _safe_read(Path(tables["p37_delta_vs_p34_csv"]).parent / "t1_delta_vs_p34.csv")
    t1_macro = t1_delta[t1_delta.get("metric", "").astype(str).eq("macro_f1")] if not t1_delta.empty else pd.DataFrame()
    if not t1_macro.empty and (t1_macro["delta_positive_is_better"] > 0).any():
        accepted["accepted"].append({"scope": "T1", "reason": "P37 improves macro-F1 on at least one split."})
    else:
        rejected["rejected"].append({"scope": "T1", "reason": "No P37 macro-F1 improvement over P34 in available rows."})
    p35_delta = _safe_read(tables["p37_delta_vs_p35_csv"])
    if not p35_delta.empty and (p35_delta["delta_positive_is_better"] > 0).any():
        accepted["accepted"].append({"scope": "public_route", "reason": "P37 public route improves at least one P35 public metric."})
    else:
        rejected["rejected"].append({"scope": "public_route", "reason": "No accepted public route improvement over P35 in available rows."})
    return accepted, rejected


def _write_gpu_summary(run_root: Path, nested_roots: Mapping[str, str | None]) -> Path:
    rows = []
    for name, root in nested_roots.items():
        if not root:
            continue
        path = Path(root) / "gpu_perf_summary.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"source": name, **payload})
    path = run_root / "gpu_perf_summary.json"
    _write_json(path, {"sources": rows, "source_count": len(rows)})
    return path


def _write_run_log(
    run_root: Path,
    config: StageIMetricCalibrationConfig,
    status: str,
    nested_roots: Mapping[str, str | None],
) -> None:
    lines = [
        f"{_utc_now()} INFO stage=P37 run_id={config.run_id} status={status}",
        "P37 did not overwrite P30/P31/P34/P35/P36 artifacts.",
    ]
    lines.extend(f"{key}={value}" for key, value in nested_roots.items())
    (run_root / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resume_command(config: StageIMetricCalibrationConfig) -> str:
    return (
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/task_eval/evidence/run_metric_calibration.py "
        f"--run-id {config.run_id} --device cuda --require-cuda --resume --skip-completed "
        "--tensor-cache auto --auto-batch-size --amp bf16"
    )


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.write_bytes(source.read_bytes())
    elif not target.exists():
        pd.DataFrame().to_csv(target, index=False)


def _safe_read(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")
