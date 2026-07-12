"""P28 chronaris_public_fusion refresh and long-confirm workflow."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronaris-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chronaris.evaluation import (
    evaluate_classification_predictions,
    evaluate_regression_predictions,
)
from chronaris.evaluation.public_datasets.pipelines.deep_baseline import (
    StageIDeepBaselineConfig,
    StageIDeepBaselineRunResult,
    run_task_eval_deep_baseline,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.pipelines.torch_runtime import resolve_torch_device_name

REPO_ROOT = Path(__file__).resolve().parents[5]
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

BASELINES = {
    "nasa_csm": {
        "combined": {
            "macro_f1": {
                "public_baseline": 0.4549915569394342,
                "classical_baseline": 0.3741024976765464,
                "mult": 0.3023466874108648,
                "contiformer": 0.30226373483816565,
            },
            "balanced_accuracy": {
                "public_baseline": 0.5590685276024547,
                "classical_baseline": 0.37649783482140214,
                "mult": 0.3333333333333333,
                "contiformer": 0.33304745568896515,
            },
        },
        "benchmark_only": {
            "macro_f1": {
                "public_baseline": 0.7445135090687961,
                "classical_baseline": 0.4641561353985333,
                "mult": 0.3024598393574297,
                "contiformer": 0.30243657372519467,
            }
        },
        "loft_only": {
            "macro_f1": {
                "public_baseline": 0.3643360690716937,
                "classical_baseline": 0.3723186247618541,
                "mult": 0.30222579780101905,
                "contiformer": 0.3002832861189802,
            }
        },
    },
    "uab_workload_dataset": {
        "n_back": {
            "rmse": {
                "public_baseline": 4.610314175862912,
                "classical_baseline": 10.223413421909578,
                "mult": 5.828201477861089,
                "contiformer": 4.654091035125815,
            }
        },
        "heat_the_chair": {
            "rmse": {
                "public_baseline": 1.433140268927459,
                "classical_baseline": 1.8639105911346199,
                "mult": 2.8251064718014294,
                "contiformer": 1.4567585837447001,
            }
        },
    },
}


@dataclass(frozen=True, slots=True)
class PublicFusionRefreshCandidate:
    candidate_id: str
    hidden_dim: int
    layers: int
    num_heads: int
    dropout: float
    fusion_event_bias_weight: float
    fusion_lag_window_points: int | None
    fusion_normalize_states: bool
    learning_rate: float
    batch_size: int
    regression_loss: str = "mse"
    huber_delta: float = 1.0
    target_transform: str = "none"
    weight_decay: float = 0.0
    gradient_clip_max_norm: float | None = 1.0


@dataclass(frozen=True, slots=True)
class StageIPublicFusionRefreshConfig:
    run_id: str
    dataset_prepared_roots: Mapping[str, str]
    artifact_root: str = "docs/artifacts/runs"
    report_root: str = "docs/artifacts/runs"
    datasets: tuple[str, ...] = ("nasa_csm", "uab_workload_dataset")
    screen_epochs: int = 5
    confirm_epochs: int = 20
    screen_max_folds: int | None = 2
    confirm_max_folds: int | None = None
    screen_candidate_limit: int = 8
    confirm_top_k: int = 1
    screen_seed: int = 42
    confirm_seeds: tuple[int, ...] = (42,)
    device: str = "auto"
    require_cuda: bool = True
    resume: bool = False
    resume_run_id: str | None = None
    resume_from_progress: str | None = None
    skip_completed: bool = True
    allow_partial: bool = False
    screen_only: bool = False
    confirm_only: bool = False
    candidate_filter: tuple[str, ...] = ()
    dataset_filter: tuple[str, ...] = ()
    heartbeat_seconds: float = 60.0
    batch_log_interval: int = 20
    batch_size: int | None = None
    learning_rate: float | None = None
    train_sampling_policy: str | None = None
    locked_configuration_path: str | None = None
    promotion_evidence_path: str | None = None
    external_confirmation_only: bool = False


@dataclass(frozen=True, slots=True)
class StageIPublicFusionRefreshResult:
    run_id: str
    artifact_root: str
    summary_path: str
    evidence_manifest_path: str
    report_path: str
    summary: Mapping[str, object]


def run_task_eval_public_fusion_refresh(
    config: StageIPublicFusionRefreshConfig,
) -> StageIPublicFusionRefreshResult:
    artifact_root = _resolve_path(config.artifact_root) / config.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=artifact_root,
        run_id=config.run_id,
        stage_name="task_eval_public_fusion_refresh",
        logger=LOGGER,
        initial_progress={
            "artifact_root": str(artifact_root),
            "datasets": list(config.datasets),
            "dataset_filter": list(config.dataset_filter),
            "candidate_filter": list(config.candidate_filter),
            "resume": config.resume,
            "resume_run_id": config.resume_run_id,
            "resume_from_progress": config.resume_from_progress,
            "skip_completed": config.skip_completed,
            "allow_partial": config.allow_partial,
            "screen_only": config.screen_only,
            "confirm_only": config.confirm_only,
            "heartbeat_seconds": config.heartbeat_seconds,
            "batch_log_interval": config.batch_log_interval,
        },
    ) as progress:
        try:
            result = _run_task_eval_public_fusion_refresh_core(
                config=config,
                artifact_root=artifact_root,
                progress=progress,
            )
            progress.finish(
                status=result.summary.get("status", "completed"),
                summary_path=result.summary_path,
                evidence_manifest_path=result.evidence_manifest_path,
                report_path=result.report_path,
            )
            return result
        except BaseException as exc:
            blocked_path = artifact_root / "blocked_reason.json"
            partial_path = artifact_root / "partial_summary.json"
            payload = {
                "run_id": config.run_id,
                "status": "blocked",
                "generated_at_utc": _utc_now(),
                "blocked_reason": str(exc),
                "blocked_type": type(exc).__name__,
                "artifact_root": str(artifact_root),
                "resume_command": (
                    "python scripts/task_eval/public/run_public_fusion_refresh.py "
                    f"--run-id {config.run_id} --resume --skip-completed --allow-partial"
                ),
            }
            blocked_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            partial_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            progress.update(
                "blocked",
                status="blocked",
                blocked_reason=str(exc),
                blocked_reason_path=str(blocked_path),
                partial_summary_path=str(partial_path),
            )
            if config.allow_partial:
                report_path = (
                    _resolve_path(config.report_root)
                    / f"task-eval-public-fusion-refresh-{config.run_id}.md"
                )
                manifest_path = artifact_root / "evidence_manifest.json"
                summary = {
                    **payload,
                    "summary_path": str(partial_path),
                    "evidence_manifest_path": str(manifest_path),
                    "report_path": str(report_path),
                }
                manifest_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                return StageIPublicFusionRefreshResult(
                    run_id=config.run_id,
                    artifact_root=str(artifact_root),
                    summary_path=str(partial_path),
                    evidence_manifest_path=str(manifest_path),
                    report_path=str(report_path),
                    summary=summary,
                )
            raise


def _run_task_eval_public_fusion_refresh_core(
    *,
    config: StageIPublicFusionRefreshConfig,
    artifact_root: Path,
    progress,
) -> StageIPublicFusionRefreshResult:
    runtime_device = resolve_torch_device_name(config.device)
    if runtime_device != "cuda" and config.require_cuda:
        raise RuntimeError(
            "task_eval_public_fusion_refresh requires CUDA but resolved "
            f"runtime_device={runtime_device}."
        )

    candidates, candidate_grid = build_public_fusion_refresh_candidates(
        limit=config.screen_candidate_limit,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )
    if config.external_confirmation_only:
        candidate, bridge = _locked_public_bridge_candidate(config)
        candidates = (candidate,)
        candidate_grid.update(bridge)
    if config.candidate_filter:
        selected_candidate_ids = set(config.candidate_filter)
        candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.candidate_id in selected_candidate_ids
        )
        candidate_grid["candidate_filter"] = list(config.candidate_filter)
        candidate_grid["filtered_screen_candidates"] = [
            asdict(candidate) for candidate in candidates
        ]
    dataset_order = config.dataset_filter or config.datasets
    selected_datasets = tuple(
        dataset for dataset in dataset_order if dataset in config.dataset_prepared_roots
    )
    if not selected_datasets:
        raise ValueError("no requested datasets have prepared roots.")
    if not candidates:
        raise ValueError("candidate filter selected no refresh candidates.")

    config_path = artifact_root / "fusion_refresh_config.json"
    candidate_grid_path = artifact_root / "candidate_grid.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": config.run_id,
                "runtime_device": runtime_device,
                "config": _config_dict(config),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_grid_path.write_text(
        json.dumps(candidate_grid, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    screen_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    training_curve_rows: list[dict[str, object]] = []
    screen_path = artifact_root / "screen_leaderboard.csv"
    if config.confirm_only:
        if not screen_path.exists():
            raise FileNotFoundError(
                f"confirm-only requested but screen leaderboard is missing: {screen_path}"
            )
        screen_frame = _sort_leaderboard(pd.read_csv(screen_path))
        screen_rows = screen_frame.to_dict(orient="records")
        progress.update(
            "screen_loaded_for_confirm",
            screen_leaderboard_csv=str(screen_path),
            screen_row_count=len(screen_rows),
        )
    else:
        for dataset_id in selected_datasets:
            progress.update("dataset_screen_start", dataset_id=dataset_id)
            LOGGER.info(
                "P28 screen dataset_start run_id=%s dataset=%s candidate_count=%d",
                config.run_id,
                dataset_id,
                len(candidates),
            )
            for candidate in candidates:
                progress.update(
                    "candidate_screen_start",
                    dataset_id=dataset_id,
                    candidate_id=candidate.candidate_id,
                    seed=config.screen_seed,
                )
                result = _run_candidate(
                    stage="screen",
                    artifact_root=artifact_root,
                    dataset_id=dataset_id,
                    prepared_root=config.dataset_prepared_roots[dataset_id],
                    candidate=candidate,
                    epochs=config.screen_epochs,
                    max_folds=config.screen_max_folds,
                    seed=config.screen_seed,
                    device=config.device,
                    train_sampling_policy=config.train_sampling_policy,
                    heartbeat_seconds=config.heartbeat_seconds,
                    batch_log_interval=config.batch_log_interval,
                    skip_completed=config.resume and config.skip_completed,
                )
                screen_row = _extract_leaderboard_row(
                    dataset_id=dataset_id,
                    stage="screen",
                    candidate=candidate,
                    seed=config.screen_seed,
                    summary=result.summary,
                    summary_path=result.summary_path,
                    artifact_root=result.artifact_root,
                )
                screen_rows.append(screen_row)
                fold_metric_rows.extend(
                    _write_candidate_fold_metrics(
                        result=result,
                        stage="screen",
                        dataset_id=dataset_id,
                        candidate_id=candidate.candidate_id,
                        seed=config.screen_seed,
                    )
                )
                training_curve_rows.extend(
                    _read_training_curves(
                        result.summary,
                        stage="screen",
                        dataset_id=dataset_id,
                        candidate_id=candidate.candidate_id,
                        seed=config.screen_seed,
                    )
                )
                _write_partial_tables(
                    artifact_root,
                    screen_rows=screen_rows,
                    confirm_rows=[],
                    fold_metric_rows=fold_metric_rows,
                    training_curve_rows=training_curve_rows,
                )
                progress.update(
                    "candidate_screen_done",
                    dataset_id=dataset_id,
                    candidate_id=candidate.candidate_id,
                    seed=config.screen_seed,
                    summary_path=result.summary_path,
                    completed_candidate_count=len(screen_rows),
                )
                LOGGER.info(
                    "P28 screen candidate_done run_id=%s dataset=%s candidate=%s seed=%s summary=%s",
                    config.run_id,
                    dataset_id,
                    candidate.candidate_id,
                    config.screen_seed,
                    result.summary_path,
                )
            progress.update("dataset_screen_done", dataset_id=dataset_id)

        screen_frame = _sort_leaderboard(pd.DataFrame(screen_rows))
        screen_frame.to_csv(screen_path, index=False)
        progress.update(
            "screen_done",
            screen_leaderboard_csv=str(screen_path),
            screen_row_count=len(screen_frame),
        )

    confirm_rows: list[dict[str, object]] = []
    if not config.screen_only:
        for dataset_id in selected_datasets:
            top_candidates = _select_confirm_candidates(
                screen_frame,
                candidates,
                dataset_id=dataset_id,
                top_k=config.confirm_top_k,
            )
            progress.update(
                "dataset_confirm_start",
                dataset_id=dataset_id,
                top_candidate_ids=[candidate.candidate_id for candidate in top_candidates],
            )
            LOGGER.info(
                "P28 confirm dataset_start run_id=%s dataset=%s top_candidates=%s",
                config.run_id,
                dataset_id,
                [candidate.candidate_id for candidate in top_candidates],
            )
            for candidate in top_candidates:
                for seed in config.confirm_seeds:
                    progress.update(
                        "candidate_confirm_start",
                        dataset_id=dataset_id,
                        candidate_id=candidate.candidate_id,
                        seed=seed,
                    )
                    result = _run_candidate(
                        stage="confirm",
                        artifact_root=artifact_root,
                        dataset_id=dataset_id,
                        prepared_root=config.dataset_prepared_roots[dataset_id],
                        candidate=candidate,
                        epochs=config.confirm_epochs,
                        max_folds=config.confirm_max_folds,
                        seed=seed,
                        device=config.device,
                        train_sampling_policy=config.train_sampling_policy,
                        heartbeat_seconds=config.heartbeat_seconds,
                        batch_log_interval=config.batch_log_interval,
                        skip_completed=config.resume and config.skip_completed,
                    )
                    confirm_row = _extract_leaderboard_row(
                        dataset_id=dataset_id,
                        stage="confirm",
                        candidate=candidate,
                        seed=seed,
                        summary=result.summary,
                        summary_path=result.summary_path,
                        artifact_root=result.artifact_root,
                    )
                    confirm_rows.append(confirm_row)
                    fold_metric_rows.extend(
                        _write_candidate_fold_metrics(
                            result=result,
                            stage="confirm",
                            dataset_id=dataset_id,
                            candidate_id=candidate.candidate_id,
                            seed=seed,
                        )
                    )
                    training_curve_rows.extend(
                        _read_training_curves(
                            result.summary,
                            stage="confirm",
                            dataset_id=dataset_id,
                            candidate_id=candidate.candidate_id,
                            seed=seed,
                        )
                    )
                    _write_partial_tables(
                        artifact_root,
                        screen_rows=screen_rows,
                        confirm_rows=confirm_rows,
                        fold_metric_rows=fold_metric_rows,
                        training_curve_rows=training_curve_rows,
                    )
                    progress.update(
                        "candidate_confirm_done",
                        dataset_id=dataset_id,
                        candidate_id=candidate.candidate_id,
                        seed=seed,
                        summary_path=result.summary_path,
                        completed_confirm_candidate_count=len(confirm_rows),
                    )
                    LOGGER.info(
                        "P28 confirm candidate_done run_id=%s dataset=%s candidate=%s seed=%s summary=%s",
                        config.run_id,
                        dataset_id,
                        candidate.candidate_id,
                        seed,
                        result.summary_path,
                    )
            progress.update("dataset_confirm_done", dataset_id=dataset_id)

    confirm_frame = pd.DataFrame(confirm_rows)
    confirm_frame = _sort_leaderboard(confirm_frame)
    confirm_path = artifact_root / "confirm_leaderboard.csv"
    fold_metrics_path = artifact_root / "fold_metrics.csv"
    training_curves_path = artifact_root / "training_curves.csv"
    confirm_frame.to_csv(confirm_path, index=False)
    pd.DataFrame(fold_metric_rows).to_csv(fold_metrics_path, index=False)
    pd.DataFrame(training_curve_rows).to_csv(training_curves_path, index=False)

    best_by_dataset_task = _build_best_by_dataset_task(confirm_frame)
    best_path = artifact_root / "best_by_dataset_task.json"
    best_path.write_text(
        json.dumps(best_by_dataset_task, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    figure_paths = _render_refresh_figures(
        artifact_root=artifact_root,
        screen_frame=screen_frame,
        confirm_frame=confirm_frame,
        training_curves=pd.DataFrame(training_curve_rows),
        best_by_dataset_task=best_by_dataset_task,
    )
    missing_figures = [
        {"figure": name, "path": path, "reason": "file_not_generated"}
        for name, path in figure_paths.items()
        if not Path(path).exists()
    ]
    summary_path = artifact_root / "fusion_refresh_summary.json"
    manifest_path = artifact_root / "evidence_manifest.json"
    report_path = _resolve_path(config.report_root) / f"task-eval-public-fusion-refresh-{config.run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    status = _infer_refresh_status(
        selected_datasets=selected_datasets,
        screen_only=config.screen_only,
        confirm_frame=confirm_frame,
        missing_figures=missing_figures,
    )
    completed_fold_count = _completed_fold_count(fold_metric_rows)
    expected_confirm_candidate_count = (
        len(selected_datasets) * max(config.confirm_top_k, 1) * len(config.confirm_seeds)
        if not config.screen_only
        else 0
    )
    summary = {
        "run_id": config.run_id,
        "status": status,
        "generated_at_utc": _utc_now(),
        "runtime_device": runtime_device,
        "artifact_root": str(artifact_root),
        "config_path": str(config_path),
        "candidate_grid_path": str(candidate_grid_path),
        "screen_leaderboard_csv": str(screen_path),
        "confirm_leaderboard_csv": str(confirm_path),
        "fold_metrics_csv": str(fold_metrics_path),
        "training_curves_csv": str(training_curves_path),
        "best_by_dataset_task_path": str(best_path),
        "best_by_dataset_task": best_by_dataset_task,
        "figure_paths": figure_paths,
        "missing_figures": missing_figures,
        "evidence_manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "progress_path": str(artifact_root / "progress.json"),
        "run_log_path": str(artifact_root / "run.log"),
        "completed_dataset_count": int(confirm_frame["dataset_id"].nunique()) if not confirm_frame.empty else 0,
        "completed_candidate_count": len(screen_rows) + len(confirm_rows),
        "completed_confirm_candidate_count": len(confirm_rows),
        "expected_confirm_candidate_count": expected_confirm_candidate_count,
        "completed_fold_count": completed_fold_count,
        "protocol": {
            "screen": {
                "epochs": config.screen_epochs,
                "max_folds": config.screen_max_folds,
                "seed": config.screen_seed,
            },
            "confirm": {
                "epochs": config.confirm_epochs,
                "max_folds": config.confirm_max_folds,
                "seeds": list(config.confirm_seeds),
            },
            "selection": (
                "single locked public-adapter bridge; no public-label model choice"
                if config.external_confirmation_only
                else "top-k by NASA combined macro-F1 or UAB mean RMSE"
            ),
            "evidence_role": "public_adapter_context_proxy_evidence",
            "external_confirmation_only": config.external_confirmation_only,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if status != "completed":
        (artifact_root / "partial_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest = {
        **summary,
        "summary_path": str(summary_path),
        "source_prepared_roots": dict(config.dataset_prepared_roots),
        "baseline_reference": BASELINES,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _render_refresh_report(config.run_id, summary, confirm_frame) + "\n",
        encoding="utf-8",
    )
    return StageIPublicFusionRefreshResult(
        run_id=config.run_id,
        artifact_root=str(artifact_root),
        summary_path=str(summary_path),
        evidence_manifest_path=str(manifest_path),
        report_path=str(report_path),
        summary=summary,
    )


def build_public_fusion_refresh_candidates(
    *,
    limit: int | None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> tuple[tuple[PublicFusionRefreshCandidate, ...], dict[str, object]]:
    grid_spec = {
        "hidden_dim": [64, 96, 128],
        "layers": [2, 3],
        "num_heads": [4],
        "dropout": [0.1, 0.2, 0.3],
        "fusion_event_bias_weight": [0.0, 0.25, 0.5, 0.75],
        "fusion_lag_window_points": [None, 4, 8, 16],
        "fusion_normalize_states": [True],
        "learning_rate": [1e-3, 3e-4],
        "batch_size": [128, 256],
        "regression_loss": ["mse", "smooth_l1", "huber"],
        "huber_delta": [0.5, 1.0, 2.0],
        "target_transform": ["none", "zscore_train", "robust_train"],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "gradient_clip_max_norm": [1.0],
    }
    full_count = int(np.prod([len(values) for values in grid_spec.values()]))
    focused = [
        _candidate(96, 2, 0.2, 0.5, 8, 1e-3, 128, "mse", 1.0, "none", 0.0),
        _candidate(64, 2, 0.1, 0.25, 16, 1e-3, 128, "smooth_l1", 1.0, "zscore_train", 1e-5),
        _candidate(128, 3, 0.2, 0.5, 8, 3e-4, 128, "huber", 1.0, "robust_train", 1e-5),
        _candidate(96, 3, 0.1, 0.75, 16, 3e-4, 256, "smooth_l1", 0.5, "zscore_train", 1e-4),
        _candidate(64, 2, 0.3, 0.0, None, 1e-3, 256, "mse", 1.0, "none", 0.0),
        _candidate(128, 2, 0.2, 0.25, 4, 3e-4, 128, "huber", 2.0, "zscore_train", 1e-5),
        _candidate(96, 2, 0.3, 0.5, 16, 1e-3, 256, "smooth_l1", 2.0, "robust_train", 1e-4),
        _candidate(64, 3, 0.2, 0.75, 8, 3e-4, 128, "mse", 1.0, "none", 1e-5),
    ]
    if limit is None or limit <= 0:
        candidates = tuple(focused)
    else:
        candidates = tuple(focused[:limit])
    if batch_size is not None or learning_rate is not None:
        candidates = tuple(
            _with_candidate_overrides(
                candidate,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            for candidate in candidates
        )
    return candidates, {
        "grid_spec": grid_spec,
        "full_candidate_count": full_count,
        "screen_candidate_policy": "focused_midterm_refresh_subset",
        "screen_candidate_limit": limit,
        "batch_size_override": batch_size,
        "learning_rate_override": learning_rate,
        "screen_candidates": [asdict(candidate) for candidate in candidates],
    }


def _locked_public_bridge_candidate(config):
    if not config.locked_configuration_path or not config.promotion_evidence_path:
        raise PermissionError(
            "external public confirmation requires lock and promotion evidence"
        )
    lock = json.loads(
        _resolve_path(config.locked_configuration_path).read_text(encoding="utf-8")
    )
    promotion = json.loads(
        _resolve_path(config.promotion_evidence_path).read_text(encoding="utf-8")
    )
    if (
        lock.get("format") != "chronaris.v2_locked_configuration.v1"
        or lock.get("configuration_locked") is not True
        or lock.get("selection_uses_downstream_labels") is not False
    ):
        raise PermissionError("public confirmation requires a clean v2 lock")
    if promotion.get("status") != "completed" or promotion.get(
        "locked_results_returned_to_development", False
    ):
        raise PermissionError("public confirmation requires completed one-shot audit")
    source = lock["candidate"]
    candidate = _candidate(
        int(source["internal_hidden_dim"]),
        2,
        float(source["dropout"]),
        0.0,
        None,
        float(source["learning_rate"]),
        config.batch_size or 128,
        "mse",
        1.0,
        "none",
        0.0,
    )
    return candidate, {
        "screen_candidate_policy": "single_locked_v2_public_adapter_bridge",
        "screen_candidate_limit": 1,
        "screen_candidates": [asdict(candidate)],
        "locked_source_candidate_id": source["candidate_id"],
        "public_label_candidate_selection": False,
        "bridge_scope": (
            "public context stream adapter; not equivalent to Dingxin vehicle stream"
        ),
    }


def _candidate(
    hidden_dim: int,
    layers: int,
    dropout: float,
    event_bias: float,
    lag_window: int | None,
    learning_rate: float,
    batch_size: int,
    regression_loss: str,
    huber_delta: float,
    target_transform: str,
    weight_decay: float,
) -> PublicFusionRefreshCandidate:
    candidate_id = (
        f"fusion_h{hidden_dim}_l{layers}_hd4_do{_num(dropout)}"
        f"_bias{_num(event_bias)}_lag{lag_window if lag_window is not None else 'none'}"
        f"_lr{_num(learning_rate)}_bs{batch_size}_{regression_loss}"
        f"_td{_num(huber_delta)}_{target_transform}_wd{_num(weight_decay)}"
    )
    return PublicFusionRefreshCandidate(
        candidate_id=candidate_id,
        hidden_dim=hidden_dim,
        layers=layers,
        num_heads=4,
        dropout=dropout,
        fusion_event_bias_weight=event_bias,
        fusion_lag_window_points=lag_window,
        fusion_normalize_states=True,
        learning_rate=learning_rate,
        batch_size=batch_size,
        regression_loss=regression_loss,
        huber_delta=huber_delta,
        target_transform=target_transform,
        weight_decay=weight_decay,
    )


def _with_candidate_overrides(
    candidate: PublicFusionRefreshCandidate,
    *,
    batch_size: int | None,
    learning_rate: float | None,
) -> PublicFusionRefreshCandidate:
    if batch_size is None and learning_rate is None:
        return candidate
    return _candidate(
        hidden_dim=candidate.hidden_dim,
        layers=candidate.layers,
        dropout=candidate.dropout,
        event_bias=candidate.fusion_event_bias_weight,
        lag_window=candidate.fusion_lag_window_points,
        learning_rate=(
            float(learning_rate)
            if learning_rate is not None
            else candidate.learning_rate
        ),
        batch_size=int(batch_size) if batch_size is not None else candidate.batch_size,
        regression_loss=candidate.regression_loss,
        huber_delta=candidate.huber_delta,
        target_transform=candidate.target_transform,
        weight_decay=candidate.weight_decay,
    )


def _run_candidate(
    *,
    stage: str,
    artifact_root: Path,
    dataset_id: str,
    prepared_root: str,
    candidate: PublicFusionRefreshCandidate,
    epochs: int,
    max_folds: int | None,
    seed: int,
    device: str,
    train_sampling_policy: str | None,
    heartbeat_seconds: float,
    batch_log_interval: int,
    skip_completed: bool,
) -> StageIDeepBaselineRunResult:
    run_root = artifact_root / stage / dataset_id / f"{candidate.candidate_id}__seed{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    candidate_config = {
        "stage": stage,
        "dataset_id": dataset_id,
        "prepared_root": prepared_root,
        "seed": int(seed),
        "epochs": int(epochs),
        "max_folds": max_folds,
        "device": device,
        "candidate": asdict(candidate),
        "train_sampling_policy": _dataset_sampling_policy(
            dataset_id,
            override=train_sampling_policy,
        ),
        "heartbeat_seconds": float(heartbeat_seconds),
        "batch_log_interval": int(batch_log_interval),
    }
    (run_root / "config.json").write_text(
        json.dumps(candidate_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    cached = _load_completed_candidate(run_root, dataset_id=dataset_id)
    if skip_completed and cached is not None:
        LOGGER.info(
            "P28 %s candidate_skip_completed dataset=%s candidate=%s seed=%s summary=%s",
            stage,
            dataset_id,
            candidate.candidate_id,
            seed,
            cached.summary_path,
        )
        return cached
    return run_task_eval_deep_baseline(
        StageIDeepBaselineConfig(
            model_name="chronaris_public_fusion",
            dataset_id=dataset_id,
            profile="window_v2",
            prepared_artifact_root=prepared_root,
            artifact_root=str(run_root),
            epochs=epochs,
            learning_rate=candidate.learning_rate,
            batch_size=candidate.batch_size,
            hidden_dim=candidate.hidden_dim,
            num_heads=candidate.num_heads,
            layers=candidate.layers,
            dropout=candidate.dropout,
            fusion_event_bias_weight=candidate.fusion_event_bias_weight,
            fusion_lag_window_points=candidate.fusion_lag_window_points,
            fusion_normalize_states=candidate.fusion_normalize_states,
            max_folds=max_folds,
            seed=seed,
            device=device,
            train_sampling_policy=_dataset_sampling_policy(
                dataset_id,
                override=train_sampling_policy,
            ),
            regression_loss=candidate.regression_loss,
            huber_delta=candidate.huber_delta,
            target_transform=(
                candidate.target_transform if dataset_id == "uab_workload_dataset" else "none"
            ),
            gradient_clip_max_norm=candidate.gradient_clip_max_norm,
            weight_decay=candidate.weight_decay,
            heartbeat_seconds=heartbeat_seconds,
            batch_log_interval=batch_log_interval,
        )
    )


def _dataset_sampling_policy(dataset_id: str, *, override: str | None) -> str:
    if override:
        return override
    return "balanced_class" if dataset_id == "nasa_csm" else "none"


def _load_completed_candidate(
    run_root: Path,
    *,
    dataset_id: str,
) -> StageIDeepBaselineRunResult | None:
    summary_path = run_root / "deep_baseline_summary.json"
    predictions_path = run_root / "fold_predictions.csv"
    report_path = run_root / "deep_baseline_report.md"
    if not summary_path.exists() or not predictions_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return StageIDeepBaselineRunResult(
        dataset_id=dataset_id,
        model_name="chronaris_public_fusion",
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        report_path=str(report_path),
        predictions_path=str(predictions_path),
        summary=summary,
    )


def _extract_leaderboard_row(
    *,
    dataset_id: str,
    stage: str,
    candidate: PublicFusionRefreshCandidate,
    seed: int,
    summary: Mapping[str, object],
    summary_path: str,
    artifact_root: str,
) -> dict[str, object]:
    row = {
        "stage": stage,
        "dataset_id": dataset_id,
        "candidate_id": candidate.candidate_id,
        "seed": int(seed),
        "summary_path": summary_path,
        "artifact_root": artifact_root,
        **asdict(candidate),
    }
    if dataset_id == "nasa_csm":
        groups = summary["objective"]["groups"]
        combined = groups["combined"]
        row.update(
            {
                "primary_metric": "combined_macro_f1",
                "selection_score": float(combined["macro_f1"]),
                "secondary_score": float(combined["balanced_accuracy"]),
                "combined_macro_f1": float(combined["macro_f1"]),
                "combined_balanced_accuracy": float(combined["balanced_accuracy"]),
                "benchmark_only_macro_f1": float(groups["benchmark_only"]["macro_f1"]),
                "loft_only_macro_f1": float(groups["loft_only"]["macro_f1"]),
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
    mean_mae = (
        float(subjective["n_back"]["mae"])
        + float(subjective["heat_the_chair"]["mae"])
    ) / 2.0
    row.update(
        {
            "primary_metric": "mean_rmse",
            "selection_score": mean_rmse,
            "secondary_score": mean_mae,
            "mean_rmse": mean_rmse,
            "mean_mae": mean_mae,
            "n_back_rmse": float(subjective["n_back"]["rmse"]),
            "n_back_mae": float(subjective["n_back"]["mae"]),
            "heat_the_chair_rmse": float(subjective["heat_the_chair"]["rmse"]),
            "heat_the_chair_mae": float(subjective["heat_the_chair"]["mae"]),
            "n_back_macro_f1": float(objective["n_back"]["macro_f1"]),
            "heat_the_chair_macro_f1": float(objective["heat_the_chair"]["macro_f1"]),
            "sample_count": int(
                subjective["n_back"]["sample_count"]
                + subjective["heat_the_chair"]["sample_count"]
            ),
            "fold_count": int(
                max(subjective["n_back"]["fold_count"], subjective["heat_the_chair"]["fold_count"])
            ),
        }
    )
    return row


def _sort_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    pieces = []
    for dataset_id, subset in frame.groupby("dataset_id", sort=False):
        ascending = dataset_id == "uab_workload_dataset"
        pieces.append(
            subset.sort_values(
                ["selection_score", "secondary_score", "candidate_id"],
                ascending=[ascending, ascending, True],
            )
        )
    return pd.concat(pieces, axis=0, ignore_index=True)


def _select_confirm_candidates(
    screen_frame: pd.DataFrame,
    candidates: Sequence[PublicFusionRefreshCandidate],
    *,
    dataset_id: str,
    top_k: int,
) -> tuple[PublicFusionRefreshCandidate, ...]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    subset = screen_frame[screen_frame["dataset_id"] == dataset_id]
    return tuple(
        candidate_by_id[candidate_id]
        for candidate_id in subset["candidate_id"].head(max(top_k, 1)).tolist()
    )


def _extract_fold_metrics(
    predictions_path: str,
    *,
    stage: str,
    dataset_id: str,
    candidate_id: str,
    seed: int,
) -> list[dict[str, object]]:
    frame = pd.read_csv(predictions_path)
    rows: list[dict[str, object]] = []
    for (track, group, split_group), fold in frame.groupby(
        ["track", "evaluation_group", "split_group"],
        sort=False,
    ):
        if track == "objective":
            labels = sorted(set(fold["y_true"].astype(int)) | set(fold["y_pred"].astype(int)))
            metrics = evaluate_classification_predictions(fold, label_order=labels)
            metric_payload = {
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        else:
            metrics = evaluate_regression_predictions(fold)
            metric_payload = {"rmse": metrics["rmse"], "mae": metrics["mae"]}
        rows.append(
            {
                "stage": stage,
                "dataset_id": dataset_id,
                "candidate_id": candidate_id,
                "seed": int(seed),
                "track": track,
                "evaluation_group": group,
                "split_group": split_group,
                "sample_count": int(metrics["sample_count"]),
                **metric_payload,
            }
        )
    return rows


def _write_candidate_fold_metrics(
    *,
    result: StageIDeepBaselineRunResult,
    stage: str,
    dataset_id: str,
    candidate_id: str,
    seed: int,
) -> list[dict[str, object]]:
    rows = _extract_fold_metrics(
        result.predictions_path,
        stage=stage,
        dataset_id=dataset_id,
        candidate_id=candidate_id,
        seed=seed,
    )
    pd.DataFrame(rows).to_csv(Path(result.artifact_root) / "fold_metrics.csv", index=False)
    return rows


def _write_partial_tables(
    artifact_root: Path,
    *,
    screen_rows: Sequence[Mapping[str, object]],
    confirm_rows: Sequence[Mapping[str, object]],
    fold_metric_rows: Sequence[Mapping[str, object]],
    training_curve_rows: Sequence[Mapping[str, object]],
) -> None:
    if screen_rows:
        _sort_leaderboard(pd.DataFrame(screen_rows)).to_csv(
            artifact_root / "screen_leaderboard.partial.csv",
            index=False,
        )
    if confirm_rows:
        _sort_leaderboard(pd.DataFrame(confirm_rows)).to_csv(
            artifact_root / "confirm_leaderboard.partial.csv",
            index=False,
        )
    if fold_metric_rows:
        pd.DataFrame(fold_metric_rows).to_csv(
            artifact_root / "fold_metrics.partial.csv",
            index=False,
        )
    if training_curve_rows:
        pd.DataFrame(training_curve_rows).to_csv(
            artifact_root / "training_curves.partial.csv",
            index=False,
        )


def _read_training_curves(
    summary: Mapping[str, object],
    *,
    stage: str,
    dataset_id: str,
    candidate_id: str,
    seed: int,
) -> list[dict[str, object]]:
    path = summary.get("training_curves_path")
    if not path or not Path(path).exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    frame["stage"] = stage
    frame["dataset_id"] = dataset_id
    frame["candidate_id"] = candidate_id
    frame["seed"] = int(seed)
    return frame.to_dict(orient="records")


def _build_best_by_dataset_task(confirm_frame: pd.DataFrame) -> dict[str, object]:
    best: dict[str, object] = {}
    if confirm_frame.empty:
        return best
    for dataset_id, subset in confirm_frame.groupby("dataset_id", sort=False):
        ordered = _sort_leaderboard(subset).reset_index(drop=True)
        row = ordered.iloc[0].to_dict()
        summary = json.loads(Path(row["summary_path"]).read_text(encoding="utf-8"))
        if dataset_id == "nasa_csm":
            best[dataset_id] = {}
            for group, metrics in summary["objective"]["groups"].items():
                best[dataset_id][group] = {
                    "candidate_id": row["candidate_id"],
                    "seed": int(row["seed"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "sample_count": int(metrics["sample_count"]),
                    "fold_count": int(metrics["fold_count"]),
                    "summary_path": row["summary_path"],
                    "artifact_root": row["artifact_root"],
                    "protocol": "p28_refresh_full_loso_confirm"
                    if _is_full_confirm(metrics)
                    else "p28_refresh_bounded_confirm",
                    "confusion_matrix_plot": (
                        summary["objective"].get("plot_paths", {}).get(
                            f"objective_{group}_confusion_matrix"
                        )
                    ),
                }
        else:
            best[dataset_id] = {}
            subjective = summary["subjective"]["groups"]
            objective = summary["objective"]["groups"]
            for group, metrics in subjective.items():
                objective_metrics = objective.get(group, {})
                best[dataset_id][group] = {
                    "candidate_id": row["candidate_id"],
                    "seed": int(row["seed"]),
                    "rmse": float(metrics["rmse"]),
                    "mae": float(metrics["mae"]),
                    "sample_count": int(metrics["sample_count"]),
                    "fold_count": int(metrics["fold_count"]),
                    "objective_macro_f1": float(objective_metrics.get("macro_f1", float("nan"))),
                    "objective_balanced_accuracy": float(
                        objective_metrics.get("balanced_accuracy", float("nan"))
                    ),
                    "objective_sample_count": int(objective_metrics.get("sample_count", 0)),
                    "objective_fold_count": int(objective_metrics.get("fold_count", 0)),
                    "summary_path": row["summary_path"],
                    "artifact_root": row["artifact_root"],
                    "protocol": "p28_refresh_full_loso_confirm"
                    if _is_full_confirm(metrics)
                    else "p28_refresh_bounded_confirm",
                }
    return best


def _render_refresh_figures(
    *,
    artifact_root: Path,
    screen_frame: pd.DataFrame,
    confirm_frame: pd.DataFrame,
    training_curves: pd.DataFrame,
    best_by_dataset_task: Mapping[str, object],
) -> dict[str, str]:
    figure_paths = {
        "fig_public_fusion_refresh_screen_leaderboard": str(
            artifact_root / "fig_public_fusion_refresh_screen_leaderboard.png"
        ),
        "fig_public_fusion_refresh_confirm_vs_baselines": str(
            artifact_root / "fig_public_fusion_refresh_confirm_vs_baselines.png"
        ),
        "fig_public_fusion_refresh_delta_heatmap": str(
            artifact_root / "fig_public_fusion_refresh_delta_heatmap.png"
        ),
        "fig_public_fusion_config_sensitivity": str(
            artifact_root / "fig_public_fusion_config_sensitivity.png"
        ),
        "fig_public_fusion_training_curves_best": str(
            artifact_root / "fig_public_fusion_training_curves_best.png"
        ),
        "fig_public_fusion_best_confusion_nasa_combined": str(
            artifact_root / "fig_public_fusion_best_confusion_nasa_combined.png"
        ),
        "fig_public_fusion_win_summary": str(
            artifact_root / "fig_public_fusion_win_summary.png"
        ),
    }
    _plot_screen_leaderboard(screen_frame, figure_paths["fig_public_fusion_refresh_screen_leaderboard"])
    _plot_confirm_vs_baselines(
        best_by_dataset_task,
        figure_paths["fig_public_fusion_refresh_confirm_vs_baselines"],
    )
    _plot_refresh_delta_heatmap(
        best_by_dataset_task,
        figure_paths["fig_public_fusion_refresh_delta_heatmap"],
    )
    _plot_config_sensitivity(
        screen_frame,
        figure_paths["fig_public_fusion_config_sensitivity"],
    )
    _plot_training_curves(
        training_curves,
        confirm_frame,
        figure_paths["fig_public_fusion_training_curves_best"],
    )
    _copy_best_nasa_confusion(
        best_by_dataset_task,
        figure_paths["fig_public_fusion_best_confusion_nasa_combined"],
    )
    _plot_refresh_win_summary(
        best_by_dataset_task,
        figure_paths["fig_public_fusion_win_summary"],
    )
    return figure_paths


def _plot_screen_leaderboard(frame: pd.DataFrame, path: str) -> None:
    if frame.empty or "dataset_id" not in frame:
        fig, axis = plt.subplots(figsize=(6, 3))
        axis.text(0.5, 0.5, "no screen rows", ha="center", va="center")
        axis.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, max(frame["dataset_id"].nunique(), 1), figsize=(12, 5), squeeze=False)
    for axis, (dataset_id, subset) in zip(axes[0], frame.groupby("dataset_id", sort=False), strict=False):
        top = subset.head(10).copy()
        labels = top["candidate_id"].str.replace("fusion_", "", regex=False)
        values = top["selection_score"].astype(float)
        axis.barh(np.arange(len(top)), values, color="#2f6f9f")
        axis.set_yticks(np.arange(len(top)))
        axis.set_yticklabels(labels, fontsize=7)
        axis.invert_yaxis()
        axis.set_title(f"{dataset_id} screen top candidates")
        axis.set_xlabel(str(top["primary_metric"].iloc[0]) if not top.empty else "score")
        for index, value in enumerate(values):
            axis.text(value, index, f"{value:.4f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_confirm_vs_baselines(best: Mapping[str, object], path: str) -> None:
    rows: list[tuple[str, list[tuple[str, float]]]] = []
    nasa = ((best.get("nasa_csm") or {}).get("combined") or {})
    if nasa:
        rows.append(
            (
                "NASA combined macro-F1",
                [
                    ("public", BASELINES["nasa_csm"]["combined"]["macro_f1"]["public_baseline"]),
                    ("MulT", BASELINES["nasa_csm"]["combined"]["macro_f1"]["mult"]),
                    ("ContiFormer", BASELINES["nasa_csm"]["combined"]["macro_f1"]["contiformer"]),
                    ("Chronaris refresh", float(nasa["macro_f1"])),
                ],
            )
        )
    uab = best.get("uab_workload_dataset") or {}
    for group in ("n_back", "heat_the_chair"):
        if group in uab:
            rows.append(
                (
                    f"UAB {group} RMSE",
                    [
                        ("public", BASELINES["uab_workload_dataset"][group]["rmse"]["public_baseline"]),
                        ("MulT", BASELINES["uab_workload_dataset"][group]["rmse"]["mult"]),
                        ("ContiFormer", BASELINES["uab_workload_dataset"][group]["rmse"]["contiformer"]),
                        ("Chronaris refresh", float(uab[group]["rmse"])),
                    ],
                )
            )
    if not rows:
        fig, axis = plt.subplots(figsize=(6, 3))
        axis.text(0.5, 0.5, "no confirm rows", ha="center", va="center")
        axis.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(8, max(4, len(rows) * 2.6)), squeeze=False)
    for axis, (title, values) in zip(axes[:, 0], rows, strict=False):
        x = np.arange(len(values))
        bars = axis.bar(x, [value for _, value in values], color=["#7f8790", "#879f6a", "#9f7a66", "#1f7a5a"])
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels([label for label, _ in values], rotation=15, ha="right")
        for bar, (_, value) in zip(bars, values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_refresh_delta_heatmap(best: Mapping[str, object], path: str) -> None:
    labels: list[str] = []
    rows: list[list[float]] = []
    columns = ["vs public", "vs MulT", "vs ContiFormer", "vs classical"]
    nasa = ((best.get("nasa_csm") or {}).get("combined") or {})
    if nasa:
        labels.append("NASA combined macro-F1")
        value = float(nasa["macro_f1"])
        base = BASELINES["nasa_csm"]["combined"]["macro_f1"]
        rows.append(
            [
                value - base["public_baseline"],
                value - base["mult"],
                value - base["contiformer"],
                value - base["classical_baseline"],
            ]
        )
    uab = best.get("uab_workload_dataset") or {}
    for group in ("n_back", "heat_the_chair"):
        if group in uab:
            labels.append(f"UAB {group} RMSE")
            value = float(uab[group]["rmse"])
            base = BASELINES["uab_workload_dataset"][group]["rmse"]
            rows.append(
                [
                    base["public_baseline"] - value,
                    base["mult"] - value,
                    base["contiformer"] - value,
                    base["classical_baseline"] - value,
                ]
            )
    if not rows:
        labels = ["no confirm rows"]
    data = np.asarray(rows or [[0.0] * len(columns)], dtype=float)
    fig, axis = plt.subplots(figsize=(8, max(3, len(labels) * 0.7)))
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    image = axis.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    axis.set_title("P28 refresh improvement over baselines")
    axis.set_xticks(np.arange(len(columns)))
    axis.set_xticklabels(columns, rotation=15, ha="right")
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels)
    for row_index in range(data.shape[0]):
        for col_index in range(data.shape[1]):
            axis.text(col_index, row_index, f"{data[row_index, col_index]:+.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_config_sensitivity(frame: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fields = ["hidden_dim", "dropout", "fusion_event_bias_weight", "fusion_lag_window_points"]
    for axis, field in zip(axes.ravel(), fields, strict=True):
        if frame.empty:
            axis.text(0.5, 0.5, "no screen rows", ha="center", va="center")
            axis.axis("off")
            continue
        values = frame[field].fillna(-1).astype(float)
        axis.scatter(values, frame["selection_score"].astype(float), alpha=0.8)
        axis.set_xlabel(field)
        axis.set_ylabel("selection_score")
    fig.suptitle("Public fusion config sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_training_curves(training_curves: pd.DataFrame, confirm_frame: pd.DataFrame, path: str) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.5))
    if not training_curves.empty and not confirm_frame.empty:
        for _, row in confirm_frame.head(4).iterrows():
            subset = training_curves[
                (training_curves["stage"] == "confirm")
                & (training_curves["dataset_id"] == row["dataset_id"])
                & (training_curves["candidate_id"] == row["candidate_id"])
                & (training_curves["seed"] == row["seed"])
            ]
            if subset.empty:
                continue
            grouped = subset.groupby("epoch", sort=True)["train_loss"].mean()
            axis.plot(grouped.index, grouped.values, marker="o", label=f"{row['dataset_id']} {row['candidate_id'][:18]}")
    axis.set_title("Best confirm training loss curves")
    axis.set_xlabel("epoch")
    axis.set_ylabel("train loss")
    axis.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _copy_best_nasa_confusion(best: Mapping[str, object], path: str) -> None:
    source = ((best.get("nasa_csm") or {}).get("combined") or {}).get("confusion_matrix_plot")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if source and Path(source).exists():
        shutil.copyfile(source, output)
        return
    fig, axis = plt.subplots(figsize=(4, 3))
    axis.text(0.5, 0.5, "NASA combined confusion matrix unavailable", ha="center", va="center")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_refresh_win_summary(best: Mapping[str, object], path: str) -> None:
    records: list[dict[str, object]] = []
    nasa = ((best.get("nasa_csm") or {}).get("combined") or {})
    if nasa:
        value = float(nasa["macro_f1"])
        for label, baseline in BASELINES["nasa_csm"]["combined"]["macro_f1"].items():
            delta = value - float(baseline)
            records.append(
                {
                    "row": "NASA combined macro-F1",
                    "baseline": label.replace("_baseline", ""),
                    "status": _status_from_delta(delta, threshold=0.02),
                    "delta": delta,
                }
            )
    uab = best.get("uab_workload_dataset") or {}
    for group in ("n_back", "heat_the_chair"):
        metrics = uab.get(group) or {}
        if not metrics:
            continue
        value = float(metrics["rmse"])
        for label, baseline in BASELINES["uab_workload_dataset"][group]["rmse"].items():
            delta = float(baseline) - value
            rel = abs(delta) / abs(float(baseline)) if abs(float(baseline)) > 1e-12 else 0.0
            records.append(
                {
                    "row": f"UAB {group} RMSE",
                    "baseline": label.replace("_baseline", ""),
                    "status": "T" if rel < 0.02 else ("W" if delta > 0 else "L"),
                    "delta": delta,
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(
            [{"row": "no confirm rows", "baseline": "none", "status": "T", "delta": 0.0}]
        )
    rows = list(dict.fromkeys(frame["row"]))
    baselines = list(dict.fromkeys(frame["baseline"]))
    status_to_num = {"L": -1, "T": 0, "W": 1}
    data = np.full((len(rows), len(baselines)), np.nan)
    lookup = {
        (record["row"], record["baseline"]): record
        for record in frame.to_dict(orient="records")
    }
    for row_index, row_label in enumerate(rows):
        for col_index, baseline in enumerate(baselines):
            record = lookup.get((row_label, baseline))
            if record:
                data[row_index, col_index] = status_to_num[record["status"]]
    fig, axis = plt.subplots(figsize=(8, max(3.5, 0.55 * len(rows))))
    image = axis.imshow(data, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axis.set_title("P28 Chronaris W/T/L summary")
    axis.set_xticks(np.arange(len(baselines)))
    axis.set_xticklabels(baselines, rotation=15, ha="right")
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(rows)
    for row_index, row_label in enumerate(rows):
        for col_index, baseline in enumerate(baselines):
            record = lookup.get((row_label, baseline))
            if not record:
                continue
            axis.text(
                col_index,
                row_index,
                f"{record['status']}\n{float(record['delta']):+.4f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _status_from_delta(delta: float, *, threshold: float) -> str:
    if abs(delta) < threshold:
        return "T"
    return "W" if delta > 0 else "L"


def _infer_refresh_status(
    *,
    selected_datasets: Sequence[str],
    screen_only: bool,
    confirm_frame: pd.DataFrame,
    missing_figures: Sequence[Mapping[str, object]],
) -> str:
    if screen_only:
        return "partial"
    if confirm_frame.empty:
        return "partial"
    confirmed = set(confirm_frame["dataset_id"].astype(str).tolist())
    if any(dataset_id not in confirmed for dataset_id in selected_datasets):
        return "partial"
    if "nasa_csm" in selected_datasets:
        nasa = confirm_frame[confirm_frame["dataset_id"] == "nasa_csm"]
        if nasa.empty or int(nasa["fold_count"].max()) < 16:
            return "partial"
    blocking_missing = [
        item
        for item in missing_figures
        if item.get("figure") in {"fig_public_fusion_refresh_confirm_vs_baselines"}
    ]
    if blocking_missing:
        return "partial"
    return "completed"


def _completed_fold_count(rows: Sequence[Mapping[str, object]]) -> int:
    if not rows:
        return 0
    keys = {
        (
            row.get("stage"),
            row.get("dataset_id"),
            row.get("candidate_id"),
            row.get("seed"),
            row.get("track"),
            row.get("evaluation_group"),
            row.get("split_group"),
        )
        for row in rows
    }
    return len(keys)


def _render_refresh_report(run_id: str, summary: Mapping[str, object], confirm_frame: pd.DataFrame) -> str:
    best = summary.get("best_by_dataset_task") or {}
    nasa = ((best.get("nasa_csm") or {}).get("combined") or {})
    uab = best.get("uab_workload_dataset") or {}
    lines = [
        f"# task evaluation Public Fusion Refresh - {run_id}",
        "",
        "## Executive Summary",
        "",
    ]
    if nasa:
        value = float(nasa["macro_f1"])
        public = BASELINES["nasa_csm"]["combined"]["macro_f1"]["public_baseline"]
        delta = value - public
        rel = delta / abs(public) * 100.0
        lines.append(
            "P28 refresh enlarged chronaris_public_fusion from the previous light screen "
            f"setting to confirm runs. On NASA combined attention-state classification, "
            f"the best refreshed chronaris_public_fusion achieved macro-F1={value:.4f}, "
            f"outperforming the public baseline by {delta:+.4f} absolute / {rel:+.1f}% relative."
        )
    if "n_back" in uab and "heat_the_chair" in uab:
        mean_rmse = (float(uab["n_back"]["rmse"]) + float(uab["heat_the_chair"]["rmse"])) / 2.0
        lines.append(
            f"On UAB subjective workload regression, the best refreshed candidate achieved "
            f"mean RMSE={mean_rmse:.4f}, with per-task RMSE "
            f"{float(uab['n_back']['rmse']):.4f} / {float(uab['heat_the_chair']['rmse']):.4f}."
        )
    lines.extend(
        [
            "",
            "## Leaderboards",
            "",
            f"- status: `{summary.get('status', 'unknown')}`",
            f"- screen_leaderboard: `{summary['screen_leaderboard_csv']}`",
            f"- confirm_leaderboard: `{summary['confirm_leaderboard_csv']}`",
            f"- fold_metrics: `{summary['fold_metrics_csv']}`",
            f"- training_curves: `{summary['training_curves_csv']}`",
            f"- run_log: `{summary.get('run_log_path', '')}`",
            f"- progress: `{summary.get('progress_path', '')}`",
            "",
            "## Best confirm rows",
            "",
            _markdown_table(
                confirm_frame.head(20),
                [
                    "dataset_id",
                    "candidate_id",
                    "seed",
                    "primary_metric",
                    "selection_score",
                    "secondary_score",
                    "fold_count",
                    "summary_path",
                ],
            ),
            "",
            "## Figure index",
            "",
        ]
    )
    for name, path in (summary.get("figure_paths") or {}).items():
        lines.append(f"- `{name}`: `{path}`")
    missing_figures = summary.get("missing_figures") or []
    if missing_figures:
        lines.extend(["", "Missing figures:"])
        for item in missing_figures:
            lines.append(
                f"- `{item.get('figure')}`: {item.get('reason')} ({item.get('path')})"
            )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- config: `{summary['config_path']}`",
            f"- candidate_grid: `{summary['candidate_grid_path']}`",
            f"- evidence_manifest: `{summary['evidence_manifest_path']}`",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame[columns].iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, float):
                cells.append(f"{value:.4f}" if math.isfinite(value) else "")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _config_dict(config: StageIPublicFusionRefreshConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["dataset_prepared_roots"] = dict(config.dataset_prepared_roots)
    payload["datasets"] = list(config.datasets)
    payload["confirm_seeds"] = list(config.confirm_seeds)
    return payload


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _num(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _is_full_confirm(metrics: Mapping[str, object]) -> bool:
    return int(metrics.get("fold_count", 0)) >= 16
