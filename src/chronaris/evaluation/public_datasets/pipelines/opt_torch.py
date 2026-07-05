"""Torch-native GPU-first UAB regression runner for task evaluation public-opt."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from chronaris.evaluation.public_datasets.pipelines.opt_data import (
    PUBLIC_OPT_PROFILE,
    build_task_eval_public_opt_feature_frame,
)
from chronaris.evaluation.public_datasets.pipelines.opt_reference import (
    load_public_opt_prepared_dataset,
)
from chronaris.modeling.common.run_observer import (
    StageIRunProgress,
    open_task_eval_run_observer,
)
from chronaris.evaluation.public_datasets.pipelines.opt_torch_catalog import (
    TorchUABCandidateSpec,
    build_torch_uab_candidates,
    is_better_torch_screen_row,
    validate_torch_uab_config,
)
from chronaris.evaluation.public_datasets.pipelines.opt_torch_candidates import (
    _build_feature_profile_bundle,
    _run_torch_uab_candidate,
)
from chronaris.evaluation.public_datasets.pipelines.opt_torch_reporting import (
    build_torch_uab_acceptance,
    load_torch_uab_reference_deep_summary,
    load_torch_uab_reference_public_opt_summary,
    render_torch_uab_report,
)
from chronaris.evaluation.public_datasets.pipelines.opt_torch_selection import (
    _build_torch_uab_full_shortlist_rows,
    _group_metric_or_nan,
    _normalize_torch_uab_selected_subsets,
    _select_terminal_torch_uab_result,
    _validate_torch_uab_runtime_config,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name

UAB_TORCH_DATASET_ID = "uab_workload_dataset"
UAB_TORCH_ACCEPTANCE_THRESHOLDS = {
    "n_back": 4.6541,
    "heat_the_chair": 1.4568,
}
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class StageIPublicOptTorchUABConfig:
    run_id: str
    prepared_artifact_root: str
    artifact_root: str = "docs/artifacts/runs"
    report_root: str = "docs/artifacts"
    dataset_id: str = UAB_TORCH_DATASET_ID
    profile: str = PUBLIC_OPT_PROFILE
    device: str = "auto"
    seed: int = 42
    batch_size: int = 256
    epochs: int = 20
    patience: int = 4
    screen_max_folds: int | None = 2
    full_max_folds: int | None = None
    run_full_loso: bool = True
    full_candidate_limit: int = 2
    full_group_winner_limit: int = 1
    ensemble_policy: str = "none"
    prediction_aggregation_policy: str = "none"
    supervision_granularity: str = "window"
    require_cuda: bool = False
    candidate_catalog: str = "default"
    selected_subsets: tuple[str, ...] = ("n_back", "heat_the_chair")
    learning_rates: tuple[float, ...] = (1e-3, 3e-4)
    weight_decays: tuple[float, ...] = (1e-4, 1e-3)
    feature_profiles: tuple[str, ...] = (
        "full",
        "residual_only",
        "physiology_only",
        "physiology_lowdim",
        "physiology_scalar_only",
    )
    reference_public_opt_summary_path: str | None = None
    reference_deep_comparison_summary_path: str | None = None


@dataclass(frozen=True, slots=True)
class StageIPublicOptTorchUABRunResult:
    run_id: str
    artifact_root: str
    feature_frame_path: str
    predictions_path: str
    summary_path: str
    report_path: str
    summary: Mapping[str, object]


def run_task_eval_public_opt_torch_uab(
    config: StageIPublicOptTorchUABConfig,
) -> StageIPublicOptTorchUABRunResult:
    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="task_eval_public_opt_torch",
        logger=LOGGER,
        initial_progress={
            "dataset_id": config.dataset_id,
            "prepared_artifact_root": str(Path(config.prepared_artifact_root)),
            "artifact_root": str(run_root),
            "requested_device": config.device,
            "require_cuda": config.require_cuda,
            "candidate_catalog": config.candidate_catalog,
            "selected_subsets": list(config.selected_subsets),
        },
    ) as progress:
        return _run_task_eval_public_opt_torch_uab_observed(
            config=config,
            run_root=run_root,
            progress=progress,
        )


def _run_task_eval_public_opt_torch_uab_observed(
    *,
    config: StageIPublicOptTorchUABConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageIPublicOptTorchUABRunResult:
    validate_torch_uab_config(
        dataset_id=config.dataset_id,
        expected_dataset_id=UAB_TORCH_DATASET_ID,
        profile=config.profile,
        expected_profile=PUBLIC_OPT_PROFILE,
        feature_profiles=config.feature_profiles,
        learning_rates=config.learning_rates,
        weight_decays=config.weight_decays,
        candidate_catalog=config.candidate_catalog,
    )
    _validate_torch_uab_runtime_config(config)
    selected_subsets = _normalize_torch_uab_selected_subsets(config.selected_subsets)
    runtime_device = resolve_torch_device_name(config.device)
    progress.update(
        "device_resolved",
        runtime_device=runtime_device,
        requested_device=config.device,
    )
    LOGGER.info(
        "task_eval_public_opt_torch start run_id=%s requested_device=%s resolved_device=%s",
        config.run_id,
        config.device,
        runtime_device,
    )
    if runtime_device != "cuda":
        if config.require_cuda:
            raise RuntimeError(
                "task_eval_public_opt_torch requires CUDA but resolved runtime_device="
                f"{runtime_device}. Use --device cuda on a CUDA host or pass "
                "--allow-cpu-debug for explicit debugging only."
            )
        LOGGER.warning(
            "task_eval_public_opt_torch using CPU fallback run_id=%s requested_device=%s",
            config.run_id,
            config.device,
        )

    prepared = load_public_opt_prepared_dataset(config.prepared_artifact_root)
    if prepared["dataset_id"] != UAB_TORCH_DATASET_ID:
        raise ValueError(
            f"prepared dataset mismatch: expected {UAB_TORCH_DATASET_ID}, got {prepared['dataset_id']}"
        )
    feature_result = build_task_eval_public_opt_feature_frame(
        prepared["entries"],
        prepared["bundle"],
        dataset_id=UAB_TORCH_DATASET_ID,
        profile=config.profile,
    )
    if feature_result.track != "subjective":
        raise ValueError("torch UAB runner only supports subjective regression track.")
    LOGGER.info(
        "task_eval_public_opt_torch prepared dataset_id=%s samples=%d subsets=%s feature_profiles=%s",
        prepared["dataset_id"],
        len(feature_result.feature_frame),
        sorted(feature_result.feature_frame["subset_id"].astype(str).unique()),
        config.feature_profiles,
    )
    progress.update(
        "feature_frame_ready",
        feature_frame_shape=list(feature_result.feature_frame.shape),
        subsets=sorted(feature_result.feature_frame["subset_id"].astype(str).unique()),
    )

    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    feature_frame_path = run_root / "public_opt_torch_feature_frame.parquet"
    predictions_path = run_root / "public_opt_torch_predictions.csv"
    summary_path = run_root / "public_opt_torch_summary.json"
    report_path = report_root / f"task-eval-public-opt-{config.run_id}.md"
    progress.update(
        "output_paths_ready",
        feature_frame_path=str(feature_frame_path),
        predictions_path=str(predictions_path),
        summary_path=str(summary_path),
        report_path=str(report_path),
    )

    feature_result.feature_frame.to_parquet(feature_frame_path, index=False)
    candidates = build_torch_uab_candidates(
        feature_profiles=config.feature_profiles,
        learning_rates=config.learning_rates,
        weight_decays=config.weight_decays,
        candidate_catalog=config.candidate_catalog,
    )
    progress.update("candidate_catalog_ready", candidate_count=len(candidates))
    feature_bundles_by_profile = {
        feature_profile: _build_feature_profile_bundle(
            feature_result,
            feature_profile=feature_profile,
        )
        for feature_profile in sorted({candidate.feature_profile for candidate in candidates})
    }

    screen_rows: list[dict[str, object]] = []
    screen_candidate_details: dict[str, object] = {}
    best_candidate: TorchUABCandidateSpec | None = None
    best_screen_result: tuple[dict[str, object], pd.DataFrame] | None = None
    best_screen_row: dict[str, object] | None = None
    for candidate_index, candidate in enumerate(candidates, start=1):
        LOGGER.info(
            "task_eval_public_opt_torch screen candidate %d/%d candidate_id=%s",
            candidate_index,
            len(candidates),
            candidate.candidate_id,
        )
        progress.update(
            "candidate_start",
            candidate_index=candidate_index,
            candidate_count=len(candidates),
            candidate=candidate.candidate_id,
            model_family=candidate.model_family,
            feature_profile=candidate.feature_profile,
        )
        candidate_bundle = feature_bundles_by_profile[candidate.feature_profile]
        subset_result, predictions = _run_torch_uab_candidate(
            feature_bundle=candidate_bundle,
            candidate=candidate,
            selected_subsets=selected_subsets,
            seed=config.seed,
            device=runtime_device,
            batch_size=config.batch_size,
            epochs=config.epochs,
            patience=config.patience,
            max_folds=config.screen_max_folds,
            prediction_aggregation_policy=config.prediction_aggregation_policy,
            supervision_granularity=config.supervision_granularity,
            progress=progress,
        )
        screen_candidate_details[candidate.candidate_id] = subset_result
        row = {
            "candidate_id": candidate.candidate_id,
            "model_family": candidate.model_family,
            "feature_profile": candidate.feature_profile,
            "learning_rate": candidate.learning_rate,
            "weight_decay": candidate.weight_decay,
            "dropout": candidate.dropout,
            "hidden_dims": list(candidate.hidden_dims),
            "screen_mean_rmse": subset_result["mean_rmse"],
            "screen_mean_mae": subset_result["mean_mae"],
            "n_back_rmse": _group_metric_or_nan(subset_result["groups"], "n_back", "rmse"),
            "n_back_mae": _group_metric_or_nan(subset_result["groups"], "n_back", "mae"),
            "heat_the_chair_rmse": _group_metric_or_nan(
                subset_result["groups"],
                "heat_the_chair",
                "rmse",
            ),
            "heat_the_chair_mae": _group_metric_or_nan(
                subset_result["groups"],
                "heat_the_chair",
                "mae",
            ),
        }
        screen_rows.append(row)
        LOGGER.info(
            "task_eval_public_opt_torch screen candidate done candidate_id=%s mean_rmse=%.4f mean_mae=%.4f",
            candidate.candidate_id,
            float(subset_result["mean_rmse"]),
            float(subset_result["mean_mae"]),
        )
        progress.update(
            "candidate_done",
            candidate=candidate.candidate_id,
            mean_rmse=float(subset_result["mean_rmse"]),
            mean_mae=float(subset_result["mean_mae"]),
        )
        if best_screen_row is None or is_better_torch_screen_row(row, best_screen_row):
            best_candidate = candidate
            best_screen_result = (subset_result, predictions)
            best_screen_row = row

    if best_candidate is None or best_screen_result is None:
        raise ValueError("torch UAB screen produced no candidate results.")

    leaderboard = pd.DataFrame(screen_rows).sort_values(
        ["screen_mean_rmse", "screen_mean_mae", "candidate_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    leaderboard_path = run_root / "candidate_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    LOGGER.info(
        "task_eval_public_opt_torch screen complete winner=%s leaderboard_path=%s",
        best_candidate.candidate_id,
        leaderboard_path,
    )

    selected_result = best_screen_result[0]
    selected_predictions = best_screen_result[1]
    full_run_completed = False
    full_candidate_results: dict[str, dict[str, object]] = {}
    full_candidate_predictions: dict[str, pd.DataFrame] = {}
    if config.run_full_loso:
        shortlist_rows = _build_torch_uab_full_shortlist_rows(
            leaderboard=leaderboard,
            full_candidate_limit=config.full_candidate_limit,
            group_winner_limit=config.full_group_winner_limit,
            ensemble_policy=config.ensemble_policy,
            selected_subsets=selected_subsets,
            candidate_catalog=config.candidate_catalog,
        )
        candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        for shortlist_index, row in enumerate(shortlist_rows, start=1):
            candidate_id = str(row["candidate_id"])
            candidate = candidates_by_id[candidate_id]
            LOGGER.info(
                "task_eval_public_opt_torch full LOSO candidate %d/%d candidate_id=%s",
                shortlist_index,
                len(shortlist_rows),
                candidate_id,
            )
            progress.update(
                "full_candidate_start",
                candidate_index=shortlist_index,
                candidate_count=len(shortlist_rows),
                candidate=candidate_id,
            )
            full_bundle = feature_bundles_by_profile[candidate.feature_profile]
            candidate_result, candidate_predictions = _run_torch_uab_candidate(
                feature_bundle=full_bundle,
                candidate=candidate,
                selected_subsets=selected_subsets,
                seed=config.seed,
                device=runtime_device,
                batch_size=config.batch_size,
                epochs=config.epochs,
                patience=config.patience,
                max_folds=config.full_max_folds,
                prediction_aggregation_policy=config.prediction_aggregation_policy,
                supervision_granularity=config.supervision_granularity,
                progress=progress,
            )
            full_candidate_results[candidate_id] = candidate_result
            full_candidate_predictions[candidate_id] = candidate_predictions
            LOGGER.info(
                "task_eval_public_opt_torch full LOSO done candidate_id=%s mean_rmse=%.4f mean_mae=%.4f",
                candidate_id,
                float(candidate_result["mean_rmse"]),
                float(candidate_result["mean_mae"]),
            )
            progress.update(
                "full_candidate_done",
                candidate=candidate_id,
                mean_rmse=float(candidate_result["mean_rmse"]),
                mean_mae=float(candidate_result["mean_mae"]),
            )
        selected_result, selected_predictions = _select_terminal_torch_uab_result(
            candidate_results=full_candidate_results,
            candidate_predictions=full_candidate_predictions,
            ensemble_policy=config.ensemble_policy,
            selected_subsets=selected_subsets,
        )
        full_run_completed = True
        LOGGER.info(
            "task_eval_public_opt_torch selected selection run_id=%s selection=%s",
            config.run_id,
            selected_result["selection_policy"],
        )

    selected_predictions.to_csv(predictions_path, index=False)
    reference_public_opt = load_torch_uab_reference_public_opt_summary(
        config.reference_public_opt_summary_path
    )
    reference_deep = load_torch_uab_reference_deep_summary(
        config.reference_deep_comparison_summary_path,
        dataset_id=UAB_TORCH_DATASET_ID,
    )
    acceptance = build_torch_uab_acceptance(
        selected_result["groups"],
        thresholds=UAB_TORCH_ACCEPTANCE_THRESHOLDS,
    )
    public_mainline_status = (
        "UAB closed"
        if acceptance["all_passed"]
        else "NASA closed, UAB partial"
    )
    summary = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "run_id": config.run_id,
        "dataset_id": UAB_TORCH_DATASET_ID,
        "profile": config.profile,
            "runtime_device": runtime_device,
            "prepared_artifact_root": str(Path(config.prepared_artifact_root)),
            "artifact_root": str(run_root),
            "feature_frame_path": str(feature_frame_path),
            "leaderboard_path": str(leaderboard_path),
            "predictions_path": str(predictions_path),
            "run_log_path": str(run_root / "run.log"),
            "progress_path": str(run_root / "progress.json"),
            "screen_config": {
            "screen_max_folds": config.screen_max_folds,
            "full_max_folds": config.full_max_folds,
            "run_full_loso": config.run_full_loso,
            "full_candidate_limit": config.full_candidate_limit,
            "full_group_winner_limit": config.full_group_winner_limit,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "patience": config.patience,
            "learning_rates": list(config.learning_rates),
            "weight_decays": list(config.weight_decays),
                "feature_profiles": list(config.feature_profiles),
                "device": config.device,
                "require_cuda": config.require_cuda,
                "candidate_catalog": config.candidate_catalog,
                "selected_subsets": list(selected_subsets),
                "ensemble_policy": config.ensemble_policy,
                "prediction_aggregation_policy": config.prediction_aggregation_policy,
                "supervision_granularity": config.supervision_granularity,
        },
        "candidate_count": len(candidates),
        "screen_leaderboard": leaderboard.to_dict(orient="records"),
        "screen_candidate_details": screen_candidate_details,
        "winning_candidate": {
            "candidate_id": best_candidate.candidate_id,
            "model_family": best_candidate.model_family,
            "feature_profile": best_candidate.feature_profile,
            "hidden_dims": list(best_candidate.hidden_dims),
            "dropout": best_candidate.dropout,
            "learning_rate": best_candidate.learning_rate,
            "weight_decay": best_candidate.weight_decay,
        },
        "full_run_completed": full_run_completed,
        "full_candidate_details": full_candidate_results,
        "selected_result": selected_result,
        "reference_public_opt": reference_public_opt,
        "reference_deep_models": reference_deep,
        "acceptance": acceptance,
        "public_mainline_status": public_mainline_status,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_torch_uab_report(summary) + "\n", encoding="utf-8")
    progress.finish(
        summary_path=str(summary_path),
        report_path=str(report_path),
        predictions_path=str(predictions_path),
        leaderboard_path=str(leaderboard_path),
    )
    LOGGER.info(
        "task_eval_public_opt_torch finished run_id=%s status=%s summary_path=%s report_path=%s",
        config.run_id,
        public_mainline_status,
        summary_path,
        report_path,
    )
    return StageIPublicOptTorchUABRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        feature_frame_path=str(feature_frame_path),
        predictions_path=str(predictions_path),
        summary_path=str(summary_path),
        report_path=str(report_path),
        summary=summary,
    )
