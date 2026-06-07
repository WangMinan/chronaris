"""Torch-native GPU-first UAB regression runner for Stage I public-opt."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import HuberRegressor, Ridge
from torch import nn
from torch.utils.data import DataLoader, Dataset

from chronaris.evaluation import evaluate_regression_predictions
from chronaris.pipelines.stage_i.stage_i_baseline_models import build_loso_splits
from chronaris.pipelines.stage_i.stage_i_public_opt_data import (
    PUBLIC_OPT_PROFILE,
    build_stage_i_public_opt_feature_frame,
)
from chronaris.pipelines.stage_i.stage_i_public_opt_reference import (
    load_public_opt_prepared_dataset,
)
from chronaris.pipelines.stage_i.stage_i_public_opt_postprocess import (
    apply_public_opt_regression_prediction_aggregation,
    validate_public_opt_prediction_aggregation_policy,
)
from chronaris.pipelines.stage_i.stage_i_public_opt_shared import (
    safe_public_opt_regression_fallback,
    sanitize_public_opt_metrics,
    sanitize_public_opt_regression_outputs,
)
from chronaris.pipelines.stage_i.stage_i_run_observer import (
    StageIRunProgress,
    open_stage_i_run_observer,
)
from chronaris.pipelines.stage_i.stage_i_public_opt_torch_catalog import (
    TorchUABCandidateSpec,
    build_torch_uab_candidates,
    is_better_torch_screen_row,
    validate_torch_uab_config,
)
from chronaris.pipelines.stage_i.stage_i_public_opt_torch_reporting import (
    build_torch_uab_acceptance,
    load_torch_uab_reference_deep_summary,
    load_torch_uab_reference_public_opt_summary,
    render_torch_uab_report,
)
from chronaris.pipelines.stage_i.stage_i_public_opt_torch_supervision import (
    build_torch_uab_supervision_view,
    broadcast_torch_uab_session_predictions,
    validate_torch_uab_supervision_granularity,
)
from chronaris.pipelines.torch_runtime import resolve_torch_device_name, seed_torch

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
    artifact_root: str = "docs/artifacts/assets/stage_i_public_opt_torch"
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

@dataclass(frozen=True, slots=True)
class _TorchFeatureBundle:
    feature_profile: str
    feature_columns: tuple[str, ...]
    physiology_indices: tuple[int, ...]
    context_indices: tuple[int, ...]
    feature_matrix: np.ndarray
    feature_frame: pd.DataFrame
    subset_bundles: Mapping[str, "_TorchSubsetBundle"]


@dataclass(frozen=True, slots=True)
class _TorchSubsetBundle:
    subset_id: str
    subset_frame: pd.DataFrame
    subset_matrix: np.ndarray
    split_groups: np.ndarray
    targets: np.ndarray
    loso_splits: tuple[object, ...]


class _TabularRegressionDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        physiology_indices: Sequence[int],
        context_indices: Sequence[int],
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.physiology = torch.as_tensor(
            features[:, physiology_indices],
            dtype=torch.float32,
        )
        self.context = torch.as_tensor(features[:, context_indices], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.features[index],
            self.physiology[index],
            self.context[index],
            self.targets[index],
        )


class _HuberMLP(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        first_hidden, second_hidden = tuple(hidden_dims)
        self.network = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.LayerNorm(second_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        physiology_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        del physiology_features, context_features
        return self.network(features)


class _LinearHuber(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        physiology_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        del physiology_features, context_features
        return self.head(features)


class _ResidualGatedMLP(nn.Module):
    def __init__(
        self,
        physiology_dim: int,
        context_dim: int,
        *,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        first_hidden, second_hidden = tuple(hidden_dims)
        self.physiology_encoder = nn.Sequential(
            nn.Linear(physiology_dim, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(first_hidden * 2, first_hidden),
            nn.GELU(),
            nn.Linear(first_hidden, first_hidden),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.LayerNorm(second_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        physiology_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        del features
        encoded_physiology = self.physiology_encoder(physiology_features)
        encoded_context = self.context_encoder(context_features)
        gate = self.gate(torch.cat((encoded_physiology, encoded_context), dim=-1))
        fused = gate * encoded_physiology + (1.0 - gate) * encoded_context
        return self.head(fused)


def run_stage_i_public_opt_torch_uab(
    config: StageIPublicOptTorchUABConfig,
) -> StageIPublicOptTorchUABRunResult:
    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_public_opt_torch",
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
        return _run_stage_i_public_opt_torch_uab_observed(
            config=config,
            run_root=run_root,
            progress=progress,
        )


def _run_stage_i_public_opt_torch_uab_observed(
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
        "stage_i_public_opt_torch start run_id=%s requested_device=%s resolved_device=%s",
        config.run_id,
        config.device,
        runtime_device,
    )
    if runtime_device != "cuda":
        if config.require_cuda:
            raise RuntimeError(
                "stage_i_public_opt_torch requires CUDA but resolved runtime_device="
                f"{runtime_device}. Use --device cuda on a CUDA host or pass "
                "--allow-cpu-debug for explicit debugging only."
            )
        LOGGER.warning(
            "stage_i_public_opt_torch using CPU fallback run_id=%s requested_device=%s",
            config.run_id,
            config.device,
        )

    prepared = load_public_opt_prepared_dataset(config.prepared_artifact_root)
    if prepared["dataset_id"] != UAB_TORCH_DATASET_ID:
        raise ValueError(
            f"prepared dataset mismatch: expected {UAB_TORCH_DATASET_ID}, got {prepared['dataset_id']}"
        )
    feature_result = build_stage_i_public_opt_feature_frame(
        prepared["entries"],
        prepared["bundle"],
        dataset_id=UAB_TORCH_DATASET_ID,
        profile=config.profile,
    )
    if feature_result.track != "subjective":
        raise ValueError("torch UAB runner only supports subjective regression track.")
    LOGGER.info(
        "stage_i_public_opt_torch prepared dataset_id=%s samples=%d subsets=%s feature_profiles=%s",
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
    report_path = report_root / f"stage-i-public-opt-{config.run_id}.md"
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
    best_candidate: _TorchUABCandidateSpec | None = None
    best_screen_result: tuple[dict[str, object], pd.DataFrame] | None = None
    best_screen_row: dict[str, object] | None = None
    for candidate_index, candidate in enumerate(candidates, start=1):
        LOGGER.info(
            "stage_i_public_opt_torch screen candidate %d/%d candidate_id=%s",
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
            "stage_i_public_opt_torch screen candidate done candidate_id=%s mean_rmse=%.4f mean_mae=%.4f",
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
        "stage_i_public_opt_torch screen complete winner=%s leaderboard_path=%s",
        best_candidate.candidate_id,
        leaderboard_path,
    )

    final_result = best_screen_result[0]
    final_predictions = best_screen_result[1]
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
                "stage_i_public_opt_torch full LOSO candidate %d/%d candidate_id=%s",
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
                "stage_i_public_opt_torch full LOSO done candidate_id=%s mean_rmse=%.4f mean_mae=%.4f",
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
        final_result, final_predictions = _select_final_torch_uab_result(
            candidate_results=full_candidate_results,
            candidate_predictions=full_candidate_predictions,
            ensemble_policy=config.ensemble_policy,
            selected_subsets=selected_subsets,
        )
        full_run_completed = True
        LOGGER.info(
            "stage_i_public_opt_torch final selection run_id=%s selection=%s",
            config.run_id,
            final_result["selection_policy"],
        )

    final_predictions.to_csv(predictions_path, index=False)
    reference_public_opt = load_torch_uab_reference_public_opt_summary(
        config.reference_public_opt_summary_path
    )
    reference_deep = load_torch_uab_reference_deep_summary(
        config.reference_deep_comparison_summary_path,
        dataset_id=UAB_TORCH_DATASET_ID,
    )
    acceptance = build_torch_uab_acceptance(
        final_result["groups"],
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
        "final_result": final_result,
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
        "stage_i_public_opt_torch finished run_id=%s status=%s summary_path=%s report_path=%s",
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


def _build_feature_profile_bundle(
    feature_result,
    *,
    feature_profile: str,
) -> _TorchFeatureBundle:
    feature_columns = tuple(feature_result.feature_groups[feature_profile])
    feature_frame = feature_result.feature_frame
    feature_matrix = feature_frame.loc[:, list(feature_columns)].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    feature_matrix = np.nan_to_num(
        feature_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=False,
    )
    physiology_allowed = set(feature_result.feature_groups["physiology_only"])
    context_allowed = set(feature_result.feature_groups["context_only"])
    physiology_indices = tuple(
        index for index, name in enumerate(feature_columns) if name in physiology_allowed
    )
    context_indices = tuple(
        index for index, name in enumerate(feature_columns) if name in context_allowed
    )
    if not physiology_indices or not context_indices:
        raise ValueError(
            f"feature profile {feature_profile} has empty branch features after filter."
        )
    subset_labels = feature_frame["subset_id"].astype(str).to_numpy()
    subset_bundles = {
        subset_id: _build_subset_feature_bundle(
            subset_id=subset_id,
            feature_frame=feature_frame,
            feature_matrix=feature_matrix,
            subset_labels=subset_labels,
        )
        for subset_id in ("n_back", "heat_the_chair")
    }
    return _TorchFeatureBundle(
        feature_profile=feature_profile,
        feature_columns=feature_columns,
        physiology_indices=physiology_indices,
        context_indices=context_indices,
        feature_matrix=feature_matrix,
        feature_frame=feature_frame,
        subset_bundles=subset_bundles,
    )


def _build_subset_feature_bundle(
    *,
    subset_id: str,
    feature_frame: pd.DataFrame,
    feature_matrix: np.ndarray,
    subset_labels: np.ndarray,
) -> _TorchSubsetBundle:
    subset_indices = np.flatnonzero(subset_labels == subset_id)
    if subset_indices.size == 0:
        raise ValueError(f"torch UAB feature frame is missing subset_id={subset_id}")
    subset_frame = feature_frame.iloc[subset_indices].reset_index(drop=True)
    subset_matrix = feature_matrix[subset_indices].copy()
    split_groups = subset_frame["split_group"].astype(str).to_numpy()
    targets = subset_frame["y_true"].to_numpy(dtype=np.float32, copy=True)
    return _TorchSubsetBundle(
        subset_id=subset_id,
        subset_frame=subset_frame,
        subset_matrix=subset_matrix,
        split_groups=split_groups,
        targets=targets,
        loso_splits=build_loso_splits(split_groups),
    )


def _run_torch_uab_candidate(
    *,
    feature_bundle: _TorchFeatureBundle,
    candidate: TorchUABCandidateSpec,
    selected_subsets: tuple[str, ...],
    seed: int,
    device: str,
    batch_size: int,
    epochs: int,
    patience: int,
    max_folds: int | None,
    prediction_aggregation_policy: str,
    supervision_granularity: str,
    progress: StageIRunProgress | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    group_metrics: dict[str, object] = {}
    for subset_id in selected_subsets:
        LOGGER.info(
            "stage_i_public_opt_torch candidate=%s subset=%s start",
            candidate.candidate_id,
            subset_id,
        )
        if progress is not None:
            progress.update(
                "subset_start",
                dataset_id=UAB_TORCH_DATASET_ID,
                candidate=candidate.candidate_id,
                subset=subset_id,
            )
        subset_bundle = feature_bundle.subset_bundles[subset_id]
        subset_frame = subset_bundle.subset_frame
        subset_matrix = subset_bundle.subset_matrix
        loso_splits = subset_bundle.loso_splits
        if max_folds is not None:
            loso_splits = loso_splits[:max_folds]
        frames: list[pd.DataFrame] = []
        targets = subset_bundle.targets
        for fold_index, split in enumerate(loso_splits):
            held_out_groups = ",".join(
                sorted(
                    subset_frame.iloc[split.test_indices]["split_group"]
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )
            LOGGER.info(
                "stage_i_public_opt_torch candidate=%s subset=%s fold=%d/%d train=%d test=%d held_out=%s",
                candidate.candidate_id,
                subset_id,
                fold_index + 1,
                len(loso_splits),
                len(split.train_indices),
                len(split.test_indices),
                held_out_groups,
            )
            if progress is not None:
                progress.update(
                    "fold_start",
                    dataset_id=UAB_TORCH_DATASET_ID,
                    candidate=candidate.candidate_id,
                    subset=subset_id,
                    fold_index=fold_index + 1,
                    fold_count=len(loso_splits),
                    train_count=len(split.train_indices),
                    test_count=len(split.test_indices),
                    held_out_groups=held_out_groups,
                )
            fold_predictions = _run_one_torch_uab_fold(
                subset_frame=subset_frame,
                subset_matrix=subset_matrix,
                candidate=candidate,
                physiology_indices=feature_bundle.physiology_indices,
                context_indices=feature_bundle.context_indices,
                seed=seed + fold_index,
                device=device,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                train_indices=split.train_indices,
                test_indices=split.test_indices,
                targets=targets,
                supervision_granularity=supervision_granularity,
            )
            frames.append(fold_predictions)
        predictions = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
        predictions = apply_public_opt_regression_prediction_aggregation(
            predictions,
            policy=prediction_aggregation_policy,
        )
        metrics = sanitize_public_opt_metrics(evaluate_regression_predictions(predictions))
        group_metrics[subset_id] = metrics
        prediction_frames.append(predictions)
        LOGGER.info(
            "stage_i_public_opt_torch candidate=%s subset=%s done rmse=%.4f mae=%.4f",
            candidate.candidate_id,
            subset_id,
            float(metrics["rmse"]),
            float(metrics["mae"]),
        )
        if progress is not None:
            progress.update(
                "subset_done",
                dataset_id=UAB_TORCH_DATASET_ID,
                candidate=candidate.candidate_id,
                subset=subset_id,
                rmse=float(metrics["rmse"]),
                mae=float(metrics["mae"]),
            )
    predictions = pd.concat(prediction_frames, axis=0, ignore_index=True)
    mean_rmse = float(
        np.mean([float(group_metrics[group]["rmse"]) for group in selected_subsets], dtype=np.float64)
    )
    mean_mae = float(
        np.mean([float(group_metrics[group]["mae"]) for group in selected_subsets], dtype=np.float64)
    )
    return {
        "candidate_id": candidate.candidate_id,
        "model_family": candidate.model_family,
        "feature_profile": candidate.feature_profile,
        "hidden_dims": list(candidate.hidden_dims),
        "dropout": candidate.dropout,
        "learning_rate": candidate.learning_rate,
        "weight_decay": candidate.weight_decay,
        "groups": group_metrics,
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
    }, predictions


def _run_one_torch_uab_fold(
    *,
    subset_frame: pd.DataFrame,
    subset_matrix: np.ndarray,
    candidate: TorchUABCandidateSpec,
    physiology_indices: Sequence[int],
    context_indices: Sequence[int],
    seed: int,
    device: str,
    batch_size: int,
    epochs: int,
    patience: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    targets: np.ndarray,
    supervision_granularity: str,
) -> pd.DataFrame:
    seed_torch(seed, device=device)
    train_view = build_torch_uab_supervision_view(
        subset_frame=subset_frame.iloc[train_indices].copy(),
        subset_matrix=subset_matrix[train_indices],
        targets=targets[train_indices],
        supervision_granularity=supervision_granularity,
    )
    test_frame = subset_frame.iloc[test_indices].copy()
    test_view = build_torch_uab_supervision_view(
        subset_frame=test_frame,
        subset_matrix=subset_matrix[test_indices],
        targets=targets[test_indices],
        supervision_granularity=supervision_granularity,
    )
    train_X = train_view.matrix
    train_y = train_view.targets
    test_X = test_view.matrix
    fallback_value = safe_public_opt_regression_fallback(train_y)
    train_scale = _fit_standardizer(train_X)
    train_X = _apply_standardizer(train_X, train_scale)
    test_X = _apply_standardizer(test_X, train_scale)

    train_groups = train_view.split_groups
    if candidate.model_family in {
        "heat_residual_correction",
        "heat_affine_calibrated_blend",
    }:
        predictions = _predict_heat_specialist_fold(
            candidate=candidate,
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            train_groups=train_groups,
            seed=seed,
            fallback_value=fallback_value,
        )
        if supervision_granularity == "session_pooled_broadcast":
            predictions = broadcast_torch_uab_session_predictions(
                target_frame=test_frame,
                pooled_frame=test_view.frame,
                pooled_predictions=predictions,
            )
        return _torch_uab_prediction_frame(
            test_frame=test_frame,
            candidate=candidate,
            predictions=predictions,
        )

    inner_train_idx, inner_val_idx = _split_train_validation_groups(train_groups, seed=seed)
    model = _build_torch_uab_model(
        candidate=candidate,
        input_dim=train_X.shape[1],
        physiology_dim=len(physiology_indices),
        context_dim=len(context_indices),
    ).to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )
    criterion = nn.HuberLoss()

    train_dataset = _TabularRegressionDataset(
        train_X[inner_train_idx],
        train_y[inner_train_idx],
        physiology_indices,
        context_indices,
    )
    val_dataset = (
        _TabularRegressionDataset(
            train_X[inner_val_idx],
            train_y[inner_val_idx],
            physiology_indices,
            context_indices,
        )
        if inner_val_idx is not None and len(inner_val_idx) > 0
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(batch_size, 1),
        shuffle=True,
        drop_last=False,
        pin_memory=device == "cuda",
    )
    non_blocking = device == "cuda"

    best_state = None
    best_metric = float("inf")
    stale_epochs = 0
    for epoch in range(epochs):
        del epoch
        model.train()
        for features, physiology_features, context_features, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                features.to(device=device, non_blocking=non_blocking),
                physiology_features.to(device=device, non_blocking=non_blocking),
                context_features.to(device=device, non_blocking=non_blocking),
            )
            loss = criterion(
                predictions,
                batch_targets.to(device=device, non_blocking=non_blocking).view(-1, 1),
            )
            loss.backward()
            optimizer.step()
        validation_metric = _validation_rmse(
            model=model,
            dataset=val_dataset,
            device=device,
            fallback_value=fallback_value,
        )
        if validation_metric < best_metric - 1e-6:
            best_metric = validation_metric
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if val_dataset is not None and stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(best_state)

    predictions = _predict_torch_uab(
        model=model,
        features=test_X,
        physiology_indices=physiology_indices,
        context_indices=context_indices,
        device=device,
        fallback_value=fallback_value,
    )
    if supervision_granularity == "session_pooled_broadcast":
        predictions = broadcast_torch_uab_session_predictions(
            target_frame=test_frame,
            pooled_frame=test_view.frame,
            pooled_predictions=predictions,
        )
    return _torch_uab_prediction_frame(
        test_frame=test_frame,
        candidate=candidate,
        predictions=predictions,
    )


def _torch_uab_prediction_frame(
    *,
    test_frame: pd.DataFrame,
    candidate: TorchUABCandidateSpec,
    predictions: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track": "subjective",
            "dataset_id": test_frame["dataset_id"].astype(str).to_numpy(),
            "profile": test_frame["profile"].astype(str).to_numpy(),
            "evaluation_group": test_frame["subset_id"].astype(str).to_numpy(),
            "subset_id": test_frame["subset_id"].astype(str).to_numpy(),
            "candidate_id": np.full(len(test_frame), candidate.candidate_id, dtype=object),
            "model_name": np.full(len(test_frame), candidate.model_family, dtype=object),
            "feature_profile": np.full(
                len(test_frame),
                candidate.feature_profile,
                dtype=object,
            ),
            "split_group": test_frame["split_group"].astype(str).to_numpy(),
            "sample_id": test_frame["sample_id"].astype(str).to_numpy(),
            "subject_id": test_frame["subject_id"].astype(str).to_numpy(),
            "session_id": test_frame["session_id"].astype(str).to_numpy(),
            "y_true": test_frame["y_true"].to_numpy(dtype=float, copy=True),
            "y_pred": predictions.astype(float, copy=False),
        }
    )


def _predict_heat_specialist_fold(
    *,
    candidate: TorchUABCandidateSpec,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    train_groups: np.ndarray,
    seed: int,
    fallback_value: float,
) -> np.ndarray:
    if should_use_public_opt_regression_fallback_local(train_y):
        return np.full((len(test_X),), fallback_value, dtype=np.float32)
    if candidate.model_family == "heat_residual_correction":
        baseline = _fit_ridge_safe(train_X, train_y, alpha=1.0)
        baseline_train = np.asarray(baseline.predict(train_X), dtype=np.float32)
        baseline_test = np.asarray(baseline.predict(test_X), dtype=np.float32)
        residual_y = train_y - baseline_train
        correction = _fit_huber_safe(
            train_X,
            residual_y,
            alpha=max(candidate.weight_decay, 1e-6),
        )
        predicted = baseline_test + np.asarray(correction.predict(test_X), dtype=np.float32)
    elif candidate.model_family == "heat_affine_calibrated_blend":
        predicted = _predict_affine_calibrated_blend(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            train_groups=train_groups,
            seed=seed,
            weight_decay=candidate.weight_decay,
        )
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unsupported heat specialist family: {candidate.model_family}")
    predicted, _ = sanitize_public_opt_regression_outputs(
        np.asarray(predicted, dtype=np.float32),
        fallback_value=fallback_value,
    )
    return predicted


def should_use_public_opt_regression_fallback_local(train_y: np.ndarray) -> bool:
    finite_values = np.asarray(train_y, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size <= 1:
        return True
    return bool(np.allclose(finite_values, finite_values[0]))


def _predict_affine_calibrated_blend(
    *,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    train_groups: np.ndarray,
    seed: int,
    weight_decay: float,
) -> np.ndarray:
    inner_train_idx, inner_val_idx = _split_train_validation_groups(train_groups, seed=seed)
    if inner_val_idx is None or len(inner_val_idx) == 0:
        inner_train_idx = np.arange(len(train_y), dtype=int)
        inner_val_idx = inner_train_idx

    baseline = _fit_ridge_safe(train_X[inner_train_idx], train_y[inner_train_idx], alpha=1.0)
    residual_model = _fit_huber_safe(
        train_X[inner_train_idx],
        train_y[inner_train_idx],
        alpha=max(weight_decay, 1e-6),
    )
    val_baseline = np.asarray(baseline.predict(train_X[inner_val_idx]), dtype=np.float32)
    val_raw = np.asarray(residual_model.predict(train_X[inner_val_idx]), dtype=np.float32)
    test_baseline = np.asarray(baseline.predict(test_X), dtype=np.float32)
    test_raw = np.asarray(residual_model.predict(test_X), dtype=np.float32)
    affine_a, affine_b = _fit_affine_calibration(
        predictions=val_raw,
        targets=train_y[inner_val_idx],
    )
    val_calibrated = affine_a * val_raw + affine_b
    test_calibrated = affine_a * test_raw + affine_b
    blend_weight = _select_blend_weight(
        calibrated=val_calibrated,
        baseline=val_baseline,
        targets=train_y[inner_val_idx],
    )
    return blend_weight * test_calibrated + (1.0 - blend_weight) * test_baseline


def _fit_ridge_safe(train_X: np.ndarray, train_y: np.ndarray, *, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(train_X, train_y)
    return model


def _fit_huber_safe(train_X: np.ndarray, train_y: np.ndarray, *, alpha: float) -> object:
    try:
        model = HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=500)
        model.fit(train_X, train_y)
        return model
    except Exception:  # pragma: no cover - rare sklearn convergence fallback
        return _fit_ridge_safe(train_X, train_y, alpha=1.0)


def _fit_affine_calibration(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float]:
    design = np.column_stack(
        [np.asarray(predictions, dtype=np.float64), np.ones(len(predictions), dtype=np.float64)]
    )
    try:
        coefficients, *_ = np.linalg.lstsq(design, np.asarray(targets, dtype=np.float64), rcond=None)
    except np.linalg.LinAlgError:
        return 1.0, 0.0
    return float(coefficients[0]), float(coefficients[1])


def _select_blend_weight(
    *,
    calibrated: np.ndarray,
    baseline: np.ndarray,
    targets: np.ndarray,
) -> float:
    best_weight = 1.0
    best_rmse = float("inf")
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        predictions = weight * calibrated + (1.0 - weight) * baseline
        rmse = float(
            np.sqrt(np.mean(np.square(predictions - targets), dtype=np.float64))
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = float(weight)
    return best_weight


def _build_torch_uab_model(
    *,
    candidate: TorchUABCandidateSpec,
    input_dim: int,
    physiology_dim: int,
    context_dim: int,
) -> nn.Module:
    if candidate.model_family == "linear_huber":
        return _LinearHuber(input_dim)
    if candidate.model_family == "mlp_huber_small":
        return _HuberMLP(
            input_dim,
            hidden_dims=candidate.hidden_dims,
            dropout=candidate.dropout,
        )
    if candidate.model_family == "mlp_huber_wide":
        return _HuberMLP(
            input_dim,
            hidden_dims=candidate.hidden_dims,
            dropout=candidate.dropout,
        )
    if candidate.model_family == "residual_gated_mlp":
        return _ResidualGatedMLP(
            physiology_dim,
            context_dim,
            hidden_dims=candidate.hidden_dims,
            dropout=candidate.dropout,
        )
    raise ValueError(f"unsupported torch UAB model family: {candidate.model_family}")


def _build_torch_uab_full_shortlist_rows(
    *,
    leaderboard: pd.DataFrame,
    full_candidate_limit: int,
    group_winner_limit: int,
    ensemble_policy: str,
    selected_subsets: tuple[str, ...],
    candidate_catalog: str,
) -> list[dict[str, object]]:
    shortlist_size = max(
        full_candidate_limit,
        2 if ensemble_policy == "mean_top2" else 1,
    )
    selected_candidate_ids: list[str] = []
    _extend_shortlist_with_ordered_ids(
        selected_candidate_ids,
        leaderboard["candidate_id"].head(shortlist_size).astype(str).tolist(),
    )
    if candidate_catalog == "heat_specialist":
        _extend_shortlist_with_sorted_candidates(
            selected_candidate_ids,
            leaderboard=leaderboard,
            sort_columns=("screen_mean_mae", "screen_mean_rmse", "candidate_id"),
        )
        _extend_shortlist_with_sorted_candidates(
            selected_candidate_ids,
            leaderboard=leaderboard.loc[
                leaderboard["candidate_id"]
                .astype(str)
                .str.startswith("heat_affine_calibrated_blend")
            ].reset_index(drop=True),
            sort_columns=("screen_mean_rmse", "screen_mean_mae", "candidate_id"),
        )
    if group_winner_limit > 0:
        for subset_id in selected_subsets:
            metric_name = f"{subset_id}_rmse"
            if metric_name not in leaderboard.columns:
                continue
            ordered_group = leaderboard.sort_values(
                [metric_name, "screen_mean_rmse", "screen_mean_mae", "candidate_id"],
                ascending=[True, True, True, True],
            )
            _extend_shortlist_with_ordered_ids(
                selected_candidate_ids,
                ordered_group["candidate_id"]
                .head(group_winner_limit)
                .astype(str)
                .tolist(),
            )
    by_candidate_id = {
        str(row["candidate_id"]): row
        for row in leaderboard.to_dict(orient="records")
    }
    return [by_candidate_id[candidate_id] for candidate_id in selected_candidate_ids]


def _extend_shortlist_with_sorted_candidates(
    selected_candidate_ids: list[str],
    *,
    leaderboard: pd.DataFrame,
    sort_columns: tuple[str, ...],
) -> None:
    if leaderboard.empty:
        return
    ordered = leaderboard.sort_values(
        list(sort_columns),
        ascending=[True] * len(sort_columns),
    )
    _extend_shortlist_with_ordered_ids(
        selected_candidate_ids,
        ordered["candidate_id"].head(1).astype(str).tolist(),
    )


def _extend_shortlist_with_ordered_ids(
    selected_candidate_ids: list[str],
    candidate_ids: Sequence[str],
) -> None:
    for candidate_id in candidate_ids:
        if candidate_id not in selected_candidate_ids:
            selected_candidate_ids.append(candidate_id)


def _validate_torch_uab_runtime_config(config: StageIPublicOptTorchUABConfig) -> None:
    if config.full_candidate_limit < 1:
        raise ValueError("torch UAB full_candidate_limit must be >= 1.")
    if config.full_group_winner_limit < 0:
        raise ValueError("torch UAB full_group_winner_limit must be >= 0.")
    if config.ensemble_policy not in {"none", "mean_top2"}:
        raise ValueError(
            f"unsupported torch UAB ensemble_policy: {config.ensemble_policy}"
        )
    validate_public_opt_prediction_aggregation_policy(
        config.prediction_aggregation_policy
    )
    validate_torch_uab_supervision_granularity(config.supervision_granularity)
    selected_subsets = _normalize_torch_uab_selected_subsets(config.selected_subsets)
    if config.candidate_catalog == "heat_specialist" and selected_subsets != (
        "heat_the_chair",
    ):
        raise ValueError(
            "candidate_catalog=heat_specialist only supports selected_subsets=('heat_the_chair',)."
        )


def _normalize_torch_uab_selected_subsets(
    selected_subsets: Sequence[str],
) -> tuple[str, ...]:
    if not selected_subsets:
        raise ValueError("torch UAB selected_subsets must not be empty.")
    normalized = tuple(dict.fromkeys(str(value) for value in selected_subsets))
    allowed = {"n_back", "heat_the_chair"}
    unsupported = sorted(set(normalized) - allowed)
    if unsupported:
        raise ValueError(
            "unsupported torch UAB selected_subsets: " + ", ".join(unsupported)
        )
    return normalized


def _group_metric_or_nan(
    groups: Mapping[str, object],
    subset_id: str,
    metric_name: str,
) -> float:
    payload = groups.get(subset_id)
    if not isinstance(payload, Mapping):
        return float("nan")
    value = payload.get(metric_name)
    return float(value) if value is not None else float("nan")


def _select_final_torch_uab_result(
    *,
    candidate_results: Mapping[str, Mapping[str, object]],
    candidate_predictions: Mapping[str, pd.DataFrame],
    ensemble_policy: str,
    selected_subsets: tuple[str, ...],
) -> tuple[dict[str, object], pd.DataFrame]:
    final_groups: dict[str, object] = {}
    final_predictions: list[pd.DataFrame] = []
    selection_details: dict[str, object] = {}
    for subset_id in selected_subsets:
        candidate_metrics = {
            candidate_id: result["groups"][subset_id]
            for candidate_id, result in candidate_results.items()
        }
        candidate_frames = {
            candidate_id: predictions.loc[
                predictions["subset_id"].astype(str) == subset_id
            ].reset_index(drop=True)
            for candidate_id, predictions in candidate_predictions.items()
        }
        best_candidate_id = min(
            candidate_metrics,
            key=lambda candidate_id: (
                float(candidate_metrics[candidate_id]["rmse"]),
                float(candidate_metrics[candidate_id]["mae"]),
                candidate_id,
            ),
        )
        selected_metrics = dict(candidate_metrics[best_candidate_id])
        selected_frame = candidate_frames[best_candidate_id].copy()
        selection_payload = {
            "selected_source_type": "candidate",
            "selected_source_id": best_candidate_id,
            "selected_metrics": {
                "rmse": float(selected_metrics["rmse"]),
                "mae": float(selected_metrics["mae"]),
            },
        }
        if ensemble_policy == "mean_top2":
            ensemble_frame, ensemble_metrics, ensemble_members = _build_torch_regression_ensemble(
                candidate_prediction_frames=candidate_frames,
                candidate_metrics=candidate_metrics,
            )
            if (
                ensemble_frame is not None
                and ensemble_metrics is not None
                and _is_better_regression_metrics(ensemble_metrics, selected_metrics)
            ):
                selected_metrics = ensemble_metrics
                selected_frame = ensemble_frame
                selection_payload = {
                    "selected_source_type": "mean_top2_ensemble",
                    "selected_source_id": "mean_top2_ensemble",
                    "selected_members": list(ensemble_members),
                    "selected_metrics": {
                        "rmse": float(selected_metrics["rmse"]),
                        "mae": float(selected_metrics["mae"]),
                    },
                }
        final_groups[subset_id] = selected_metrics
        final_predictions.append(selected_frame)
        selection_details[subset_id] = selection_payload
    merged_predictions = pd.concat(final_predictions, axis=0, ignore_index=True)
    mean_rmse = float(
        np.mean([float(final_groups[subset_id]["rmse"]) for subset_id in selected_subsets], dtype=np.float64)
    )
    mean_mae = float(
        np.mean([float(final_groups[subset_id]["mae"]) for subset_id in selected_subsets], dtype=np.float64)
    )
    return {
        "selection_policy": {
            "ensemble_policy": ensemble_policy,
            "selection_scope": "per_subset_best_of_full_candidates",
        },
        "groups": final_groups,
        "group_selections": selection_details,
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
    }, merged_predictions


def _build_torch_regression_ensemble(
    *,
    candidate_prediction_frames: Mapping[str, pd.DataFrame],
    candidate_metrics: Mapping[str, Mapping[str, object]],
) -> tuple[pd.DataFrame | None, dict[str, object] | None, tuple[str, str] | None]:
    if len(candidate_prediction_frames) < 2:
        return None, None, None
    top_two = tuple(
        sorted(
            candidate_metrics,
            key=lambda candidate_id: (
                float(candidate_metrics[candidate_id]["rmse"]),
                float(candidate_metrics[candidate_id]["mae"]),
                candidate_id,
            ),
        )[:2]
    )
    merged = _merge_prediction_frames(
        [candidate_prediction_frames[candidate_id] for candidate_id in top_two]
    )
    if merged.empty:
        return None, None, None
    ensemble = merged.loc[
        :,
        [
            "track",
            "dataset_id",
            "profile",
            "evaluation_group",
            "subset_id",
            "split_group",
            "sample_id",
            "subject_id",
            "session_id",
            "y_true",
        ],
    ].copy()
    ensemble["candidate_id"] = "mean_top2_ensemble"
    ensemble["model_name"] = "mean_top2_ensemble"
    ensemble["feature_profile"] = "ensemble"
    ensemble["y_pred"] = merged[
        [f"y_pred__{index}" for index in range(len(top_two))]
    ].mean(axis=1)
    ensemble = ensemble[
        [
            "track",
            "dataset_id",
            "profile",
            "evaluation_group",
            "subset_id",
            "candidate_id",
            "model_name",
            "feature_profile",
            "split_group",
            "sample_id",
            "subject_id",
            "session_id",
            "y_true",
            "y_pred",
        ]
    ]
    metrics = sanitize_public_opt_metrics(evaluate_regression_predictions(ensemble))
    return ensemble, metrics, top_two


def _merge_prediction_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    base_keys = [
        "track",
        "dataset_id",
        "profile",
        "evaluation_group",
        "subset_id",
        "split_group",
        "sample_id",
        "subject_id",
        "session_id",
        "y_true",
    ]
    merged = None
    for index, frame in enumerate(frames):
        renamed = frame.loc[:, base_keys + ["y_pred"]].rename(
            columns={"y_pred": f"y_pred__{index}"}
        )
        merged = renamed if merged is None else merged.merge(renamed, on=base_keys, how="inner")
    if merged is None:
        return pd.DataFrame()
    return merged


def _is_better_regression_metrics(
    candidate_metrics: Mapping[str, object],
    incumbent_metrics: Mapping[str, object],
) -> bool:
    return (
        float(candidate_metrics["rmse"]),
        float(candidate_metrics["mae"]),
    ) < (
        float(incumbent_metrics["rmse"]),
        float(incumbent_metrics["mae"]),
    )


def _fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64)
    std = np.std(values, axis=0, dtype=np.float64)
    std = np.where(std <= 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_standardizer(
    values: np.ndarray,
    standardizer: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    mean, std = standardizer
    normalized = (values - mean.reshape(1, -1)) / std.reshape(1, -1)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _split_train_validation_groups(
    split_groups: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(split_groups) <= 1:
        indices = np.arange(len(split_groups), dtype=int)
        return indices, None
    unique_groups = np.unique(split_groups)
    if unique_groups.size < 2:
        indices = np.arange(len(split_groups), dtype=int)
        return indices, None
    rng = np.random.default_rng(seed)
    validation_group = str(rng.choice(unique_groups, size=1, replace=False)[0])
    validation_mask = split_groups == validation_group
    if not np.any(validation_mask) or np.all(validation_mask):
        indices = np.arange(len(split_groups), dtype=int)
        return indices, None
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    return train_indices.astype(int), validation_indices.astype(int)


def _validation_rmse(
    *,
    model: nn.Module,
    dataset: _TabularRegressionDataset | None,
    device: str,
    fallback_value: float,
) -> float:
    if dataset is None or len(dataset) == 0:
        return 0.0
    non_blocking = device == "cuda"
    predictions = _predict_torch_uab_from_tensors(
        model=model,
        feature_tensor=dataset.features.to(device=device, non_blocking=non_blocking),
        physiology_tensor=dataset.physiology.to(device=device, non_blocking=non_blocking),
        context_tensor=dataset.context.to(device=device, non_blocking=non_blocking),
        fallback_value=fallback_value,
    )
    truth = dataset.targets.cpu().numpy()
    return float(np.sqrt(np.mean(np.square(predictions - truth), dtype=np.float64)))


def _predict_torch_uab_from_tensors(
    *,
    model: nn.Module,
    feature_tensor: torch.Tensor,
    physiology_tensor: torch.Tensor,
    context_tensor: torch.Tensor,
    fallback_value: float,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        outputs = model(feature_tensor, physiology_tensor, context_tensor).reshape(-1)
    predictions, _ = sanitize_public_opt_regression_outputs(
        outputs.detach().cpu().numpy(),
        fallback_value=fallback_value,
    )
    return predictions


def _predict_torch_uab(
    *,
    model: nn.Module,
    features: np.ndarray,
    physiology_indices: Sequence[int],
    context_indices: Sequence[int],
    device: str,
    fallback_value: float,
    use_pre_sliced_branches: bool = False,
    physiology_values: np.ndarray | None = None,
    context_values: np.ndarray | None = None,
) -> np.ndarray:
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    if use_pre_sliced_branches:
        if physiology_values is None or context_values is None:
            raise ValueError("pre-sliced branch prediction requires physiology/context arrays.")
        physiology_tensor = torch.as_tensor(
            physiology_values,
            dtype=torch.float32,
            device=device,
        )
        context_tensor = torch.as_tensor(
            context_values,
            dtype=torch.float32,
            device=device,
        )
    else:
        physiology_tensor = feature_tensor[:, physiology_indices]
        context_tensor = feature_tensor[:, context_indices]
    return _predict_torch_uab_from_tensors(
        model=model,
        feature_tensor=feature_tensor,
        physiology_tensor=physiology_tensor,
        context_tensor=context_tensor,
        fallback_value=fallback_value,
    )
