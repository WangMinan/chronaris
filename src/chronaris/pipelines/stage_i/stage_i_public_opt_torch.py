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
from chronaris.pipelines.stage_i.stage_i_public_opt_shared import (
    safe_public_opt_regression_fallback,
    sanitize_public_opt_metrics,
    sanitize_public_opt_regression_outputs,
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
    artifact_root: str = "docs/reports/assets/stage_i_public_opt_torch"
    report_root: str = "docs/reports"
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
    ensemble_policy: str = "none"
    learning_rates: tuple[float, ...] = (1e-3, 3e-4)
    weight_decays: tuple[float, ...] = (1e-4, 1e-3)
    feature_profiles: tuple[str, ...] = ("full", "residual_only")
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
    feature_columns: tuple[str, ...]
    physiology_indices: tuple[int, ...]
    context_indices: tuple[int, ...]
    feature_matrix: np.ndarray
    feature_frame: pd.DataFrame


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
    validate_torch_uab_config(
        dataset_id=config.dataset_id,
        expected_dataset_id=UAB_TORCH_DATASET_ID,
        profile=config.profile,
        expected_profile=PUBLIC_OPT_PROFILE,
        feature_profiles=config.feature_profiles,
        learning_rates=config.learning_rates,
        weight_decays=config.weight_decays,
    )
    _validate_torch_uab_runtime_config(config)
    runtime_device = resolve_torch_device_name(config.device)
    LOGGER.info(
        "stage_i_public_opt_torch start run_id=%s requested_device=%s resolved_device=%s",
        config.run_id,
        config.device,
        runtime_device,
    )
    if runtime_device != "cuda":
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

    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    feature_frame_path = run_root / "public_opt_torch_feature_frame.parquet"
    predictions_path = run_root / "public_opt_torch_predictions.csv"
    summary_path = run_root / "public_opt_torch_summary.json"
    report_path = report_root / f"stage-i-public-opt-{config.run_id}.md"

    feature_result.feature_frame.to_parquet(feature_frame_path, index=False)
    candidates = build_torch_uab_candidates(
        feature_profiles=config.feature_profiles,
        learning_rates=config.learning_rates,
        weight_decays=config.weight_decays,
    )

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
        candidate_bundle = _build_candidate_feature_bundle(feature_result, candidate)
        subset_result, predictions = _run_torch_uab_candidate(
            feature_bundle=candidate_bundle,
            candidate=candidate,
            seed=config.seed,
            device=runtime_device,
            batch_size=config.batch_size,
            epochs=config.epochs,
            patience=config.patience,
            max_folds=config.screen_max_folds,
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
            "n_back_rmse": subset_result["groups"]["n_back"]["rmse"],
            "n_back_mae": subset_result["groups"]["n_back"]["mae"],
            "heat_the_chair_rmse": subset_result["groups"]["heat_the_chair"]["rmse"],
            "heat_the_chair_mae": subset_result["groups"]["heat_the_chair"]["mae"],
        }
        screen_rows.append(row)
        LOGGER.info(
            "stage_i_public_opt_torch screen candidate done candidate_id=%s mean_rmse=%.4f mean_mae=%.4f",
            candidate.candidate_id,
            float(subset_result["mean_rmse"]),
            float(subset_result["mean_mae"]),
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
        shortlist_size = max(config.full_candidate_limit, 2 if config.ensemble_policy == "mean_top2" else 1)
        shortlist_rows = leaderboard.head(shortlist_size).to_dict(orient="records")
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
            full_bundle = _build_candidate_feature_bundle(feature_result, candidate)
            candidate_result, candidate_predictions = _run_torch_uab_candidate(
                feature_bundle=full_bundle,
                candidate=candidate,
                seed=config.seed,
                device=runtime_device,
                batch_size=config.batch_size,
                epochs=config.epochs,
                patience=config.patience,
                max_folds=config.full_max_folds,
            )
            full_candidate_results[candidate_id] = candidate_result
            full_candidate_predictions[candidate_id] = candidate_predictions
            LOGGER.info(
                "stage_i_public_opt_torch full LOSO done candidate_id=%s mean_rmse=%.4f mean_mae=%.4f",
                candidate_id,
                float(candidate_result["mean_rmse"]),
                float(candidate_result["mean_mae"]),
            )
        final_result, final_predictions = _select_final_torch_uab_result(
            candidate_results=full_candidate_results,
            candidate_predictions=full_candidate_predictions,
            ensemble_policy=config.ensemble_policy,
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
        "screen_config": {
            "screen_max_folds": config.screen_max_folds,
            "full_max_folds": config.full_max_folds,
            "run_full_loso": config.run_full_loso,
            "full_candidate_limit": config.full_candidate_limit,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "patience": config.patience,
            "learning_rates": list(config.learning_rates),
            "weight_decays": list(config.weight_decays),
            "feature_profiles": list(config.feature_profiles),
            "device": config.device,
            "ensemble_policy": config.ensemble_policy,
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


def _build_candidate_feature_bundle(
    feature_result,
    candidate: TorchUABCandidateSpec,
) -> _TorchFeatureBundle:
    feature_columns = tuple(feature_result.feature_groups[candidate.feature_profile])
    feature_frame = feature_result.feature_frame.copy()
    feature_matrix = feature_frame.loc[:, list(feature_columns)].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
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
            f"candidate {candidate.candidate_id} has empty branch features after profile filter."
        )
    return _TorchFeatureBundle(
        feature_columns=feature_columns,
        physiology_indices=physiology_indices,
        context_indices=context_indices,
        feature_matrix=feature_matrix,
        feature_frame=feature_frame,
    )


def _run_torch_uab_candidate(
    *,
    feature_bundle: _TorchFeatureBundle,
    candidate: TorchUABCandidateSpec,
    seed: int,
    device: str,
    batch_size: int,
    epochs: int,
    patience: int,
    max_folds: int | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    group_metrics: dict[str, object] = {}
    subset_order = ("n_back", "heat_the_chair")
    for subset_id in subset_order:
        LOGGER.info(
            "stage_i_public_opt_torch candidate=%s subset=%s start",
            candidate.candidate_id,
            subset_id,
        )
        subset_frame = feature_bundle.feature_frame.loc[
            feature_bundle.feature_frame["subset_id"].astype(str) == subset_id
        ].reset_index(drop=True)
        subset_indices = feature_bundle.feature_frame.index[
            feature_bundle.feature_frame["subset_id"].astype(str) == subset_id
        ].to_numpy(dtype=int, copy=False)
        subset_matrix = feature_bundle.feature_matrix[subset_indices]
        split_groups = subset_frame["split_group"].astype(str).to_numpy()
        loso_splits = build_loso_splits(split_groups)
        if max_folds is not None:
            loso_splits = loso_splits[:max_folds]
        frames: list[pd.DataFrame] = []
        targets = subset_frame["y_true"].to_numpy(dtype=np.float32, copy=True)
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
            )
            frames.append(fold_predictions)
        predictions = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
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
    predictions = pd.concat(prediction_frames, axis=0, ignore_index=True)
    mean_rmse = float(
        np.mean([float(group_metrics[group]["rmse"]) for group in subset_order], dtype=np.float64)
    )
    mean_mae = float(
        np.mean([float(group_metrics[group]["mae"]) for group in subset_order], dtype=np.float64)
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
) -> pd.DataFrame:
    seed_torch(seed, device=device)
    train_X = subset_matrix[train_indices]
    train_y = targets[train_indices]
    test_X = subset_matrix[test_indices]
    fallback_value = safe_public_opt_regression_fallback(train_y)
    train_scale = _fit_standardizer(train_X)
    train_X = _apply_standardizer(train_X, train_scale)
    test_X = _apply_standardizer(test_X, train_scale)

    train_groups = subset_frame.iloc[train_indices]["split_group"].astype(str).to_numpy()
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
    )

    best_state = None
    best_metric = float("inf")
    stale_epochs = 0
    for epoch in range(epochs):
        del epoch
        model.train()
        for features, physiology_features, context_features, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                features.to(device=device),
                physiology_features.to(device=device),
                context_features.to(device=device),
            )
            loss = criterion(
                predictions,
                batch_targets.to(device=device).view(-1, 1),
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
    test_frame = subset_frame.iloc[test_indices].copy()
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
            "y_true": test_frame["y_true"].to_numpy(dtype=float, copy=True),
            "y_pred": predictions.astype(float, copy=False),
        }
    )


def _build_torch_uab_model(
    *,
    candidate: TorchUABCandidateSpec,
    input_dim: int,
    physiology_dim: int,
    context_dim: int,
) -> nn.Module:
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


def _validate_torch_uab_runtime_config(config: StageIPublicOptTorchUABConfig) -> None:
    if config.full_candidate_limit < 1:
        raise ValueError("torch UAB full_candidate_limit must be >= 1.")
    if config.ensemble_policy not in {"none", "mean_top2"}:
        raise ValueError(
            f"unsupported torch UAB ensemble_policy: {config.ensemble_policy}"
        )


def _select_final_torch_uab_result(
    *,
    candidate_results: Mapping[str, Mapping[str, object]],
    candidate_predictions: Mapping[str, pd.DataFrame],
    ensemble_policy: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    subset_order = ("n_back", "heat_the_chair")
    final_groups: dict[str, object] = {}
    final_predictions: list[pd.DataFrame] = []
    selection_details: dict[str, object] = {}
    for subset_id in subset_order:
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
        np.mean([float(final_groups[subset_id]["rmse"]) for subset_id in subset_order], dtype=np.float64)
    )
    mean_mae = float(
        np.mean([float(final_groups[subset_id]["mae"]) for subset_id in subset_order], dtype=np.float64)
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
    predictions = _predict_torch_uab(
        model=model,
        features=dataset.features.cpu().numpy(),
        physiology_indices=np.arange(dataset.physiology.shape[1], dtype=int),
        context_indices=np.arange(dataset.context.shape[1], dtype=int),
        device=device,
        fallback_value=fallback_value,
        use_pre_sliced_branches=True,
        physiology_values=dataset.physiology.cpu().numpy(),
        context_values=dataset.context.cpu().numpy(),
    )
    truth = dataset.targets.cpu().numpy()
    return float(np.sqrt(np.mean(np.square(predictions - truth), dtype=np.float64)))


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
    model.eval()
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
    with torch.no_grad():
        outputs = model(feature_tensor, physiology_tensor, context_tensor).reshape(-1)
    predictions, _ = sanitize_public_opt_regression_outputs(
        outputs.detach().cpu().numpy(),
        fallback_value=fallback_value,
    )
    return predictions
