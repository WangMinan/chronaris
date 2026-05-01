"""Stage I baseline pipelines for public benchmark datasets."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.evaluation import evaluate_classification_predictions, evaluate_regression_predictions
from chronaris.pipelines.stage_i_baseline_models import (
    StageIBaselineLosoSplit,
    StageIBaselineModelSpec,
    build_fold_cache,
    build_loso_splits,
    clone_estimator,
    objective_model_specs,
    subjective_model_specs,
)
from chronaris.pipelines.stage_i_baseline_reporting import (
    render_stage_i_baseline_report,
    render_uab_baseline_report,
    summarize_auxiliary_flight,
    write_best_model_plots,
    write_dataset_diagnostic_plots,
)

warnings.filterwarnings(
    "ignore",
    message=".*is_sparse is deprecated and will be removed in a future version.*",
    category=DeprecationWarning,
)

UAB_DATASET_ID = "uab_workload_dataset"
NASA_DATASET_ID = "nasa_csm"


@dataclass(frozen=True, slots=True)
class StageIBaselineArtifacts:
    """Machine outputs from one Stage I baseline run."""

    objective_metrics: Mapping[str, object] | None
    subjective_metrics: Mapping[str, object] | None
    fold_predictions: pd.DataFrame
    plot_paths: Mapping[str, str]


def run_stage_i_baselines(
    feature_table: pd.DataFrame,
    *,
    dataset_id: str,
    profile: str,
    task_family: str,
    artifact_root: str | Path,
) -> StageIBaselineArtifacts:
    """Run one Stage I baseline suite."""

    if dataset_id == UAB_DATASET_ID and task_family == "workload":
        return _run_uab_workload_suite(feature_table, profile=profile, artifact_root=artifact_root)
    if dataset_id == NASA_DATASET_ID and task_family == "attention_state":
        return _run_nasa_attention_suite(feature_table, profile=profile, artifact_root=artifact_root)
    raise ValueError(
        f"unsupported Stage I baseline request: dataset_id={dataset_id}, profile={profile}, task_family={task_family}"
    )


def run_uab_baselines(
    feature_table: pd.DataFrame,
    *,
    artifact_root: str | Path,
) -> StageIBaselineArtifacts:
    """Backwards-compatible wrapper for UAB Stage I runs."""

    sample_granularities = sorted(set(feature_table["sample_granularity"].dropna().astype(str)))
    profile = "session_v1" if sample_granularities == ["session"] else "window_v2"
    return run_stage_i_baselines(
        feature_table,
        dataset_id=UAB_DATASET_ID,
        profile=profile,
        task_family="workload",
        artifact_root=artifact_root,
    )


def write_baseline_artifacts(
    artifacts: StageIBaselineArtifacts,
    *,
    artifact_root: str | Path,
) -> None:
    """Persist JSON/CSV outputs for one baseline run."""

    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    if artifacts.objective_metrics is not None:
        (root / "objective_metrics.json").write_text(
            json.dumps(artifacts.objective_metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if artifacts.subjective_metrics is not None:
        (root / "subjective_metrics.json").write_text(
            json.dumps(artifacts.subjective_metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    artifacts.fold_predictions.to_csv(root / "fold_predictions.csv", index=False)


def _run_uab_workload_suite(
    feature_table: pd.DataFrame,
    *,
    profile: str,
    artifact_root: str | Path,
) -> StageIBaselineArtifacts:
    feature_sets = _uab_feature_sets(feature_table)
    objective_groups = {
        "n_back": lambda frame: _subset_primary(frame, subset_id="n_back"),
        "heat_the_chair": lambda frame: _subset_primary(frame, subset_id="heat_the_chair"),
    }
    subjective_groups = objective_groups

    objective_metrics, objective_predictions = _run_objective_suite(
        feature_table,
        dataset_id=UAB_DATASET_ID,
        profile=profile,
        feature_sets=feature_sets,
        evaluation_groups=objective_groups,
    )
    subjective_metrics, subjective_predictions = _run_subjective_suite(
        feature_table,
        dataset_id=UAB_DATASET_ID,
        profile=profile,
        feature_sets=feature_sets,
        evaluation_groups=subjective_groups,
    )
    objective_metrics["auxiliary_flight_summary"] = summarize_auxiliary_flight(
        feature_table.loc[feature_table["subset_id"] == "flight_simulator"].copy()
    )
    predictions = [frame for frame in (objective_predictions, subjective_predictions) if not frame.empty]
    plot_paths = write_best_model_plots(
        artifact_root=Path(artifact_root),
        objective_metrics=objective_metrics,
        objective_predictions=objective_predictions,
        subjective_metrics=subjective_metrics,
        subjective_predictions=subjective_predictions,
    )
    plot_paths.update(
        write_dataset_diagnostic_plots(
            artifact_root=Path(artifact_root),
            feature_table=feature_table,
            objective_metrics=objective_metrics,
            subjective_metrics=subjective_metrics,
        )
    )
    return StageIBaselineArtifacts(
        objective_metrics=objective_metrics,
        subjective_metrics=subjective_metrics,
        fold_predictions=pd.concat(predictions, axis=0, ignore_index=True) if predictions else pd.DataFrame(),
        plot_paths=plot_paths,
    )


def _run_nasa_attention_suite(
    feature_table: pd.DataFrame,
    *,
    profile: str,
    artifact_root: str | Path,
) -> StageIBaselineArtifacts:
    feature_sets = _nasa_feature_sets(feature_table)
    objective_groups = {
        "benchmark_only": lambda frame: _subset_primary(frame, subset_id="benchmark"),
        "loft_only": lambda frame: _subset_primary(frame, subset_id="loft"),
        "combined": lambda frame: frame.loc[
            (frame["training_role"] == "primary") & (frame["subset_id"].isin(["benchmark", "loft"]))
        ].copy(),
    }
    objective_metrics, objective_predictions = _run_objective_suite(
        feature_table,
        dataset_id=NASA_DATASET_ID,
        profile=profile,
        feature_sets=feature_sets,
        evaluation_groups=objective_groups,
        label_order=(1, 2, 5),
    )
    plot_paths = write_best_model_plots(
        artifact_root=Path(artifact_root),
        objective_metrics=objective_metrics,
        objective_predictions=objective_predictions,
        subjective_metrics=None,
        subjective_predictions=pd.DataFrame(),
    )
    plot_paths.update(
        write_dataset_diagnostic_plots(
            artifact_root=Path(artifact_root),
            feature_table=feature_table,
            objective_metrics=objective_metrics,
            subjective_metrics=None,
        )
    )
    return StageIBaselineArtifacts(
        objective_metrics=objective_metrics,
        subjective_metrics=None,
        fold_predictions=objective_predictions,
        plot_paths=plot_paths,
    )


def _run_objective_suite(
    feature_table: pd.DataFrame,
    *,
    dataset_id: str,
    profile: str,
    feature_sets: Mapping[str, tuple[str, ...]],
    evaluation_groups: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
    label_order: Sequence[int] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    model_specs = objective_model_specs(profile)
    return _run_grouped_track(
        feature_table,
        dataset_id=dataset_id,
        profile=profile,
        track="objective",
        feature_sets=feature_sets,
        evaluation_groups=evaluation_groups,
        target_column="objective_label_value",
        model_specs=model_specs,
        scorer=lambda predictions, _: evaluate_classification_predictions(
            predictions,
            label_order=label_order or sorted(set(predictions["y_true"].astype(int))),
        ),
        result_picker=lambda metrics: max(
            metrics,
            key=lambda name: (metrics[name]["macro_f1"], metrics[name]["balanced_accuracy"]),
        ),
    )


def _run_subjective_suite(
    feature_table: pd.DataFrame,
    *,
    dataset_id: str,
    profile: str,
    feature_sets: Mapping[str, tuple[str, ...]],
    evaluation_groups: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
) -> tuple[dict[str, object], pd.DataFrame]:
    model_specs = subjective_model_specs(profile)
    return _run_grouped_track(
        feature_table,
        dataset_id=dataset_id,
        profile=profile,
        track="subjective",
        feature_sets=feature_sets,
        evaluation_groups=evaluation_groups,
        target_column="subjective_target_value",
        model_specs=model_specs,
        scorer=lambda predictions, _: evaluate_regression_predictions(predictions),
        result_picker=lambda metrics: min(
            metrics,
            key=lambda name: (metrics[name]["rmse"], metrics[name]["mae"]),
        ),
    )


def _run_grouped_track(
    feature_table: pd.DataFrame,
    *,
    dataset_id: str,
    profile: str,
    track: str,
    feature_sets: Mapping[str, tuple[str, ...]],
    evaluation_groups: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
    target_column: str,
    model_specs: Mapping[str, StageIBaselineModelSpec],
    scorer: Callable[[pd.DataFrame, pd.DataFrame], dict[str, object]],
    result_picker: Callable[[Mapping[str, dict[str, object]]], str],
) -> tuple[dict[str, object], pd.DataFrame]:
    primary_feature_set = "all_sensors" if dataset_id == NASA_DATASET_ID else "eeg_ecg"
    ablation_results: dict[str, dict[str, object]] = {}
    primary_results: dict[str, object] = {}
    prediction_frames: list[pd.DataFrame] = []

    for group_name, frame_builder in evaluation_groups.items():
        group_frame = frame_builder(feature_table)
        group_frame = group_frame.loc[group_frame[target_column].notna()].copy().reset_index(drop=True)
        if group_frame.empty:
            continue
        split_groups = group_frame["split_group"].astype(str).to_numpy()
        loso_splits = build_loso_splits(split_groups)
        required_preprocessing = {spec.preprocessing for spec in model_specs.values()}
        feature_matrices = {
            feature_set_name: group_frame.loc[:, list(feature_columns)].to_numpy(dtype=float, copy=True)
            for feature_set_name, feature_columns in feature_sets.items()
            if feature_columns
        }
        feature_set_results: dict[str, object] = {}
        for feature_set_name, feature_matrix in feature_matrices.items():
            fold_cache = build_fold_cache(
                feature_matrix=feature_matrix,
                loso_splits=loso_splits,
                preprocess_modes=required_preprocessing,
            )
            model_metrics: dict[str, object] = {}
            for model_name, spec in model_specs.items():
                predictions = _run_loso_predictions(
                    data=group_frame,
                    target_column=target_column,
                    model_spec=spec,
                    fold_cache=fold_cache,
                    track=track,
                    model_name=model_name,
                    evaluation_group=group_name,
                    feature_set=feature_set_name,
                )
                prediction_frames.append(predictions)
                model_metrics[model_name] = scorer(predictions, group_frame)
            best_model_name = result_picker(model_metrics)
            feature_set_results[feature_set_name] = {
                "sample_count": int(len(group_frame)),
                "fold_count": int(group_frame["split_group"].nunique()),
                "best_model_name": best_model_name,
                "models": model_metrics,
            }
        ablation_results[group_name] = feature_set_results
        primary_results[group_name] = feature_set_results[primary_feature_set]

    metrics = {
        "track": track,
        "dataset_id": dataset_id,
        "profile": profile,
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "primary_feature_set": primary_feature_set,
        "feature_set_columns": {name: len(columns) for name, columns in feature_sets.items()},
        "primary_results": primary_results,
        "ablation_results": ablation_results,
    }
    return metrics, pd.concat(prediction_frames, axis=0, ignore_index=True) if prediction_frames else pd.DataFrame()


def _run_loso_predictions(
    *,
    data: pd.DataFrame,
    target_column: str,
    model_spec: StageIBaselineModelSpec,
    fold_cache: Sequence[tuple[StageIBaselineLosoSplit, Mapping[str, tuple[object, object]]]],
    track: str,
    model_name: str,
    evaluation_group: str,
    feature_set: str,
) -> pd.DataFrame:
    dataset_id = str(data["dataset_id"].iloc[0])
    y_all = data[target_column].to_numpy()
    sample_ids = data["sample_id"].to_numpy()
    subset_ids = data["subset_id"].to_numpy()
    subject_ids = data["subject_id"].to_numpy()
    prediction_frames: list[pd.DataFrame] = []

    for split, prepared_matrices in fold_cache:
        estimator = clone_estimator(model_spec)
        train_X, test_X = prepared_matrices[model_spec.preprocessing]
        estimator.fit(train_X, y_all[split.train_indices])
        predicted = estimator.predict(test_X)
        test_indices = split.test_indices
        y_true = y_all[test_indices]
        prediction_frames.append(
            pd.DataFrame(
                {
                    "track": track,
                    "dataset_id": dataset_id,
                    "evaluation_group": evaluation_group,
                    "feature_set": feature_set,
                    "model_name": model_name,
                    "subset_id": subset_ids[test_indices],
                    "split_group": split.split_group,
                    "sample_id": sample_ids[test_indices],
                    "subject_id": subject_ids[test_indices],
                    "y_true": y_true.astype(float if track == "subjective" else int, copy=False),
                    "y_pred": predicted.astype(float, copy=False)
                    if track == "subjective"
                    else np.rint(predicted).astype(int, copy=False),
                }
            )
        )
    return pd.concat(prediction_frames, axis=0, ignore_index=True)


def _subset_primary(frame: pd.DataFrame, *, subset_id: str) -> pd.DataFrame:
    return frame.loc[(frame["training_role"] == "primary") & (frame["subset_id"] == subset_id)].copy()


def _uab_feature_sets(feature_table: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    eeg_columns = tuple(column for column in feature_table.columns if column.startswith("eeg__"))
    ecg_columns = tuple(column for column in feature_table.columns if column.startswith("ecg__"))
    return {
        "eeg_ecg": tuple((*eeg_columns, *ecg_columns)),
        "eeg_only": eeg_columns,
        "ecg_only": ecg_columns,
    }


def _nasa_feature_sets(feature_table: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    eeg_columns = tuple(column for column in feature_table.columns if column.startswith("eeg__"))
    peripheral_columns = tuple(column for column in feature_table.columns if column.startswith("peripheral__"))
    return {
        "all_sensors": tuple((*eeg_columns, *peripheral_columns)),
        "eeg_only": eeg_columns,
        "peripheral_only": peripheral_columns,
    }
