"""Stage I baseline pipelines for public benchmark datasets."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

warnings.filterwarnings(
    "ignore",
    message=".*is_sparse is deprecated and will be removed in a future version.*",
    category=DeprecationWarning,
)

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVR

from chronaris.evaluation import (
    evaluate_classification_predictions,
    evaluate_regression_predictions,
    save_confusion_matrix_plot,
    save_regression_plot,
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


def render_stage_i_baseline_report(
    *,
    artifact_root: str | Path,
    dataset_summary: Mapping[str, object],
    objective_metrics: Mapping[str, object] | None,
    subjective_metrics: Mapping[str, object] | None,
    plot_paths: Mapping[str, str],
) -> str:
    """Render one markdown report for a Stage I baseline run."""

    dataset_id = str(dataset_summary["dataset_id"])
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    title = "# Stage I NASA CSM Attention Baseline" if dataset_id == NASA_DATASET_ID else "# Stage I UAB Baseline"
    lines = [
        title,
        "",
        f"- 生成时间：{generated_at}",
        f"- 机器产物根目录：`{artifact_root}`",
        "",
        "## 数据摘要",
        "",
        f"- 样本总数：`{dataset_summary['entry_count']}`",
        f"- recording 数：`{dataset_summary['recording_count']}`",
        f"- window 数：`{dataset_summary['window_count']}`",
        f"- split_group 数：`{dataset_summary['split_group_count']}`",
        f"- 特征总数：`{dataset_summary['feature_count']}`",
        f"- 模态特征数：`{json.dumps(dataset_summary['feature_group_counts'], ensure_ascii=False)}`",
        f"- subset 计数：`{json.dumps(dataset_summary['subset_counts'], ensure_ascii=False)}`",
        "",
    ]

    if objective_metrics is not None:
        lines.extend(_render_track_section("客观/分类主结果", objective_metrics, metric_fields=("macro_f1", "balanced_accuracy")))
    if subjective_metrics is not None:
        lines.extend(_render_track_section("主观/回归主结果", subjective_metrics, metric_fields=("rmse", "mae", "spearman")))

    if dataset_id == UAB_DATASET_ID and objective_metrics is not None:
        flight_summary = objective_metrics.get("auxiliary_flight_summary", {})
        lines.extend(
            [
                "## Flight 辅助摘要",
                "",
                f"- flight window 数：`{flight_summary.get('sample_count', 0)}`",
                f"- flight split_group 数：`{flight_summary.get('subject_count', 0)}`",
                f"- 理论难度分布：`{json.dumps(flight_summary.get('theoretical_difficulty_distribution', {}), ensure_ascii=False)}`",
                f"- 感知难度均值：`{flight_summary.get('perceived_difficulty_mean', float('nan')):.4f}`",
                "",
            ]
        )

    if plot_paths:
        lines.extend(["## 图表", ""])
        for name, path in sorted(plot_paths.items()):
            lines.append(f"- `{name}`: `{path}`")
        lines.append("")
    return "\n".join(lines)


def render_uab_baseline_report(
    *,
    artifact_root: str | Path,
    dataset_summary: Mapping[str, object],
    objective_metrics: Mapping[str, object],
    subjective_metrics: Mapping[str, object],
    plot_paths: Mapping[str, str],
) -> str:
    """Backwards-compatible UAB report wrapper."""

    return render_stage_i_baseline_report(
        artifact_root=artifact_root,
        dataset_summary=dataset_summary,
        objective_metrics=objective_metrics,
        subjective_metrics=subjective_metrics,
        plot_paths=plot_paths,
    )


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
    objective_metrics["auxiliary_flight_summary"] = _summarize_auxiliary_flight(
        feature_table.loc[feature_table["subset_id"] == "flight_simulator"].copy()
    )
    predictions = [frame for frame in (objective_predictions, subjective_predictions) if not frame.empty]
    plot_paths = _write_best_model_plots(
        artifact_root=Path(artifact_root),
        objective_metrics=objective_metrics,
        objective_predictions=objective_predictions,
        subjective_metrics=subjective_metrics,
        subjective_predictions=subjective_predictions,
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
    plot_paths = _write_best_model_plots(
        artifact_root=Path(artifact_root),
        objective_metrics=objective_metrics,
        objective_predictions=objective_predictions,
        subjective_metrics=None,
        subjective_predictions=pd.DataFrame(),
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
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42)),
            ]
        ),
        "linear_svc": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearSVC(class_weight="balanced", max_iter=5_000)),
            ]
        ),
        "random_forest_classifier": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ]
        ),
    }
    return _run_grouped_track(
        feature_table,
        dataset_id=dataset_id,
        profile=profile,
        track="objective",
        feature_sets=feature_sets,
        evaluation_groups=evaluation_groups,
        target_column="objective_label_value",
        models=models,
        scorer=lambda predictions, _: evaluate_classification_predictions(predictions, label_order=label_order or sorted(set(predictions["y_true"].astype(int)))),
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
    models = {
        "ridge_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "svr": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVR(C=1.0, epsilon=0.1)),
            ]
        ),
        "random_forest_regressor": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
            ]
        ),
    }
    return _run_grouped_track(
        feature_table,
        dataset_id=dataset_id,
        profile=profile,
        track="subjective",
        feature_sets=feature_sets,
        evaluation_groups=evaluation_groups,
        target_column="subjective_target_value",
        models=models,
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
    models: Mapping[str, Pipeline],
    scorer: Callable[[pd.DataFrame, pd.DataFrame], dict[str, object]],
    result_picker: Callable[[Mapping[str, dict[str, object]]], str],
) -> tuple[dict[str, object], pd.DataFrame]:
    primary_feature_set = "all_sensors" if dataset_id == NASA_DATASET_ID else "eeg_ecg"
    ablation_results: dict[str, dict[str, object]] = {}
    primary_results: dict[str, object] = {}
    prediction_frames: list[pd.DataFrame] = []

    for group_name, frame_builder in evaluation_groups.items():
        group_frame = frame_builder(feature_table)
        feature_set_results: dict[str, object] = {}
        for feature_set_name, feature_columns in feature_sets.items():
            if not feature_columns:
                continue
            model_metrics: dict[str, object] = {}
            for model_name, model in models.items():
                predictions = _run_loso_predictions(
                    data=group_frame,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    model=clone(model),
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
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "primary_feature_set": primary_feature_set,
        "feature_set_columns": {name: len(columns) for name, columns in feature_sets.items()},
        "primary_results": primary_results,
        "ablation_results": ablation_results,
    }
    return metrics, pd.concat(prediction_frames, axis=0, ignore_index=True)


def _run_loso_predictions(
    *,
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    model: Pipeline,
    track: str,
    model_name: str,
    evaluation_group: str,
    feature_set: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    unique_groups = tuple(sorted(data["split_group"].astype(str).unique()))
    for split_group in unique_groups:
        train = data.loc[data["split_group"] != split_group].copy()
        test = data.loc[data["split_group"] == split_group].copy()
        model.fit(train[list(feature_columns)], train[target_column])
        predictions = model.predict(test[list(feature_columns)])
        for sample_id, subset_id, subject_id, y_true, y_pred in zip(
            test["sample_id"],
            test["subset_id"],
            test["subject_id"],
            test[target_column],
            predictions,
            strict=True,
        ):
            rows.append(
                {
                    "track": track,
                    "dataset_id": str(test["dataset_id"].iloc[0]),
                    "evaluation_group": evaluation_group,
                    "feature_set": feature_set,
                    "model_name": model_name,
                    "subset_id": subset_id,
                    "split_group": split_group,
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "y_true": float(y_true) if track == "subjective" else int(y_true),
                    "y_pred": float(y_pred) if track == "subjective" else int(round(float(y_pred))),
                }
            )
    return pd.DataFrame(rows)


def _write_best_model_plots(
    *,
    artifact_root: Path,
    objective_metrics: Mapping[str, object] | None,
    objective_predictions: pd.DataFrame,
    subjective_metrics: Mapping[str, object] | None,
    subjective_predictions: pd.DataFrame,
) -> dict[str, str]:
    plot_root = artifact_root / "plots"
    plot_paths: dict[str, str] = {}
    if objective_metrics is not None:
        for evaluation_group, payload in objective_metrics["primary_results"].items():
            best_model = payload["best_model_name"]
            metrics = payload["models"][best_model]
            plot_key = f"objective_confusion_matrix_{evaluation_group}"
            plot_paths[plot_key] = save_confusion_matrix_plot(
                metrics,
                path=plot_root / f"{plot_key}.png",
                title=f"{evaluation_group} objective ({best_model})",
            )
    if subjective_metrics is not None:
        for evaluation_group, payload in subjective_metrics["primary_results"].items():
            best_model = payload["best_model_name"]
            subset_predictions = subjective_predictions.loc[
                (subjective_predictions["evaluation_group"] == evaluation_group)
                & (subjective_predictions["feature_set"] == subjective_metrics["primary_feature_set"])
                & (subjective_predictions["model_name"] == best_model)
            ].copy()
            plot_key = f"subjective_regression_{evaluation_group}"
            plot_paths[plot_key] = save_regression_plot(
                subset_predictions,
                path=plot_root / f"{plot_key}.png",
                title=f"{evaluation_group} subjective ({best_model})",
            )
    return plot_paths


def _render_track_section(
    title: str,
    metrics: Mapping[str, object],
    *,
    metric_fields: Sequence[str],
) -> list[str]:
    lines = [f"## {title}", "", f"- 主特征集：`{metrics['primary_feature_set']}`", ""]
    for evaluation_group, payload in metrics["primary_results"].items():
        best_name = payload["best_model_name"]
        best_metrics = payload["models"][best_name]
        lines.extend(
            [
                f"### {evaluation_group}",
                "",
                f"- 最优模型：`{best_name}`",
                *(f"- {field}：`{best_metrics[field]:.4f}`" for field in metric_fields),
                f"- 样本数：`{payload['sample_count']}`",
                f"- ablation：`{json.dumps(_render_ablation_metrics(metrics['ablation_results'][evaluation_group], metric_fields[0]), ensure_ascii=False)}`",
                "",
            ]
        )
    return lines


def _render_ablation_metrics(feature_set_results: Mapping[str, object], rank_field: str) -> dict[str, float]:
    rendered: dict[str, float] = {}
    for feature_set_name, payload in feature_set_results.items():
        best_model_name = payload["best_model_name"]
        rendered[feature_set_name] = float(payload["models"][best_model_name][rank_field])
    return rendered


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


def _summarize_auxiliary_flight(flight_frame: pd.DataFrame) -> dict[str, object]:
    theoretical = (
        flight_frame["objective_label_value"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    perceived = flight_frame["subjective_target_value"].dropna().astype(float)
    return {
        "sample_count": int(len(flight_frame)),
        "subject_count": int(flight_frame["subject_id"].nunique()),
        "theoretical_difficulty_distribution": {str(key): int(value) for key, value in theoretical.items()},
        "perceived_difficulty_mean": float(perceived.mean()) if not perceived.empty else float("nan"),
    }
