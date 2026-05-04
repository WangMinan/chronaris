"""Minimal public-opt runner for Stage I UAB subjective regression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from chronaris.dataset import (
    load_stage_i_sequence_bundle,
    load_stage_i_sequence_entries,
    load_stage_i_sequence_summary,
)
from chronaris.evaluation import evaluate_regression_predictions
from chronaris.pipelines.stage_i_baseline_models import build_loso_splits
from chronaris.pipelines.stage_i_public_opt_data import (
    PUBLIC_OPT_DATASET_ID,
    PUBLIC_OPT_HEAD_FEATURES,
    PUBLIC_OPT_PROFILE,
    PUBLIC_OPT_SUBSET_ORDER,
    build_stage_i_public_opt_feature_frame,
)


@dataclass(frozen=True, slots=True)
class StageIPublicOptConfig:
    run_id: str
    prepared_artifact_root: str
    artifact_root: str = "docs/reports/assets/stage_i_public_opt"
    report_root: str = "docs/reports"
    dataset_id: str = PUBLIC_OPT_DATASET_ID
    profile: str = PUBLIC_OPT_PROFILE
    seed: int = 42


@dataclass(frozen=True, slots=True)
class StageIPublicOptRunResult:
    run_id: str
    dataset_id: str
    profile: str
    artifact_root: str
    feature_frame_path: str
    predictions_path: str
    summary_path: str
    report_path: str
    summary: Mapping[str, object]


def run_stage_i_public_opt(
    config: StageIPublicOptConfig,
) -> StageIPublicOptRunResult:
    prepared = _load_prepared_dataset(config.prepared_artifact_root)
    if prepared["dataset_id"] != config.dataset_id:
        raise ValueError(
            f"prepared dataset mismatch: expected {config.dataset_id}, got {prepared['dataset_id']}"
        )
    prepared_profile = str(prepared["summary"]["profile"])
    if prepared_profile != config.profile:
        raise ValueError(
            f"prepared profile mismatch: expected {config.profile}, got {prepared_profile}"
        )
    feature_result = build_stage_i_public_opt_feature_frame(
        prepared["entries"],
        prepared["bundle"],
        dataset_id=config.dataset_id,
        profile=config.profile,
    )
    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"stage-i-public-opt-{config.run_id}.md"
    feature_frame_path = run_root / "public_opt_feature_frame.parquet"
    predictions_path = run_root / "public_opt_predictions.csv"
    summary_path = run_root / "public_opt_summary.json"

    predictions, subset_results = _run_public_opt_regression(
        feature_frame=feature_result.feature_frame,
        head_feature_columns=feature_result.head_feature_columns,
    )
    summary = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "run_id": config.run_id,
        "dataset_id": config.dataset_id,
        "profile": config.profile,
        "track": "subjective",
        "subset_order": list(feature_result.subset_order),
        "heads": list(PUBLIC_OPT_HEAD_FEATURES),
        "artifact_root": str(run_root),
        "prepared_artifact_root": str(Path(config.prepared_artifact_root)),
        "subset_results": subset_results,
    }

    feature_result.feature_frame.to_parquet(feature_frame_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        render_stage_i_public_opt_report(
            summary=summary,
            feature_frame=feature_result.feature_frame,
        )
        + "\n",
        encoding="utf-8",
    )
    return StageIPublicOptRunResult(
        run_id=config.run_id,
        dataset_id=config.dataset_id,
        profile=config.profile,
        artifact_root=str(run_root),
        feature_frame_path=str(feature_frame_path),
        predictions_path=str(predictions_path),
        summary_path=str(summary_path),
        report_path=str(report_path),
        summary=summary,
    )


def render_stage_i_public_opt_report(
    *,
    summary: Mapping[str, object],
    feature_frame: pd.DataFrame,
) -> str:
    subset_rows = []
    subset_results = summary["subset_results"]
    for subset_id in summary["subset_order"]:
        payload = subset_results.get(subset_id)
        if payload is None:
            continue
        best_head = payload["best_head"]
        best_metrics = payload["heads"][best_head]
        subset_rows.append(
            (
                subset_id,
                payload["sample_count"],
                payload["fold_count"],
                best_head,
                best_metrics["mae"],
                best_metrics["rmse"],
                best_metrics["r2"],
                best_metrics["spearman"],
            )
        )

    lines = [
        "# Stage I Public Opt Minimal Run",
        "",
        "## 运行口径",
        "",
        f"- run_id：`{summary['run_id']}`",
        f"- dataset_id：`{summary['dataset_id']}`",
        f"- profile：`{summary['profile']}`",
        f"- track：`{summary['track']}`",
        f"- prepared asset root：`{summary['prepared_artifact_root']}`",
        f"- output artifact root：`{summary['artifact_root']}`",
        f"- generated_at_utc：`{summary['generated_at_utc']}`",
        "",
        "## 样本范围",
        "",
        f"- 总样本数：`{len(feature_frame)}`",
        f"- subset：`{', '.join(summary['subset_order'])}`",
        f"- split_group 数：`{feature_frame['split_group'].nunique()}`",
        f"- subject 数：`{feature_frame['subject_id'].nunique()}`",
        "",
        "## Subset 指标",
        "",
        "| subset | sample_count | fold_count | best_head | mae | rmse | r2 | spearman |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for subset_id, sample_count, fold_count, best_head, mae, rmse, r2, spearman in subset_rows:
        lines.append(
            f"| {subset_id} | {sample_count} | {fold_count} | {best_head} | "
            f"{_fmt_float(mae)} | {_fmt_float(rmse)} | {_fmt_float(r2)} | {_fmt_float(spearman)} |"
        )

    for subset_id in summary["subset_order"]:
        payload = subset_results.get(subset_id)
        if payload is None:
            continue
        lines.extend(
            [
                "",
                f"### {subset_id}",
                "",
                f"- best_head：`{payload['best_head']}`",
                f"- sample_count：`{payload['sample_count']}`",
                f"- fold_count：`{payload['fold_count']}`",
                "",
                "| head | mae | rmse | r2 | spearman |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for head_name in summary["heads"]:
            metrics = payload["heads"][head_name]
            lines.append(
                f"| {head_name} | {_fmt_float(metrics['mae'])} | {_fmt_float(metrics['rmse'])} | "
                f"{_fmt_float(metrics['r2'])} | {_fmt_float(metrics['spearman'])} |"
            )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 本轮仅实现 `chronaris public opt` 的最小可跑版，目标是打通 `UAB subjective regression` 路径并形成可比较工件。",
            "- 本报告不替换既有 Stage I 公开 benchmark 历史结论，也不改写 `Phase 3` / `MulT` / `ContiFormer` 的收口事实。",
        ]
    )
    return "\n".join(lines)


def _run_public_opt_regression(
    *,
    feature_frame: pd.DataFrame,
    head_feature_columns: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    prediction_frames: list[pd.DataFrame] = []
    subset_results: dict[str, object] = {}

    for subset_id in PUBLIC_OPT_SUBSET_ORDER:
        subset_frame = feature_frame.loc[
            feature_frame["subset_id"] == subset_id
        ].copy()
        if subset_frame.empty:
            continue
        split_groups = subset_frame["split_group"].astype(str).to_numpy()
        loso_splits = build_loso_splits(split_groups)
        head_metrics: dict[str, dict[str, object]] = {}
        for head_name, feature_columns in head_feature_columns.items():
            predictions = _run_one_head(
                subset_frame=subset_frame,
                subset_id=subset_id,
                head_name=head_name,
                feature_columns=tuple(feature_columns),
                loso_splits=loso_splits,
            )
            metrics = _sanitize_regression_metrics(
                evaluate_regression_predictions(predictions)
            )
            head_metrics[head_name] = metrics
            prediction_frames.append(predictions)
        best_head = min(
            head_metrics,
            key=lambda name: (
                float(head_metrics[name]["rmse"]),
                float(head_metrics[name]["mae"]),
            ),
        )
        subset_results[subset_id] = {
            "sample_count": int(len(subset_frame)),
            "fold_count": int(subset_frame["split_group"].nunique()),
            "best_head": best_head,
            "heads": head_metrics,
        }

    predictions = (
        pd.concat(prediction_frames, axis=0, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return predictions, subset_results


def _run_one_head(
    *,
    subset_frame: pd.DataFrame,
    subset_id: str,
    head_name: str,
    feature_columns: Sequence[str],
    loso_splits,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    feature_matrix = subset_frame.loc[:, list(feature_columns)].to_numpy(
        dtype=float,
        copy=True,
    )
    feature_matrix = np.nan_to_num(
        feature_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y_all = subset_frame["y_true"].to_numpy(dtype=float, copy=True)
    for split in loso_splits:
        train_X = feature_matrix[split.train_indices]
        test_X = feature_matrix[split.test_indices]
        train_y = y_all[split.train_indices]
        fallback_value = _safe_regression_fallback(train_y)
        if _should_use_fold_fallback(train_y):
            predicted = np.full(
                shape=(len(split.test_indices),),
                fill_value=fallback_value,
                dtype=np.float32,
            )
        else:
            predicted = _fit_regression_head(
                head_name=head_name,
                train_X=train_X,
                train_y=train_y,
                test_X=test_X,
            )
            predicted, nonfinite_mask = _sanitize_regression_outputs(
                predicted,
                fallback_value=fallback_value,
            )
            if np.any(nonfinite_mask):
                predicted = predicted.astype(np.float32, copy=False)
        test_frame = subset_frame.iloc[split.test_indices].copy()
        prediction_frame = pd.DataFrame(
            {
                "track": "subjective",
                "dataset_id": test_frame["dataset_id"].astype(str).to_numpy(),
                "profile": test_frame["profile"].astype(str).to_numpy(),
                "evaluation_group": np.full(len(test_frame), subset_id, dtype=object),
                "subset_id": test_frame["subset_id"].astype(str).to_numpy(),
                "head_name": np.full(len(test_frame), head_name, dtype=object),
                "model_name": np.full(len(test_frame), head_name, dtype=object),
                "split_group": test_frame["split_group"].astype(str).to_numpy(),
                "sample_id": test_frame["sample_id"].astype(str).to_numpy(),
                "subject_id": test_frame["subject_id"].astype(str).to_numpy(),
                "y_true": test_frame["y_true"].to_numpy(dtype=float, copy=True),
                "y_pred": predicted.astype(float, copy=False),
            }
        )
        rows.append(prediction_frame)
    return pd.concat(rows, axis=0, ignore_index=True)


def _fit_regression_head(
    *,
    head_name: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
) -> np.ndarray:
    if head_name == "physiology_persistence":
        model = Ridge(alpha=1.0)
        model.fit(train_X, train_y)
        return np.asarray(model.predict(test_X), dtype=np.float32)
    if head_name == "ridge_residual":
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_X)
        scaled_test = scaler.transform(test_X)
        model = Ridge(alpha=1.0)
        model.fit(scaled_train, train_y)
        return np.asarray(model.predict(scaled_test), dtype=np.float32)
    raise ValueError(f"unsupported public opt head: {head_name}")


def _load_prepared_dataset(artifact_root: str | Path) -> dict[str, object]:
    root = Path(artifact_root)
    entries = load_stage_i_sequence_entries(root / "task_manifest.jsonl")
    bundle = load_stage_i_sequence_bundle(root / "sequence_bundle.npz")
    summary = load_stage_i_sequence_summary(root / "dataset_summary.json")
    schema = json.loads((root / "sequence_schema.json").read_text(encoding="utf-8"))
    return {
        "artifact_root": str(root),
        "dataset_id": summary.dataset_id,
        "entries": entries,
        "bundle": bundle,
        "summary": summary.to_dict(),
        "schema": schema,
    }


def _safe_regression_fallback(values: np.ndarray) -> float:
    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0
    return float(np.mean(finite_values, dtype=np.float64))


def _should_use_fold_fallback(train_y: np.ndarray) -> bool:
    finite_values = np.asarray(train_y, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return True
    return bool(np.allclose(finite_values, finite_values[0]))


def _sanitize_regression_outputs(
    values: np.ndarray,
    *,
    fallback_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    sanitized = np.asarray(values, dtype=np.float32).copy()
    nonfinite_mask = ~np.isfinite(sanitized)
    if np.any(nonfinite_mask):
        sanitized[nonfinite_mask] = float(fallback_value)
    return sanitized, nonfinite_mask


def _sanitize_regression_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            sanitized[key] = 0.0
        else:
            sanitized[key] = value
    return sanitized


def _fmt_float(value: float) -> str:
    if not np.isfinite(value):
        return "0.0000"
    return f"{float(value):.4f}"
