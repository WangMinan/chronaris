"""Leakage-safe Stage I private proxy ablation protocol."""

from __future__ import annotations

import csv
import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from chronaris.pipelines.stage_i.common.baseline_models import StageIBaselineLosoSplit, build_loso_splits
from chronaris.pipelines.stage_i.common.run_observer import StageIRunProgress, open_stage_i_run_observer
from chronaris.pipelines.stage_i.private.benchmark_data import (
    CLASS_LABEL_TO_ID,
    TASK_MANEUVER,
    TASK_RESPONSE,
    TASK_RETRIEVAL,
    build_variant_feature_frames,
    derive_private_proxy_task_entries,
    load_aligned_private_records,
    merge_task_features,
)
from chronaris.pipelines.stage_i.private.benchmark_feature_helpers import aggregate_field_score
from chronaris.pipelines.stage_i.private.feature_utils import cosine_similarity_numpy
from chronaris.pipelines.stage_i.private.leakage_audit import (
    LabelFeatureOverlapAudit,
    audit_label_feature_overlap,
    write_label_feature_overlap_audit,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_private_leakage_safe_ablation"
DEFAULT_REPORT_ROOT = "docs/artifacts/stage_i"
PROTOCOL = "leakage_safe_v1"
MAX_T3_DISTRIBUTION_ROWS = 20000


@dataclass(frozen=True, slots=True)
class StageILeakageSafeAblationConfig:
    run_id: str
    e_run_manifest_path: str
    f_run_manifest_path: str
    output_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    protocol: str = PROTOCOL
    leakage_safe: bool = True
    exclude_label_source_features: bool = True
    exclude_temporal_identity_features: bool = True
    seeds: tuple[int, ...] = (17, 29, 43, 71, 97)
    split_strategy: tuple[str, ...] = ("leave_one_view_out", "leave_one_sortie_out")
    target_variant_name: str = "chronaris_opt"
    lag_window_points: int = 3
    residual_mode: str = "raw_window_stats"
    git_commit: str | None = None


@dataclass(frozen=True, slots=True)
class StageILeakageSafeAblationRunResult:
    run_id: str
    artifact_root: str
    summary_path: str
    report_path: str
    label_feature_audit_json_path: str
    label_feature_audit_csv_path: str
    seed_metrics_path: str
    split_manifest_path: str
    model_backbone_csv_path: str
    task_adapter_csv_path: str
    model_backbone_figure_path: str
    task_adapter_figure_path: str
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _AblationSpec:
    variant_name: str
    display_name_cn: str
    source_variant: str
    ablation_group: str
    feature_family: str
    component: str
    model_mode: str = "learned"


MODEL_BACKBONE_SPECS = (
    _AblationSpec("naive_time_sync", "朴素时间同步", "f_full", "model_backbone", "dual_projection", "naive_time_sync"),
    _AblationSpec("continuous_dual_state", "双流连续表示", "f_full", "model_backbone", "dual_projection", "continuous_state_backbone"),
    _AblationSpec("remove_physics_constraint", "移除物理约束", "e_baseline", "model_backbone", "dual_projection", "remove_physics_constraint"),
    _AblationSpec("remove_causal_mask", "移除因果掩码", "chronaris_opt_no_causal_mask", "model_backbone", "full_safe", "remove_causal_mask"),
    _AblationSpec("remove_semantic_event_fusion", "移除语义事件融合", "g_min", "model_backbone", "fused_only", "remove_semantic_event_fusion"),
    _AblationSpec("full_model", "完整方案", "chronaris_opt", "model_backbone", "full_safe", "full_model"),
)
TASK_ADAPTER_SPECS = (
    _AblationSpec("full_leakage_safe_task_input", "完整防泄漏任务输入", "chronaris_opt", "task_adapter", "full_safe", "full_task_input"),
    _AblationSpec("remove_task_head", "移除任务头", "chronaris_opt", "task_adapter", "full_safe", "remove_task_head", "no_task_head"),
    _AblationSpec("remove_raw_window_stats_residual", "移除原始窗口统计残差", "chronaris_opt", "task_adapter", "full_safe", "remove_raw_window_stats_residual"),
    _AblationSpec("remove_temporal_position_features", "移除时间位置特征", "chronaris_opt", "task_adapter", "full_safe", "remove_temporal_position_features"),
    _AblationSpec("only_fused_latent", "仅融合潜态", "chronaris_opt", "task_adapter", "fused_only", "fused_latent_only"),
    _AblationSpec("single_modality_only", "仅单模态表示", "f_full", "task_adapter", "physiology_only", "single_modality_only"),
)


def run_stage_i_leakage_safe_ablation(
    config: StageILeakageSafeAblationConfig,
) -> StageILeakageSafeAblationRunResult:
    run_root = Path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_stage_i_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="stage_i_private_leakage_safe_ablation",
        logger=LOGGER,
        initial_progress={"artifact_root": str(run_root), "protocol": config.protocol, "seed_count": len(config.seeds)},
    ) as progress:
        return _run_observed(config=config, run_root=run_root, progress=progress)


def _run_observed(
    *,
    config: StageILeakageSafeAblationConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageILeakageSafeAblationRunResult:
    records = load_aligned_private_records(
        e_run_manifest_path=config.e_run_manifest_path,
        f_run_manifest_path=config.f_run_manifest_path,
    )
    task_payload = derive_private_proxy_task_entries(records)
    frames, _diagnostics = build_variant_feature_frames(
        records,
        enable_optimized_chronaris=True,
        target_variant_name=config.target_variant_name,
        lag_window_points=config.lag_window_points,
        residual_mode=config.residual_mode,
    )
    progress.update("sources_loaded", sample_count=len(records), task_count=len(task_payload["by_task"]))
    split_manifest = _build_split_manifest(records, task_payload["by_task"], config.split_strategy)
    split_manifest_path = run_root / "split_manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    specs = (*MODEL_BACKBONE_SPECS, *TASK_ADAPTER_SPECS)
    audit_rows: list[LabelFeatureOverlapAudit] = []
    seed_rows: list[dict[str, object]] = []
    t3_distribution_rows: list[dict[str, object]] = []
    for spec in specs:
        source_frame = frames[spec.source_variant]
        safe_frame = _filter_feature_frame(source_frame, feature_family=spec.feature_family)
        input_features = _feature_names(safe_frame)
        for task_name, entries in task_payload["by_task"].items():
            audit = audit_label_feature_overlap(
                task_name=str(task_name),
                label_source_fields=_label_source_fields(str(task_name), task_payload["summary"]),
                input_feature_fields=input_features,
                derived_input_features=_derived_features(str(task_name)),
                forbidden_feature_families=_forbidden_families(str(task_name)),
                leakage_safe=config.leakage_safe,
            )
            audit_rows.append(audit)
            for seed in config.seeds:
                if str(task_name) == TASK_RETRIEVAL:
                    result, distribution = _evaluate_retrieval(entries, safe_frame, spec=spec, seed=seed)
                    seed_rows.append(result)
                    t3_distribution_rows.extend(distribution)
                    continue
                for strategy in config.split_strategy:
                    seed_rows.append(_evaluate_supervised(
                        entries,
                        safe_frame,
                        records=records,
                        task_name=str(task_name),
                        spec=spec,
                        seed=seed,
                        split_strategy=strategy,
                        task_summary=task_payload["summary"],
                    ))
        progress.update("ablation_spec_finished", variant_name=spec.variant_name)

    audit_json_path, audit_csv_path = write_label_feature_overlap_audit(audit_rows, output_root=run_root)
    seed_metrics_path = run_root / "seed_metrics.csv"
    _write_csv(seed_rows, seed_metrics_path)
    cross_view_path = _write_metric_subset(seed_rows, run_root / "cross_view_metrics.csv", "leave_one_view_out")
    cross_sortie_path = _write_metric_subset(seed_rows, run_root / "cross_sortie_metrics.csv", "leave_one_sortie_out")
    t3_distribution_summary = {"raw_row_count": len(t3_distribution_rows), "written_row_count": 0, "truncated": False}
    if t3_distribution_rows:
        compact_t3_distribution_rows = _compact_t3_similarity_distribution(t3_distribution_rows)
        t3_distribution_summary = {
            "raw_row_count": len(t3_distribution_rows),
            "written_row_count": len(compact_t3_distribution_rows),
            "truncated": len(compact_t3_distribution_rows) < len(t3_distribution_rows),
            "max_rows": MAX_T3_DISTRIBUTION_ROWS,
            "sampling_policy": "retain all positive pairs, deterministically downsample negative pairs by sorted key",
        }
        _write_csv(compact_t3_distribution_rows, run_root / "t3_similarity_distribution.csv")

    ablation_rows = _aggregate_ablation_rows(seed_rows)
    model_rows = [row for row in ablation_rows if row["ablation_group"] == "model_backbone"]
    adapter_rows = [row for row in ablation_rows if row["ablation_group"] == "task_adapter"]
    model_csv = run_root / "model_backbone_ablation.csv"
    adapter_csv = run_root / "task_adapter_ablation.csv"
    _write_csv(model_rows, model_csv)
    _write_csv(adapter_rows, adapter_csv)
    model_json = run_root / "model_backbone_ablation.json"
    adapter_json = run_root / "task_adapter_ablation.json"
    model_json.write_text(json.dumps({"rows": model_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    adapter_json.write_text(json.dumps({"rows": adapter_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model_png = run_root / "model_backbone_ablation.png"
    adapter_png = run_root / "task_adapter_ablation.png"
    _plot_ablation_heatmap(model_rows, model_png, "模型骨干结构消融")
    _plot_ablation_heatmap(adapter_rows, adapter_png, "任务适配层消融")
    _plot_t2_errors(seed_rows, run_root / "t2_error_distribution.png")
    if t3_distribution_rows:
        _plot_t3_similarity(compact_t3_distribution_rows, run_root / "t3_similarity_distribution.png")

    summary = _build_summary(
        config=config,
        run_root=run_root,
        records=records,
        audit_rows=audit_rows,
        seed_rows=seed_rows,
        ablation_rows=ablation_rows,
        paths={
            "label_feature_overlap_audit_json": str(audit_json_path),
            "label_feature_overlap_audit_csv": str(audit_csv_path),
            "seed_metrics_csv": str(seed_metrics_path),
            "split_manifest_json": str(split_manifest_path),
            "cross_view_metrics_csv": str(cross_view_path),
            "cross_sortie_metrics_csv": str(cross_sortie_path),
            "model_backbone_ablation_csv": str(model_csv),
            "model_backbone_ablation_json": str(model_json),
            "model_backbone_ablation_png": str(model_png),
            "task_adapter_ablation_csv": str(adapter_csv),
            "task_adapter_ablation_json": str(adapter_json),
            "task_adapter_ablation_png": str(adapter_png),
            "t2_error_distribution_png": str(run_root / "t2_error_distribution.png"),
            "t3_similarity_distribution_csv": str(run_root / "t3_similarity_distribution.csv"),
            "t3_similarity_distribution_png": str(run_root / "t3_similarity_distribution.png"),
        },
        t3_distribution_summary=t3_distribution_summary,
    )
    summary_path = run_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"stage-i-private-leakage-safe-ablation-{config.run_id}.md"
    report_path.write_text(render_leakage_safe_ablation_report(summary) + "\n", encoding="utf-8")
    progress.finish(summary_path=str(summary_path), report_path=str(report_path), row_count=len(ablation_rows))
    return StageILeakageSafeAblationRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        report_path=str(report_path),
        label_feature_audit_json_path=str(audit_json_path),
        label_feature_audit_csv_path=str(audit_csv_path),
        seed_metrics_path=str(seed_metrics_path),
        split_manifest_path=str(split_manifest_path),
        model_backbone_csv_path=str(model_csv),
        task_adapter_csv_path=str(adapter_csv),
        model_backbone_figure_path=str(model_png),
        task_adapter_figure_path=str(adapter_png),
        summary=summary,
    )


def _evaluate_supervised(
    entries,
    frame: pd.DataFrame,
    *,
    records: pd.DataFrame,
    task_name: str,
    spec: _AblationSpec,
    seed: int,
    split_strategy: str,
    task_summary: Mapping[str, object],
) -> dict[str, object]:
    task_type = str(entries[0].task_type)
    merged = merge_task_features(entries, frame, task_type=task_type)
    if merged.empty:
        return _skipped_row(task_name, task_type, spec, seed, split_strategy, "empty_safe_feature_frame")
    splits = _splits_for_strategy(merged, split_strategy)
    predictions: list[dict[str, object]] = []
    skipped_folds: list[dict[str, object]] = []
    current_response = _current_response_by_sample(records, task_summary)
    for fold_index, split in enumerate(splits):
        train = merged.iloc[split.train_indices]
        test = merged.iloc[split.test_indices]
        if task_type == "classification" and not _classification_fold_complete(train, test):
            skipped_folds.append({"split_group": split.split_group, "reason": "incomplete_class_labels"})
            continue
        feature_columns = [column for column in merged.columns if column.startswith("feat__")]
        if not feature_columns:
            skipped_folds.append({"split_group": split.split_group, "reason": "no_safe_features"})
            continue
        y_true = test["y_label"].to_numpy()
        if spec.model_mode == "no_task_head":
            y_pred = _no_task_head_predictions(train, test, task_type, current_response)
        else:
            y_pred = _fit_predict(train, test, feature_columns, task_type, seed + fold_index)
        for row_index, predicted in zip(test.index, y_pred, strict=True):
            sample_id = str(merged.loc[row_index, "sample_id"])
            predictions.append(
                {
                    "split_group": split.split_group,
                    "sample_id": sample_id,
                    "y_true": float(merged.loc[row_index, "y_label"]),
                    "y_pred": float(predicted),
                    "persistence_pred": current_response.get(sample_id),
                }
            )
    if not predictions:
        return _skipped_row(task_name, task_type, spec, seed, split_strategy, "all_folds_skipped", skipped_folds)
    metrics = _classification_metrics(predictions) if task_type == "classification" else _regression_metrics(predictions)
    return {
        **_row_base(task_name, task_type, spec, seed, split_strategy),
        "status": "completed",
        "valid_fold_count": len({row["split_group"] for row in predictions}),
        "skipped_fold_count": len(skipped_folds),
        "skipped_reason": "",
        **metrics,
    }


def _evaluate_retrieval(entries, frame: pd.DataFrame, *, spec: _AblationSpec, seed: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    merged = merge_task_features(entries, frame, task_type="retrieval")
    if merged.empty:
        return _skipped_row(TASK_RETRIEVAL, "retrieval", spec, seed, "candidate_pool", "empty_safe_feature_frame"), []
    rows: list[dict[str, object]] = []
    distribution: list[dict[str, object]] = []
    invalid_single_view = 0
    for row in merged.itertuples(index=False):
        if not row.paired_sample_id:
            invalid_single_view += 1
            continue
        candidates = merged.loc[
            (merged["sample_id"] != row.sample_id)
            & (merged["sortie_id"] == row.sortie_id)
            & (merged["pilot_id"] != row.pilot_id)
        ].copy()
        if row.paired_sample_id not in set(candidates["sample_id"]):
            invalid_single_view += 1
            continue
        similarities = cosine_similarity_numpy(row.feature_vector, np.stack(candidates["feature_vector"].to_list(), axis=0))
        candidates = candidates.assign(similarity=similarities)
        ranked = candidates.sort_values("similarity", ascending=False).reset_index(drop=True)
        rank = int(ranked.index[ranked["sample_id"] == row.paired_sample_id][0]) + 1
        rows.append({"rank": rank, "candidate_count": len(ranked), "top1": int(rank == 1), "top3": int(rank <= 3), "top5": int(rank <= 5), "mrr": 1.0 / rank})
        for candidate in ranked.itertuples(index=False):
            distribution.append({
                "seed": seed,
                "variant_name": spec.variant_name,
                "query_sample_id": row.sample_id,
                "candidate_sample_id": candidate.sample_id,
                "is_positive": int(candidate.sample_id == row.paired_sample_id),
                "similarity": float(candidate.similarity),
            })
    if not rows:
        return _skipped_row(TASK_RETRIEVAL, "retrieval", spec, seed, "candidate_pool", "no_cross_pilot_positive_pairs"), distribution
    positives = [item["similarity"] for item in distribution if item["is_positive"]]
    negatives = [item["similarity"] for item in distribution if not item["is_positive"]]
    return {
        **_row_base(TASK_RETRIEVAL, "retrieval", spec, seed, "candidate_pool"),
        "status": "completed",
        "valid_fold_count": 1,
        "skipped_fold_count": invalid_single_view,
        "skipped_reason": "",
        "primary_metric_name": "top1_accuracy",
        "primary_metric_value": float(np.mean([row["top1"] for row in rows])),
        "metric_direction": "higher_is_better",
        "candidate_pool_policy": "same_sortie_cross_pilot",
        "top1_accuracy": float(np.mean([row["top1"] for row in rows])),
        "top3_accuracy": float(np.mean([row["top3"] for row in rows])),
        "top5_accuracy": float(np.mean([row["top5"] for row in rows])),
        "mrr": float(np.mean([row["mrr"] for row in rows])),
        "valid_query_count": len(rows),
        "candidate_count": int(sum(row["candidate_count"] for row in rows)),
        "positive_similarity_mean": float(np.mean(positives)) if positives else float("nan"),
        "negative_similarity_mean": float(np.mean(negatives)) if negatives else float("nan"),
    }, distribution


def _filter_feature_frame(frame: pd.DataFrame, *, feature_family: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        features = dict(row.get("feature_values") or {})
        if feature_family == "dual_projection":
            allowed = {key: value for key, value in features.items() if key.startswith(("phys__dim_", "veh__dim_"))}
        elif feature_family == "fused_only":
            allowed = {key: value for key, value in features.items() if key.startswith(("fused__dim_", "opt_fused__dim_"))}
        elif feature_family == "physiology_only":
            allowed = {key: value for key, value in features.items() if key.startswith("phys__dim_")}
        else:
            allowed = {
                key: value
                for key, value in features.items()
                if key.startswith(("phys__dim_", "veh__dim_", "fused__dim_", "opt_fused__dim_", "diag"))
                and "window_index" not in key
                and "window_fraction" not in key
                and "start_offset" not in key
                and "end_offset" not in key
            }
        updated = dict(row)
        updated["feature_values"] = allowed
        rows.append(updated)
    return pd.DataFrame(rows)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, feature_columns: Sequence[str], task_type: str, seed: int) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_x = scaler.fit_transform(imputer.fit_transform(train[list(feature_columns)].to_numpy(dtype=float)))
    test_x = scaler.transform(imputer.transform(test[list(feature_columns)].to_numpy(dtype=float)))
    if task_type == "classification":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        model.fit(train_x, train["y_label"].to_numpy(dtype=int))
        return model.predict(test_x)
    model = Ridge(alpha=1.0)
    model.fit(train_x, train["y_label"].to_numpy(dtype=float))
    return model.predict(test_x)


def _no_task_head_predictions(train: pd.DataFrame, test: pd.DataFrame, task_type: str, current_response: Mapping[str, float]) -> np.ndarray:
    if task_type == "classification":
        labels = train["y_label"].to_numpy(dtype=int)
        values, counts = np.unique(labels, return_counts=True)
        return np.full(len(test), values[int(np.argmax(counts))], dtype=float)
    fallback = float(train["y_label"].mean())
    return np.asarray([current_response.get(str(row.sample_id), fallback) for row in test.itertuples(index=False)], dtype=float)


def _classification_metrics(predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    y_true = np.asarray([row["y_true"] for row in predictions], dtype=int)
    y_pred = np.asarray([row["y_pred"] for row in predictions], dtype=int)
    labels = [0, 1, 2]
    precision, recall, f1, _support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "primary_metric_name": "macro_f1",
        "primary_metric_value": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_precision": _metric_map(labels, precision),
        "per_class_recall": _metric_map(labels, recall),
        "per_class_f1": _metric_map(labels, f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "metric_direction": "higher_is_better",
    }


def _regression_metrics(predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    y_true = np.asarray([row["y_true"] for row in predictions], dtype=float)
    y_pred = np.asarray([row["y_pred"] for row in predictions], dtype=float)
    persistence = np.asarray([
        row["persistence_pred"] if row["persistence_pred"] is not None else np.nan
        for row in predictions
    ], dtype=float)
    rmse = _rmse(y_true, y_pred)
    persistence_rmse = _rmse(y_true[~np.isnan(persistence)], persistence[~np.isnan(persistence)])
    denominator = float(np.max(y_true) - np.min(y_true)) if y_true.size else 0.0
    nrmse = rmse / denominator if denominator > 0 else float("nan")
    improvement = (persistence_rmse - rmse) / persistence_rmse if persistence_rmse > 0 else float("nan")
    return {
        "primary_metric_name": "rmse",
        "primary_metric_value": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nrmse": nrmse,
        "nrmse_normalization": "target_range",
        "persistence_rmse": persistence_rmse,
        "persistence_improvement_rate": improvement,
        "metric_direction": "lower_is_better",
    }


def _aggregate_ablation_rows(seed_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    completed = [row for row in seed_rows if row.get("status") == "completed"]
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in completed:
        if row["task_name"] in {TASK_MANEUVER, TASK_RESPONSE} and row["split_strategy"] != "leave_one_view_out":
            continue
        grouped.setdefault((str(row["variant_name"]), str(row["task_name"])), []).append(row)
    base_by_group_task: dict[tuple[str, str], float] = {}
    for rows in grouped.values():
        first = rows[0]
        if first["variant_name"] in {"full_model", "full_leakage_safe_task_input"}:
            base_by_group_task[(first["ablation_group"], first["task_name"])] = float(np.mean([row["primary_metric_value"] for row in rows]))
    output: list[dict[str, object]] = []
    for (variant_name, task_name), rows in sorted(grouped.items()):
        first = rows[0]
        values = [float(row["primary_metric_value"]) for row in rows]
        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        baseline = base_by_group_task.get((first["ablation_group"], task_name), mean_value)
        direction = str(first["metric_direction"])
        delta = mean_value - baseline if direction == "lower_is_better" else baseline - mean_value
        relative = (delta / abs(baseline) * 100.0) if baseline else 0.0
        output.append({
            "task_name": task_name,
            "task_type": first["task_type"],
            "variant_name": variant_name,
            "display_name_cn": first["display_name_cn"],
            "source_variant": first["source_variant"],
            "component": first["component"],
            "ablation_group": first["ablation_group"],
            "feature_family": first["feature_family"],
            "protocol": PROTOCOL,
            "leakage_safe": True,
            "primary_metric_name": first["primary_metric_name"],
            "primary_metric_value": mean_value,
            "primary_metric_std": std_value,
            "metric_direction": direction,
            "delta_vs_full": delta,
            "relative_delta_percent": relative,
            "seed_count": len({row["seed"] for row in rows}),
            "valid_fold_count": int(sum(int(row.get("valid_fold_count") or 0) for row in rows)),
            "split_strategy": first["split_strategy"],
            "evidence_layer": "private_proxy_leakage_safe",
            "metric_definition": _metric_definition(str(task_name), str(first["primary_metric_name"])),
            **_retrieval_summary_metrics(rows),
        })
    return output


def _retrieval_summary_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    first = rows[0]
    if first.get("task_type") != "retrieval":
        return {}
    metrics: dict[str, object] = {}
    for key in ("top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr", "positive_similarity_mean", "negative_similarity_mean"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            metrics[key] = float(np.mean(values))
    for key in ("valid_query_count", "candidate_count"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            metrics[key] = int(round(float(np.mean(values))))
    if first.get("candidate_pool_policy"):
        metrics["candidate_pool_policy"] = first["candidate_pool_policy"]
    return metrics


def _build_summary(
    *,
    config: StageILeakageSafeAblationConfig,
    run_root: Path,
    records: pd.DataFrame,
    audit_rows: Sequence[LabelFeatureOverlapAudit],
    seed_rows: Sequence[Mapping[str, object]],
    ablation_rows: Sequence[Mapping[str, object]],
    paths: Mapping[str, str],
    t3_distribution_summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "protocol": config.protocol,
        "leakage_safe": config.leakage_safe,
        "exclude_label_source_features": config.exclude_label_source_features,
        "exclude_temporal_identity_features": config.exclude_temporal_identity_features,
        "seed": list(config.seeds),
        "split_strategy": list(config.split_strategy),
        "git_commit": config.git_commit,
        "records": {"sample_count": int(len(records)), "view_count": int(records["view_id"].nunique()), "sortie_count": int(records["sortie_id"].nunique())},
        "audit_status": "pass" if all(row.audit_status == "pass" for row in audit_rows) else "failed",
        "status": "completed",
        "paths": dict(paths),
        "rows": list(ablation_rows),
        "seed_metric_rows": len(seed_rows),
        "task_status": _task_status(seed_rows),
        "t3_similarity_distribution": dict(t3_distribution_summary),
        "nrmse_normalization": "target_range",
        "boundary": "T1/T2/T3 remain private proxy tasks; this protocol audits label-feature overlap and excludes identity/time leakage.",
    }


def render_leakage_safe_ablation_report(summary: Mapping[str, object]) -> str:
    lines = [
        f"# Stage I Leakage-Safe Private Proxy Ablation - {summary['run_id']}",
        "",
        f"- protocol: `{summary['protocol']}`",
        f"- leakage_safe: `{summary['leakage_safe']}`",
        f"- audit_status: `{summary['audit_status']}`",
        f"- seeds: `{summary['seed']}`",
        f"- split_strategy: `{summary['split_strategy']}`",
        f"- label-feature audit: `{summary['paths']['label_feature_overlap_audit_json']}`",
        "",
        "## 读取边界",
        "",
        "`T1/T2/T3` 仍属于 private proxy 组件诊断。`leakage_safe_v1` 不覆盖历史结果，而是新增排除标签源字段、确定性派生特征、样本身份与窗口位置的评价协议。",
        "",
        "## 消融汇总",
        "",
        "| group | task | component | metric | mean | std | relative_delta_percent |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary.get("rows", []):
        lines.append(
            f"| `{row['ablation_group']}` | `{row['task_name']}` | {row['display_name_cn']} | "
            f"`{row['primary_metric_name']}` | {float(row['primary_metric_value']):.6f} | "
            f"{float(row['primary_metric_std']):.6f} | {float(row['relative_delta_percent']):.3f} |"
        )
    t3_rows = [row for row in summary.get("rows", []) if row.get("task_name") == TASK_RETRIEVAL]
    if t3_rows:
        lines.extend(
            [
                "",
                "## T3 检索诊断",
                "",
                "T3 使用 `same_sortie_cross_pilot` 候选池：候选集合限定为同一 sortie 的另一名飞行员窗口；`pilot_id/window_index` 仍不进入特征向量。",
                "",
                "| component | candidate_policy | query_count | candidate_count | top1 | top3 | top5 | mrr |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(t3_rows, key=lambda item: float(item.get("primary_metric_value") or 0.0), reverse=True):
            lines.append(
                f"| {row['display_name_cn']} | `{row.get('candidate_pool_policy', 'candidate_pool')}` | "
                f"{int(row.get('valid_query_count') or 0)} | {int(row.get('candidate_count') or 0)} | "
                f"{float(row.get('top1_accuracy') or 0.0):.6f} | "
                f"{float(row.get('top3_accuracy') or 0.0):.6f} | "
                f"{float(row.get('top5_accuracy') or 0.0):.6f} | "
                f"{float(row.get('mrr') or 0.0):.6f} |"
            )
    lines.extend(["", "## 产物", ""])
    for key, value in summary.get("paths", {}).items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)


def _build_split_manifest(records: pd.DataFrame, by_task: Mapping[str, Sequence[object]], strategies: Sequence[str]) -> dict[str, object]:
    manifest: dict[str, object] = {"strategies": {}, "task_label_counts": {}}
    for strategy in strategies:
        groups = records["view_id"] if strategy == "leave_one_view_out" else records["sortie_id"]
        manifest["strategies"][strategy] = [
            {
                "split_group": split.split_group,
                "train_count": int(len(split.train_indices)),
                "test_count": int(len(split.test_indices)),
            }
            for split in build_loso_splits(groups.to_numpy(dtype=object))
        ]
    for task_name, entries in by_task.items():
        valid = [entry for entry in entries if entry.label_value is not None]
        manifest["task_label_counts"][task_name] = {"entry_count": len(entries), "valid_label_count": len(valid)}
    return manifest


def _splits_for_strategy(frame: pd.DataFrame, split_strategy: str) -> tuple[StageIBaselineLosoSplit, ...]:
    group_column = "view_id" if split_strategy == "leave_one_view_out" else "sortie_id"
    return build_loso_splits(frame[group_column].to_numpy(dtype=object))


def _classification_fold_complete(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    labels = set(CLASS_LABEL_TO_ID.values())
    return labels.issubset(set(train["y_label"].astype(int))) and labels.issubset(set(test["y_label"].astype(int)))


def _current_response_by_sample(records: pd.DataFrame, task_summary: Mapping[str, object]) -> dict[str, float]:
    fields = task_summary.get("selected_physiology_fields", ())
    return {
        str(row.sample_id): float(score)
        for row in records.itertuples(index=False)
        if (score := aggregate_field_score(row.raw_physiology_stats, fields)) is not None
    }


def _label_source_fields(task_name: str, summary: Mapping[str, object]) -> tuple[str, ...]:
    if task_name == TASK_MANEUVER:
        return tuple(summary.get("selected_vehicle_fields", ()))
    if task_name == TASK_RESPONSE:
        return tuple(f"next_window::{field}" for field in summary.get("selected_physiology_fields", ()))
    return ("sample_id", "sortie_id", "pilot_id", "view_id", "window_index")


def _derived_features(task_name: str) -> tuple[str, ...]:
    if task_name == TASK_MANEUVER:
        return ("vehicle_proxy_score", "maneuver_label_quantile", "maneuver_intensity_class")
    if task_name == TASK_RESPONSE:
        return ("next_window_physiology_response", "target_window_physiology")
    return ("paired_sample_id", "window_index", "start_offset", "view_id")


def _forbidden_families(task_name: str) -> tuple[str, ...]:
    base = ("sample_id", "sortie_id", "pilot_id", "view_id", "window_index", "start_offset", "window_fraction")
    if task_name == TASK_MANEUVER:
        return (*base, "vehicle_proxy_score", "label_threshold")
    if task_name == TASK_RESPONSE:
        return (*base, "target_window", "future")
    return base


def _feature_names(frame: pd.DataFrame) -> tuple[str, ...]:
    names: set[str] = set()
    for features in frame.get("feature_values", pd.Series(dtype=object)):
        names.update(str(key) for key in dict(features or {}).keys())
    return tuple(sorted(names))


def _row_base(task_name: str, task_type: str, spec: _AblationSpec, seed: int, split_strategy: str) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "leakage_safe": True,
        "seed": int(seed),
        "split_strategy": split_strategy,
        "task_name": task_name,
        "task_type": task_type,
        "variant_name": spec.variant_name,
        "display_name_cn": spec.display_name_cn,
        "source_variant": spec.source_variant,
        "ablation_group": spec.ablation_group,
        "feature_family": spec.feature_family,
        "component": spec.component,
        "model_mode": spec.model_mode,
    }


def _skipped_row(task_name: str, task_type: str, spec: _AblationSpec, seed: int, split_strategy: str, reason: str, details: object = None) -> dict[str, object]:
    return {**_row_base(task_name, task_type, spec, seed, split_strategy), "status": "skipped", "skipped_reason": reason, "skip_details": details or "", "valid_fold_count": 0, "skipped_fold_count": 0}


def _metric_map(labels: Sequence[int], values: Sequence[float]) -> dict[str, float]:
    label_names = {0: "low", 1: "medium", 2: "high"}
    return {label_names.get(label, str(label)): float(value) for label, value in zip(labels, values, strict=True)}


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(math.sqrt(float(np.mean((y_true - y_pred) ** 2))))


def _task_status(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    status: dict[str, object] = {}
    for task_name in (TASK_MANEUVER, TASK_RESPONSE, TASK_RETRIEVAL):
        task_rows = [row for row in rows if row.get("task_name") == task_name]
        status[task_name] = {
            "completed_seed_rows": sum(row.get("status") == "completed" for row in task_rows),
            "skipped_seed_rows": sum(row.get("status") == "skipped" for row in task_rows),
        }
    return status


def _metric_definition(task_name: str, metric_name: str) -> str:
    if task_name == TASK_MANEUVER:
        return "Macro-F1 and balanced accuracy; higher is better."
    if task_name == TASK_RESPONSE:
        return "RMSE/MAE/NRMSE with target_range normalization and persistence baseline; lower RMSE is better."
    return "Top-k and MRR over leakage-safe candidate vectors; higher is better."


def _write_metric_subset(rows: Sequence[Mapping[str, object]], path: Path, strategy: str) -> str:
    subset = [row for row in rows if row.get("split_strategy") == strategy]
    _write_csv(subset, path)
    return str(path)


def _compact_t3_similarity_distribution(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    positives = [dict(row) for row in rows if int(row.get("is_positive") or 0) == 1]
    negatives = sorted(
        (dict(row) for row in rows if int(row.get("is_positive") or 0) == 0),
        key=lambda row: (
            str(row.get("variant_name")),
            int(row.get("seed") or 0),
            str(row.get("query_sample_id")),
            str(row.get("candidate_sample_id")),
        ),
    )
    if len(positives) + len(negatives) <= MAX_T3_DISTRIBUTION_ROWS:
        return positives + negatives
    negative_budget = max(0, MAX_T3_DISTRIBUTION_ROWS - len(positives))
    if negative_budget == 0:
        return positives[:MAX_T3_DISTRIBUTION_ROWS]
    if negative_budget >= len(negatives):
        return positives + negatives
    step = len(negatives) / negative_budget
    sampled_negatives = [negatives[min(int(index * step), len(negatives) - 1)] for index in range(negative_budget)]
    return positives + sampled_negatives


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return str(path)


def _plot_ablation_heatmap(rows: Sequence[Mapping[str, object]], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    _configure_plot_font(rcParams)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    tasks = list(dict.fromkeys(frame["task_name"].tolist()))
    variants = list(dict.fromkeys(frame["display_name_cn"].tolist()))
    heat = np.full((len(variants), len(tasks)), np.nan, dtype=float)
    labels = [["" for _ in tasks] for _ in variants]
    for row in frame.to_dict(orient="records"):
        y = variants.index(row["display_name_cn"])
        x = tasks.index(row["task_name"])
        heat[y, x] = float(row["relative_delta_percent"])
        labels[y][x] = f"{float(row['delta_vs_full']):.3g}\n{float(row['relative_delta_percent']):.1f}%"
    fig, ax = plt.subplots(figsize=(10.5, max(4.6, len(variants) * 0.58)))
    im = ax.imshow(np.nan_to_num(heat, nan=0.0), cmap="YlOrBr", aspect="auto")
    ax.set_xticks(range(len(tasks)), [_task_label(task) for task in tasks], fontsize=9)
    ax.set_yticks(range(len(variants)), variants, fontsize=9.5)
    for y in range(len(variants)):
        for x in range(len(tasks)):
            ax.text(x, y, labels[y][x], ha="center", va="center", fontsize=8.2, color="#20252b")
    ax.set_title(title, fontsize=15, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, label="相对退化百分比")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_t2_errors(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    _configure_plot_font(rcParams)

    values = [float(row["primary_metric_value"]) for row in rows if row.get("task_name") == TASK_RESPONSE and row.get("status") == "completed" and row.get("primary_metric_name") == "rmse"]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(values, bins=min(10, max(3, len(values) // 2)), color="#5a7ca8", edgecolor="#20252b")
    ax.set_title("T2 RMSE 多种子分布", fontsize=13, weight="bold")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("seed rows")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_t3_similarity(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    _configure_plot_font(rcParams)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for label, group in frame.groupby("is_positive"):
        ax.hist(group["similarity"].astype(float), bins=12, alpha=0.65, label="正样本" if int(label) else "负样本")
    ax.set_title("T3 正负样本相似度分布", fontsize=13, weight="bold")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("candidate pairs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _task_label(task_name: str) -> str:
    return {
        TASK_MANEUVER: "T1\n宏平均F1",
        TASK_RESPONSE: "T2\nRMSE",
        TASK_RETRIEVAL: "T3\nTop-1",
    }.get(task_name, task_name)


def _configure_plot_font(rc_params) -> None:
    from matplotlib import font_manager

    for family in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        rc_params["font.family"] = [family]
        break
    else:
        rc_params["font.family"] = ["DejaVu Sans"]
    rc_params["axes.unicode_minus"] = False


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    return value


def _json_default(value: object):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def resolve_git_commit(*, cwd: str | Path = ".") -> str | None:
    try:
        payload = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return payload.stdout.strip() or None
