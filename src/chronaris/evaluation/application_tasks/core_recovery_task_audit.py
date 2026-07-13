"""Independent inner-train audit for the Chronaris core-task recovery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chronaris.evaluation.application_tasks.core_recovery_splits import (
    build_task_decision_splits,
)
from chronaris.evaluation.application_tasks.core_recovery_target_data import (
    fit_core_recovery_maneuver_targets,
    fit_core_recovery_response_targets,
    load_core_recovery_target_source,
)
from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    load_dingxin_fold_pretraining_data,
)
from chronaris.modeling.fusion_encoders.causal_query import causal_query_stream
from chronaris.representation import TrainOnlyRobustNormalizer


MANEUVER_CLASS_MAPPING = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True, slots=True)
class CoreTaskAuditConfig:
    run_id: str = "2026-07-13_dingxin-core-task-audit"
    compact_output_root: str = "docs/artifacts/runs"
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    snapshot_root: str = "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    context_catalog_path: str = (
        "docs/artifacts/runs/2026-07-11_dingxin-context-bindings/context_catalog.csv"
    )
    e_run_manifest_path: str = (
        "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json"
    )
    f_run_manifest_path: str = (
        "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json"
    )
    random_state: int = 17


@dataclass(frozen=True, slots=True)
class CoreTaskAuditResult:
    run_id: str
    status: str
    primary_maneuver_target: str
    valid_outer_fold_count: int
    classification_metric_count: int
    response_metric_count: int
    report_path: str
    decision_path: str


def run_core_task_audit(config: CoreTaskAuditConfig) -> CoreTaskAuditResult:
    root = Path(config.compact_output_root) / config.run_id
    root.mkdir(parents=True, exist_ok=True)
    context_catalog = pd.read_csv(config.context_catalog_path)
    split_payload = json.loads(
        (Path(config.inner_split_root) / "split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source = load_core_recovery_target_source(
        fixed_audit_root=config.fixed_audit_root,
        snapshot_root=config.snapshot_root,
        e_run_manifest_path=config.e_run_manifest_path,
        f_run_manifest_path=config.f_run_manifest_path,
    )
    classification_rows = []
    response_rows = []
    split_rows = []
    main_folds = [
        row
        for row in split_payload["folds"]
        if row["outer_split_strategy"] == "leave_one_view_out"
    ]
    for fold in main_folds:
        fold_id = str(fold["fold_id"])
        task_splits = build_task_decision_splits(
            context_catalog=context_catalog,
            eligible_context_ids=fold["train_sample_ids"],
        )
        data = load_dingxin_fold_pretraining_data(
            fold_id=fold_id,
            snapshot_root=config.snapshot_root,
            fixed_audit_root=config.fixed_audit_root,
            inner_split_root=config.inner_split_root,
        )
        for task_split in task_splits:
            split_rows.append(
                {
                    "outer_fold_id": fold_id,
                    **task_split.to_dict(),
                    "outer_test_accessed": False,
                }
            )
            train_ids = task_split.train_context_ids
            evaluation_ids = task_split.evaluation_context_ids
            batch = data.load_batch(train_ids + evaluation_ids)
            normalizer = TrainOnlyRobustNormalizer().fit(
                batch,
                train_sample_ids=train_ids,
                held_out_sample_ids=evaluation_ids,
            )
            normalized = normalizer.transform(batch)
            feature_sets = {
                (modality, history_s): _summary_features(
                    normalized,
                    modality=modality,
                    history_s=history_s,
                )
                for modality in ("physiology_only", "vehicle_only", "dual_stream")
                for history_s in (5.0, 30.0)
            }
            for target_mode in ("current_5s", "future_5s"):
                try:
                    maneuver_targets, _maneuver_thresholds = fit_core_recovery_maneuver_targets(
                        source,
                        train_context_ids=train_ids,
                        evaluation_context_ids=evaluation_ids,
                        maneuver_target_mode=target_mode,
                    )
                except ValueError as error:
                    for modality, history_s in feature_sets:
                        classification_rows.append(
                            _unavailable_maneuver_row(
                                fold_id=fold_id,
                                split_id=task_split.split_id,
                                target_mode=target_mode,
                                modality=modality,
                                history_s=history_s,
                                reason=str(error),
                            )
                        )
                    continue
                for (modality, history_s), features in feature_sets.items():
                    row = _evaluate_maneuver(
                        fold_id=fold_id,
                        split_id=task_split.split_id,
                        target_mode=target_mode,
                        modality=modality,
                        history_s=history_s,
                        sample_ids=batch.sample_ids,
                        features=features,
                        targets=maneuver_targets,
                        random_state=config.random_state,
                    )
                    classification_rows.append(row)
            try:
                response_targets, _response_thresholds = fit_core_recovery_response_targets(
                    source,
                    train_context_ids=train_ids,
                    evaluation_context_ids=evaluation_ids,
                )
            except ValueError as error:
                for modality, history_s in feature_sets:
                    response_rows.append(
                        _unavailable_response_row(
                            fold_id=fold_id,
                            split_id=task_split.split_id,
                            modality=modality,
                            history_s=history_s,
                            reason=str(error),
                        )
                    )
            else:
                for (modality, history_s), features in feature_sets.items():
                    response_rows.append(
                        _evaluate_response(
                            fold_id=fold_id,
                            split_id=task_split.split_id,
                            modality=modality,
                            history_s=history_s,
                            sample_ids=batch.sample_ids,
                            features=features,
                            targets=response_targets,
                        )
                    )
    classification = pd.DataFrame(classification_rows)
    response = pd.DataFrame(response_rows)
    splits = pd.DataFrame(split_rows)
    decision = _task_decision(classification, splits)
    status = "completed" if decision["decision_status"] == "locked" else "insufficient"
    classification.to_csv(root / "maneuver_audit_metrics.csv", index=False)
    response.to_csv(root / "physiology_response_audit_metrics.csv", index=False)
    splits.to_json(root / "task_decision_splits.json", orient="records", indent=2, force_ascii=False)
    (root / "task_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(root / "report.md", config=config, decision=decision, classification=classification, response=response)
    evidence = {
        "run_id": config.run_id,
        "status": status,
        "primary_maneuver_target": decision["primary_maneuver_target"],
        "outer_test_accessed": False,
        "confirmed_metrics_changed": False,
        "classification_metric_count": len(classification),
        "response_metric_count": len(response),
        "source_hashes": dict(source.source_hashes),
    }
    (root / "evidence_manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CoreTaskAuditResult(
        run_id=config.run_id,
        status=status,
        primary_maneuver_target=str(decision["primary_maneuver_target"]),
        valid_outer_fold_count=int(decision["valid_outer_fold_count"]),
        classification_metric_count=len(classification),
        response_metric_count=len(response),
        report_path=str(root / "report.md"),
        decision_path=str(root / "task_decision.json"),
    )


def _summary_features(batch, *, modality: str, history_s: float) -> np.ndarray:
    if modality not in {"physiology_only", "vehicle_only", "dual_stream"}:
        raise ValueError("unsupported task-audit modality")
    streams = []
    if modality in {"physiology_only", "dual_stream"}:
        streams.append(causal_query_stream(batch, stream_name="physiology"))
    if modality in {"vehicle_only", "dual_stream"}:
        streams.append(causal_query_stream(batch, stream_name="vehicle"))
    query = batch.query_timestamps_s
    start = query[:, -1:] - history_s
    time_mask = query >= start
    parts = []
    for stream in streams:
        selected_mask = stream.feature_mask & time_mask.unsqueeze(-1)
        selected = torch.where(selected_mask, stream.values, torch.zeros_like(stream.values))
        count = selected_mask.sum(dim=1).clamp_min(1)
        mean = selected.sum(dim=1) / count
        centered = torch.where(
            selected_mask,
            stream.values - mean.unsqueeze(1),
            torch.zeros_like(stream.values),
        )
        std = torch.sqrt(centered.square().sum(dim=1) / count)
        last = stream.values[:, -1]
        available_ratio = selected_mask.to(stream.values.dtype).mean(dim=1)
        last_age = torch.where(
            stream.feature_mask[:, -1],
            stream.observation_age_s[:, -1].clamp(max=30.0) / 30.0,
            torch.ones_like(stream.observation_age_s[:, -1]),
        )
        parts.extend((mean, std, last, available_ratio, last_age))
    return torch.cat(parts, dim=-1).detach().cpu().numpy().astype(np.float32)


def _evaluate_maneuver(**values) -> dict[str, object]:
    targets = values["targets"]
    label_by_id = {
        str(row.context_id): MANEUVER_CLASS_MAPPING[str(row.class_label)]
        for row in targets.itertuples(index=False)
        if row.status == "completed"
    }
    role_by_id = {str(row.context_id): str(row.role) for row in targets.itertuples(index=False)}
    train_positions, train_labels = _positions_and_values(
        values["sample_ids"], label_by_id, role_by_id, "train"
    )
    evaluation_positions, evaluation_labels = _positions_and_values(
        values["sample_ids"], label_by_id, role_by_id, "test"
    )
    status = "completed"
    macro_f1 = None
    if len(set(train_labels)) < 2 or not evaluation_labels:
        status = "unavailable_class_coverage"
    else:
        model = make_pipeline(
            StandardScaler(with_mean=False),
            OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=values["random_state"],
                    solver="liblinear",
                )
            ),
        )
        model.fit(values["features"][train_positions], train_labels)
        prediction = model.predict(values["features"][evaluation_positions])
        macro_f1 = float(f1_score(evaluation_labels, prediction, labels=(0, 1, 2), average="macro", zero_division=0))
    return {
        "outer_fold_id": values["fold_id"],
        "task_decision_split_id": values["split_id"],
        "maneuver_target_mode": values["target_mode"],
        "modality": values["modality"],
        "history_s": values["history_s"],
        "train_count": len(train_labels),
        "evaluation_count": len(evaluation_labels),
        "macro_f1": macro_f1,
        "status": status,
        "outer_test_accessed": False,
    }


def _evaluate_response(**values) -> dict[str, object]:
    targets = values["targets"]
    score_by_id = {
        str(row.context_id): float(row.score)
        for row in targets.itertuples(index=False)
        if row.status == "completed"
    }
    high_by_id = {
        str(row.context_id): int(row.binary_target)
        for row in targets.itertuples(index=False)
        if row.status == "completed"
    }
    role_by_id = {str(row.context_id): str(row.role) for row in targets.itertuples(index=False)}
    train_positions, train_values = _positions_and_values(values["sample_ids"], score_by_id, role_by_id, "train")
    eval_positions, eval_values = _positions_and_values(values["sample_ids"], score_by_id, role_by_id, "evaluation")
    _, train_high = _positions_and_values(values["sample_ids"], high_by_id, role_by_id, "train")
    _, eval_high = _positions_and_values(values["sample_ids"], high_by_id, role_by_id, "evaluation")
    status = "completed"
    rmse = None
    auprc = None
    if len(train_values) < 4 or not eval_values:
        status = "unavailable_target_coverage"
    else:
        regression = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=10.0))
        regression.fit(values["features"][train_positions], np.log1p(train_values))
        prediction = np.expm1(regression.predict(values["features"][eval_positions])).clip(min=0)
        rmse = float(mean_squared_error(eval_values, prediction) ** 0.5)
        if len(set(train_high)) == 2 and len(set(eval_high)) == 2:
            classifier = make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2_000, solver="liblinear"),
            )
            classifier.fit(values["features"][train_positions], train_high)
            probability = classifier.predict_proba(values["features"][eval_positions])[:, 1]
            auprc = float(average_precision_score(eval_high, probability))
    return {
        "outer_fold_id": values["fold_id"],
        "task_decision_split_id": values["split_id"],
        "modality": values["modality"],
        "history_s": values["history_s"],
        "train_count": len(train_values),
        "evaluation_count": len(eval_values),
        "rmse": rmse,
        "auprc": auprc,
        "status": status,
        "outer_test_accessed": False,
    }


def _unavailable_maneuver_row(**values) -> dict[str, object]:
    return {
        "outer_fold_id": values["fold_id"],
        "task_decision_split_id": values["split_id"],
        "maneuver_target_mode": values["target_mode"],
        "modality": values["modality"],
        "history_s": values["history_s"],
        "train_count": 0,
        "evaluation_count": 0,
        "macro_f1": None,
        "status": "unavailable_target_fit",
        "reason": values["reason"],
        "outer_test_accessed": False,
    }


def _unavailable_response_row(**values) -> dict[str, object]:
    return {
        "outer_fold_id": values["fold_id"],
        "task_decision_split_id": values["split_id"],
        "modality": values["modality"],
        "history_s": values["history_s"],
        "train_count": 0,
        "evaluation_count": 0,
        "rmse": None,
        "auprc": None,
        "status": "unavailable_target_fit",
        "reason": values["reason"],
        "outer_test_accessed": False,
    }


def _positions_and_values(sample_ids, target_by_id, role_by_id, role):
    positions = []
    targets = []
    for index, sample_id in enumerate(sample_ids):
        if role_by_id.get(sample_id) == role and sample_id in target_by_id:
            positions.append(index)
            targets.append(target_by_id[sample_id])
    return np.asarray(positions, dtype=np.int64), targets


def _task_decision(classification: pd.DataFrame, splits: pd.DataFrame) -> dict[str, object]:
    completed = classification[classification["status"] == "completed"].copy()
    fold_count = completed["outer_fold_id"].nunique()
    if fold_count < 2:
        return {
            "decision_status": "insufficient",
            "primary_maneuver_target": "current_5s",
            "valid_outer_fold_count": int(fold_count),
            "reason": "fewer_than_two_valid_outer_folds",
            "outer_test_accessed": False,
        }
    fold_means = completed.groupby(
        ["outer_fold_id", "maneuver_target_mode", "modality", "history_s"],
        as_index=False,
    )["macro_f1"].mean()
    complete_fold_ids = []
    for fold_id, frame in fold_means.groupby("outer_fold_id"):
        keys = {
            (row.maneuver_target_mode, row.modality, float(row.history_s))
            for row in frame.itertuples(index=False)
        }
        expected = {
            (target, modality, history)
            for target in ("current_5s", "future_5s")
            for modality in ("physiology_only", "vehicle_only", "dual_stream")
            for history in (5.0, 30.0)
        }
        if expected <= keys:
            complete_fold_ids.append(str(fold_id))
    fold_means = fold_means[fold_means["outer_fold_id"].isin(complete_fold_ids)]
    if len(complete_fold_ids) < 2:
        return {
            "decision_status": "locked",
            "primary_maneuver_target": "current_5s",
            "future_target_promoted": False,
            "valid_outer_fold_count": len(complete_fold_ids),
            "valid_task_decision_split_count": int(len(splits)),
            "reason": "future_task_has_fewer_than_two_complete_outer_folds",
            "outer_test_accessed": False,
            "decision_irreversible_after_lock": True,
        }
    lookup = {
        (row.outer_fold_id, row.maneuver_target_mode, row.modality, float(row.history_s)): float(row.macro_f1)
        for row in fold_means.itertuples(index=False)
    }
    current_history = []
    future_history = []
    future_dual = []
    future_scores = []
    directional_fold_count = 0
    for fold_id in sorted(fold_means["outer_fold_id"].unique()):
        current_gain = lookup[(fold_id, "current_5s", "dual_stream", 30.0)] - lookup[(fold_id, "current_5s", "dual_stream", 5.0)]
        future_gain = lookup[(fold_id, "future_5s", "dual_stream", 30.0)] - lookup[(fold_id, "future_5s", "dual_stream", 5.0)]
        best_single = max(
            lookup[(fold_id, "future_5s", "physiology_only", 30.0)],
            lookup[(fold_id, "future_5s", "vehicle_only", 30.0)],
        )
        dual_gain = lookup[(fold_id, "future_5s", "dual_stream", 30.0)] - best_single
        current_history.append(current_gain)
        future_history.append(future_gain)
        future_dual.append(dual_gain)
        future_scores.append(lookup[(fold_id, "future_5s", "dual_stream", 30.0)])
        if future_gain >= 0.03 and dual_gain >= 0.03:
            directional_fold_count += 1
    summary = {
        "current_history_gain": float(np.mean(current_history)),
        "future_history_gain": float(np.mean(future_history)),
        "future_dual_stream_gain": float(np.mean(future_dual)),
        "future_dual_stream_macro_f1": float(np.mean(future_scores)),
        "future_directional_outer_fold_count": directional_fold_count,
    }
    promote = (
        summary["current_history_gain"] < 0.02
        and summary["future_history_gain"] >= 0.03
        and summary["future_dual_stream_gain"] >= 0.03
        and summary["future_dual_stream_macro_f1"] >= 0.75
        and directional_fold_count >= 2
    )
    return {
        "decision_status": "locked",
        "primary_maneuver_target": "future_5s" if promote else "current_5s",
        "future_target_promoted": promote,
        "valid_outer_fold_count": len(complete_fold_ids),
        "valid_task_decision_split_count": int(len(splits)),
        "criteria": {
            "current_history_gain_less_than": 0.02,
            "future_history_gain_at_least": 0.03,
            "future_dual_stream_gain_at_least": 0.03,
            "future_macro_f1_at_least": 0.75,
            "directional_outer_folds_at_least": 2,
        },
        "observed": summary,
        "outer_test_accessed": False,
        "decision_irreversible_after_lock": True,
    }


def _write_report(path: Path, *, config, decision, classification, response) -> None:
    lines = [
        "# 鼎新核心任务审计",
        "",
        "本审计只使用每个外层折的 inner-train 再划分数据，用于锁定机动任务定义和评估任务可学习性；外层测试未读取。",
        "",
        "## 任务决定",
        "",
        f"- 锁定的机动任务：`{decision['primary_maneuver_target']}`。",
        f"- 有效外层折：{decision['valid_outer_fold_count']}。",
        f"- 未来任务是否晋升：{decision.get('future_target_promoted', False)}。",
        "",
        "## 审计范围",
        "",
        f"- 机动分类诊断记录：{len(classification)} 条。",
        f"- 生理响应诊断记录：{len(response)} 条。",
        "- 比较 5 秒与 30 秒历史、生理单流、航电单流和双流直接观测。",
        "- 这些数值是训练内诊断，不进入论文确认主表。",
        "",
        "## 下一步",
        "",
        "使用锁定任务定义运行冻结消费者筛选；只有开发门禁通过后才允许训练任务感知 Chronaris。",
        "",
        f"协议配置：`{json.dumps(asdict(config), ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
