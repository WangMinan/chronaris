"""One-shot three-seed outer confirmation for the locked core-task recovery."""

from __future__ import annotations

import csv
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error

from chronaris.evaluation.application_tasks.consumer_model_selection import (
    classifier_classes,
    fit_classifier,
)
from chronaris.evaluation.application_tasks.core_recovery_consumer_screen import (
    _transform_sequences,
)
from chronaris.evaluation.application_tasks.core_recovery_outer_audit import (
    inspect_outer_support_isolation,
)
from chronaris.evaluation.application_tasks.core_recovery_run import (
    _validate_source_checkpoint,
)
from chronaris.evaluation.application_tasks.core_recovery_protocol import (
    sha256_file,
    stable_mapping_sha256,
)
from chronaris.evaluation.application_tasks.core_recovery_training import (
    CoreRecoveryFrozenEncoding,
    build_core_recovery_method_model,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_targets import (
    MANEUVER_TASK,
    RESPONSE_TASK,
    DingxinFoldConsumerTargets,
    load_dingxin_nested_fold_consumer_targets,
)
from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    load_dingxin_fold_pretraining_data,
)
from chronaris.representation import load_fusion_stream_batch


@dataclass(frozen=True, slots=True)
class CoreRecoveryConfirmationConfig:
    run_id: str = "2026-07-13_chronaris-core-task-recovery-confirmation"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    protocol_root: str = "docs/artifacts/runs/2026-07-13_chronaris-core-task-recovery"
    source_checkpoint_root: str = (
        "artifacts/application_evaluation/2026-07-12_dingxin-locked-pretraining-coalesced/"
        "checkpoints"
    )
    baseline_representation_root: str = (
        "artifacts/application_evaluation/"
        "2026-07-12_dingxin-locked-representations-coalesced/representations"
    )
    nested_target_path: str = (
        "docs/artifacts/runs/2026-07-11_dingxin-nested-targets/nested_targets.csv"
    )
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    snapshot_root: str = "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class CoreRecoveryConfirmationResult:
    run_id: str
    status: str
    method_seed_fold_count: int
    metric_count: int
    promotion_passed: bool
    report_path: str
    overall_summary_path: str


def run_core_recovery_confirmation(
    config: CoreRecoveryConfirmationConfig,
) -> CoreRecoveryConfirmationResult:
    root = Path(config.compact_output_root) / config.run_id
    root.mkdir(parents=True, exist_ok=True)
    lock, authorization = _load_and_validate_authorization(config)
    support_audit = inspect_outer_support_isolation(
        context_manifest_path=(
            Path(config.fixed_audit_root) / "context_sample_manifest.jsonl"
        ),
        split_manifest_path=Path(config.inner_split_root) / "split_manifest.json",
        fold_ids=tuple(lock["confirmation_folds"]),
    )
    if not support_audit["valid"]:
        raise PermissionError(
            "outer confirmation support isolation failed before held-out access"
        )
    progress_path = root / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "status": "running",
                "locked_configuration_sha256": lock[
                    "locked_configuration_sha256"
                ],
                "outer_test_accessed": True,
                "started_at_unix_s": time.time(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _mark_authorization_consumed(config, authorization, run_id=config.run_id)
    metrics = []
    predictions = []
    inventory = []
    methods = tuple(lock["methods"])
    for seed in lock["confirmation_seeds"]:
        for fold_id in lock["confirmation_folds"]:
            targets = load_dingxin_nested_fold_consumer_targets(
                fold_id=fold_id,
                nested_target_path=config.nested_target_path,
            )
            chronaris_encodings = _load_chronaris_encodings(
                config,
                seed=int(seed),
                fold_id=fold_id,
            )
            for method_name in methods:
                encodings = (
                    chronaris_encodings
                    if method_name == "chronaris"
                    else _load_baseline_encodings(
                        config,
                        seed=int(seed),
                        fold_id=fold_id,
                        method_name=method_name,
                    )
                )
                method_metrics, method_predictions = _evaluate_locked_method(
                    method_name=method_name,
                    seed=int(seed),
                    fold_id=fold_id,
                    encodings=encodings,
                    targets=targets,
                    task_configs=lock["methods"][method_name],
                )
                metrics.extend(method_metrics)
                predictions.extend(method_predictions)
                inventory.append(
                    {
                        "seed": seed,
                        "fold_id": fold_id,
                        "method_name": method_name,
                        "status": "completed",
                        "outer_test_accessed": True,
                    }
                )
    seed_summary, overall = _aggregate_metrics(metrics)
    promotion = _promotion_audit(overall, lock)
    _write_csv(root / "metric_long.csv", metrics)
    _write_csv(root / "prediction_rows.csv", predictions)
    _write_csv(root / "method_inventory.csv", inventory)
    _write_csv(root / "seed_summary.csv", seed_summary)
    overall_path = root / "overall_summary.csv"
    _write_csv(overall_path, overall)
    (root / "promotion_audit.json").write_text(
        json.dumps(promotion, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = root / "report.md"
    _write_report(report_path, overall, promotion)
    evidence = {
        "format": "chronaris.core_task_recovery_confirmation.v1",
        "run_id": config.run_id,
        "status": "completed",
        "locked_configuration_sha256": lock["locked_configuration_sha256"],
        "method_seed_fold_count": len(inventory),
        "metric_count": len(metrics),
        "promotion_passed": promotion["promotion_passed"],
        "outer_test_access_count": 1,
        "historical_confirmed_artifacts_overwritten": False,
    }
    (root / "evidence_manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    progress_path.write_text(
        json.dumps({**evidence, "finished_at_unix_s": time.time()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return CoreRecoveryConfirmationResult(
        run_id=config.run_id,
        status="completed",
        method_seed_fold_count=len(inventory),
        metric_count=len(metrics),
        promotion_passed=bool(promotion["promotion_passed"]),
        report_path=str(report_path),
        overall_summary_path=str(overall_path),
    )


def _load_chronaris_encodings(config, *, seed: int, fold_id: str):
    data = load_dingxin_fold_pretraining_data(
        fold_id=fold_id,
        snapshot_root=config.snapshot_root,
        fixed_audit_root=config.fixed_audit_root,
        inner_split_root=config.inner_split_root,
    )
    development_batch = data.load_batch(
        data.fold.train_sample_ids + data.fold.validation_sample_ids
    )
    checkpoint = (
        Path(config.source_checkpoint_root)
        / f"seed_{seed}"
        / fold_id
        / "chronaris"
        / "best.pt"
    )
    model, payload = build_core_recovery_method_model(
        source_checkpoint_path=checkpoint,
        batch=development_batch,
        train_sample_ids=data.fold.train_sample_ids,
        validation_sample_ids=data.fold.validation_sample_ids,
        physiology_target_count=1,
        device=config.device,
    )
    _validate_source_checkpoint(
        payload,
        fold_id=fold_id,
        expected_method="chronaris",
    )
    model.eval()
    with torch.inference_mode():
        development = model.encode_frozen(development_batch)
        held_out = model.encode_frozen(data.load_batch(data.fold.held_out_sample_ids))
    return {
        "train": development.select(data.fold.train_sample_ids),
        "held_out": held_out,
    }


def _load_baseline_encodings(config, *, seed: int, fold_id: str, method_name: str):
    base = (
        Path(config.baseline_representation_root)
        / f"seed_{seed}"
        / fold_id
        / method_name
        / fold_id
    )
    result = {}
    for role in ("train", "held_out"):
        output = load_fusion_stream_batch(base / role)
        result[role] = CoreRecoveryFrozenEncoding(
            sample_ids=output.sample_ids,
            valid_mask=output.valid_mask,
            sequence_embedding=output.sequence_embedding,
        )
    return result


def _evaluate_locked_method(
    *,
    method_name: str,
    seed: int,
    fold_id: str,
    encodings,
    targets: DingxinFoldConsumerTargets,
    task_configs,
):
    feature_cache = {}

    def features(task: str):
        configuration = task_configs[task]
        key = (configuration["sequence_mode"], int(configuration["n_kernels"]))
        if key not in feature_cache:
            feature_cache[key] = _task_features(
                encodings,
                mode=key[0],
                n_kernels=key[1],
                random_state=seed,
            )
        return feature_cache[key]

    metrics = []
    predictions = []
    train_positions = {
        sample_id: index for index, sample_id in enumerate(encodings["train"].sample_ids)
    }
    held_positions = {
        sample_id: index
        for index, sample_id in enumerate(encodings["held_out"].sample_ids)
    }
    maneuver_train_ids = targets.sample_ids(role="train", task=MANEUVER_TASK)
    maneuver_held_ids = targets.sample_ids(role="held_out", task=MANEUVER_TASK)
    response_train_ids = targets.sample_ids(role="train", task=RESPONSE_TASK)
    response_held_ids = targets.sample_ids(role="held_out", task=RESPONSE_TASK)
    maneuver_train, maneuver_held = features("maneuver")
    maneuver_train = maneuver_train[_positions(train_positions, maneuver_train_ids)]
    maneuver_held = maneuver_held[_positions(held_positions, maneuver_held_ids)]
    c_value = _parse_float(task_configs["maneuver"]["head_config_id"], "logistic_c")
    classifier, _ = fit_classifier(
        maneuver_train,
        targets.maneuver_classes(maneuver_train_ids),
        None,
        None,
        c_values=(c_value,),
        random_state=seed,
        scaler_with_mean=False,
        classification_labels=(0, 1, 2),
        solver="liblinear_ovr",
    )
    maneuver_prediction = classifier.predict(maneuver_held)
    maneuver_value = float(
        f1_score(
            targets.maneuver_classes(maneuver_held_ids),
            maneuver_prediction,
            labels=(0, 1, 2),
            average="macro",
            zero_division=0,
        )
    )
    metrics.append(_metric_row(seed, fold_id, method_name, "maneuver", "macro_f1", maneuver_value, "higher"))
    predictions.extend(_prediction_rows(seed, fold_id, method_name, "maneuver", maneuver_held_ids, targets.maneuver_classes(maneuver_held_ids), maneuver_prediction))

    response_train, response_held = features("response")
    response_train = response_train[_positions(train_positions, response_train_ids)]
    response_held = response_held[_positions(held_positions, response_held_ids)]
    response_config = task_configs["response"]["head_config_id"]
    minimum_leaf = int(_parse_float(response_config, "extra_trees_leaf"))
    regressor = ExtraTreesRegressor(
        n_estimators=256,
        min_samples_leaf=minimum_leaf,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    ).fit(response_train, targets.response_values(response_train_ids))
    response_prediction = regressor.predict(response_held)
    response_value = float(
        mean_squared_error(
            targets.response_values(response_held_ids),
            response_prediction,
        )
        ** 0.5
    )
    metrics.append(_metric_row(seed, fold_id, method_name, "response", "rmse", response_value, "lower"))
    predictions.extend(_prediction_rows(seed, fold_id, method_name, "response", response_held_ids, targets.response_values(response_held_ids), response_prediction))

    high_train, high_held = features("high_response")
    high_train = high_train[_positions(train_positions, response_train_ids)]
    high_held = high_held[_positions(held_positions, response_held_ids)]
    high_c = _parse_float(task_configs["high_response"]["head_config_id"], "logistic_c")
    high_classifier, _ = fit_classifier(
        high_train,
        targets.high_response_classes(response_train_ids),
        None,
        None,
        c_values=(high_c,),
        random_state=seed,
        scaler_with_mean=False,
        classification_labels=(0, 1),
        solver="liblinear_ovr",
    )
    probability = high_classifier.predict_proba(high_held)
    classes = classifier_classes(high_classifier)
    positive = int(np.flatnonzero(classes == 1)[0])
    high_value = float(
        average_precision_score(
            targets.high_response_classes(response_held_ids),
            probability[:, positive],
        )
    )
    metrics.append(_metric_row(seed, fold_id, method_name, "high_response", "auprc", high_value, "higher"))
    predictions.extend(_prediction_rows(seed, fold_id, method_name, "high_response", response_held_ids, targets.high_response_classes(response_held_ids), probability[:, positive]))
    return metrics, predictions


def _task_features(encodings, *, mode: str, n_kernels: int, random_state: int):
    train = encodings["train"]
    held = encodings["held_out"]
    if mode == "direct_continuous_concat":
        direct_train, direct_held, _ = _transform_sequences(
            train.observed_sequence,
            held.observed_sequence,
            n_kernels=n_kernels,
            random_state=random_state,
        )
        continuous_train, continuous_held, _ = _transform_sequences(
            train.continuous_sequence,
            held.continuous_sequence,
            n_kernels=n_kernels,
            random_state=random_state,
        )
        return (
            np.concatenate((direct_train, continuous_train), axis=1),
            np.concatenate((direct_held, continuous_held), axis=1),
        )
    if mode == "direct_only":
        train_sequence = train.maneuver_observed_sequence
        held_sequence = held.maneuver_observed_sequence
    elif mode == "continuous_only":
        train_sequence = train.continuous_sequence
        held_sequence = held.continuous_sequence
    elif mode == "direct_observed":
        train_sequence = train.observed_sequence
        held_sequence = held.observed_sequence
    elif mode == "frozen_sequence":
        train_sequence = train.sequence_embedding
        held_sequence = held.sequence_embedding
    else:
        raise ValueError(f"unsupported locked sequence mode: {mode}")
    transformed_train, transformed_held, _ = _transform_sequences(
        train_sequence,
        held_sequence,
        n_kernels=n_kernels,
        random_state=random_state,
    )
    return transformed_train, transformed_held


def _aggregate_metrics(metrics):
    frame = pd.DataFrame(metrics)
    seed_rows = []
    for key, group in frame.groupby(
        ["seed", "method_name", "task", "metric", "direction"],
        sort=True,
    ):
        values = group["value"].astype(float).tolist()
        seed_rows.append(
            {
                "seed": key[0],
                "method_name": key[1],
                "task": key[2],
                "metric": key[3],
                "direction": key[4],
                "fold_count": len(values),
                "mean": statistics.fmean(values),
                "worst_fold": min(values) if key[4] == "higher" else max(values),
            }
        )
    seed_frame = pd.DataFrame(seed_rows)
    overall = []
    for key, group in seed_frame.groupby(
        ["method_name", "task", "metric", "direction"],
        sort=True,
    ):
        values = group["mean"].astype(float).tolist()
        overall.append(
            {
                "method_name": key[0],
                "task": key[1],
                "metric": key[2],
                "direction": key[3],
                "seed_count": len(values),
                "mean": statistics.fmean(values),
                "std_across_seeds": float(np.std(values, ddof=0)),
            }
        )
    overall_frame = pd.DataFrame(overall)
    overall_frame["rank"] = np.nan
    for _task, indices in overall_frame.groupby("task", sort=True).groups.items():
        direction = str(overall_frame.loc[indices[0], "direction"])
        overall_frame.loc[indices, "rank"] = overall_frame.loc[
            indices, "mean"
        ].rank(
            method="min",
            ascending=direction == "lower",
        )
    return seed_rows, overall_frame.to_dict("records")


def _promotion_audit(overall, lock):
    frame = pd.DataFrame(overall)
    thresholds = lock["hard_confirmation_thresholds"]
    checks = []
    mapping = {
        "maneuver": ("maneuver_macro_f1", "higher"),
        "response": ("response_rmse", "lower"),
        "high_response": ("high_response_auprc", "higher"),
    }
    for task, (threshold_key, direction) in mapping.items():
        row = frame[
            (frame["method_name"] == "chronaris") & (frame["task"] == task)
        ].iloc[0]
        threshold = float(thresholds[threshold_key]["strict"])
        value = float(row["mean"])
        threshold_passed = value > threshold if direction == "higher" else value < threshold
        checks.append(
            {
                "task": task,
                "value": value,
                "rank": int(row["rank"]),
                "strict_threshold": threshold,
                "threshold_passed": threshold_passed,
                "rank_first": int(row["rank"]) == 1,
            }
        )
    return {
        "checks": checks,
        "promotion_passed": all(
            row["threshold_passed"] and row["rank_first"] for row in checks
        ),
    }


def _load_and_validate_authorization(config):
    root = Path(config.protocol_root)
    lock = json.loads((root / "locked_configuration.json").read_text(encoding="utf-8"))
    authorization = json.loads(
        (root / "outer_test_access_lock.json").read_text(encoding="utf-8")
    )
    if authorization.get("authorized_open_count") != 1:
        raise PermissionError("outer confirmation does not have exactly one authorization")
    if authorization.get("consumed"):
        raise PermissionError("outer confirmation authorization was already consumed")
    if authorization.get("locked_configuration_sha256") != lock.get(
        "locked_configuration_sha256"
    ):
        raise PermissionError("outer confirmation authorization hash changed")
    lock_for_hash = dict(lock)
    embedded_hash = str(lock_for_hash.pop("locked_configuration_sha256"))
    if stable_mapping_sha256(lock_for_hash) != embedded_hash:
        raise PermissionError("locked outer configuration content changed")
    for path_key, hash_key in (
        ("development_result_path", "development_result_sha256"),
        ("consumer_summary_path", "consumer_summary_sha256"),
        ("nested_target_path", "nested_target_sha256"),
    ):
        if sha256_file(lock[path_key]) != lock[hash_key]:
            raise PermissionError(f"locked confirmation input changed: {path_key}")
    if str(config.nested_target_path) != str(lock["nested_target_path"]):
        raise PermissionError("confirmation target path differs from the locked path")
    return lock, authorization


def _mark_authorization_consumed(config, authorization, *, run_id: str):
    path = Path(config.protocol_root) / "outer_test_access_lock.json"
    path.write_text(
        json.dumps(
            {
                **authorization,
                "consumed": True,
                "consumed_by_run_id": run_id,
                "actual_open_count": 1,
                "outer_test_opened": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_float(config: str, key: str) -> float:
    match = re.search(rf"(?:^|;){re.escape(key)}=([0-9.]+)", config)
    if match is None:
        raise ValueError(f"locked head configuration lacks {key}: {config}")
    return float(match.group(1))


def _positions(index, sample_ids):
    return np.asarray([index[value] for value in sample_ids], dtype=np.int64)


def _metric_row(seed, fold_id, method, task, metric, value, direction):
    return {
        "seed": seed,
        "fold_id": fold_id,
        "method_name": method,
        "task": task,
        "metric": metric,
        "value": value,
        "direction": direction,
        "role": "held_out",
    }


def _prediction_rows(seed, fold_id, method, task, sample_ids, truth, prediction):
    return [
        {
            "seed": seed,
            "fold_id": fold_id,
            "method_name": method,
            "task": task,
            "sample_id": sample_id,
            "truth": float(truth[index]),
            "prediction": float(prediction[index]),
            "role": "held_out",
        }
        for index, sample_id in enumerate(sample_ids)
    ]


def _write_csv(path: Path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, overall, promotion):
    frame = pd.DataFrame(overall)
    task_labels = {
        "maneuver": "鼎新机动强度分类",
        "response": "鼎新未来生理响应预测",
        "high_response": "鼎新高生理响应识别",
    }
    lines = [
        "# Chronaris 核心任务恢复一次性确认",
        "",
        "本结果来自锁定配置后的三个随机种子和三个鼎新外层主折。",
        "",
    ]
    for task in ("maneuver", "response", "high_response"):
        rows = frame[frame["task"] == task].sort_values("rank")
        leader = rows.iloc[0]
        chronaris = rows[rows["method_name"] == "chronaris"].iloc[0]
        lines.append(
            f"- {task_labels[task]}：Chronaris={chronaris['mean']:.6f}"
            f"（第 {int(chronaris['rank'])}），"
            f"领先方法={leader['method_name']}（{leader['mean']:.6f}）。"
        )
    lines.extend(
        (
            "",
            f"- 严格晋级：`{str(promotion['promotion_passed']).lower()}`。",
            "- 历史确认产物未覆盖，本 run 仅新增一次性确认结果。",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")
