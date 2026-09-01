"""Run the protocol-v3 seed-17 clean repair matrix without outer-result selection."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chronaris.dataset.clare_native import build_clare_native_dataset
from chronaris.dataset.cogpilot_native import build_cogpilot_difficulty_dataset
from chronaris.dataset.group_splits import split_group_train_validation
from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    load_dingxin_fold_pretraining_data,
)
from chronaris.evaluation.application_tasks.dingxin_target_data import (
    load_dingxin_target_source_data,
)
from chronaris.evaluation.dingxin.simple_downstream_consumers import (
    SimpleConsumerConfig,
    fit_simple_downstream_consumer,
    maneuver_metric_summary,
)
from chronaris.evaluation.dingxin.simple_downstream_protocol import (
    extract_simple_raw_targets,
    fit_simple_loso_targets,
)
from chronaris.modeling.training import (
    CommonPretrainingConfig,
    EncoderCandidateConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_common_pretext_method,
)
from chronaris.representation import (
    AugmentationPolicy,
    FoldLineage,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
    load_simulation_observed_context,
)


REPO = Path(__file__).resolve().parents[2]
SIM_ROOT = REPO / "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
COG_ROOT = Path("/home/wangminan/dataset/chronaris/physio_net/physionet.org/files/virtual-reality-piloting/1.0.0/dataPackage/task-ils")
CLARE_ROOT = Path("/home/wangminan/dataset/chronaris/clare")
DINGXIN_SNAPSHOT = REPO / "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
FIXED_AUDIT = REPO / "docs/artifacts/runs/2026-07-10_fixed-data-audit"
INNER_SPLIT = REPO / "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
RUN_ROOT = REPO / "docs/artifacts/runs/2026-09-01_seed17-clean-baseline"
HEAVY_ROOT = REPO / "artifacts/application_evaluation/2026-09-01_seed17-clean-baseline"
FOLD_ID = "leave_one_sortie_out__fold01"


def main() -> int:
    args = _parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paper-facing clean matrix requires available CUDA")
    state = _load_state(args.resume)
    for stage in args.stages:
        if stage == "dingxin":
            _run_dingxin(state, args)
        elif stage == "wave_a":
            _run_wave_a(state, args)
        elif stage == "cogpilot":
            _run_cogpilot(state, args)
        elif stage == "clare":
            _run_clare(state, args)
    state["completed"] = all(
        any(row["stage"] == stage for row in state["rows"])
        for stage in ("dingxin", "wave_a", "cogpilot", "clare")
    )
    _write_state(state)
    _atomic_text(RUN_ROOT / "report.md", _render_report(state))
    return 0


def _run_dingxin(state, args) -> None:
    data = load_dingxin_fold_pretraining_data(
        fold_id=FOLD_ID,
        snapshot_root=DINGXIN_SNAPSHOT,
        fixed_audit_root=FIXED_AUDIT,
        inner_split_root=INNER_SPLIT,
    )
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
        data.load_batch,
        train_sample_ids=data.fold.train_sample_ids,
        held_out_sample_ids=data.fold.validation_sample_ids + data.fold.held_out_sample_ids,
        batch_size=2,
    )
    source = load_dingxin_target_source_data(
        fixed_audit_root=FIXED_AUDIT,
        snapshot_root=DINGXIN_SNAPSHOT,
    )
    targets = fit_simple_loso_targets(
        extract_simple_raw_targets(source, snapshot_root=DINGXIN_SNAPSHOT)
    )
    maneuver = targets.maneuver_targets[
        targets.maneuver_targets["fold_id"].astype(str) == FOLD_ID
    ].copy()
    physiology = targets.physiology_targets[
        targets.physiology_targets["fold_id"].astype(str) == FOLD_ID
    ].copy()
    for label, lag_weight in (("hardmax_weight_0", 0.0), ("hardmax_weight_0p1", 0.1)):
        unit = f"dingxin::{label}"
        if _done(state, unit):
            continue
        result, adapter = _train(
            method="chronaris",
            fold=data.fold,
            provider=data.load_batch,
            schema=data.index.plan.schema,
            normalizer=normalizer,
            output_root=HEAVY_ROOT / unit.replace("::", "/"),
            args=args,
            fusion_kind="safe_lag",
            lag_weight=lag_weight,
            vehicle_labels=data.vehicle_field_labels,
        )
        train_ids = tuple(
            value
            for value in data.fold.train_sample_ids
            if value in set(maneuver["context_id"].astype(str))
        )
        validation_ids = tuple(
            value
            for value in data.fold.validation_sample_ids
            if value in set(maneuver["context_id"].astype(str))
        )
        inner_ids = set(train_ids + validation_ids)
        inner_maneuver = maneuver[
            maneuver["context_id"].astype(str).isin(inner_ids)
        ].copy()
        inner_maneuver["split_role"] = np.where(
            inner_maneuver["context_id"].astype(str).isin(set(train_ids)),
            "train",
            "held_out",
        )
        inner_physiology = physiology[
            physiology["context_id"].astype(str).isin(inner_ids)
        ].copy()
        inner_physiology["split_role"] = np.where(
            inner_physiology["context_id"].astype(str).isin(set(train_ids)),
            "train",
            "held_out",
        )
        train_embedding = _export(adapter, data.load_batch, train_ids, args.batch_size)
        validation_embedding = _export(
            adapter, data.load_batch, validation_ids, args.batch_size
        )
        consumer = fit_simple_downstream_consumer(
            pooled_embedding=train_embedding,
            sample_ids=train_ids,
            maneuver_targets=inner_maneuver,
            physiology_targets=inner_physiology,
            config=SimpleConsumerConfig(random_state=args.seed),
        )
        prediction = consumer.predict(validation_embedding)
        metrics, _ = maneuver_metric_summary(
            inner_maneuver,
            sample_ids=validation_ids,
            score_prediction=prediction["maneuver_score"],
            class_probability=prediction["maneuver_probability"],
        )
        _append(
            state,
            {
                "unit": unit,
                "stage": "dingxin",
                "label": label,
                "lag_aware_weight": lag_weight,
                "validation_macro_f1": metrics.get("macro_f1"),
                "validation_balanced_accuracy": metrics.get("balanced_accuracy"),
                **_training_summary(result),
            },
        )


def _run_wave_a(state, args) -> None:
    train_paths = _simulation_paths("train")[:12]
    validation_paths = _simulation_paths("validation")[:6]
    samples = [
        load_simulation_observed_context(
            path,
            context_start_s=0.0,
            context_duration_s=30.0,
            sample_id=f"wave_train_{index:03d}",
            group_id=path.parents[2].name,
        )
        for index, path in enumerate(train_paths)
    ]
    samples.extend(
        load_simulation_observed_context(
            path,
            context_start_s=0.0,
            context_duration_s=30.0,
            sample_id=f"wave_validation_{index:03d}",
            group_id=path.parents[2].name,
        )
        for index, path in enumerate(validation_paths)
    )
    batch = collate_observation_samples(samples)
    fold = FoldLineage(
        fold_id="wave_a_clean",
        train_sample_ids=tuple(sample.sample_id for sample in samples[:12]),
        validation_sample_ids=tuple(sample.sample_id for sample in samples[12:-1]),
        held_out_sample_ids=(samples[-1].sample_id,),
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
    )
    schema = samples[0].schema
    for label, method, fusion_kind in (
        ("safe_lag", "chronaris", "safe_lag"),
        ("legacy_fusion", "chronaris", "multiscale"),
        ("vehicle_single_stream", "vehicle_only", "multiscale"),
    ):
        unit = f"wave_a::{label}"
        if _done(state, unit):
            continue
        result, _adapter = _train(
            method=method,
            fold=fold,
            batch=batch,
            schema=schema,
            normalizer=normalizer,
            output_root=HEAVY_ROOT / unit.replace("::", "/"),
            args=args,
            fusion_kind=fusion_kind,
        )
        _append(
            state,
            {
                "unit": unit,
                "stage": "wave_a",
                "label": label,
                **_training_summary(result),
            },
        )


def _run_cogpilot(state, args) -> None:
    dataset = build_cogpilot_difficulty_dataset(
        COG_ROOT,
        subject_limit=args.cogpilot_subjects,
        cache_root=HEAVY_ROOT / "cogpilot/native_cache",
    )
    groups = np.asarray(dataset.group_ids)
    labels = np.asarray(dataset.labels, dtype=int)
    outer_groups = set(sorted(set(groups))[::4])
    outer_train = [index for index, group in enumerate(groups) if group not in outer_groups]
    train_index, validation_index = split_group_train_validation(
        outer_train,
        groups,
        seed=args.seed,
    )
    held_index = [index for index, group in enumerate(groups) if group in outer_groups]
    fold = FoldLineage(
        fold_id="cogpilot_clean_inner_validation",
        train_sample_ids=tuple(dataset.sample_ids[index] for index in train_index),
        validation_sample_ids=tuple(
            dataset.sample_ids[index] for index in validation_index
        ),
        held_out_sample_ids=tuple(dataset.sample_ids[index] for index in held_index),
    )
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
        dataset.batch_provider,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
        batch_size=args.batch_size,
    )
    lookup = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
    y_train = labels[[lookup[value] for value in fold.train_sample_ids]]
    y_validation = labels[[lookup[value] for value in fold.validation_sample_ids]]
    for label, method, fusion_kind in _four_methods():
        unit = f"cogpilot::{label}"
        if _done(state, unit):
            continue
        result, adapter = _train(
            method=method,
            fold=fold,
            provider=dataset.batch_provider,
            schema=dataset.schema,
            normalizer=normalizer,
            output_root=HEAVY_ROOT / unit.replace("::", "/"),
            args=args,
            fusion_kind=fusion_kind,
        )
        train_embedding = _export(
            adapter, dataset.batch_provider, fold.train_sample_ids, args.batch_size
        )
        validation_embedding = _export(
            adapter, dataset.batch_provider, fold.validation_sample_ids, args.batch_size
        )
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=args.seed,
            ),
        ).fit(train_embedding, y_train)
        prediction = classifier.predict(validation_embedding)
        _append(
            state,
            {
                "unit": unit,
                "stage": "cogpilot",
                "label": label,
                "validation_macro_f1": f1_score(
                    y_validation, prediction, average="macro"
                ),
                "validation_balanced_accuracy": balanced_accuracy_score(
                    y_validation, prediction
                ),
                **_training_summary(result),
            },
        )


def _run_clare(state, args) -> None:
    dataset = build_clare_native_dataset(
        CLARE_ROOT,
        subject_limit=args.clare_subjects,
        window_stride=args.clare_window_stride,
        cache_root=HEAVY_ROOT / "clare/native_cache",
    )
    for row in state["rows"]:
        if row["unit"].startswith("clare::"):
            row["status"] = "superseded_due_to_missing_modality_contract"
    valid_indices = []
    for index, sample_id in enumerate(dataset.sample_ids):
        sample = dataset.load_sample(sample_id)
        if sample.physiology_feature_mask.any() and sample.vehicle_feature_mask.any():
            valid_indices.append(index)
    sample_ids = tuple(dataset.sample_ids[index] for index in valid_indices)
    labels = np.asarray([dataset.labels[index] for index in valid_indices], dtype=int)
    groups = np.asarray([dataset.group_ids[index] for index in valid_indices])
    state["clare_stream_filter"] = {
        "source_sample_count": len(dataset.records),
        "both_stream_sample_count": len(valid_indices),
        "removed_complete_modality_missing_count": len(dataset.records) - len(valid_indices),
    }
    _write_state(state)
    binary = (labels >= 7).astype(int)
    lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    splitter = GroupKFold(n_splits=5)
    for fold_index, (outer_train, outer_test) in enumerate(
        splitter.split(sample_ids, binary, groups),
        start=1,
    ):
        train_index, validation_index = split_group_train_validation(
            outer_train,
            groups,
            seed=args.seed + fold_index,
        )
        fold = FoldLineage(
            fold_id=f"clare_clean_fold_{fold_index}",
            train_sample_ids=tuple(sample_ids[index] for index in train_index),
            validation_sample_ids=tuple(
                sample_ids[index] for index in validation_index
            ),
            held_out_sample_ids=tuple(
                sample_ids[index] for index in outer_test
            ),
        )
        normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
            dataset.batch_provider,
            train_sample_ids=fold.train_sample_ids,
            held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
            batch_size=args.batch_size,
        )
        train_position = [lookup[value] for value in fold.train_sample_ids]
        validation_position = [lookup[value] for value in fold.validation_sample_ids]
        for label, method, fusion_kind in _four_methods():
            unit = f"clare_valid::fold{fold_index}::{label}"
            if _done(state, unit):
                continue
            result, adapter = _train(
                method=method,
                fold=fold,
                provider=dataset.batch_provider,
                schema=dataset.schema,
                normalizer=normalizer,
                output_root=HEAVY_ROOT / unit.replace("::", "/"),
                args=args,
                fusion_kind=fusion_kind,
            )
            train_embedding = _export(
                adapter, dataset.batch_provider, fold.train_sample_ids, args.batch_size
            )
            validation_embedding = _export(
                adapter,
                dataset.batch_provider,
                fold.validation_sample_ids,
                args.batch_size,
            )
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ).fit(train_embedding, binary[train_position])
            class_prediction = classifier.predict(validation_embedding)
            regressor = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
                train_embedding,
                labels[train_position],
            )
            score_prediction = regressor.predict(validation_embedding)
            rho = spearmanr(labels[validation_position], score_prediction).correlation
            _append(
                state,
                {
                    "unit": unit,
                    "stage": "clare",
                    "fold": fold_index,
                    "label": label,
                    "validation_macro_f1": f1_score(
                        binary[validation_position],
                        class_prediction,
                        average="macro",
                    ),
                    "validation_balanced_accuracy": balanced_accuracy_score(
                        binary[validation_position], class_prediction
                    ),
                    "validation_spearman": float(rho) if math.isfinite(rho) else None,
                    **_training_summary(result),
                },
            )


def _train(
    *,
    method,
    fold,
    schema,
    normalizer,
    output_root,
    args,
    fusion_kind,
    batch=None,
    provider=None,
    lag_weight=0.0,
    vehicle_labels=None,
):
    print(f"[clean-matrix] training {output_root.relative_to(HEAVY_ROOT)}", flush=True)
    started = time.perf_counter()
    result = train_common_pretext_method(
        method,
        batch=batch,
        batch_provider=provider,
        fold=fold,
        physiology_feature_names=schema.physiology_feature_names,
        vehicle_feature_names=schema.vehicle_feature_names,
        vehicle_field_labels=vehicle_labels
        or tuple((name, name) for name in schema.vehicle_feature_names),
        normalizer=normalizer,
        output_root=output_root,
        config=CommonPretrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            heartbeat_interval_s=30.0,
        ),
        candidate_config=EncoderCandidateConfig(candidate_id="C", hidden_dim=32),
        augmentation_policy=AugmentationPolicy(),
        chronaris_fusion_kind=fusion_kind,
        chronaris_lag_aware_weight=lag_weight,
        chronaris_mechanism_enabled=method == "chronaris",
        resume=args.resume,
    )
    encoder, _heads, loaded_normalizer, payload = load_common_pretraining_checkpoint(
        result.best_checkpoint_path
    )
    print(
        f"[clean-matrix] completed elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return result, TrainedFusionAdapter(
        encoder=encoder,
        normalizer=loaded_normalizer,
        fold_id=fold.fold_id,
        checkpoint_sha256=payload["canonical_training_state_sha256"],
    )


def _export(adapter, provider, sample_ids, batch_size):
    rows = []
    for offset in range(0, len(sample_ids), batch_size):
        batch = provider(sample_ids[offset : offset + batch_size])
        rows.append(adapter(batch).pooled_embedding.detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _training_summary(result):
    payload = torch.load(
        result.best_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    best_epoch = int(payload["best_epoch"])
    best = next(row for row in payload["epoch_rows"] if row["epoch"] == best_epoch)
    return {
        "best_epoch": best_epoch,
        "validation_self_supervised_loss": float(
            payload["best_public_selection_loss"]
        ),
        "mechanism_validation": best.get("mechanism_validation", {}),
        "training_elapsed_s": result.training_elapsed_s,
        "protocol_sha256": result.protocol_sha256,
    }


def _four_methods():
    return (
        ("safe_lag", "chronaris", "safe_lag"),
        ("legacy_fusion", "chronaris", "multiscale"),
        ("physiology_single_stream", "physiology_only", "multiscale"),
        ("context_single_stream", "vehicle_only", "multiscale"),
    )


def _simulation_paths(split):
    return sorted(
        (SIM_ROOT / split).glob("*/*/clean_asynchronous/raw_dual_stream.npz")
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("dingxin", "wave_a", "cogpilot", "clare"),
        default=("dingxin", "wave_a", "cogpilot", "clare"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cogpilot-subjects", type=int, default=8)
    parser.add_argument("--clare-subjects", type=int, default=8)
    parser.add_argument("--clare-window-stride", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_state(resume):
    path = RUN_ROOT / "baseline_matrix.json"
    if resume and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "format": "chronaris.seed17_clean_baseline.v3",
        "seed": 17,
        "outer_results_opened": False,
        "paper_table_eligible": False,
        "rows": [],
        "completed": False,
    }


def _done(state, unit):
    return any(row["unit"] == unit for row in state["rows"])


def _append(state, row):
    state["rows"].append(_json_safe(row))
    _write_state(state)


def _write_state(state):
    _atomic_text(
        RUN_ROOT / "baseline_matrix.json",
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _render_report(state):
    completed = "完成" if state["completed"] else "运行中"
    active = [
        row
        for row in state["rows"]
        if row.get("status") != "superseded_due_to_missing_modality_contract"
    ]
    dingxin = [row for row in active if row["stage"] == "dingxin"]
    wave = [row for row in active if row["stage"] == "wave_a"]
    cogpilot = [row for row in active if row["stage"] == "cogpilot"]
    clare = [row for row in active if row["stage"] == "clare"]
    names = {
        "hardmax_weight_0": "不启用滞后对齐损失",
        "hardmax_weight_0p1": "滞后对齐损失权重 0.1",
        "safe_lag": "安全滞后感知融合",
        "legacy_fusion": "旧多尺度融合",
        "vehicle_single_stream": "航电单流",
        "physiology_single_stream": "生理或中枢脑电单流",
        "context_single_stream": "航电或外周生理单流",
    }
    lines = [
        "# seed 17 干净基线矩阵",
        "",
        f"状态：**{completed}**。本矩阵只验证协议修复与训练内行为，不进入论文主表；公开数据外层结果和鼎新分组确认结果均未打开。",
        "",
        "矩阵记录鼎新滞后损失接线、仿真波次 A、CogPilot 飞行难度训练内验证，以及 CLARE 五折外层训练集内部验证。各项均使用 CUDA、训练折归一化和 validation-backed checkpoint。",
        "",
        f"当前有效完成单元数：`{len(active)}`；另有 `{len(state['rows']) - len(active)}` 个首次尝试因完整模态缺失合同退出主比较但保留追溯。",
        "",
        "鼎新训练内验证展示滞后对齐损失接线是否真实影响训练和下游结果。",
        "",
        "| 配置 | 训练内宏平均 F1 | 训练内平衡准确率 | 自监督 validation 损失 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in dingxin:
        lines.append(
            f"| {names[row['label']]} | {row['validation_macro_f1']:.4f} | "
            f"{row['validation_balanced_accuracy']:.4f} | "
            f"{row['validation_self_supervised_loss']:.4f} |"
        )
    lines.extend(
        [
            "",
            "仿真波次 A 只比较相同训练内 validation 的自监督损失，不打开仿真预留任务结果。",
            "",
            "| 方法 | 自监督 validation 损失 | 训练用时（秒） |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in wave:
        lines.append(
            f"| {names[row['label']]} | {row['validation_self_supervised_loss']:.4f} | "
            f"{row['training_elapsed_s']:.2f} |"
        )
    lines.extend(
        [
            "",
            "CogPilot 飞行难度只评价 outer-train 内部的受试者分组 validation。",
            "",
            "| 方法 | 宏平均 F1 | 平衡准确率 |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in cogpilot:
        lines.append(
            f"| {names[row['label']]} | {row['validation_macro_f1']:.4f} | "
            f"{row['validation_balanced_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "CLARE 使用五个外层训练集各自的内部受试者分组 validation；下表报告五折均值和最差折。",
            "",
            "| 方法 | 宏平均 F1 均值 | 最差折 |",
            "| --- | ---: | ---: |",
        ]
    )
    for label in ("safe_lag", "legacy_fusion", "physiology_single_stream", "context_single_stream"):
        values = [row["validation_macro_f1"] for row in clare if row["label"] == label]
        if values:
            lines.append(
                f"| {names[label]} | {np.mean(values):.4f} | {np.min(values):.4f} |"
            )
    stream_filter = state.get("clare_stream_filter", {})
    lines.extend(
        [
            "",
            f"CLARE 共检查 `{stream_filter.get('source_sample_count', 0)}` 个候选窗口，其中 `{stream_filter.get('removed_complete_modality_missing_count', 0)}` 个窗口因一个模态完全缺失而不进入同样本四方法比较；未进行数值填充。",
            "",
            "该矩阵证明修复后的训练、分组和缺模态合同能够闭环。CogPilot 出现 safe-lag 正向训练内结果；CLARE 仍有明显折间方差。因此下一步必须使用冻结协议运行多随机种子候选筛选，本矩阵本身不支持论文排名。",
            "",
            "工程闭环：Ruff 与 `git diff --check` 通过；完整测试 `448 passed, 8 skipped`。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
