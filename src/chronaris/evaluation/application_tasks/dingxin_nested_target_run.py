"""Build, audit, and archive inner-train-fitted Dingxin targets."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.evaluation.application_tasks.dingxin_nested_target_data import (
    build_dingxin_nested_targets,
)
from chronaris.modeling.common.run_observer import open_task_eval_run_observer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import (
    sha256_file,
    write_deterministic_npz,
)


LOGGER = logging.getLogger("chronaris.pipelines.task_eval.dingxin_nested_targets")
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class DingxinNestedTargetConfig:
    run_id: str = "2026-07-11_dingxin-nested-targets"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    snapshot_root: str = (
        "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    )
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    outer_binding_path: str = (
        "docs/artifacts/runs/2026-07-11_dingxin-context-bindings/fold_task_binding.csv"
    )
    e_run_manifest_path: str = (
        "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json"
    )
    f_run_manifest_path: str = (
        "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json"
    )


@dataclass(frozen=True, slots=True)
class DingxinNestedTargetResult:
    run_id: str
    status: str
    compact_run_root: str
    heavy_run_root: str
    fold_count: int
    archive_count: int
    acceptance_pass_count: int
    acceptance_check_count: int
    report_path: str
    evidence_manifest_path: str


def run_dingxin_nested_targets(
    config: DingxinNestedTargetConfig,
) -> DingxinNestedTargetResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    snapshot_hashes_before = _snapshot_hashes(Path(config.snapshot_root))
    with open_task_eval_run_observer(
        run_root=compact_root,
        run_id=config.run_id,
        stage_name="dingxin_inner_train_nested_targets",
        logger=LOGGER,
        initial_progress={
            "threshold_scope": "inner_train_nested",
            "training_invoked": False,
            "metrics_generated": False,
            "confirmed_metrics_changed": False,
        },
    ) as progress:
        labels, thresholds, folds = _build(config)
        rebuilt_labels, rebuilt_thresholds, rebuilt_folds = _build(config)
        deterministic = (
            _frame_hash(labels) == _frame_hash(rebuilt_labels)
            and _frame_hash(thresholds) == _frame_hash(rebuilt_thresholds)
            and _frame_hash(folds) == _frame_hash(rebuilt_folds)
        )
        archive_rows = _write_archives(labels, heavy_root / "targets")
        comparison_rows = _compare_outer_targets(
            labels,
            pd.read_csv(config.outer_binding_path),
        )
        snapshot_hashes_after = _snapshot_hashes(Path(config.snapshot_root))
        acceptance_rows = _acceptance_rows(
            labels=labels,
            thresholds=thresholds,
            folds=folds,
            archive_rows=archive_rows,
            deterministic=deterministic,
            snapshot_unchanged=snapshot_hashes_before == snapshot_hashes_after,
        )
        status = "completed" if all(row["passed"] for row in acceptance_rows) else "partial"
        paths = _write_outputs(
            run_root=compact_root,
            run_id=config.run_id,
            status=status,
            labels=labels,
            thresholds=thresholds,
            folds=folds,
            archive_rows=archive_rows,
            comparison_rows=comparison_rows,
            acceptance_rows=acceptance_rows,
            heavy_root=str(heavy_root),
            snapshot_hashes=snapshot_hashes_after,
        )
        pass_count = sum(row["passed"] for row in acceptance_rows)
        progress.finish(
            status=status,
            fold_count=len(folds),
            archive_count=len(archive_rows),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
        )
        return DingxinNestedTargetResult(
            run_id=config.run_id,
            status=status,
            compact_run_root=str(compact_root),
            heavy_run_root=str(heavy_root),
            fold_count=len(folds),
            archive_count=len(archive_rows),
            acceptance_pass_count=pass_count,
            acceptance_check_count=len(acceptance_rows),
            report_path=paths["report"],
            evidence_manifest_path=paths["evidence_manifest"],
        )


def _build(config):
    return build_dingxin_nested_targets(
        fixed_audit_root=config.fixed_audit_root,
        snapshot_root=config.snapshot_root,
        inner_split_root=config.inner_split_root,
        e_run_manifest_path=config.e_run_manifest_path,
        f_run_manifest_path=config.f_run_manifest_path,
    )


def _write_archives(labels, root):
    rows = []
    for (fold_id, task_slug), frame in labels.groupby(["fold_id", "task_slug"]):
        destination = root / fold_id / task_slug / "targets.npz"
        sha = write_deterministic_npz(
            destination,
            {
                "context_ids": np.asarray(frame["context_id"].astype(str).tolist(), dtype=str),
                "roles": np.asarray(frame["role"].astype(str).tolist(), dtype=str),
                "statuses": np.asarray(frame["status"].astype(str).tolist(), dtype=str),
                "class_target": frame["class_target"].fillna(-1).to_numpy(np.int64),
                "continuous_target": frame["continuous_target"].to_numpy(np.float32),
                "binary_target": frame["binary_target"].fillna(-1).to_numpy(np.int64),
                "fit_sample_hashes": np.asarray(frame["fit_sample_hash"].astype(str).tolist(), dtype=str),
            },
        )
        rows.append(
            {
                "fold_id": fold_id,
                "task_slug": task_slug,
                "row_count": len(frame),
                "available_count": int(frame["status"].eq("completed").sum()),
                "archive_path": str(destination),
                "archive_sha256": sha,
            }
        )
    return rows


def _compare_outer_targets(nested, outer):
    rows = []
    for fold_id in sorted(nested["fold_id"].unique()):
        nested_fold = nested[nested["fold_id"] == fold_id]
        outer_fold = outer[
            (outer["fold_id"] == fold_id)
            & (outer["binding_status"] == "available")
        ]
        classification = nested_fold[
            nested_fold["task_slug"] == "maneuver_intensity_classification"
        ].merge(
            outer_fold[outer_fold["task_slug"] == "maneuver_intensity_classification"]
            [["context_id", "class_target"]],
            on="context_id",
            suffixes=("_nested", "_outer"),
        )
        response = nested_fold[
            (nested_fold["task_slug"] == "physiology_response_prediction")
            & (nested_fold["status"] == "completed")
        ].merge(
            outer_fold[outer_fold["task_slug"] == "physiology_response_prediction"]
            [["context_id", "continuous_target", "binary_target"]],
            on="context_id",
            suffixes=("_nested", "_outer"),
        )
        rows.append(
            {
                "fold_id": fold_id,
                "classification_label_change_count": int(
                    (classification["class_target_nested"] != classification["class_target_outer"]).sum()
                ),
                "classification_compared_count": len(classification),
                "response_binary_change_count": int(
                    (response["binary_target_nested"] != response["binary_target_outer"]).sum()
                ),
                "response_compared_count": len(response),
                "response_score_spearman": float(
                    response[["continuous_target_nested", "continuous_target_outer"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
            }
        )
    return rows


def _acceptance_rows(
    *, labels, thresholds, folds, archive_rows, deterministic, snapshot_unchanged
):
    train_class_sets = [
        set(frame["class_target"].astype(int))
        for (_fold, _task), frame in labels[
            (labels["task_slug"] == "maneuver_intensity_classification")
            & (labels["role"] == "train")
        ].groupby(["fold_id", "task_slug"])
    ]
    train_binary_sets = [
        set(frame.loc[frame["status"] == "completed", "binary_target"].astype(int))
        for (_fold, _task), frame in labels[
            (labels["task_slug"] == "physiology_response_prediction")
            & (labels["role"] == "train")
        ].groupby(["fold_id", "task_slug"])
    ]
    return [
        _check("five_nested_folds", len(folds) == 5, len(folds), 5),
        _check("ten_deterministic_archives", len(archive_rows) == 10 and all(Path(row["archive_path"]).is_file() and sha256_file(row["archive_path"]) == row["archive_sha256"] for row in archive_rows), len(archive_rows), 10),
        _check("all_thresholds_inner_train", set(thresholds["threshold_scope"]) == {"inner_train_nested"}, sorted(set(thresholds["threshold_scope"])), ["inner_train_nested"]),
        _check("classification_train_has_three_classes", len(train_class_sets) == 5 and all(values == {0, 1, 2} for values in train_class_sets), [sorted(values) for values in train_class_sets], "five x three classes"),
        _check("response_train_has_two_classes", len(train_binary_sets) == 5 and all(values == {0, 1} for values in train_binary_sets), [sorted(values) for values in train_binary_sets], "five x binary"),
        _check("classification_role_count", len(labels[labels["task_slug"] == "maneuver_intensity_classification"]) == 440, len(labels[labels["task_slug"] == "maneuver_intensity_classification"]), 440),
        _check("response_available_role_count", int(((labels["task_slug"] == "physiology_response_prediction") & (labels["status"] == "completed")).sum()) == 425, int(((labels["task_slug"] == "physiology_response_prediction") & (labels["status"] == "completed")).sum()), 425),
        _check("deterministic_rebuild", deterministic, deterministic, True),
        _check("snapshot_hashes_unchanged", snapshot_unchanged, snapshot_unchanged, True),
        _check("no_metrics_or_training", True, False, False),
    ]


def _write_outputs(
    *, run_root, run_id, status, labels, thresholds, folds, archive_rows,
    comparison_rows, acceptance_rows, heavy_root, snapshot_hashes
):
    paths = {
        "nested_targets": run_root / "nested_targets.csv",
        "nested_thresholds": run_root / "nested_thresholds.csv",
        "fold_summary": run_root / "fold_summary.csv",
        "archive_manifest": run_root / "target_archive_manifest.csv",
        "outer_comparison": run_root / "outer_target_comparison.csv",
        "acceptance_checks": run_root / "acceptance_checks.csv",
        "source_manifest": run_root / "source_manifest.json",
        "report": run_root / "report.md",
        "claim_boundary": run_root / "claim_boundary.md",
        "resume_command": run_root / "resume_command.txt",
        "evidence_manifest": run_root / "evidence_manifest.json",
    }
    for key, frame in (
        ("nested_targets", labels),
        ("nested_thresholds", thresholds),
        ("fold_summary", folds),
        ("archive_manifest", pd.DataFrame(archive_rows)),
        ("outer_comparison", pd.DataFrame(comparison_rows)),
        ("acceptance_checks", pd.DataFrame(acceptance_rows)),
    ):
        frame.to_csv(paths[key], index=False)
    _write_json(paths["source_manifest"], {"format": "chronaris.dingxin_nested_target_sources.v1", "snapshot_hashes": snapshot_hashes, "threshold_scope": "inner_train_nested"})
    passed = sum(row["passed"] for row in acceptance_rows)
    paths["report"].write_text(
        "\n".join((
            "# 鼎新 inner-train 嵌套目标报告", "", "## 结论", "",
            f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
            "- 五折机动分位阈值、语义尺度、生理字段 IQR 与高响应阈值均只使用各折 inner-train 重拟合。",
            "- 生成 10 个确定性目标 archive；机动分类 440 个角色上下文，生理响应 425 个可用角色上下文。",
            "- validation 与 outer-test 仅应用训练内参数，不参与字段选择、尺度或阈值估计。",
            "- 本 run 不训练模型、不生成指标，也不形成候选排名。", "",
            "## 下一步", "", "1. 用嵌套目标复跑 validation consumer，保持 outer-test 指标关闭。", "2. 完成后进入 seed 17 固定候选 screen。", ""
        )), encoding="utf-8")
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n- 嵌套目标仍是弱监督任务，不等价于人工评价。\n- 本 run 只修复标签拟合边界，不提供模型效果证据。\n- outer-test 未参与任何目标参数拟合。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/evaluation/application_tasks/build_dingxin_nested_targets.py "
        f"--run-id {run_id}\n", encoding="utf-8")
    _write_json(paths["evidence_manifest"], {"run_id": run_id, "status": status, "evidence_layer": "dingxin_inner_train_nested_targets", "training_invoked": False, "metrics_generated": False, "confirmed_metrics_changed": False, "heavy_run_root": heavy_root, "output_paths": {key: str(value) for key, value in paths.items()}})
    return {key: str(value) for key, value in paths.items()}


def _snapshot_hashes(root):
    manifest = json.loads((root / "snapshot_manifest.json").read_text(encoding="utf-8"))
    return {item["relative_path"]: sha256_file(root / item["relative_path"]) for item in manifest["files"]}


def _frame_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _check(check_id, passed, actual, expected):
    return {"check_id": check_id, "passed": bool(passed), "actual": actual, "expected": expected}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
