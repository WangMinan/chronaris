"""Audit outer-fold time support and invalidate overlap-contaminated results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.core_recovery_splits import (
    context_support_intervals,
)


@dataclass(frozen=True, slots=True)
class CoreRecoveryOuterAuditConfig:
    protocol_root: str = (
        "docs/artifacts/runs/2026-07-13_chronaris-core-task-recovery"
    )
    confirmation_root: str = (
        "docs/artifacts/runs/2026-07-13_chronaris-core-task-recovery-confirmation"
    )
    context_manifest_path: str = (
        "docs/artifacts/runs/2026-07-10_fixed-data-audit/"
        "context_sample_manifest.jsonl"
    )
    split_manifest_path: str = (
        "docs/artifacts/runs/2026-07-11_dingxin-inner-splits/split_manifest.json"
    )


def inspect_outer_support_isolation(
    *,
    context_manifest_path: str | Path,
    split_manifest_path: str | Path,
    fold_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    contexts = pd.read_json(context_manifest_path, lines=True)
    intervals = {
        row.context_id: row for row in context_support_intervals(contexts)
    }
    split = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    selected = set(fold_ids) if fold_ids is not None else None
    rows = []
    for plan in split["folds"]:
        fold_id = str(plan["fold_id"])
        if selected is not None and fold_id not in selected:
            continue
        train = [intervals[str(value)] for value in plan["train_sample_ids"]]
        held_out = [intervals[str(value)] for value in plan["held_out_sample_ids"]]
        overlap_pairs = [
            (left, right)
            for left in train
            for right in held_out
            if left.overlaps(right)
        ]
        train_overlap = {left.context_id for left, _right in overlap_pairs}
        held_overlap = {right.context_id for _left, right in overlap_pairs}
        exact_anchor_pairs = sum(
            left.sortie_id == right.sortie_id
            and left.anchor_offset_ms == right.anchor_offset_ms
            for left in train
            for right in held_out
        )
        rows.append(
            {
                "fold_id": fold_id,
                "train_context_count": len(train),
                "held_out_context_count": len(held_out),
                "overlap_pair_count": len(overlap_pairs),
                "overlapping_train_context_count": len(train_overlap),
                "overlapping_held_out_context_count": len(held_overlap),
                "exact_sortie_anchor_pair_count": exact_anchor_pairs,
                "support_isolated": not overlap_pairs,
            }
        )
    if selected is not None and {row["fold_id"] for row in rows} != selected:
        raise ValueError("outer support audit did not find every requested fold")
    valid = bool(rows) and all(row["support_isolated"] for row in rows)
    return {
        "format": "chronaris.core_task_recovery_outer_support_audit.v1",
        "valid": valid,
        "folds": rows,
        "invalid_reason": (
            None if valid else "outer_train_and_held_out_time_support_overlap"
        ),
    }


def audit_and_invalidate_confirmation(
    config: CoreRecoveryOuterAuditConfig,
) -> dict[str, object]:
    root = Path(config.confirmation_root)
    audit = inspect_outer_support_isolation(
        context_manifest_path=config.context_manifest_path,
        split_manifest_path=config.split_manifest_path,
        fold_ids=tuple(f"leave_one_view_out__fold0{index}" for index in range(1, 4)),
    )
    pd.DataFrame(audit["folds"]).to_csv(
        root / "outer_support_overlap_audit.csv",
        index=False,
    )
    (root / "outer_support_overlap_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if audit["valid"]:
        return audit

    clean_folds = [
        row["fold_id"] for row in audit["folds"] if row["support_isolated"]
    ]
    metrics = pd.read_csv(root / "metric_long.csv")
    diagnostic = (
        metrics[metrics["fold_id"].isin(clean_folds)]
        .groupby(
            ["method_name", "task", "metric", "direction"],
            as_index=False,
            sort=True,
        )
        .agg(mean=("value", "mean"), std=("value", "std"), count=("value", "count"))
    )
    diagnostic.to_csv(root / "isolated_fold_diagnostic.csv", index=False)

    promotion_path = root / "promotion_audit.json"
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion["raw_metric_promotion_passed"] = bool(promotion["promotion_passed"])
    promotion["promotion_passed"] = False
    promotion["protocol_valid"] = False
    promotion["invalid_reason"] = audit["invalid_reason"]
    promotion_path.write_text(
        json.dumps(promotion, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    evidence_path = root / "evidence_manifest.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence.update(
        {
            "status": "invalid_protocol_overlap",
            "promotion_passed": False,
            "claim_eligible": False,
            "invalid_reason": audit["invalid_reason"],
            "isolated_fold_ids": clean_folds,
            "outer_support_audit_path": str(
                root / "outer_support_overlap_audit.json"
            ),
        }
    )
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    progress_path = root / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update(evidence)
    progress_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _update_protocol_root(config, audit)
    _write_invalid_report(root / "report.md", audit, diagnostic)
    return audit


def _write_invalid_report(path: Path, audit, diagnostic: pd.DataFrame) -> None:
    labels = {
        "maneuver": "鼎新机动强度分类宏平均 F1（Macro-F1）",
        "response": "鼎新未来生理响应预测均方根误差（RMSE）",
        "high_response": "鼎新高生理响应识别精确率—召回率曲线下面积（AUPRC）",
    }
    chronaris = diagnostic[diagnostic["method_name"] == "chronaris"]
    lines = [
        "# Chronaris 核心任务恢复一次性确认审计",
        "",
        "结论：本次一次性外层确认存在训练与外层留出集时间支持重叠，",
        "原始三折汇总不可用于论文结论或模型晋级。唯一授权已经消费，不重跑、不调参。",
        "",
    ]
    for index, row in enumerate(audit["folds"], start=1):
        lines.append(
            f"- 第 {index} 个留一视图折：重叠训练上下文 "
            f"{row['overlapping_train_context_count']}/"
            f"{row['train_context_count']}，精确同架次同锚点配对 "
            f"{row['exact_sortie_anchor_pair_count']}。"
        )
    lines.extend(("", "仅将无重叠折作为失败诊断，不作为正式汇总：", ""))
    for row in chronaris.itertuples(index=False):
        lines.append(
            f"- {labels[str(row.task)]}：{float(row.mean):.6f}"
            f"（3 个随机种子，1 个无重叠折）。"
        )
    lines.extend(
        (
            "",
            "后续若需要重新确认，必须先重建按架次隔离或带完整 35 秒支持清除的外层协议，",
            "并由新的预注册授权启动；不得复用本次外层结果继续选模。",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _update_protocol_root(config: CoreRecoveryOuterAuditConfig, audit) -> None:
    root = Path(config.protocol_root)
    evidence_path = root / "evidence_manifest.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence.update(
        {
            "status": "closed_invalid_protocol_overlap",
            "outer_test_opened": True,
            "outer_test_access_count": 1,
            "outer_test_authorization_consumed": True,
            "confirmed_metrics_changed": False,
            "simulation_secondary_started": False,
            "claim_eligible": False,
            "invalid_reason": audit["invalid_reason"],
            "confirmation_run": (
                "docs/artifacts/runs/"
                "2026-07-13_chronaris-core-task-recovery-confirmation"
            ),
        }
    )
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "report.md").write_text(
        "\n".join(
            (
                "# Chronaris 核心任务恢复协议收口",
                "",
                "唯一外层授权已经消费；一次性确认完成后发现前两个留一视图折",
                "存在训练与外层留出集完整支持区间重叠。确认产物不可用于论文或模型晋级，",
                "仿真次级确认未启动，历史确认数值未修改。",
                "",
            )
        ),
        encoding="utf-8",
    )
