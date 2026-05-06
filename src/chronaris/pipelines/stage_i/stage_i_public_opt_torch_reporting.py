"""Reporting and reference helpers for torch-native UAB public-opt runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from chronaris.pipelines.stage_i.stage_i_public_opt_reporting import fmt_public_opt_float


def load_torch_uab_reference_public_opt_summary(path_like: str | None) -> dict[str, object]:
    if not path_like:
        return {}
    path = Path(path_like)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    subset_results = payload.get("subset_results") or {}
    return {
        subset_id: {
            "best_head": result["best_head"],
            "rmse": float(result["heads"][result["best_head"]]["rmse"]),
            "mae": float(result["heads"][result["best_head"]]["mae"]),
        }
        for subset_id, result in subset_results.items()
    }


def load_torch_uab_reference_deep_summary(
    path_like: str | None,
    *,
    dataset_id: str,
) -> dict[str, object]:
    if not path_like:
        return {}
    path = Path(path_like)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset_payload = payload.get("datasets", {}).get(dataset_id)
    if not dataset_payload or dataset_payload.get("status") != "completed":
        return {}
    result: dict[str, object] = {}
    for model_name, model_payload in dataset_payload.get("models", {}).items():
        groups = (
            model_payload.get("summary", {})
            .get("subjective", {})
            .get("groups", {})
        )
        result[model_name] = {
            subset_id: {
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
            }
            for subset_id, metrics in groups.items()
        }
    return result


def build_torch_uab_acceptance(
    groups: Mapping[str, Mapping[str, object]],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    per_group = {}
    for subset_id, threshold in thresholds.items():
        rmse = float(groups[subset_id]["rmse"])
        per_group[subset_id] = {
            "threshold_rmse": threshold,
            "observed_rmse": rmse,
            "passed": rmse < threshold,
        }
    return {
        "groups": per_group,
        "all_passed": all(item["passed"] for item in per_group.values()),
    }


def render_torch_uab_report(summary: Mapping[str, object]) -> str:
    winning = summary["winning_candidate"]
    final_result = summary["final_result"]
    lines = [
        "# Stage I Public Opt UAB Torch Mainline",
        "",
        "## 运行口径",
        "",
        f"- run_id：`{summary['run_id']}`",
        f"- dataset_id：`{summary['dataset_id']}`",
        f"- profile：`{summary['profile']}`",
        f"- runtime_device：`{summary['runtime_device']}`",
        f"- prepared asset root：`{summary['prepared_artifact_root']}`",
        f"- output artifact root：`{summary['artifact_root']}`",
        f"- generated_at_utc：`{summary['generated_at_utc']}`",
        "",
        "## Screen Winner",
        "",
        f"- candidate_id：`{winning['candidate_id']}`",
        f"- model_family：`{winning['model_family']}`",
        f"- feature_profile：`{winning['feature_profile']}`",
        f"- hidden_dims：`{winning['hidden_dims']}`",
        f"- learning_rate：`{winning['learning_rate']}`",
        f"- weight_decay：`{winning['weight_decay']}`",
        f"- full_run_completed：`{summary['full_run_completed']}`",
        "",
        "## Screen Leaderboard",
        "",
        "| rank | candidate_id | feature_profile | mean_rmse | mean_mae | n_back_rmse | heat_the_chair_rmse |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(summary["screen_leaderboard"], start=1):
        lines.append(
            f"| {index} | `{row['candidate_id']}` | `{row['feature_profile']}` | "
            f"{fmt_public_opt_float(row['screen_mean_rmse'])} | "
            f"{fmt_public_opt_float(row['screen_mean_mae'])} | "
            f"{fmt_public_opt_float(row['n_back_rmse'])} | "
            f"{fmt_public_opt_float(row['heat_the_chair_rmse'])} |"
        )

    lines.extend(
        [
            "",
            "## Final LOSO Result",
            "",
            "| evaluation_group | rmse | mae | r2 | spearman |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for subset_id in ("n_back", "heat_the_chair"):
        metrics = final_result["groups"][subset_id]
        lines.append(
            f"| {subset_id} | {fmt_public_opt_float(metrics['rmse'])} | "
            f"{fmt_public_opt_float(metrics['mae'])} | {fmt_public_opt_float(metrics['r2'])} | "
            f"{fmt_public_opt_float(metrics['spearman'])} |"
        )

    acceptance = summary["acceptance"]
    lines.extend(
        [
            "",
            "## Acceptance Gate",
            "",
            "| evaluation_group | threshold_rmse | observed_rmse | passed |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for subset_id in ("n_back", "heat_the_chair"):
        payload = acceptance["groups"][subset_id]
        lines.append(
            f"| {subset_id} | {fmt_public_opt_float(payload['threshold_rmse'])} | "
            f"{fmt_public_opt_float(payload['observed_rmse'])} | `{payload['passed']}` |"
        )
    lines.append("")
    lines.append(f"- public_mainline_status：`{summary['public_mainline_status']}`")

    reference_public_opt = summary.get("reference_public_opt") or {}
    reference_deep = summary.get("reference_deep_models") or {}
    if reference_public_opt or reference_deep:
        lines.extend(["", "## 历史对照", ""])
        lines.extend(
            [
                "| evaluation_group | current_public_opt_rmse | MulT rmse | ContiFormer rmse | torch_uab_rmse |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for subset_id in ("n_back", "heat_the_chair"):
            public_opt_rmse = float(
                (reference_public_opt.get(subset_id) or {}).get("rmse", 0.0)
            )
            mult_rmse = float(
                (reference_deep.get("mult") or {}).get(subset_id, {}).get("rmse", 0.0)
            )
            contiformer_rmse = float(
                (reference_deep.get("contiformer") or {})
                .get(subset_id, {})
                .get("rmse", 0.0)
            )
            torch_rmse = float(final_result["groups"][subset_id]["rmse"])
            lines.append(
                f"| {subset_id} | {fmt_public_opt_float(public_opt_rmse)} | "
                f"{fmt_public_opt_float(mult_rmse)} | {fmt_public_opt_float(contiformer_rmse)} | "
                f"{fmt_public_opt_float(torch_rmse)} |"
            )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- NASA `public opt round 1` 继续冻结为当前公开主线的已闭合部分。",
            "- UAB 由当前 torch-native runner 接替 CPU-heavy `sklearn` 扩搜；若本次 gate 未全过，则主结论仍按 `NASA closed, UAB partial` 维护。",
            "- `chronaris_public_fusion` 保持 secondary exploratory branch，不替代当前 paper-facing public mainline。",
        ]
    )
    return "\n".join(lines)
