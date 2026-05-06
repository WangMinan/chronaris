"""Unified public-mainline report builder for Stage I."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from chronaris.pipelines.stage_i.stage_i_public_opt_reporting import fmt_public_opt_float


@dataclass(frozen=True, slots=True)
class StageIPublicMainlineReportConfig:
    run_id: str
    uab_summary_path: str
    nasa_summary_path: str
    deep_comparison_summary_path: str
    artifact_root: str = "docs/reports/assets/stage_i_public_mainline"
    report_root: str = "docs/reports"
    public_fusion_screen_summary_path: str | None = None
    public_fusion_nasa_confirm_summary_path: str | None = None
    public_fusion_uab_confirm_summary_path: str | None = None


@dataclass(frozen=True, slots=True)
class StageIPublicMainlineReportRunResult:
    run_id: str
    artifact_root: str
    summary_path: str
    report_path: str
    summary: Mapping[str, object]


def run_stage_i_public_mainline_report(
    config: StageIPublicMainlineReportConfig,
) -> StageIPublicMainlineReportRunResult:
    run_root = Path(config.artifact_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "public_mainline_summary.json"
    report_path = report_root / f"stage-i-public-mainline-{config.run_id}.md"

    uab_payload = json.loads(Path(config.uab_summary_path).read_text(encoding="utf-8"))
    nasa_payload = json.loads(Path(config.nasa_summary_path).read_text(encoding="utf-8"))
    deep_payload = json.loads(
        Path(config.deep_comparison_summary_path).read_text(encoding="utf-8")
    )
    fusion_screen_payload = _load_optional_json(config.public_fusion_screen_summary_path)
    fusion_nasa_confirm = _load_optional_json(config.public_fusion_nasa_confirm_summary_path)
    fusion_uab_confirm = _load_optional_json(config.public_fusion_uab_confirm_summary_path)

    uab_summary = _extract_uab_status(uab_payload, deep_payload)
    nasa_summary = _extract_nasa_status(nasa_payload)
    fusion_summary = _extract_public_fusion_status(
        fusion_screen_payload,
        fusion_nasa_confirm,
        fusion_uab_confirm,
    )
    if uab_summary["strict_mainline_closed"] and nasa_summary["mainline_closed"]:
        public_mainline_status = "public opt closed"
    elif nasa_summary["mainline_closed"]:
        public_mainline_status = "NASA closed, UAB partial"
    else:
        public_mainline_status = "public mainline open"

    summary = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "source_paths": {
            "uab_summary_path": config.uab_summary_path,
            "nasa_summary_path": config.nasa_summary_path,
            "deep_comparison_summary_path": config.deep_comparison_summary_path,
            "public_fusion_screen_summary_path": config.public_fusion_screen_summary_path,
            "public_fusion_nasa_confirm_summary_path": config.public_fusion_nasa_confirm_summary_path,
            "public_fusion_uab_confirm_summary_path": config.public_fusion_uab_confirm_summary_path,
        },
        "public_mainline_status": public_mainline_status,
        "uab": uab_summary,
        "nasa": nasa_summary,
        "public_fusion": fusion_summary,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_public_mainline_report(summary) + "\n", encoding="utf-8")
    return StageIPublicMainlineReportRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        report_path=str(report_path),
        summary=summary,
    )


def _load_optional_json(path_like: str | None) -> dict[str, object]:
    if not path_like:
        return {}
    path = Path(path_like)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_uab_status(
    uab_payload: Mapping[str, object],
    deep_payload: Mapping[str, object],
) -> dict[str, object]:
    if "final_result" in uab_payload:
        groups = uab_payload["final_result"]["groups"]
        source_type = "torch_uab"
        acceptance = uab_payload.get("acceptance") or {}
    else:
        subset_results = uab_payload.get("subset_results") or {}
        groups = {
            subset_id: result["heads"][result["best_head"]]
            for subset_id, result in subset_results.items()
        }
        source_type = "legacy_public_opt"
        acceptance = {}
    deep_groups = {}
    uab_deep = (
        deep_payload.get("datasets", {})
        .get("uab_workload_dataset", {})
        .get("models", {})
    )
    for model_name, model_payload in uab_deep.items():
        subjective_groups = (
            model_payload.get("summary", {})
            .get("subjective", {})
            .get("groups", {})
        )
        deep_groups[model_name] = {
            subset_id: {
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
            }
            for subset_id, metrics in subjective_groups.items()
        }
    per_group = {}
    strict_mainline_closed = True
    for subset_id in ("n_back", "heat_the_chair"):
        public_rmse = float(groups[subset_id]["rmse"])
        public_mae = float(groups[subset_id]["mae"])
        best_deep_name, best_deep_rmse = min(
            (
                (model_name, float(metrics.get(subset_id, {}).get("rmse", float("inf"))))
                for model_name, metrics in deep_groups.items()
            ),
            key=lambda item: item[1],
        )
        margin = best_deep_rmse - public_rmse
        clean_win = margin > 1e-4
        strict_mainline_closed = strict_mainline_closed and clean_win
        payload = {
            "public_rmse": public_rmse,
            "public_mae": public_mae,
            "best_deep_model": best_deep_name,
            "best_deep_rmse": best_deep_rmse,
            "margin_vs_best_deep": margin,
            "clean_win": clean_win,
        }
        if acceptance:
            payload["acceptance_gate"] = acceptance.get("groups", {}).get(subset_id)
        per_group[subset_id] = payload
    return {
        "source_type": source_type,
        "strict_mainline_closed": strict_mainline_closed,
        "public_mainline_status": (
            "closed" if strict_mainline_closed else "partial"
        ),
        "groups": per_group,
    }


def _extract_nasa_status(nasa_payload: Mapping[str, object]) -> dict[str, object]:
    combined = nasa_payload["subset_results"]["combined"]
    combined_best = combined["heads"][combined["best_head"]]
    winning_margins = nasa_payload.get("winning_margin_vs_deep") or {}
    combined_margin = winning_margins.get("combined") or {}
    mainline_closed = bool(combined_margin.get("gate_passed", False))
    return {
        "mainline_closed": mainline_closed,
        "combined_best_head": combined["best_head"],
        "combined_macro_f1": float(combined_best["macro_f1"]),
        "combined_balanced_accuracy": float(combined_best["balanced_accuracy"]),
        "winning_margins": winning_margins,
    }


def _extract_public_fusion_status(
    screen_payload: Mapping[str, object],
    nasa_confirm: Mapping[str, object],
    uab_confirm: Mapping[str, object],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "screen_available": bool(screen_payload),
        "secondary_only": True,
    }
    if screen_payload:
        nasa_ranking = (screen_payload.get("per_dataset_rankings") or {}).get("nasa_csm") or []
        uab_ranking = (
            screen_payload.get("per_dataset_rankings") or {}
        ).get("uab_workload_dataset") or []
        if nasa_ranking:
            summary["screen_nasa_best"] = nasa_ranking[0]
        if uab_ranking:
            summary["screen_uab_best"] = uab_ranking[0]
    if nasa_confirm:
        combined = nasa_confirm.get("objective", {}).get("groups", {}).get("combined") or {}
        summary["nasa_confirm_combined_macro_f1"] = float(combined.get("macro_f1", 0.0))
        summary["nasa_confirm_combined_balanced_accuracy"] = float(
            combined.get("balanced_accuracy", 0.0)
        )
        summary["nasa_clears_040_gate"] = float(combined.get("macro_f1", 0.0)) > 0.40
    if uab_confirm:
        subjective = uab_confirm.get("subjective", {}).get("groups", {})
        if subjective:
            summary["uab_confirm_mean_rmse"] = float(
                (
                    float(subjective["n_back"]["rmse"])
                    + float(subjective["heat_the_chair"]["rmse"])
                )
                / 2.0
            )
    return summary


def _render_public_mainline_report(summary: Mapping[str, object]) -> str:
    uab = summary["uab"]
    nasa = summary["nasa"]
    fusion = summary["public_fusion"]
    lines = [
        "# Stage I Public Mainline Report",
        "",
        f"- generated_at_utc：`{summary['generated_at_utc']}`",
        f"- public_mainline_status：`{summary['public_mainline_status']}`",
        "",
        "## Mainline Decision",
        "",
        "- NASA `public opt round 1` 继续冻结为当前公开主线的已闭合部分。",
        (
            "- UAB 当前已由 torch-native branch 补位并满足严格门槛。"
            if uab["strict_mainline_closed"]
            else "- UAB 当前仍未形成严格双组 clean win，因此公开主线状态保持 `NASA closed, UAB partial`。"
        ),
        "- `chronaris_public_fusion` 继续作为 secondary exploratory branch，不作为当前论文主线。",
        "",
        "## NASA",
        "",
        f"- combined_best_head：`{nasa['combined_best_head']}`",
        f"- combined_macro_f1：`{fmt_public_opt_float(nasa['combined_macro_f1'])}`",
        f"- combined_balanced_accuracy：`{fmt_public_opt_float(nasa['combined_balanced_accuracy'])}`",
        f"- mainline_closed：`{nasa['mainline_closed']}`",
        "",
        "## UAB",
        "",
        "| evaluation_group | public_rmse | public_mae | best_deep_model | best_deep_rmse | margin_vs_best_deep | clean_win |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for subset_id in ("n_back", "heat_the_chair"):
        payload = uab["groups"][subset_id]
        lines.append(
            f"| {subset_id} | {fmt_public_opt_float(payload['public_rmse'])} | "
            f"{fmt_public_opt_float(payload['public_mae'])} | `{payload['best_deep_model']}` | "
            f"{fmt_public_opt_float(payload['best_deep_rmse'])} | {fmt_public_opt_float(payload['margin_vs_best_deep'])} | "
            f"`{payload['clean_win']}` |"
        )
    lines.extend(["", "## Public Fusion", ""])
    if not fusion.get("screen_available"):
        lines.append("- 本轮未提供 `public_fusion` screen/confirm 结果。")
    else:
        if fusion.get("screen_nasa_best"):
            row = fusion["screen_nasa_best"]
            lines.append(
                "- NASA coarse best："
                f"`{row['candidate_id']}`，combined macro-F1="
                f"`{fmt_public_opt_float(row['combined_macro_f1'])}`"
            )
        if "nasa_confirm_combined_macro_f1" in fusion:
            lines.append(
                "- NASA confirm：combined macro-F1="
                f"`{fmt_public_opt_float(fusion['nasa_confirm_combined_macro_f1'])}`，"
                f"gate>0.40=`{fusion.get('nasa_clears_040_gate', False)}`"
            )
        if "uab_confirm_mean_rmse" in fusion:
            lines.append(
                "- UAB confirm：mean RMSE="
                f"`{fmt_public_opt_float(fusion['uab_confirm_mean_rmse'])}`"
            )
        lines.append("- 该分支继续保留为 exploratory only。")
    return "\n".join(lines)
