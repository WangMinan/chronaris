"""Unified public-mainline report builder for Stage I."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from chronaris.pipelines.stage_i.public.opt_reporting import fmt_public_opt_float

PUBLIC_ADAPTER_EVIDENCE_ROLE = "public_adapter_evidence"


@dataclass(frozen=True, slots=True)
class StageIPublicMainlineReportConfig:
    run_id: str
    uab_summary_path: str
    nasa_summary_path: str
    deep_comparison_summary_path: str
    artifact_root: str = "docs/artifacts/assets/stage_i_public_mainline"
    report_root: str = "docs/artifacts"
    extra_uab_summary_paths: tuple[str, ...] = ()
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

    uab_payloads = [
        (config.uab_summary_path, json.loads(Path(config.uab_summary_path).read_text(encoding="utf-8")))
    ]
    for path_like in config.extra_uab_summary_paths:
        uab_payloads.append(
            (path_like, json.loads(Path(path_like).read_text(encoding="utf-8")))
        )
    nasa_payload = json.loads(Path(config.nasa_summary_path).read_text(encoding="utf-8"))
    deep_payload = json.loads(
        Path(config.deep_comparison_summary_path).read_text(encoding="utf-8")
    )
    fusion_screen_payload = _load_optional_json(config.public_fusion_screen_summary_path)
    fusion_nasa_confirm = _load_optional_json(config.public_fusion_nasa_confirm_summary_path)
    fusion_uab_confirm = _load_optional_json(config.public_fusion_uab_confirm_summary_path)

    uab_summary = _extract_uab_status(uab_payloads, deep_payload)
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
        "thesis_facing_status": PUBLIC_ADAPTER_EVIDENCE_ROLE,
        "thesis_dual_stream_mainline_closed": False,
        "public_branch_semantics": {
            "uab": _public_branch_semantics(second_stream_name="task_context"),
            "nasa": _public_branch_semantics(second_stream_name="scenario_context"),
        },
        "source_paths": {
            "uab_summary_path": config.uab_summary_path,
            "extra_uab_summary_paths": list(config.extra_uab_summary_paths),
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


def _public_branch_semantics(*, second_stream_name: str) -> dict[str, object]:
    return {
        "evidence_role": PUBLIC_ADAPTER_EVIDENCE_ROLE,
        "second_stream_name": second_stream_name,
        "second_stream_role": "context_proxy",
        "second_stream_is_real_vehicle": False,
    }


def _load_optional_json(path_like: str | None) -> dict[str, object]:
    if not path_like:
        return {}
    path = Path(path_like)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_uab_status(
    uab_payloads: list[tuple[str, Mapping[str, object]]],
    deep_payload: Mapping[str, object],
) -> dict[str, object]:
    candidate_payloads = [
        _load_uab_candidate_payload(source_path=source_path, payload=payload)
        for source_path, payload in uab_payloads
    ]
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
    needs_deep_rerun = False
    for subset_id in ("n_back", "heat_the_chair"):
        best_public = min(
            (
                candidate["groups"][subset_id]
                for candidate in candidate_payloads
                if subset_id in candidate["groups"]
            ),
            key=lambda item: (float(item["rmse"]), float(item["mae"])),
        )
        public_rmse = float(best_public["rmse"])
        public_mae = float(best_public["mae"])
        best_deep_name, best_deep_metrics = min(
            (
                (
                    model_name,
                    {
                        "rmse": float(
                            metrics.get(subset_id, {}).get("rmse", float("inf"))
                        ),
                        "mae": float(
                            metrics.get(subset_id, {}).get("mae", float("inf"))
                        ),
                    },
                )
                for model_name, metrics in deep_groups.items()
            ),
            key=lambda item: (item[1]["rmse"], item[1]["mae"]),
        )
        rmse_margin = best_deep_metrics["rmse"] - public_rmse
        mae_margin = best_deep_metrics["mae"] - public_mae
        clean_win, tie_break_used = _subjective_clean_win(
            public_rmse=public_rmse,
            public_mae=public_mae,
            deep_rmse=best_deep_metrics["rmse"],
            deep_mae=best_deep_metrics["mae"],
        )
        rerun_threshold = max(0.02, 0.01 * best_deep_metrics["rmse"])
        group_needs_rerun = abs(rmse_margin) < rerun_threshold
        strict_mainline_closed = strict_mainline_closed and clean_win
        needs_deep_rerun = needs_deep_rerun or group_needs_rerun
        payload = {
            "public_rmse": public_rmse,
            "public_mae": public_mae,
            "best_public_head": best_public["best_head"],
            "best_public_source_type": best_public["source_type"],
            "best_public_source_path": best_public["source_path"],
            "best_deep_model": best_deep_name,
            "best_deep_rmse": best_deep_metrics["rmse"],
            "best_deep_mae": best_deep_metrics["mae"],
            "rmse_margin_vs_best_deep": rmse_margin,
            "mae_margin_vs_best_deep": mae_margin,
            "margin_vs_best_deep": rmse_margin,
            "clean_win": clean_win,
            "tie_break_used": tie_break_used,
            "needs_deep_rerun": group_needs_rerun,
            "rerun_threshold": rerun_threshold,
        }
        if best_public.get("acceptance_gate") is not None:
            payload["acceptance_gate"] = best_public["acceptance_gate"]
        per_group[subset_id] = payload
    return {
        "source_type": "multi_source_best_of" if len(candidate_payloads) > 1 else candidate_payloads[0]["source_type"],
        "candidate_sources": [
            {
                "source_type": candidate["source_type"],
                "source_path": candidate["source_path"],
                "prediction_aggregation_policy": candidate["prediction_aggregation_policy"],
            }
            for candidate in candidate_payloads
        ],
        "strict_mainline_closed": strict_mainline_closed,
        "public_mainline_status": (
            "closed" if strict_mainline_closed else "partial"
        ),
        "needs_deep_rerun": needs_deep_rerun,
        "groups": per_group,
    }


def _load_uab_candidate_payload(
    *,
    source_path: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    if "final_result" in payload:
        groups = payload["final_result"]["groups"]
        acceptance = payload.get("acceptance") or {}
        return {
            "source_type": "torch_uab",
            "source_path": source_path,
            "prediction_aggregation_policy": str(
                payload.get("screen_config", {}).get(
                    "prediction_aggregation_policy",
                    "none",
                )
            ),
            "groups": {
                subset_id: {
                    "rmse": float(metrics["rmse"]),
                    "mae": float(metrics["mae"]),
                    "best_head": "torch_native",
                    "source_type": "torch_uab",
                    "source_path": source_path,
                    "acceptance_gate": acceptance.get("groups", {}).get(subset_id),
                }
                for subset_id, metrics in groups.items()
            },
        }
    subset_results = payload.get("subset_results") or {}
    source_type = _infer_uab_public_opt_source_type(payload)
    return {
        "source_type": source_type,
        "source_path": source_path,
        "prediction_aggregation_policy": str(
            payload.get("prediction_aggregation_policy", "none")
        ),
        "groups": {
            subset_id: {
                "rmse": float(result["heads"][result["best_head"]]["rmse"]),
                "mae": float(result["heads"][result["best_head"]]["mae"]),
                "best_head": str(result["best_head"]),
                "source_type": source_type,
                "source_path": source_path,
                "acceptance_gate": None,
            }
            for subset_id, result in subset_results.items()
        },
    }


def _infer_uab_public_opt_source_type(payload: Mapping[str, object]) -> str:
    subset_results = payload.get("subset_results") or {}
    head_names = {
        str(head_name)
        for result in subset_results.values()
        for head_name in (result.get("heads") or {})
    }
    if str(payload.get("head_catalog", "")) == "uab_hybrid" and head_names.intersection(
        {
            "target_prior_median",
            "target_prior_trimmed_mean",
            "heat_prior_residual_guarded",
        }
    ):
        return "uab_public_adapter"
    return "legacy_public_opt"


def _subjective_clean_win(
    *,
    public_rmse: float,
    public_mae: float,
    deep_rmse: float,
    deep_mae: float,
    tolerance: float = 1e-6,
) -> tuple[bool, bool]:
    if public_rmse < deep_rmse - tolerance:
        return True, False
    if abs(public_rmse - deep_rmse) <= tolerance and public_mae < deep_mae - tolerance:
        return True, True
    return False, False


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
        f"- thesis_facing_status：`{summary['thesis_facing_status']}`",
        f"- thesis_dual_stream_mainline_closed：`{summary['thesis_dual_stream_mainline_closed']}`",
        "",
        "## Thesis-Facing Boundary",
        "",
        "- `public opt closed` 只表示公开 `adapter evidence` 已收口，不代表论文双流主线已经 fully closed。",
        "- UAB / NASA 第二模态分别是 `task_context` / `scenario_context` 的 `context proxy`，不是论文里的真实 vehicle stream。",
        "",
        "## Mainline Decision",
        "",
        "- NASA `public opt round 1` 继续冻结为当前公开 adapter evidence 的已闭合部分。",
        (
            "- UAB 当前 best-of Chronaris public adapter evidence 已满足严格门槛。"
            if uab["strict_mainline_closed"]
            else "- UAB 当前仍未形成严格双组 clean win，因此公开 adapter evidence 状态保持 `NASA closed, UAB partial`。"
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
        f"- current torch-native branch considered：`{any(source['source_type'] == 'torch_uab' for source in uab.get('candidate_sources', []))}`",
        "",
        "| evaluation_group | public_rmse | public_mae | best_source_type | best_public_head | best_deep_model | best_deep_rmse | margin_vs_best_deep | clean_win |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    tie_break_groups: list[str] = []
    for subset_id in ("n_back", "heat_the_chair"):
        payload = uab["groups"][subset_id]
        lines.append(
            f"| {subset_id} | {fmt_public_opt_float(payload['public_rmse'])} | "
            f"{fmt_public_opt_float(payload['public_mae'])} | `{payload['best_public_source_type']}` | "
            f"`{payload['best_public_head']}` | `{payload['best_deep_model']}` | "
            f"{fmt_public_opt_float(payload['best_deep_rmse'])} | {fmt_public_opt_float(payload['margin_vs_best_deep'])} | "
            f"`{payload['clean_win']}` |"
        )
        if payload.get("tie_break_used"):
            tie_break_groups.append(subset_id)
    if tie_break_groups:
        lines.extend(
            [
                "",
                "- MAE tie-break used for: "
                + ", ".join(f"`{subset_id}`" for subset_id in tie_break_groups)
                + ".",
            ]
        )
    if uab.get("needs_deep_rerun"):
        lines.extend(
            [
                "",
                "- UAB 当前 best-of Chronaris 结果已形成主线胜出，但至少一组领先幅度仍处于 `near-tie` 区间；若要做最严格公平确认，仍建议重跑对应 deep baseline。",
            ]
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
