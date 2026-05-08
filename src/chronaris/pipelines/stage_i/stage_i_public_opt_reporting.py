"""Markdown reporting helpers for Stage I public-opt runs."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def render_stage_i_public_opt_report(
    *,
    summary: Mapping[str, object],
    feature_frame: pd.DataFrame,
) -> str:
    subset_rows = []
    subset_results = summary["subset_results"]
    primary_field, secondary_field = _primary_metric_fields(str(summary["track"]))
    extra_fields = _extra_metric_fields(str(summary["track"]))
    for subset_id in summary["subset_order"]:
        payload = subset_results.get(subset_id)
        if payload is None:
            continue
        best_head = payload["best_head"]
        best_metrics = payload["heads"][best_head]
        subset_rows.append(
            _summary_row_values(
                subset_id=subset_id,
                payload=payload,
                best_metrics=best_metrics,
                track=str(summary["track"]),
                primary_field=primary_field,
                secondary_field=secondary_field,
                extra_fields=extra_fields,
            )
        )

    track_title = (
        "Stage I Public Opt UAB Subjective Run"
        if summary["track"] == "subjective"
        else "Stage I Public Opt NASA Attention Run"
    )
    lines = [
        f"# {track_title}",
        "",
        "## 运行口径",
        "",
        f"- run_id：`{summary['run_id']}`",
        f"- dataset_id：`{summary['dataset_id']}`",
        f"- profile：`{summary['profile']}`",
        f"- feature_profile：`{summary['feature_profile']}`",
        f"- head_catalog：`{summary['head_catalog']}`",
        f"- train_balance_policy：`{summary['train_balance_policy']}`",
        f"- ensemble_policy：`{summary['ensemble_policy']}`",
        f"- prediction_aggregation_policy：`{summary['prediction_aggregation_policy']}`",
        f"- track：`{summary['track']}`",
        f"- task_type：`{summary['task_type']}`",
        f"- prepared asset root：`{summary['prepared_artifact_root']}`",
        f"- output artifact root：`{summary['artifact_root']}`",
        f"- generated_at_utc：`{summary['generated_at_utc']}`",
        "",
        "## 样本范围",
        "",
        f"- 总样本数：`{len(feature_frame)}`",
        f"- evaluation groups：`{', '.join(summary['subset_order'])}`",
        f"- raw subsets：`{', '.join(sorted(feature_frame['subset_id'].astype(str).unique()))}`",
        f"- split_group 数：`{feature_frame['split_group'].nunique()}`",
        f"- subject 数：`{feature_frame['subject_id'].nunique()}`",
        f"- feature_group_sizes：`{summary['feature_group_sizes']}`",
        "",
        "## Evaluation 指标",
        "",
        _summary_table_header(track=str(summary["track"])),
        _summary_table_divider(track=str(summary["track"])),
    ]
    for row in subset_rows:
        lines.append(_render_summary_table_row(row=row, track=str(summary["track"])))

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
                _detail_table_header(track=str(summary["track"])),
                _detail_table_divider(track=str(summary["track"])),
            ]
        )
        for head_name in payload["heads"]:
            metrics = payload["heads"][head_name]
            lines.append(
                _render_detail_table_row(
                    head_name=head_name,
                    metrics=metrics,
                    track=str(summary["track"]),
                )
            )

    reference_comparison = summary.get("reference_comparison") or {}
    if reference_comparison:
        lines.extend(["", "## 参考对照", ""])
        lines.extend(render_public_opt_reference_comparison(reference_comparison))
    margin_summary = summary.get("winning_margin_vs_deep") or {}
    if margin_summary:
        lines.extend(["", "## Deep 胜出判定", ""])
        lines.extend(
            render_public_opt_winning_margins(
                margin_summary,
                bool(summary.get("needs_deep_rerun")),
            )
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- `public opt` 只迁移 Chronaris 的公开 sequence-contract 思路，不改写既有 Stage I 公开 benchmark 历史事实。",
            "- 当前主 gate 只看相对 `MulT / ContiFormer` 的胜出情况；`classical baseline` 只保留为历史背景，不作为本轮前进门槛。",
        ]
    )
    return "\n".join(lines)


def render_public_opt_reference_comparison(
    reference_comparison: Mapping[str, object],
) -> list[str]:
    track = str(reference_comparison["track"])
    primary_field, secondary_field = _primary_metric_fields(track)
    lines = [
        f"- dataset_id：`{reference_comparison['dataset_id']}`",
        f"- track：`{track}`",
        "",
        _reference_table_header(track),
        _reference_table_divider(track),
    ]
    for group_name, payload in reference_comparison["groups"].items():
        public_metrics = payload["public_opt"]["best_metrics"]
        deep_models = payload.get("deep_models") or {}
        mult_metrics = deep_models.get("mult") or {}
        contiformer_metrics = deep_models.get("contiformer") or {}
        lines.append(
            _reference_table_row(
                group_name=group_name,
                best_head=payload["public_opt"]["best_head"],
                public_metrics=public_metrics,
                mult_metrics=mult_metrics,
                contiformer_metrics=contiformer_metrics,
                primary_field=primary_field,
                secondary_field=secondary_field,
            )
        )
    return lines


def render_public_opt_winning_margins(
    margin_summary: Mapping[str, object],
    needs_deep_rerun: bool,
) -> list[str]:
    lines = [
        "| evaluation_group | best_public_head | best_deep_model | margin_vs_best_deep | gate_passed | needs_deep_rerun |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for group_name, payload in margin_summary.items():
        lines.append(
            f"| {group_name} | {payload['best_public_head']} | {payload['best_deep_model']} | "
            f"{fmt_public_opt_float(payload['margin_vs_best_deep'])} | "
            f"`{payload['gate_passed']}` | `{payload['needs_deep_rerun']}` |"
        )
    lines.extend(["", f"- overall_needs_deep_rerun：`{needs_deep_rerun}`"])
    return lines


def fmt_public_opt_float(value: float | int | object) -> str:
    numeric = float(value)
    if not np.isfinite(numeric):
        return "0.0000"
    return f"{numeric:.4f}"


def _primary_metric_fields(track: str) -> tuple[str, str]:
    if track == "subjective":
        return "rmse", "mae"
    return "macro_f1", "balanced_accuracy"


def _extra_metric_fields(track: str) -> tuple[str, ...]:
    if track == "subjective":
        return ("r2", "spearman")
    return ()


def _summary_table_header(track: str) -> str:
    if track == "subjective":
        return "| evaluation_group | sample_count | fold_count | best_head | mae | rmse | r2 | spearman |"
    return "| evaluation_group | sample_count | fold_count | best_head | macro_f1 | balanced_accuracy |"


def _summary_table_divider(track: str) -> str:
    if track == "subjective":
        return "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"
    return "| --- | ---: | ---: | --- | ---: | ---: |"


def _render_summary_table_row(*, row: tuple[object, ...], track: str) -> str:
    if track == "subjective":
        (
            evaluation_group,
            sample_count,
            fold_count,
            best_head,
            mae,
            rmse,
            r2,
            spearman,
        ) = row
        return (
            f"| {evaluation_group} | {sample_count} | {fold_count} | {best_head} | "
            f"{fmt_public_opt_float(mae)} | {fmt_public_opt_float(rmse)} | "
            f"{fmt_public_opt_float(r2)} | {fmt_public_opt_float(spearman)} |"
        )
    evaluation_group, sample_count, fold_count, best_head, macro_f1, balanced_accuracy = row
    return (
        f"| {evaluation_group} | {sample_count} | {fold_count} | {best_head} | "
        f"{fmt_public_opt_float(macro_f1)} | {fmt_public_opt_float(balanced_accuracy)} |"
    )


def _summary_row_values(
    *,
    subset_id: str,
    payload: Mapping[str, object],
    best_metrics: Mapping[str, object],
    track: str,
    primary_field: str,
    secondary_field: str,
    extra_fields: Sequence[str],
) -> tuple[object, ...]:
    row = [subset_id, payload["sample_count"], payload["fold_count"], payload["best_head"]]
    if track == "subjective":
        row.extend([best_metrics[secondary_field], best_metrics[primary_field]])
    else:
        row.extend([best_metrics[primary_field], best_metrics[secondary_field]])
    row.extend(best_metrics[field] for field in extra_fields)
    return tuple(row)


def _detail_table_header(track: str) -> str:
    if track == "subjective":
        return "| head | mae | rmse | r2 | spearman |"
    return "| head | macro_f1 | balanced_accuracy |"


def _detail_table_divider(track: str) -> str:
    if track == "subjective":
        return "| --- | ---: | ---: | ---: | ---: |"
    return "| --- | ---: | ---: |"


def _render_detail_table_row(
    *,
    head_name: str,
    metrics: Mapping[str, object],
    track: str,
) -> str:
    if track == "subjective":
        return (
            f"| {head_name} | {fmt_public_opt_float(metrics['mae'])} | "
            f"{fmt_public_opt_float(metrics['rmse'])} | {fmt_public_opt_float(metrics['r2'])} | "
            f"{fmt_public_opt_float(metrics['spearman'])} |"
        )
    return (
        f"| {head_name} | {fmt_public_opt_float(metrics['macro_f1'])} | "
        f"{fmt_public_opt_float(metrics['balanced_accuracy'])} |"
    )


def _reference_table_header(track: str) -> str:
    if track == "subjective":
        return (
            "| evaluation_group | public_opt best_head | public_opt rmse | public_opt mae | "
            "MulT rmse | MulT mae | ContiFormer rmse | ContiFormer mae |"
        )
    return (
        "| evaluation_group | public_opt best_head | public_opt macro_f1 | public_opt balanced_accuracy | "
        "MulT macro_f1 | MulT balanced_accuracy | "
        "ContiFormer macro_f1 | ContiFormer balanced_accuracy |"
    )


def _reference_table_divider(track: str) -> str:
    return "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"


def _reference_table_row(
    *,
    group_name: str,
    best_head: str,
    public_metrics: Mapping[str, object],
    mult_metrics: Mapping[str, object],
    contiformer_metrics: Mapping[str, object],
    primary_field: str,
    secondary_field: str,
) -> str:
    return (
        f"| {group_name} | {best_head} | "
        f"{fmt_public_opt_float(public_metrics.get(primary_field, 0.0))} | "
        f"{fmt_public_opt_float(public_metrics.get(secondary_field, 0.0))} | "
        f"{fmt_public_opt_float(mult_metrics.get(primary_field, 0.0))} | "
        f"{fmt_public_opt_float(mult_metrics.get(secondary_field, 0.0))} | "
        f"{fmt_public_opt_float(contiformer_metrics.get(primary_field, 0.0))} | "
        f"{fmt_public_opt_float(contiformer_metrics.get(secondary_field, 0.0))} |"
    )
