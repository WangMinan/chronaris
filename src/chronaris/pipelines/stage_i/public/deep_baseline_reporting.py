"""Markdown reporting helpers for Stage I deep baselines."""

from __future__ import annotations

from typing import Mapping

from chronaris.features.stage_i_sequences import STAGE_H_CASE_DATASET_ID


def _render_deep_baseline_report(summary: Mapping[str, object]) -> str:
    lines = [
        f"# Stage I Deep Baseline - {summary['dataset_id']} - {summary['model_name']}",
        "",
        f"- profile: `{summary['profile']}`",
        f"- artifact root: `{summary['artifact_root']}`",
        f"- prepared root: `{summary['prepared_artifact_root']}`",
        "",
    ]
    if summary["dataset_id"] == STAGE_H_CASE_DATASET_ID:
        lines.extend(
            [
                "## Real Sortie Summary",
                "",
                f"- view count: `{summary['view_count']}`",
                f"- sample count: `{summary['sample_count']}`",
                f"- smoke training target: `{summary['smoke_training_target']}`",
                "",
                "| view | sortie | pilot | verdict | samples | stability | attention entropy | top concentration | event-mask interference |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ],
        )
        for row in summary["view_metrics"]:
            lines.append(
                f"| `{row['view_id']}` | `{row['sortie_id']}` | {row['pilot_id']} | "
                f"`{row['projection_diagnostics_verdict']}` | {row['sample_count']} | "
                f"{row['representation_stability']:.6f} | {row['mean_attention_entropy']:.6f} | "
                f"{row['top_event_concentration']:.6f} | {row['event_mask_interference']:.6f} |"
            )
        if summary["pilot_metrics"]:
            lines.extend(
                [
                    "",
                    "## Same-Sortie Dual-Pilot Delta",
                    "",
                    "| sortie | reference pilot | comparison pilot | delta stability | delta entropy | delta top concentration | delta event-mask interference |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ],
            )
            for row in summary["pilot_metrics"]:
                lines.append(
                    f"| `{row['sortie_id']}` | {row['reference_pilot_id']} | {row['comparison_pilot_id']} | "
                    f"{row['delta_representation_stability']:+.6f} | {row['delta_attention_entropy']:+.6f} | "
                    f"{row['delta_top_event_concentration']:+.6f} | {row['delta_event_mask_interference']:+.6f} |"
                )
    else:
        for track_name in ("objective", "subjective"):
            track = summary.get(track_name)
            if not track:
                continue
            lines.extend([f"## {track_name.title()}", ""])
            if track_name == "objective":
                lines.extend(
                    [
                        "| group | macro-F1 | balanced accuracy | samples | folds |",
                        "| --- | ---: | ---: | ---: | ---: |",
                    ],
                )
                for group_name, metrics in track["groups"].items():
                    lines.append(
                        f"| `{group_name}` | {metrics['macro_f1']:.6f} | "
                        f"{metrics['balanced_accuracy']:.6f} | {metrics['sample_count']} | "
                        f"{metrics['fold_count']} |"
                    )
            else:
                lines.extend(
                    [
                        "| group | RMSE | MAE | R2 | Spearman | samples | folds |",
                        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                    ],
                )
                for group_name, metrics in track["groups"].items():
                    lines.append(
                        f"| `{group_name}` | {metrics['rmse']:.6f} | {metrics['mae']:.6f} | "
                        f"{metrics['r2']:.6f} | {metrics['spearman']:.6f} | "
                        f"{metrics['sample_count']} | {metrics['fold_count']} |"
                    )
            lines.append("")
        if summary.get("reference_comparison"):
            lines.extend(["## Existing Classical Reference", ""])
            reference = summary["reference_comparison"]
            if reference.get("objective"):
                lines.extend(
                    [
                        "| group | classical macro-F1 | classical balanced acc |",
                        "| --- | ---: | ---: |",
                    ],
                )
                for group_name, metrics in reference["objective"].items():
                    lines.append(
                        f"| `{group_name}` | {metrics['macro_f1']:.6f} | "
                        f"{metrics['balanced_accuracy']:.6f} |"
                    )
                lines.append("")
            if reference.get("subjective"):
                lines.extend(
                    [
                        "| group | classical RMSE | classical MAE |",
                        "| --- | ---: | ---: |",
                    ],
                )
                for group_name, metrics in reference["subjective"].items():
                    lines.append(
                        f"| `{group_name}` | {metrics['rmse']:.6f} | {metrics['mae']:.6f} |"
                    )
    return "\n".join(lines)


def _render_comparison_report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Stage I Deep Comparison",
        "",
        f"- generated at UTC: `{summary['generated_at_utc']}`",
        f"- artifact root: `{summary['artifact_root']}`",
        "",
    ]
    for dataset_id in summary["dataset_order"]:
        dataset_payload = summary["datasets"].get(dataset_id, {"status": "not_run"})
        lines.append(f"## {dataset_id}")
        lines.append("")
        if dataset_payload.get("status") != "completed":
            lines.append("- 本轮未运行。")
            lines.append("")
            continue
        lines.extend(
            [
                "| model | artifact root | summary path | report path |",
                "| --- | --- | --- | --- |",
            ],
        )
        for model_name, payload in dataset_payload["models"].items():
            lines.append(
                f"| `{model_name}` | `{payload['artifact_root']}` | "
                f"`{payload['summary_path']}` | `{payload['report_path']}` |"
            )
        lines.append("")
    return "\n".join(lines)
