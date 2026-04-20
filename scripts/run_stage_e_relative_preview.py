"""Run one overlap-focused Stage E preview experiment with relative reconstruction loss."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from chronaris.access import (
    DirectInfluxScopeConfig,
    InfluxSettings,
    OverlapPreviewSortieLoaderConfig,
    build_overlap_preview_sortie_loader,
)
from chronaris.dataset.builder import SortieDatasetBuilder
from chronaris.evaluation import (
    render_alignment_projection_diagnostics_markdown,
    summarize_alignment_projection_diagnostics,
)
from chronaris.features.experiment_input import E0InputConfig
from chronaris.models.alignment import AlignmentPrototypeConfig, ReferenceGridConfig
from chronaris.pipelines.alignment_experiment import AlignmentExperimentPipeline
from chronaris.pipelines.alignment_preview import AlignmentPreviewConfig, AlignmentPreviewPipeline
from chronaris.pipelines.e0_preview import E0PreviewPipeline
from chronaris.schema.models import SortieLocator

# Keep matplotlib cache in a writable location inside CLI/sandbox runs.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-chronaris")


def _utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _extract_secret(md_text: str, key: str) -> str:
    pattern = re.compile(rf"^\+\s+{re.escape(key)}:\s*(.+)$", re.MULTILINE)
    matched = pattern.search(md_text)
    if not matched:
        raise RuntimeError(f"Missing secret key in docs/SECRETS.md: {key}")
    return matched.group(1).strip()


def _resolve_influx_settings() -> InfluxSettings:
    url = None
    org = None
    token = None

    for env_key, target in (
        ("CHRONARIS_INFLUX_URL", "url"),
        ("CHRONARIS_INFLUX_ORG", "org"),
        ("CHRONARIS_INFLUX_TOKEN", "token"),
    ):
        value = os.environ.get(env_key)
        if value:
            if target == "url":
                url = value
            elif target == "org":
                org = value
            else:
                token = value

    if not (url and org and token):
        secrets_path = REPO_ROOT / "docs" / "SECRETS.md"
        secrets_text = secrets_path.read_text(encoding="utf-8")
        url = url or _extract_secret(secrets_text, "influxdb.url")
        org = org or _extract_secret(secrets_text, "influxdb.org")
        token = token or _extract_secret(secrets_text, "influxdb.token")

    return InfluxSettings(
        url=url,
        org=org,
        token_env=None,
        token_value=token,
    )


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _render_visual_artifacts(
    *,
    report_path: Path,
    experiment_result,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preview_result = experiment_result.preview_result
    train_history = preview_result.train_history
    validation_history = preview_result.validation_history
    if not train_history:
        return {}

    assets_dir = report_path.parent / "assets" / report_path.stem
    assets_dir.mkdir(parents=True, exist_ok=True)

    image_paths: dict[str, Path] = {}
    epoch_indexes = list(range(1, len(train_history) + 1))

    fig_total, ax_total = plt.subplots(figsize=(8, 4.5))
    ax_total.plot(
        epoch_indexes,
        [item.total for item in train_history],
        marker="o",
        linewidth=2.0,
        label="train total",
    )
    if validation_history:
        ax_total.plot(
            epoch_indexes,
            [item.total for item in validation_history],
            marker="o",
            linewidth=2.0,
            label="validation total",
        )
    ax_total.set_title("Stage E Relative-MSE Total Loss")
    ax_total.set_xlabel("Epoch")
    ax_total.set_ylabel("Loss")
    ax_total.grid(True, alpha=0.25)
    ax_total.legend()
    total_path = assets_dir / "train_validation_total_loss.png"
    fig_total.tight_layout()
    fig_total.savefig(total_path, dpi=160)
    plt.close(fig_total)
    image_paths["train_validation_total_loss"] = total_path

    fig_alignment, ax_alignment = plt.subplots(figsize=(8, 4.5))
    ax_alignment.plot(
        epoch_indexes,
        [item.alignment for item in train_history],
        marker="o",
        linewidth=2.0,
        label="train alignment",
    )
    if validation_history:
        ax_alignment.plot(
            epoch_indexes,
            [item.alignment for item in validation_history],
            marker="o",
            linewidth=2.0,
            label="validation alignment",
        )
    ax_alignment.set_title("Stage E Shared-Grid Alignment Loss")
    ax_alignment.set_xlabel("Epoch")
    ax_alignment.set_ylabel("Loss")
    ax_alignment.grid(True, alpha=0.25)
    ax_alignment.legend()
    alignment_path = assets_dir / "train_validation_alignment_loss.png"
    fig_alignment.tight_layout()
    fig_alignment.savefig(alignment_path, dpi=160)
    plt.close(fig_alignment)
    image_paths["train_validation_alignment_loss"] = alignment_path

    fig_recon, ax_recon = plt.subplots(figsize=(8, 4.5))
    ax_recon.plot(
        epoch_indexes,
        [item.physiology_reconstruction for item in train_history],
        marker="o",
        linewidth=2.0,
        label="train physiology",
    )
    ax_recon.plot(
        epoch_indexes,
        [item.vehicle_reconstruction for item in train_history],
        marker="o",
        linewidth=2.0,
        label="train vehicle",
    )
    if validation_history:
        ax_recon.plot(
            epoch_indexes,
            [item.physiology_reconstruction for item in validation_history],
            marker="o",
            linestyle="--",
            linewidth=2.0,
            label="validation physiology",
        )
        ax_recon.plot(
            epoch_indexes,
            [item.vehicle_reconstruction for item in validation_history],
            marker="o",
            linestyle="--",
            linewidth=2.0,
            label="validation vehicle",
        )
    ax_recon.set_title("Stage E Per-Stream Reconstruction Loss")
    ax_recon.set_xlabel("Epoch")
    ax_recon.set_ylabel("Loss")
    ax_recon.grid(True, alpha=0.25)
    ax_recon.legend(loc="best")
    reconstruction_path = assets_dir / "reconstruction_stream_loss.png"
    fig_recon.tight_layout()
    fig_recon.savefig(reconstruction_path, dpi=160)
    plt.close(fig_recon)
    image_paths["reconstruction_stream_loss"] = reconstruction_path

    intermediate = preview_result.intermediate_export
    if intermediate is not None and intermediate.samples:
        fig_cosine, ax_cosine = plt.subplots(figsize=(8, 4.5))
        for sample in intermediate.samples:
            offsets_s = list(sample.physiology.reference_offsets_s)
            cosine_values = [
                _cosine_similarity(phys_row, veh_row)
                for phys_row, veh_row in zip(
                    sample.physiology.reference_projected_states,
                    sample.vehicle.reference_projected_states,
                )
            ]
            label = sample.sample_id.split(":")[-1]
            ax_cosine.plot(offsets_s, cosine_values, marker="o", linewidth=1.6, label=label)
        ax_cosine.set_title("Reference Projection Cosine by Exported Sample")
        ax_cosine.set_xlabel("Reference Offset (s)")
        ax_cosine.set_ylabel("Cosine Similarity")
        ax_cosine.set_ylim(-1.05, 1.05)
        ax_cosine.grid(True, alpha=0.25)
        ax_cosine.legend(title="Sample", fontsize=8)
        cosine_path = assets_dir / "reference_projection_cosine.png"
        fig_cosine.tight_layout()
        fig_cosine.savefig(cosine_path, dpi=160)
        plt.close(fig_cosine)
        image_paths["reference_projection_cosine"] = cosine_path

    return {key: str(path) for key, path in image_paths.items()}


def _append_visual_links_to_report(
    *,
    report_markdown: str,
    report_path: Path,
    visual_artifacts: dict[str, str],
) -> str:
    if not visual_artifacts:
        return report_markdown

    labels = {
        "train_validation_total_loss": "Train/Validation Total Loss",
        "train_validation_alignment_loss": "Train/Validation Alignment Loss",
        "reconstruction_stream_loss": "Per-Stream Reconstruction Loss",
        "reference_projection_cosine": "Reference Projection Cosine",
    }
    lines = [
        report_markdown.rstrip(),
        "",
        "## Visual Artifacts",
        "",
    ]
    for key, absolute_path in visual_artifacts.items():
        image_path = Path(absolute_path)
        relative_path = image_path.relative_to(report_path.parent)
        lines.append(f"### {labels.get(key, key)}")
        lines.append("")
        lines.append(f"![{labels.get(key, key)}]({relative_path.as_posix()})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_projection_diagnostics_artifacts(
    *,
    report_path: Path,
    diagnostics_summary,
) -> dict[str, str]:
    assets_dir = report_path.parent / "assets" / report_path.stem
    assets_dir.mkdir(parents=True, exist_ok=True)

    json_path = assets_dir / "projection_diagnostics_summary.json"
    json_path.write_text(
        json.dumps(asdict(diagnostics_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = assets_dir / "projection_diagnostics_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "reference_point_count",
                "mean_projection_cosine",
                "min_projection_cosine",
                "max_projection_cosine",
                "physiology_projection_l2_mean",
                "vehicle_projection_l2_mean",
                "projection_l2_gap_mean",
                "projection_l2_ratio_mean",
            ]
        )
        for sample in diagnostics_summary.samples:
            writer.writerow(
                [
                    sample.sample_id,
                    sample.reference_point_count,
                    sample.mean_projection_cosine,
                    sample.min_projection_cosine,
                    sample.max_projection_cosine,
                    sample.physiology_projection_l2_mean,
                    sample.vehicle_projection_l2_mean,
                    sample.projection_l2_gap_mean,
                    sample.projection_l2_ratio_mean,
                ]
            )
    return {
        "projection_diagnostics_summary_json": str(json_path),
        "projection_diagnostics_samples_csv": str(csv_path),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    sortie_id = args.sortie_id
    settings = _resolve_influx_settings()
    loader = build_overlap_preview_sortie_loader(
        OverlapPreviewSortieLoaderConfig(
            sortie_id=sortie_id,
            physiology_scope=DirectInfluxScopeConfig(
                bucket=args.physiology_bucket,
                measurements=tuple(args.physiology_measurements),
                start_time_utc=_utc(args.start_time_utc),
                stop_time_utc=_utc(args.stop_time_utc),
                tag_filters={
                    "collect_task_id": args.collect_task_id,
                    "pilot_id": args.pilot_id,
                },
                point_limit_per_measurement=args.physiology_point_limit,
            ),
            vehicle_scope=DirectInfluxScopeConfig(
                bucket=args.vehicle_bucket,
                measurements=(args.vehicle_measurement,),
                start_time_utc=_utc(args.start_time_utc),
                stop_time_utc=_utc(args.stop_time_utc),
                tag_filters={"sortie_number": sortie_id},
                point_limit_per_measurement=args.vehicle_point_limit,
            ),
        ),
        influx_settings=settings,
    )

    resolved_device = _resolve_device(args.device)
    pipeline = AlignmentExperimentPipeline(
        e0_pipeline=E0PreviewPipeline(
            loader=loader,
            dataset_builder=SortieDatasetBuilder(),
            input_config=E0InputConfig(
                physiology_measurements=tuple(args.physiology_measurements),
                vehicle_measurements=(args.vehicle_measurement,),
            ),
        ),
        alignment_preview_pipeline=AlignmentPreviewPipeline(
            config=AlignmentPreviewConfig(
                prototype_config=AlignmentPrototypeConfig(
                    hidden_dim=args.hidden_dim,
                    embedding_dim=args.embedding_dim,
                    encoder_hidden_dim=args.encoder_hidden_dim,
                    decoder_hidden_dim=args.decoder_hidden_dim,
                    dynamics_hidden_dim=args.dynamics_hidden_dim,
                    projection_dim=args.projection_dim,
                    ode_method=args.ode_method,
                ),
                reference_grid_config=ReferenceGridConfig(point_count=args.reference_point_count),
                epoch_count=args.epoch_count,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=resolved_device,
                reconstruction_loss_mode="relative_mse",
                reconstruction_scale_epsilon=args.reconstruction_scale_epsilon,
                alignment_loss_mode=args.alignment_loss_mode,
                physiology_reconstruction_weight=args.physiology_weight,
                vehicle_reconstruction_weight=args.vehicle_weight,
                alignment_weight=args.alignment_weight,
                export_intermediate_states=True,
                intermediate_sample_limit=args.intermediate_sample_limit,
                intermediate_partition=args.intermediate_partition,
            )
        ),
    )

    result = pipeline.run(SortieLocator(sortie_id=sortie_id))
    report_path = Path(args.report_path)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_summary = summarize_alignment_projection_diagnostics(result.preview_result.intermediate_export)
    diagnostics_markdown = render_alignment_projection_diagnostics_markdown(
        diagnostics_summary,
        max_samples=args.diagnostic_max_samples,
    )
    diagnostics_artifacts = _write_projection_diagnostics_artifacts(
        report_path=report_path,
        diagnostics_summary=diagnostics_summary,
    )
    visual_artifacts = _render_visual_artifacts(
        report_path=report_path,
        experiment_result=result,
    )
    report_with_diagnostics = "\n".join(
        [
            result.report_markdown.rstrip(),
            "",
            diagnostics_markdown.rstrip(),
            "",
        ]
    ).rstrip() + "\n"
    report_markdown = _append_visual_links_to_report(
        report_markdown=report_with_diagnostics,
        report_path=report_path,
        visual_artifacts=visual_artifacts,
    )
    report_path.write_text(report_markdown, encoding="utf-8")

    train_final = result.preview_result.train_history[-1]
    validation_final = result.preview_result.validation_history[-1]
    test_final = result.preview_result.test_metrics
    intermediate = result.preview_result.intermediate_export

    return {
        "report_path": str(report_path),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": resolved_device,
        "sample_summary": asdict(result.sample_summary),
        "split": {
            "train": len(result.preview_result.split.train),
            "validation": len(result.preview_result.split.validation),
            "test": len(result.preview_result.split.test),
            "skipped_between_train_validation": len(result.preview_result.split.skipped_between_train_validation),
            "skipped_between_validation_test": len(result.preview_result.split.skipped_between_validation_test),
        },
        "final_train": asdict(train_final),
        "final_validation": asdict(validation_final),
        "test": asdict(test_final),
        "intermediate_export": (
            None
            if intermediate is None
            else {
                "partition": intermediate.partition,
                "sample_count": intermediate.sample_count,
                "reference_point_count": intermediate.reference_point_count,
            }
        ),
        "projection_diagnostics": asdict(diagnostics_summary),
        "diagnostic_artifacts": diagnostics_artifacts,
        "visual_artifacts": visual_artifacts,
    }


def build_parser() -> argparse.ArgumentParser:
    today = datetime.now(timezone.utc).date().isoformat()
    default_report = f"docs/reports/alignment-preview-20251005-act4-j20-22-relative-mse-{today}.md"

    parser = argparse.ArgumentParser(
        description="Run overlap-focused Stage E preview experiment with relative reconstruction loss.",
    )
    parser.add_argument("--sortie-id", default="20251005_四01_ACT-4_云_J20_22#01")
    parser.add_argument("--physiology-bucket", default="physiological_input")
    parser.add_argument("--vehicle-bucket", default="bus")
    parser.add_argument("--physiology-measurements", nargs="+", default=["eeg", "spo2"])
    parser.add_argument("--vehicle-measurement", default="BUS6000019110020")
    parser.add_argument("--collect-task-id", default="2100448")
    parser.add_argument("--pilot-id", default="10033")
    parser.add_argument("--start-time-utc", default="2025-10-05T01:35:00Z")
    parser.add_argument("--stop-time-utc", default="2025-10-05T01:38:01Z")
    parser.add_argument("--physiology-point-limit", type=int, default=500)
    parser.add_argument("--vehicle-point-limit", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epoch-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--reference-point-count", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--encoder-hidden-dim", type=int, default=64)
    parser.add_argument("--decoder-hidden-dim", type=int, default=64)
    parser.add_argument("--dynamics-hidden-dim", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--ode-method", default="euler")
    parser.add_argument("--alignment-loss-mode", default="mse")
    parser.add_argument("--reconstruction-scale-epsilon", type=float, default=1e-6)
    parser.add_argument("--physiology-weight", type=float, default=1.0)
    parser.add_argument("--vehicle-weight", type=float, default=1.0)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--intermediate-sample-limit", type=int, default=3)
    parser.add_argument("--intermediate-partition", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--diagnostic-max-samples", type=int, default=10)
    parser.add_argument("--report-path", default=default_report)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
