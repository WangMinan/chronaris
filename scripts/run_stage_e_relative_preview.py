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
    MySQLCliRunner,
    MySQLFlightTaskReader,
    MySQLRealBusContextReader,
    MySQLSettings,
    OverlapPreviewSortieLoaderConfig,
    build_overlap_preview_sortie_loader,
)
from chronaris.dataset.builder import SortieDatasetBuilder
from chronaris.evaluation import (
    AlignmentProjectionThresholdConfig,
    evaluate_alignment_projection_thresholds,
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
    pattern = re.compile(rf"^\+?\s*{re.escape(key)}:\s*(.+)$", re.MULTILINE)
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


def _resolve_mysql_settings(args: argparse.Namespace) -> MySQLSettings:
    host = os.environ.get("CHRONARIS_MYSQL_HOST")
    port = os.environ.get("CHRONARIS_MYSQL_PORT")
    user = os.environ.get("CHRONARIS_MYSQL_USER")
    password = os.environ.get("CHRONARIS_MYSQL_PASSWORD")
    if not (host and port and user and password):
        secrets_path = REPO_ROOT / "docs" / "SECRETS.md"
        secrets_text = secrets_path.read_text(encoding="utf-8")
        host = host or _extract_secret(secrets_text, "host")
        port = port or _extract_secret(secrets_text, "port")
        user = user or _extract_secret(secrets_text, "username")
        password = password or _extract_secret(secrets_text, "password")
    return MySQLSettings(
        host=host,
        port=int(port),
        database=args.mysql_database,
        user=user,
        password_env=None,
        password_value=password,
    )


def _resolve_vehicle_field_labels(args: argparse.Namespace, sortie_id: str) -> tuple[dict[str, str], dict[str, object]]:
    if args.skip_mysql_field_labels:
        return {}, {"status": "skipped", "field_count": 0, "error": None}
    try:
        mysql_runner = MySQLCliRunner(_resolve_mysql_settings(args))
        context_reader = MySQLRealBusContextReader(
            runner=mysql_runner,
            flight_task_reader=MySQLFlightTaskReader(mysql_runner),
        )
        context = context_reader.fetch_context(
            locator=SortieLocator(sortie_id=sortie_id),
            access_rule_id=args.bus_access_rule_id,
            analysis_id=args.bus_analysis_id,
        )
    except Exception as exc:  # pragma: no cover - exercised by live closure runs.
        if args.strict_mysql_field_labels:
            raise
        return {}, {"status": "unavailable", "field_count": 0, "error": str(exc)}

    labels: dict[str, str] = {}
    measurement = context.analysis.measurement or args.vehicle_measurement
    for detail in context.detail_list:
        labels[detail.col_field] = detail.col_name
        labels[f"{measurement}.{detail.col_field}"] = detail.col_name
    for structure in context.structure_list:
        labels.setdefault(structure.col_field, structure.col_name)
        labels.setdefault(f"{measurement}.{structure.col_field}", structure.col_name)
    for access_detail in context.access_rule_details:
        if access_detail.col_name:
            labels.setdefault(access_detail.col_field, access_detail.col_name)
            labels.setdefault(f"{measurement}.{access_detail.col_field}", access_detail.col_name)

    return labels, {
        "status": "loaded",
        "field_count": len(labels),
        "measurement": measurement,
        "analysis_id": context.analysis.analysis_id,
        "access_rule_id": args.bus_access_rule_id,
        "error": None,
    }


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

    physics_totals = [item.physics_total for item in train_history]
    has_physics_signal = any(value > 0.0 for value in physics_totals)
    if has_physics_signal:
        fig_physics, ax_physics = plt.subplots(figsize=(8, 4.5))
        ax_physics.plot(
            epoch_indexes,
            [item.physics_total for item in train_history],
            marker="o",
            linewidth=2.0,
            label="train physics total",
        )
        if validation_history:
            ax_physics.plot(
                epoch_indexes,
                [item.physics_total for item in validation_history],
                marker="o",
                linewidth=2.0,
                label="validation physics total",
            )
        ax_physics.set_title("Stage F(min) Physics Constraint Loss")
        ax_physics.set_xlabel("Epoch")
        ax_physics.set_ylabel("Loss")
        ax_physics.grid(True, alpha=0.25)
        ax_physics.legend(loc="best")
        physics_path = assets_dir / "train_validation_physics_loss.png"
        fig_physics.tight_layout()
        fig_physics.savefig(physics_path, dpi=160)
        plt.close(fig_physics)
        image_paths["train_validation_physics_loss"] = physics_path

        final_train = train_history[-1]
        final_validation = validation_history[-1] if validation_history else None
        test_metrics = preview_result.test_metrics
        labels = ["vehicle physics", "physiology physics"]
        train_values = [final_train.vehicle_physics, final_train.physiology_physics]
        validation_values = (
            [final_validation.vehicle_physics, final_validation.physiology_physics]
            if final_validation is not None
            else [0.0, 0.0]
        )
        test_values = [test_metrics.vehicle_physics, test_metrics.physiology_physics]

        fig_components, ax_components = plt.subplots(figsize=(8, 4.5))
        x_positions = [0, 1]
        width = 0.22
        ax_components.bar([x - width for x in x_positions], train_values, width=width, label="train")
        if final_validation is not None:
            ax_components.bar(x_positions, validation_values, width=width, label="validation")
            ax_components.bar([x + width for x in x_positions], test_values, width=width, label="test")
        else:
            ax_components.bar(x_positions, test_values, width=width, label="test")
        ax_components.set_xticks(x_positions)
        ax_components.set_xticklabels(labels)
        ax_components.set_ylabel("Loss")
        ax_components.set_title("Final Physics Constraint Components")
        ax_components.grid(True, axis="y", alpha=0.25)
        ax_components.legend(loc="best")
        component_path = assets_dir / "constraint_component_breakdown.png"
        fig_components.tight_layout()
        fig_components.savefig(component_path, dpi=160)
        plt.close(fig_components)
        image_paths["constraint_component_breakdown"] = component_path

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
        "train_validation_physics_loss": "Train/Validation Physics Loss",
        "constraint_component_breakdown": "Final Physics Constraint Components",
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
    threshold_config: AlignmentProjectionThresholdConfig,
    threshold_evaluation,
) -> dict[str, str]:
    assets_dir = report_path.parent / "assets" / report_path.stem
    assets_dir.mkdir(parents=True, exist_ok=True)

    json_path = assets_dir / "projection_diagnostics_summary.json"
    summary_payload = {
        "summary": asdict(diagnostics_summary),
        "threshold_config": asdict(threshold_config),
        "threshold_evaluation": asdict(threshold_evaluation),
    }
    json_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
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


def _state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    cpu_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state


def _write_model_checkpoint(
    *,
    report_path: Path,
    sortie_id: str,
    model: torch.nn.Module,
    normalization_mode: str,
    resolved_device: str,
    args: argparse.Namespace,
    enable_physics_constraints: bool,
    split: dict[str, int],
    final_train: dict[str, object],
    final_validation: dict[str, object],
    final_test: dict[str, object],
) -> dict[str, str]:
    assets_dir = report_path.parent / "assets" / report_path.stem
    assets_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = assets_dir / "alignment_model_checkpoint.pt"
    payload = {
        "checkpoint_format_version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sortie_id": sortie_id,
        "input_normalization_mode": normalization_mode,
        "runtime_device": resolved_device,
        "seed": args.seed,
        "prototype_config": {
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
            "encoder_hidden_dim": args.encoder_hidden_dim,
            "decoder_hidden_dim": args.decoder_hidden_dim,
            "dynamics_hidden_dim": args.dynamics_hidden_dim,
            "projection_dim": args.projection_dim,
            "ode_method": args.ode_method,
        },
        "training_config": {
            "epoch_count": args.epoch_count,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "reconstruction_loss_mode": "relative_mse",
            "reconstruction_scale_epsilon": args.reconstruction_scale_epsilon,
            "alignment_loss_mode": args.alignment_loss_mode,
            "physiology_weight": args.physiology_weight,
            "vehicle_weight": args.vehicle_weight,
            "alignment_weight": args.alignment_weight,
            "enable_physics_constraints": enable_physics_constraints,
            "physics_constraint_mode": args.physics_constraint_mode,
            "physics_constraint_family": args.physics_constraint_family,
            "vehicle_physics_weight": args.vehicle_physics_weight,
            "physiology_physics_weight": args.physiology_physics_weight,
            "physics_huber_delta": args.physics_huber_delta,
            "vehicle_envelope_quantile": args.vehicle_envelope_quantile,
            "physiology_envelope_quantile": args.physiology_envelope_quantile,
            "reference_point_count": args.reference_point_count,
            "intermediate_partition": args.intermediate_partition,
            "intermediate_sample_limit": args.intermediate_sample_limit,
        },
        "split": split,
        "final_metrics": {
            "train": final_train,
            "validation": final_validation,
            "test": final_test,
        },
        "model_state_dict": _state_dict_to_cpu(model),
    }
    torch.save(payload, checkpoint_path)
    return {
        "alignment_model_checkpoint": str(checkpoint_path),
    }


def _build_threshold_config(args: argparse.Namespace) -> AlignmentProjectionThresholdConfig:
    return AlignmentProjectionThresholdConfig(
        min_sample_count=args.threshold_min_sample_count,
        min_mean_projection_cosine=args.threshold_min_mean_cosine,
        enforce_min_projection_cosine=args.threshold_enforce_min_cosine,
        min_min_projection_cosine=args.threshold_min_cosine,
        max_mean_projection_l2_gap=args.threshold_max_mean_l2_gap,
        max_mean_projection_l2_ratio_deviation=args.threshold_max_mean_l2_ratio_deviation,
        max_projection_cosine_cv=args.threshold_max_cosine_cv,
        max_projection_l2_gap_cv=args.threshold_max_l2_gap_cv,
    )


def _render_physics_diagnostics_markdown(
    *,
    enabled: bool,
    family: str,
    mode: str,
    vehicle_metadata_summary: dict[str, object],
    train_final: dict[str, object],
    validation_final: dict[str, object],
    test_final: dict[str, object],
) -> str:
    train_components = train_final.get("physics_components") or {}
    validation_components = validation_final.get("physics_components") or {}
    test_components = test_final.get("physics_components") or {}
    lines = [
        "## Physics Constraint Diagnostics",
        "",
        f"- enabled: `{enabled}`",
        f"- family: `{family if enabled else '(disabled)'}`",
        f"- mode: `{mode if enabled else '(disabled)'}`",
        f"- vehicle metadata status: `{vehicle_metadata_summary.get('status')}`",
        f"- vehicle metadata fields: `{vehicle_metadata_summary.get('field_count')}`",
        "",
        "| metric | train | validation | test |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| vehicle physics | {train_final['vehicle_physics']:.6f} | "
            f"{validation_final['vehicle_physics']:.6f} | {test_final['vehicle_physics']:.6f} |"
        ),
        (
            f"| physiology physics | {train_final['physiology_physics']:.6f} | "
            f"{validation_final['physiology_physics']:.6f} | {test_final['physiology_physics']:.6f} |"
        ),
        (
            f"| physics total | {train_final['physics_total']:.6f} | "
            f"{validation_final['physics_total']:.6f} | {test_final['physics_total']:.6f} |"
        ),
        "",
    ]
    if train_components or validation_components or test_components:
        lines.extend(
            [
                "### Component Breakdown",
                "",
                "| component | train | validation | test |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        component_names = sorted(set(train_components) | set(validation_components) | set(test_components))
        for component_name in component_names:
            lines.append(
                f"| {component_name} | {train_components.get(component_name, 0.0):.6f} | "
                f"{validation_components.get(component_name, 0.0):.6f} | "
                f"{test_components.get(component_name, 0.0):.6f} |"
            )
        lines.append("")
    metadata_error = vehicle_metadata_summary.get("error")
    if metadata_error:
        lines.extend(["### Metadata Warning", "", f"- `{metadata_error}`", ""])
    return "\n".join(lines)


def _with_mode_suffix(report_path: str, mode: str) -> str:
    path = Path(report_path)
    return str(path.with_name(f"{path.stem}-{mode}{path.suffix}"))


def _render_normalization_comparison_markdown(
    *,
    primary_mode: str,
    primary_summary: dict[str, object],
    secondary_mode: str,
    secondary_summary: dict[str, object],
) -> str:
    primary_threshold = primary_summary["threshold_evaluation"]["verdict"]
    secondary_threshold = secondary_summary["threshold_evaluation"]["verdict"]
    primary_diag = primary_summary["projection_diagnostics"]
    secondary_diag = secondary_summary["projection_diagnostics"]
    primary_visuals = primary_summary["visual_artifacts"]
    secondary_visuals = secondary_summary["visual_artifacts"]
    primary_checkpoints = primary_summary["checkpoint_artifacts"]
    secondary_checkpoints = secondary_summary["checkpoint_artifacts"]
    return "\n".join(
        [
            "# Stage E Normalization Comparison",
            "",
            f"- primary mode: `{primary_mode}`",
            f"- secondary mode: `{secondary_mode}`",
            f"- sample count: `{primary_summary['sample_summary']['sample_count']}`",
            (
                f"- split: train `{primary_summary['split']['train']}`, "
                f"validation `{primary_summary['split']['validation']}`, "
                f"test `{primary_summary['split']['test']}`"
            ),
            "",
            "| metric | primary | secondary |",
            "| --- | ---: | ---: |",
            f"| final train total | {primary_summary['final_train']['total']:.6f} | {secondary_summary['final_train']['total']:.6f} |",
            f"| final validation total | {primary_summary['final_validation']['total']:.6f} | {secondary_summary['final_validation']['total']:.6f} |",
            f"| test total | {primary_summary['test']['total']:.6f} | {secondary_summary['test']['total']:.6f} |",
            f"| test physics total | {primary_summary['test']['physics_total']:.6f} | {secondary_summary['test']['physics_total']:.6f} |",
            f"| threshold verdict | {primary_threshold} | {secondary_threshold} |",
            "",
            "## Diagnostics Summary",
            "",
            "| metric | primary | secondary |",
            "| --- | ---: | ---: |",
            f"| mean projection cosine | {primary_diag['mean_projection_cosine']:.6f} | {secondary_diag['mean_projection_cosine']:.6f} |",
            f"| min projection cosine | {primary_diag['min_projection_cosine']:.6f} | {secondary_diag['min_projection_cosine']:.6f} |",
            f"| mean projection L2 gap | {primary_diag['mean_projection_l2_gap']:.6f} | {secondary_diag['mean_projection_l2_gap']:.6f} |",
            f"| mean projection L2 ratio | {primary_diag['mean_projection_l2_ratio']:.6f} | {secondary_diag['mean_projection_l2_ratio']:.6f} |",
            "",
            "## Decision",
            "",
            (
                f"- both threshold templates are `{primary_threshold}` / `{secondary_threshold}` "
                "under the default Stage E closure rules"
            ),
            (
                f"- recommended mode for Stage F entry: `{secondary_mode}` "
                f"(lower test total `{secondary_summary['test']['total']:.6f}` vs `{primary_summary['test']['total']:.6f}`)"
            ),
            "",
            "## Visual Artifacts",
            "",
            f"### Primary `{primary_mode}` - Total Loss",
            "",
            f"![Primary Total Loss]({Path(primary_visuals['train_validation_total_loss']).relative_to(REPO_ROOT / 'docs' / 'reports').as_posix()})",
            "",
            f"### Secondary `{secondary_mode}` - Total Loss",
            "",
            f"![Secondary Total Loss]({Path(secondary_visuals['train_validation_total_loss']).relative_to(REPO_ROOT / 'docs' / 'reports').as_posix()})",
            "",
            "## Diagnostic Artifacts",
            "",
            f"- primary summary json: `{primary_summary['diagnostic_artifacts']['projection_diagnostics_summary_json']}`",
            f"- secondary summary json: `{secondary_summary['diagnostic_artifacts']['projection_diagnostics_summary_json']}`",
            "",
            "## Checkpoints",
            "",
            f"- primary checkpoint: `{primary_checkpoints['alignment_model_checkpoint']}`",
            f"- secondary checkpoint: `{secondary_checkpoints['alignment_model_checkpoint']}`",
            "",
        ]
    ).rstrip() + "\n"


def _render_physics_comparison_markdown(
    *,
    baseline_summary: dict[str, object],
    physics_summary: dict[str, object],
) -> str:
    baseline_threshold = baseline_summary["threshold_evaluation"]["verdict"]
    physics_threshold = physics_summary["threshold_evaluation"]["verdict"]
    baseline_components = baseline_summary["test"].get("physics_components") or {}
    physics_components = physics_summary["test"].get("physics_components") or {}
    return "\n".join(
        [
            "# Stage F Full Physics Comparison",
            "",
            f"- baseline normalization: `{baseline_summary['input_normalization_mode']}`",
            f"- physics family: `{physics_summary['physics_config']['family']}`",
            f"- physics mode: `{physics_summary['physics_config']['mode']}`",
            f"- sample count: `{baseline_summary['sample_summary']['sample_count']}`",
            (
                f"- split: train `{baseline_summary['split']['train']}`, "
                f"validation `{baseline_summary['split']['validation']}`, "
                f"test `{baseline_summary['split']['test']}`"
            ),
            f"- vehicle metadata status: `{physics_summary['vehicle_field_metadata']['status']}`",
            f"- vehicle metadata fields: `{physics_summary['vehicle_field_metadata']['field_count']}`",
            "",
            "| metric | E baseline | E+F(full) |",
            "| --- | ---: | ---: |",
            f"| final train total | {baseline_summary['final_train']['total']:.6f} | {physics_summary['final_train']['total']:.6f} |",
            f"| final validation total | {baseline_summary['final_validation']['total']:.6f} | {physics_summary['final_validation']['total']:.6f} |",
            f"| test total | {baseline_summary['test']['total']:.6f} | {physics_summary['test']['total']:.6f} |",
            f"| test physics total | {baseline_summary['test']['physics_total']:.6f} | {physics_summary['test']['physics_total']:.6f} |",
            f"| threshold verdict | {baseline_threshold} | {physics_threshold} |",
            "",
            "## Test Physics Components",
            "",
            "| component | E baseline | E+F(full) |",
            "| --- | ---: | ---: |",
            *[
                f"| {component_name} | {baseline_components.get(component_name, 0.0):.6f} | "
                f"{physics_components.get(component_name, 0.0):.6f} |"
                for component_name in sorted(set(baseline_components) | set(physics_components))
            ],
            "",
            "## Decision",
            "",
            (
                f"- Stage F closure candidate verdict: `{physics_threshold}` "
                f"(baseline `{baseline_threshold}`)"
            ),
            "- physics constraints are considered active when at least one E+F(full) component is non-zero",
            "",
            "## Reports",
            "",
            f"- E baseline report: `{baseline_summary['report_path']}`",
            f"- E+F(full) report: `{physics_summary['report_path']}`",
            "",
        ]
    ).rstrip() + "\n"


def _run_once(
    args: argparse.Namespace,
    *,
    normalization_mode: str,
    enable_physics_constraints: bool | None = None,
    report_path_override: str | None = None,
) -> dict[str, object]:
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    sortie_id = args.sortie_id
    settings = _resolve_influx_settings()
    resolved_enable_physics = (
        args.enable_physics_constraints if enable_physics_constraints is None else enable_physics_constraints
    )
    vehicle_field_labels, vehicle_metadata_summary = _resolve_vehicle_field_labels(args, sortie_id)
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
                input_normalization_mode=normalization_mode,
                input_normalization_epsilon=args.input_normalization_epsilon,
                alignment_loss_mode=args.alignment_loss_mode,
                physiology_reconstruction_weight=args.physiology_weight,
                vehicle_reconstruction_weight=args.vehicle_weight,
                alignment_weight=args.alignment_weight,
                enable_physics_constraints=resolved_enable_physics,
                physics_constraint_mode=args.physics_constraint_mode,
                physics_constraint_family=args.physics_constraint_family,
                vehicle_physics_weight=args.vehicle_physics_weight,
                physiology_physics_weight=args.physiology_physics_weight,
                physics_huber_delta=args.physics_huber_delta,
                vehicle_envelope_quantile=args.vehicle_envelope_quantile,
                physiology_envelope_quantile=args.physiology_envelope_quantile,
                vehicle_field_labels=vehicle_field_labels,
                export_intermediate_states=True,
                intermediate_sample_limit=args.intermediate_sample_limit,
                intermediate_partition=args.intermediate_partition,
            )
        ),
    )

    result = pipeline.run(SortieLocator(sortie_id=sortie_id))
    report_path = Path(report_path_override or args.report_path)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_summary = summarize_alignment_projection_diagnostics(result.preview_result.intermediate_export)
    threshold_config = _build_threshold_config(args)
    threshold_evaluation = evaluate_alignment_projection_thresholds(
        diagnostics_summary,
        config=threshold_config,
    )
    diagnostics_markdown = render_alignment_projection_diagnostics_markdown(
        diagnostics_summary,
        max_samples=args.diagnostic_max_samples,
        threshold_evaluation=threshold_evaluation,
    )
    diagnostics_artifacts = _write_projection_diagnostics_artifacts(
        report_path=report_path,
        diagnostics_summary=diagnostics_summary,
        threshold_config=threshold_config,
        threshold_evaluation=threshold_evaluation,
    )
    visual_artifacts = _render_visual_artifacts(
        report_path=report_path,
        experiment_result=result,
    )
    train_final = asdict(result.preview_result.train_history[-1])
    validation_final = asdict(result.preview_result.validation_history[-1])
    test_final = asdict(result.preview_result.test_metrics)
    physics_diagnostics_markdown = _render_physics_diagnostics_markdown(
        enabled=resolved_enable_physics,
        family=args.physics_constraint_family,
        mode=args.physics_constraint_mode,
        vehicle_metadata_summary=vehicle_metadata_summary,
        train_final=train_final,
        validation_final=validation_final,
        test_final=test_final,
    )
    report_with_diagnostics = "\n".join(
        [
            result.report_markdown.rstrip(),
            "",
            physics_diagnostics_markdown.rstrip(),
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

    intermediate = result.preview_result.intermediate_export
    split_summary = {
        "train": len(result.preview_result.split.train),
        "validation": len(result.preview_result.split.validation),
        "test": len(result.preview_result.split.test),
        "skipped_between_train_validation": len(result.preview_result.split.skipped_between_train_validation),
        "skipped_between_validation_test": len(result.preview_result.split.skipped_between_validation_test),
    }
    checkpoint_artifacts = _write_model_checkpoint(
        report_path=report_path,
        sortie_id=sortie_id,
        model=result.preview_result.model,
        normalization_mode=normalization_mode,
        resolved_device=resolved_device,
        args=args,
        enable_physics_constraints=resolved_enable_physics,
        split=split_summary,
        final_train=train_final,
        final_validation=validation_final,
        final_test=test_final,
    )

    return {
        "report_path": str(report_path),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": resolved_device,
        "sample_summary": asdict(result.sample_summary),
        "split": split_summary,
        "final_train": train_final,
        "final_validation": validation_final,
        "test": test_final,
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
        "threshold_config": asdict(threshold_config),
        "threshold_evaluation": asdict(threshold_evaluation),
        "diagnostic_artifacts": diagnostics_artifacts,
        "visual_artifacts": visual_artifacts,
        "checkpoint_artifacts": checkpoint_artifacts,
        "input_normalization_mode": normalization_mode,
        "physics_config": {
            "enabled": resolved_enable_physics,
            "family": args.physics_constraint_family,
            "mode": args.physics_constraint_mode,
            "vehicle_weight": args.vehicle_physics_weight,
            "physiology_weight": args.physiology_physics_weight,
            "huber_delta": args.physics_huber_delta,
            "vehicle_envelope_quantile": args.vehicle_envelope_quantile,
            "physiology_envelope_quantile": args.physiology_envelope_quantile,
        },
        "vehicle_field_metadata": vehicle_metadata_summary,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.compare_with_physics_baseline:
        baseline_summary = _run_once(
            args,
            normalization_mode=args.input_normalization_mode,
            enable_physics_constraints=False,
            report_path_override=_with_mode_suffix(args.report_path, "e-baseline"),
        )
        physics_summary = _run_once(
            args,
            normalization_mode=args.input_normalization_mode,
            enable_physics_constraints=True,
            report_path_override=_with_mode_suffix(args.report_path, f"stage-f-{args.physics_constraint_family}"),
        )
        comparison_report_path = Path(args.report_path)
        if not comparison_report_path.is_absolute():
            comparison_report_path = REPO_ROOT / comparison_report_path
        comparison_report_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_report_path.write_text(
            _render_physics_comparison_markdown(
                baseline_summary=baseline_summary,
                physics_summary=physics_summary,
            ),
            encoding="utf-8",
        )
        return {
            "report_path": str(comparison_report_path),
            "baseline": baseline_summary,
            "physics": physics_summary,
        }

    if not args.compare_with_zscore_train:
        return _run_once(
            args,
            normalization_mode=args.input_normalization_mode,
        )

    primary_mode = args.input_normalization_mode
    secondary_mode = "zscore_train" if primary_mode != "zscore_train" else "none"
    primary_summary = _run_once(
        args,
        normalization_mode=primary_mode,
        report_path_override=_with_mode_suffix(args.report_path, primary_mode),
    )
    secondary_summary = _run_once(
        args,
        normalization_mode=secondary_mode,
        report_path_override=_with_mode_suffix(args.report_path, secondary_mode),
    )

    comparison_report_path = Path(args.report_path)
    if not comparison_report_path.is_absolute():
        comparison_report_path = REPO_ROOT / comparison_report_path
    comparison_report_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_markdown = _render_normalization_comparison_markdown(
        primary_mode=primary_mode,
        primary_summary=primary_summary,
        secondary_mode=secondary_mode,
        secondary_summary=secondary_summary,
    )
    comparison_report_path.write_text(comparison_markdown, encoding="utf-8")
    return {
        "comparison_report_path": str(comparison_report_path),
        "primary": primary_summary,
        "secondary": secondary_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    today = datetime.now(timezone.utc).date().isoformat()
    default_report = f"docs/reports/alignment-preview-20251005-act4-j20-22-relative-mse-{today}.md"

    parser = argparse.ArgumentParser(
        description=(
            "Run overlap-focused Stage E baseline / Stage F(min) preview "
            "with relative reconstruction loss and optional physics constraints."
        ),
    )
    parser.add_argument("--sortie-id", default="20251005_四01_ACT-4_云_J20_22#01")
    parser.add_argument("--physiology-bucket", default="physiological_input")
    parser.add_argument("--vehicle-bucket", default="bus")
    parser.add_argument("--physiology-measurements", nargs="+", default=["eeg", "spo2"])
    parser.add_argument("--vehicle-measurement", default="BUS6000019110020")
    parser.add_argument("--collect-task-id", default="2100448")
    parser.add_argument("--pilot-id", default="10033")
    parser.add_argument("--mysql-database", default="rjgx_backend")
    parser.add_argument("--bus-access-rule-id", type=int, default=6000019510066)
    parser.add_argument("--bus-analysis-id", type=int, default=6000019110020)
    parser.add_argument("--skip-mysql-field-labels", action="store_true")
    parser.add_argument("--strict-mysql-field-labels", action="store_true")
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
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--reconstruction-scale-epsilon", type=float, default=1e-6)
    parser.add_argument("--input-normalization-mode", choices=("none", "zscore_train"), default="none")
    parser.add_argument("--input-normalization-epsilon", type=float, default=1e-6)
    parser.add_argument("--physiology-weight", type=float, default=1.0)
    parser.add_argument("--vehicle-weight", type=float, default=1.0)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--enable-physics-constraints", action="store_true")
    parser.add_argument(
        "--physics-constraint-mode",
        choices=("feature_first_with_latent_fallback", "feature_only", "latent_only"),
        default="feature_first_with_latent_fallback",
    )
    parser.add_argument("--physics-constraint-family", choices=("minimal", "full"), default="minimal")
    parser.add_argument("--vehicle-physics-weight", type=float, default=0.1)
    parser.add_argument("--physiology-physics-weight", type=float, default=0.1)
    parser.add_argument("--physics-huber-delta", type=float, default=1.0)
    parser.add_argument("--vehicle-envelope-quantile", type=float, default=0.95)
    parser.add_argument("--physiology-envelope-quantile", type=float, default=0.95)
    parser.add_argument("--intermediate-sample-limit", type=int, default=3)
    parser.add_argument("--intermediate-partition", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--diagnostic-max-samples", type=int, default=10)
    parser.add_argument("--threshold-min-sample-count", type=int, default=1)
    parser.add_argument("--threshold-min-mean-cosine", type=float, default=0.65)
    parser.add_argument("--threshold-enforce-min-cosine", action="store_true")
    parser.add_argument("--threshold-min-cosine", type=float, default=0.10)
    parser.add_argument("--threshold-max-mean-l2-gap", type=float, default=0.25)
    parser.add_argument("--threshold-max-mean-l2-ratio-deviation", type=float, default=0.30)
    parser.add_argument("--threshold-max-cosine-cv", type=float, default=0.15)
    parser.add_argument("--threshold-max-l2-gap-cv", type=float, default=0.25)
    parser.add_argument("--compare-with-zscore-train", action="store_true")
    parser.add_argument("--compare-with-physics-baseline", action="store_true")
    parser.add_argument("--report-path", default=default_report)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
