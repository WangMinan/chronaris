"""Run feature export v1 multi-sortie export plus the partial-data sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.settings import (  # noqa: E402
    resolve_influx_settings,
    resolve_mysql_settings,
)
from chronaris.access import (  # noqa: E402
    InfluxCliRunner,
    InfluxDistinctMeasurementReader,
    MySQLCliRunner,
    MySQLCollectTaskReader,
    MySQLFlightTaskReader,
    MySQLRealBusContextReader,
    MySQLStorageAnalysisReader,
    StageHProfileResolver,
)
from chronaris.pipelines.partial_data import (  # noqa: E402
    InfluxPartialVehiclePointProvider,
    MySQLPartialVehicleMetadataProvider,
    PartialDataBuilder,
    PartialDataConfig,
    load_partial_data_entries,
)
from chronaris.pipelines.causal_fusion import StageGCausalFusionConfig  # noqa: E402
from chronaris.feature_export.export import (  # noqa: E402
    AlignmentStageHViewRunner,
    StageHExportConfig,
    StageHExportPipeline,
)


DEFAULT_SORTIES = (
    "20251005_四01_ACT-4_云_J20_22#01",
    "20251002_单01_ACT-8_翼云_J16_12#01",
)

DEFAULT_PREVIEW_SCOPES = {
    "20251005_四01_ACT-4_云_J20_22#01": (
        "2025-10-05T01:35:00Z",
        "2025-10-05T01:38:01Z",
    ),
    "20251002_单01_ACT-8_翼云_J16_12#01": (
        "2025-10-02T08:35:00Z",
        "2025-10-02T08:38:01Z",
    ),
}


def _utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-feature-export-v1")


def _default_report_path() -> str:
    return f"docs/artifacts/feature_export/feature-export-v1-{datetime.now().date().isoformat()}.md"


def _build_default_scope_overrides(use_full_clip_scope: bool) -> dict[str, tuple[datetime, datetime]]:
    if use_full_clip_scope:
        return {}
    return {
        sortie_id: (_utc(bounds[0]), _utc(bounds[1]))
        for sortie_id, bounds in DEFAULT_PREVIEW_SCOPES.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--output-root", default="docs/artifacts/runs")
    parser.add_argument("--report-path", default=_default_report_path())
    parser.add_argument("--sortie-id", dest="sortie_ids", action="append")
    parser.add_argument(
        "--export-profile",
        choices=("preview", "validation", "full_clip"),
        default="preview",
    )
    parser.add_argument("--mysql-database", default="rjgx_backend")
    parser.add_argument("--mysql-binary", default="mysql")
    parser.add_argument("--influx-binary", default="influx")
    parser.add_argument("--bus-access-rule-id", type=int, default=6000019510066)
    parser.add_argument("--preview-point-limit", type=int, default=500)
    parser.add_argument("--physiology-point-limit", type=int)
    parser.add_argument("--vehicle-point-limit", type=int)
    parser.add_argument("--partial-vehicle-point-limit", type=int)
    parser.add_argument(
        "--intermediate-partition",
        choices=("train", "validation", "test", "all"),
    )
    parser.add_argument(
        "--intermediate-sample-limit",
        help="positive integer or 'all'",
    )
    parser.add_argument(
        "--all-window-export",
        action="store_true",
        help="export all windows from train/validation/test into the intermediate bundle",
    )
    parser.add_argument("--disable-physics-constraints", action="store_true")
    parser.add_argument("--disable-causal-fusion", action="store_true")
    parser.add_argument("--disable-partial-data", action="store_true")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--backbone-run-id")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument(
        "--per-view-preview-training",
        action="store_true",
        help="force the historical per-view training export path even when a checkpoint is provided",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--partial-data-path",
        default="configs/partial-data/feature-export-seed-v1.jsonl",
    )
    parser.add_argument("--use-full-clip-scope", action="store_true")
    return parser.parse_args()


def _resolve_preview_point_limit(args: argparse.Namespace) -> int | None:
    if args.export_profile != "preview":
        return None
    return args.preview_point_limit


def _resolve_scope_overrides(args: argparse.Namespace) -> dict[str, tuple[datetime, datetime]]:
    if args.export_profile == "full_clip":
        return {}
    return _build_default_scope_overrides(args.use_full_clip_scope)


def _resolve_partial_vehicle_point_limit(args: argparse.Namespace) -> int | None:
    if args.partial_vehicle_point_limit is not None:
        return args.partial_vehicle_point_limit
    if args.vehicle_point_limit is not None:
        return args.vehicle_point_limit
    return _resolve_preview_point_limit(args)


def _resolve_intermediate_sample_limit(args: argparse.Namespace) -> int | None:
    if args.all_window_export:
        return None
    if args.intermediate_sample_limit is None:
        return StageHExportConfig(run_id="preview-config-probe", sortie_ids=("probe",)).preview_config.intermediate_sample_limit
    if str(args.intermediate_sample_limit).strip().lower() == "all":
        return None
    value = int(args.intermediate_sample_limit)
    if value <= 0:
        raise ValueError("--intermediate-sample-limit must be positive or 'all'.")
    return value


def _resolve_preview_config(args: argparse.Namespace):
    base = StageHExportConfig(run_id="preview-config-probe", sortie_ids=("probe",)).preview_config
    inference_only = _resolve_inference_only(args)
    intermediate_partition = (
        "all"
        if args.all_window_export
        else (
            args.intermediate_partition
            or ("all" if inference_only else base.intermediate_partition)
        )
    )
    return replace(
        base,
        device=args.device,
        intermediate_partition=intermediate_partition,
        intermediate_sample_limit=(
            None if inference_only and args.intermediate_sample_limit is None else _resolve_intermediate_sample_limit(args)
        ),
        enable_physics_constraints=not args.disable_physics_constraints,
    )


def _resolve_inference_only(args: argparse.Namespace) -> bool:
    if args.per_view_preview_training:
        return False
    if args.inference_only:
        return True
    return args.export_profile != "preview" and bool(args.checkpoint_path)


def main() -> int:
    args = parse_args()
    sortie_ids = tuple(args.sortie_ids or DEFAULT_SORTIES)
    influx_settings = resolve_influx_settings(REPO_ROOT / "docs/SECRETS.md")
    mysql_settings = resolve_mysql_settings(args.mysql_database, REPO_ROOT / "docs/SECRETS.md")
    mysql_runner = MySQLCliRunner(mysql_settings, mysql_binary=args.mysql_binary)
    influx_runner = InfluxCliRunner(influx_settings, influx_binary=args.influx_binary)

    profile_resolver = StageHProfileResolver(
        flight_task_reader=MySQLFlightTaskReader(mysql_runner),
        collect_task_reader=MySQLCollectTaskReader(mysql_runner),
        storage_analysis_reader=MySQLStorageAnalysisReader(mysql_runner),
        distinct_measurement_reader=InfluxDistinctMeasurementReader(influx_runner),
    )
    config = StageHExportConfig(
        run_id=args.run_id,
        sortie_ids=sortie_ids,
        output_root=args.output_root,
        report_path=args.report_path,
        export_profile=args.export_profile,
        preview_config=_resolve_preview_config(args),
        causal_fusion_enabled=not args.disable_causal_fusion,
        causal_fusion_config=StageGCausalFusionConfig(device=args.device),
        bus_access_rule_id=args.bus_access_rule_id,
        preview_point_limit_per_measurement=_resolve_preview_point_limit(args),
        physiology_point_limit_per_measurement=args.physiology_point_limit,
        vehicle_point_limit_per_measurement=args.vehicle_point_limit,
        export_scope_overrides_utc=_resolve_scope_overrides(args),
        partial_data_config=PartialDataConfig(
            point_limit_per_measurement=_resolve_partial_vehicle_point_limit(args),
        ),
        partial_data_entries=(
            ()
            if args.disable_partial_data
            else load_partial_data_entries(REPO_ROOT / args.partial_data_path)
        ),
        checkpoint_path=args.checkpoint_path,
        backbone_run_id=args.backbone_run_id,
        inference_only=_resolve_inference_only(args),
    )
    view_runner = AlignmentStageHViewRunner(
        config=config,
        influx_runner=influx_runner,
        vehicle_context_reader=MySQLRealBusContextReader(
            runner=mysql_runner,
            flight_task_reader=MySQLFlightTaskReader(mysql_runner),
        ),
    )
    pipeline = StageHExportPipeline(
        config=config,
        profile_resolver=profile_resolver,
        view_runner=view_runner,
        partial_data_builder=PartialDataBuilder(
            config=config.partial_data_config,
            chunk_provider=InfluxPartialVehiclePointProvider(
                runner=influx_runner,
                point_limit_per_measurement=config.partial_data_config.point_limit_per_measurement,
                window_duration_ms=config.partial_data_config.window_config.duration_ms,
                window_limit_per_field=config.partial_data_config.max_points_per_field_per_window,
            ).iter_chunks,
            metadata_provider=MySQLPartialVehicleMetadataProvider(
                storage_analysis_reader=MySQLStorageAnalysisReader(mysql_runner),
                real_bus_context_reader=MySQLRealBusContextReader(
                    runner=mysql_runner,
                    flight_task_reader=MySQLFlightTaskReader(mysql_runner),
                ),
                access_rule_id=args.bus_access_rule_id,
            ),
        ) if not args.disable_partial_data else None,
    )
    result = pipeline.run()
    print(
        json.dumps(
            {
                "run_manifest_path": result.run_manifest_path,
                "report_path": result.report_path,
                "output_root": result.output_root,
                "generated_view_ids": list(result.generated_view_ids),
                "partial_data_manifest_path": (
                    None
                    if result.partial_data_result is None
                    else result.partial_data_result.manifest_path
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
