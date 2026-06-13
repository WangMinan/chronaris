"""Run a bounded Stage I thesis weak-label multitask sweep on real Stage H assets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access import (  # noqa: E402
    InfluxCliRunner,
    InfluxDistinctMeasurementReader,
    InfluxSettings,
    MySQLCliRunner,
    MySQLCollectTaskReader,
    MySQLFlightTaskReader,
    MySQLSettings,
    MySQLStorageAnalysisReader,
    StageHProfileResolver,
)
from chronaris.dataset import build_stage_i_real_task_payload  # noqa: E402
from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix  # noqa: E402
from chronaris.pipelines import collect_stage_i_multitask_samples  # noqa: E402
from chronaris.pipelines.stage_i.stage_i_multitask_sweep import (  # noqa: E402
    StageIMultitaskSweepConfig,
    resolve_git_commit,
    run_stage_i_multitask_sweep,
)
from chronaris.schema.models import StreamKind  # noqa: E402
from chronaris.pipelines.stage_i.stage_i_private_benchmark_data import (  # noqa: E402
    load_aligned_private_records,
    validate_private_stage_h_run_contract,
)
from chronaris.schema.models import WindowConfig  # noqa: E402


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-multitask-sweep")


def _extract_secret(md_text: str, key: str) -> str:
    pattern = re.compile(rf"^\+?\s*{re.escape(key)}:\s*(.+)$", re.MULTILINE)
    matched = pattern.search(md_text)
    if not matched:
        raise RuntimeError(f"Missing secret key in docs/SECRETS.md: {key}")
    return matched.group(1).strip()


def _resolve_influx_settings() -> InfluxSettings:
    url = os.environ.get("CHRONARIS_INFLUX_URL")
    org = os.environ.get("CHRONARIS_INFLUX_ORG")
    token = os.environ.get("CHRONARIS_INFLUX_TOKEN")
    if not (org and token):
        secrets_text = (REPO_ROOT / "docs" / "SECRETS.md").read_text(encoding="utf-8")
        org = org or _extract_secret(secrets_text, "influxdb.org")
        token = token or _extract_secret(secrets_text, "influxdb.token")
    url = url or "http://127.0.0.1:8086"
    return InfluxSettings(url=url, org=org, token_env=None, token_value=token)


def _resolve_mysql_settings(database: str) -> MySQLSettings:
    host = os.environ.get("CHRONARIS_MYSQL_HOST")
    port = os.environ.get("CHRONARIS_MYSQL_PORT")
    user = os.environ.get("CHRONARIS_MYSQL_USER")
    password = os.environ.get("CHRONARIS_MYSQL_PASSWORD")
    if not (user and password):
        secrets_text = (REPO_ROOT / "docs" / "SECRETS.md").read_text(encoding="utf-8")
        user = user or _extract_secret(secrets_text, "username")
        password = password or _extract_secret(secrets_text, "password")
    host = host or "127.0.0.1"
    port = port or "3306"
    return MySQLSettings(
        host=host,
        port=int(port),
        database=database,
        user=user,
        password_env=None,
        password_value=password,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--e-run-manifest", required=True)
    parser.add_argument("--f-run-manifest", required=True)
    parser.add_argument("--output-root", default="docs/artifacts/assets/stage_i_multitask_sweep")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--mysql-database", default="rjgx_backend")
    parser.add_argument("--mysql-binary", default="mysql")
    parser.add_argument("--influx-binary", default="influx")
    parser.add_argument("--epoch-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--max-runs", type=int, default=4)
    parser.add_argument("--physiology-point-limit", type=int)
    parser.add_argument("--vehicle-point-limit", type=int)
    parser.add_argument(
        "--sample-source",
        choices=("live_influx", "stage_h_window_stats_proxy"),
        default="live_influx",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    e_manifest = _load_json(args.e_run_manifest)
    f_manifest = _load_json(args.f_run_manifest)
    validate_private_stage_h_run_contract(e_manifest, stage_name="E")
    validate_private_stage_h_run_contract(f_manifest, stage_name="F")
    _validate_matched_stage_h_manifests(e_manifest, f_manifest)

    records = load_aligned_private_records(
        e_run_manifest_path=args.e_run_manifest,
        f_run_manifest_path=args.f_run_manifest,
    )
    task_payload = build_stage_i_real_task_payload(records)
    if args.sample_source == "stage_h_window_stats_proxy":
        samples = _build_proxy_multitask_samples(records)
        sample_source_summary = {
            "sample_source": "stage_h_window_stats_proxy",
            "sample_count": len(samples),
            "note": "derived from Stage H raw window summaries instead of live Influx reload",
        }
    else:
        influx_runner = InfluxCliRunner(_resolve_influx_settings(), influx_binary=args.influx_binary)
        mysql_runner = MySQLCliRunner(
            _resolve_mysql_settings(args.mysql_database),
            mysql_binary=args.mysql_binary,
        )
        profile_resolver = StageHProfileResolver(
            flight_task_reader=MySQLFlightTaskReader(mysql_runner),
            collect_task_reader=MySQLCollectTaskReader(mysql_runner),
            storage_analysis_reader=MySQLStorageAnalysisReader(mysql_runner),
            distinct_measurement_reader=InfluxDistinctMeasurementReader(influx_runner),
        )
        sortie_ids = tuple(str(sortie_id) for sortie_id in f_manifest["sortie_ids"])
        profiles = profile_resolver.resolve_many(sortie_ids)
        samples, sample_source_summary = collect_stage_i_multitask_samples(
            profiles=profiles,
            window_config=_resolve_window_config(f_manifest),
            physiology_point_limit_per_measurement=args.physiology_point_limit,
            vehicle_point_limit_per_measurement=args.vehicle_point_limit,
            export_scope_overrides_utc=_resolve_scope_overrides(f_manifest),
            runner=influx_runner,
        )
    result = run_stage_i_multitask_sweep(
        StageIMultitaskSweepConfig(
            run_id=args.run_id,
            output_root=args.output_root,
            report_root=args.report_root,
            max_runs=args.max_runs,
            epoch_count=args.epoch_count,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            git_commit=resolve_git_commit(cwd=REPO_ROOT),
            source_manifests={
                "e_run_manifest_path": str(Path(args.e_run_manifest)),
                "f_run_manifest_path": str(Path(args.f_run_manifest)),
            },
        ),
        samples=samples,
        task_entries=task_payload["entries"],
        source_summary={
            "sample_collection": sample_source_summary,
            "task_payload_summary": task_payload["summary"],
        },
    )
    print(
        json.dumps(
            {
                "artifact_root": result.artifact_root,
                "summary_path": result.summary_path,
                "table_path": result.table_path,
                "report_path": result.report_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _load_json(path_like: str) -> dict[str, object]:
    return json.loads(Path(path_like).read_text(encoding="utf-8"))


def _validate_matched_stage_h_manifests(
    e_manifest: dict[str, object],
    f_manifest: dict[str, object],
) -> None:
    if tuple(e_manifest.get("sortie_ids", ())) != tuple(f_manifest.get("sortie_ids", ())):
        raise ValueError("E/F Stage H manifests must use the same sortie_ids.")
    if tuple(e_manifest.get("generated_view_ids", ())) != tuple(f_manifest.get("generated_view_ids", ())):
        raise ValueError("E/F Stage H manifests must export the same generated_view_ids.")


def _resolve_window_config(f_run_manifest: dict[str, object]) -> WindowConfig:
    config = f_run_manifest.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Stage H F manifest is missing config.")
    return WindowConfig(
        duration_ms=int(config["window_duration_ms"]),
        stride_ms=int(config["window_stride_ms"]),
    )


def _resolve_scope_overrides(
    f_run_manifest: dict[str, object],
) -> dict[str, tuple[datetime, datetime]]:
    config = f_run_manifest.get("config", {})
    if not isinstance(config, dict):
        return {}
    export_scope_overrides = config.get("export_scope_overrides_utc", {})
    if not isinstance(export_scope_overrides, dict):
        return {}
    return {
        str(sortie_id): (
            datetime.fromisoformat(str(bounds[0]).replace("Z", "+00:00")),
            datetime.fromisoformat(str(bounds[1]).replace("Z", "+00:00")),
        )
        for sortie_id, bounds in export_scope_overrides.items()
    }


def _build_proxy_multitask_samples(records) -> tuple[E0ExperimentSample, ...]:
    physiology_features = _collect_feature_names(records["raw_physiology_stats"])
    vehicle_features = _collect_feature_names(records["raw_vehicle_stats"])
    samples: list[E0ExperimentSample] = []
    offsets_ms = (0, 1000, 2000, 3000)
    for row in records.sort_values(["start_offset_ms", "sample_id"]).itertuples(index=False):
        physiology_values = _stats_to_sequence(
            row.raw_physiology_stats,
            feature_names=physiology_features,
        )
        vehicle_values = _stats_to_sequence(
            row.raw_vehicle_stats,
            feature_names=vehicle_features,
        )
        physiology = NumericStreamMatrix(
            stream_kind=StreamKind.PHYSIOLOGY,
            point_count=len(offsets_ms),
            feature_names=physiology_features,
            point_offsets_ms=offsets_ms,
            point_measurements=("stage_h_proxy",) * len(offsets_ms),
            values=physiology_values,
            dropped_fields=(),
        )
        vehicle = NumericStreamMatrix(
            stream_kind=StreamKind.VEHICLE,
            point_count=len(offsets_ms),
            feature_names=vehicle_features,
            point_offsets_ms=offsets_ms,
            point_measurements=("stage_h_proxy",) * len(offsets_ms),
            values=vehicle_values,
            dropped_fields=(),
        )
        samples.append(
            E0ExperimentSample(
                sample_id=str(row.sample_id),
                sortie_id=str(row.sortie_id),
                start_offset_ms=int(row.start_offset_ms),
                end_offset_ms=int(row.end_offset_ms),
                physiology=physiology,
                vehicle=vehicle,
            )
        )
    return tuple(samples)


def _collect_feature_names(series) -> tuple[str, ...]:
    feature_names: set[str] = set()
    for stats in series:
        features = stats.get("features", {}) if isinstance(stats, dict) else {}
        feature_names.update(str(name) for name in features)
    return tuple(sorted(feature_names))


def _stats_to_sequence(
    stats: dict[str, object],
    *,
    feature_names: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    feature_payload = stats.get("features", {}) if isinstance(stats, dict) else {}
    rows: list[list[float]] = [[] for _ in range(4)]
    for feature_name in feature_names:
        values = feature_payload.get(feature_name, {}) if isinstance(feature_payload, dict) else {}
        start = _coerce_float(values.get("start"), default=values.get("mean"))
        mean = _coerce_float(values.get("mean"), default=start)
        end = _coerce_float(values.get("end"), default=mean)
        delta = _coerce_float(values.get("delta"), default=end - start)
        rows[0].append(start)
        rows[1].append(mean)
        rows[2].append(end)
        rows[3].append(mean + delta)
    return tuple(tuple(row) for row in rows)


def _coerce_float(value: object, *, default: object = 0.0) -> float:
    if value is None:
        value = default
    if value is None:
        value = 0.0
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
