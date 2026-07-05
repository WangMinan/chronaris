"""Train one reusable task evaluation alignment backbone from feature export view samples."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access import (
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
from chronaris.modeling.training.backbone_train import (
    StageIBackboneTrainConfig,
    collect_task_eval_backbone_samples,
    run_task_eval_backbone_train,
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
    if not (url and org and token):
        secrets_text = (REPO_ROOT / "docs" / "SECRETS.md").read_text(encoding="utf-8")
        url = url or _extract_secret(secrets_text, "influxdb.url")
        org = org or _extract_secret(secrets_text, "influxdb.org")
        token = token or _extract_secret(secrets_text, "influxdb.token")
    return InfluxSettings(url=url, org=org, token_env=None, token_value=token)


def _resolve_mysql_settings(database: str) -> MySQLSettings:
    host = os.environ.get("CHRONARIS_MYSQL_HOST")
    port = os.environ.get("CHRONARIS_MYSQL_PORT")
    user = os.environ.get("CHRONARIS_MYSQL_USER")
    password = os.environ.get("CHRONARIS_MYSQL_PASSWORD")
    if not (host and port and user and password):
        secrets_text = (REPO_ROOT / "docs" / "SECRETS.md").read_text(encoding="utf-8")
        host = host or _extract_secret(secrets_text, "host")
        port = port or _extract_secret(secrets_text, "port")
        user = user or _extract_secret(secrets_text, "username")
        password = password or _extract_secret(secrets_text, "password")
    return MySQLSettings(
        host=host,
        port=int(port),
        database=database,
        user=user,
        password_env=None,
        password_value=password,
    )


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-task-eval-backbone")


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
    parser.add_argument("--sortie-id", dest="sortie_ids", action="append")
    parser.add_argument("--mysql-database", default="rjgx_backend")
    parser.add_argument("--mysql-binary", default="mysql")
    parser.add_argument("--influx-binary", default="influx")
    parser.add_argument("--preview-point-limit", type=int)
    parser.add_argument("--physiology-point-limit", type=int)
    parser.add_argument("--vehicle-point-limit", type=int)
    parser.add_argument("--epoch-count", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--disable-physics-constraints", action="store_true")
    parser.add_argument("--use-full-clip-scope", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sortie_ids = tuple(args.sortie_ids or DEFAULT_SORTIES)
    influx_settings = _resolve_influx_settings()
    mysql_settings = _resolve_mysql_settings(args.mysql_database)
    mysql_runner = MySQLCliRunner(mysql_settings, mysql_binary=args.mysql_binary)
    influx_runner = InfluxCliRunner(influx_settings, influx_binary=args.influx_binary)

    profile_resolver = StageHProfileResolver(
        flight_task_reader=MySQLFlightTaskReader(mysql_runner),
        collect_task_reader=MySQLCollectTaskReader(mysql_runner),
        storage_analysis_reader=MySQLStorageAnalysisReader(mysql_runner),
        distinct_measurement_reader=InfluxDistinctMeasurementReader(influx_runner),
    )
    profiles = profile_resolver.resolve_many(sortie_ids)

    config = StageIBackboneTrainConfig(run_id=args.run_id, output_root=args.output_root)
    preview_config = config.preview_config
    preview_config = replace(
        preview_config,
        device=args.device,
        epoch_count=args.epoch_count or preview_config.epoch_count,
        batch_size=args.batch_size or preview_config.batch_size,
        learning_rate=args.learning_rate or preview_config.learning_rate,
        enable_physics_constraints=not args.disable_physics_constraints,
    )
    config = replace(config, preview_config=preview_config)

    physiology_limit = args.physiology_point_limit
    vehicle_limit = args.vehicle_point_limit
    if args.preview_point_limit is not None:
        physiology_limit = physiology_limit or args.preview_point_limit
        vehicle_limit = vehicle_limit or args.preview_point_limit

    samples, source_summary = collect_task_eval_backbone_samples(
        profiles=profiles,
        window_config=config.window_config,
        physiology_point_limit_per_measurement=physiology_limit,
        vehicle_point_limit_per_measurement=vehicle_limit,
        export_scope_overrides_utc=_build_default_scope_overrides(args.use_full_clip_scope),
        runner=influx_runner,
    )
    result = run_task_eval_backbone_train(
        config,
        samples,
        source_summary=source_summary,
    )
    payload = {
        "run_id": args.run_id,
        "artifact_root": result.artifact_root,
        "checkpoint_path": result.checkpoint_path,
        "summary_path": result.summary_path,
        "sample_count": result.summary["sample_count"],
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
