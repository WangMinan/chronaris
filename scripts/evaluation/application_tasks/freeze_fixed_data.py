"""Freeze the existing two-sortie Dingxin raw inputs to a local ignored snapshot."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
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
)
from chronaris.evaluation.application_tasks import (  # noqa: E402
    DEFAULT_SNAPSHOT_RUN_ID,
    FixedDataSnapshotConfig,
    InfluxSnapshotPointSource,
    run_fixed_data_snapshot,
)
from chronaris.evaluation.application_tasks.fixed_data_snapshot_reporting import (  # noqa: E402
    write_snapshot_unavailable,
)
from chronaris.feature_export.profile import StageHProfileResolver  # noqa: E402
from chronaris.modeling.common.run_observer import configure_task_eval_cli_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_SNAPSHOT_RUN_ID)
    parser.add_argument("--snapshot-output-root", default="artifacts/application_evaluation")
    parser.add_argument("--compact-output-root", default="docs/artifacts/runs")
    parser.add_argument(
        "--e-run-manifest",
        default="docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json",
    )
    parser.add_argument(
        "--f-run-manifest",
        default="docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json",
    )
    parser.add_argument(
        "--label-field-manifest",
        default="docs/artifacts/runs/2026-07-10_fixed-data-audit/label_field_manifest.json",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--mysql-host", default=os.environ.get("CHRONARIS_MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--mysql-port", type=int, default=int(os.environ.get("CHRONARIS_MYSQL_PORT", "3306")))
    parser.add_argument("--mysql-database", default=os.environ.get("CHRONARIS_MYSQL_DATABASE", "rjgx_backend"))
    parser.add_argument("--mysql-user", default=os.environ.get("CHRONARIS_MYSQL_USER"))
    parser.add_argument("--mysql-password", default=os.environ.get("CHRONARIS_MYSQL_PASSWORD"))
    parser.add_argument("--mysql-binary", default="mysql")
    parser.add_argument("--influx-url", default=os.environ.get("CHRONARIS_INFLUX_URL", "http://127.0.0.1:8086"))
    parser.add_argument("--influx-org", default=os.environ.get("CHRONARIS_INFLUX_ORG"))
    parser.add_argument("--influx-token", default=os.environ.get("CHRONARIS_INFLUX_TOKEN"))
    parser.add_argument("--influx-binary", default="influx")
    parser.add_argument("--secrets-path", default=str(REPO_ROOT / "docs/SECRETS.md"))
    return parser.parse_args()


def main() -> int:
    configure_task_eval_cli_logging(sys.stderr)
    args = parse_args()
    config = FixedDataSnapshotConfig(
        run_id=args.run_id,
        snapshot_output_root=args.snapshot_output_root,
        compact_output_root=args.compact_output_root,
        e_run_manifest_path=args.e_run_manifest,
        f_run_manifest_path=args.f_run_manifest,
        label_field_manifest_path=args.label_field_manifest,
        resume=args.resume,
    )
    try:
        mysql_runner, influx_runner = _build_runners(args)
        profile_resolver = StageHProfileResolver(
            flight_task_reader=MySQLFlightTaskReader(mysql_runner),
            collect_task_reader=MySQLCollectTaskReader(mysql_runner),
            storage_analysis_reader=MySQLStorageAnalysisReader(mysql_runner),
            distinct_measurement_reader=InfluxDistinctMeasurementReader(influx_runner),
        )
        result = run_fixed_data_snapshot(
            config,
            profile_resolver=profile_resolver,
            point_source=InfluxSnapshotPointSource(influx_runner),
        )
    except Exception as exc:
        unavailable_path = write_snapshot_unavailable(config=config, error=exc)
        print(
            json.dumps(
                {
                    "run_id": config.run_id,
                    "status": "unavailable",
                    "unavailable_path": unavailable_path,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 1


def _build_runners(args: argparse.Namespace) -> tuple[MySQLCliRunner, InfluxCliRunner]:
    secrets_text = Path(args.secrets_path).read_text(encoding="utf-8")
    mysql_user = args.mysql_user or _extract_secret(secrets_text, "username")
    mysql_password = args.mysql_password or _extract_secret(secrets_text, "password")
    influx_org = args.influx_org or _extract_secret(secrets_text, "influxdb.org")
    influx_token = args.influx_token or _extract_secret(secrets_text, "influxdb.token")
    mysql_runner = MySQLCliRunner(
        MySQLSettings(
            host=args.mysql_host,
            port=args.mysql_port,
            database=args.mysql_database,
            user=mysql_user,
            password_env=None,
            password_value=mysql_password,
        ),
        mysql_binary=args.mysql_binary,
    )
    influx_runner = InfluxCliRunner(
        InfluxSettings(
            url=args.influx_url,
            org=influx_org,
            token_env=None,
            token_value=influx_token,
        ),
        influx_binary=args.influx_binary,
    )
    return mysql_runner, influx_runner


def _extract_secret(md_text: str, key: str) -> str:
    matched = re.search(rf"^\+?\s*{re.escape(key)}:\s*(.+)$", md_text, re.MULTILINE)
    if matched is None:
        raise RuntimeError(f"missing required secret key: {key}")
    return matched.group(1).strip()


if __name__ == "__main__":
    raise SystemExit(main())
