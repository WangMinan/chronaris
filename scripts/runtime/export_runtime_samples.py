#!/usr/bin/env python3
"""Export task evaluation runtime replay samples from a real feature export run manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

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
    MySQLStorageAnalysisReader,
    StageHProfileResolver,
)
from chronaris.modeling.training.backbone_train import collect_task_eval_multitask_samples  # noqa: E402
from chronaris.schema.models import WindowConfig  # noqa: E402
from chronaris.serving.runtime_inference import dump_runtime_samples_jsonl  # noqa: E402


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-task-eval-runtime-samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--output-root", default="docs/artifacts/runs")
    parser.add_argument("--mysql-database", default="rjgx_backend")
    parser.add_argument("--mysql-binary", default="mysql")
    parser.add_argument("--influx-binary", default="influx")
    parser.add_argument("--limit-samples", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_manifest = _load_json(args.run_manifest)
    influx_settings = resolve_influx_settings(REPO_ROOT / "docs/SECRETS.md", default_url="http://127.0.0.1:8086")
    mysql_settings = resolve_mysql_settings(args.mysql_database, REPO_ROOT / "docs/SECRETS.md", default_host="127.0.0.1", default_port=3306)
    mysql_runner = MySQLCliRunner(mysql_settings, mysql_binary=args.mysql_binary)
    influx_runner = InfluxCliRunner(influx_settings, influx_binary=args.influx_binary)

    profile_resolver = StageHProfileResolver(
        flight_task_reader=MySQLFlightTaskReader(mysql_runner),
        collect_task_reader=MySQLCollectTaskReader(mysql_runner),
        storage_analysis_reader=MySQLStorageAnalysisReader(mysql_runner),
        distinct_measurement_reader=InfluxDistinctMeasurementReader(influx_runner),
    )
    sortie_ids = tuple(str(sortie_id) for sortie_id in run_manifest["sortie_ids"])
    profiles = profile_resolver.resolve_many(sortie_ids)
    samples, source_summary = collect_task_eval_multitask_samples(
        profiles=profiles,
        window_config=_resolve_window_config(run_manifest),
        export_scope_overrides_utc=_resolve_scope_overrides(run_manifest),
        runner=influx_runner,
    )
    if args.limit_samples is not None:
        if args.limit_samples <= 0:
            raise ValueError("limit-samples must be positive when provided.")
        samples = samples[: args.limit_samples]
    run_root = Path(args.output_root) / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    sample_jsonl_path = run_root / "runtime_samples.jsonl"
    summary_path = run_root / "runtime_sample_export_summary.json"
    dump_runtime_samples_jsonl(samples, path=sample_jsonl_path)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": args.run_id,
        "run_manifest_path": str(Path(args.run_manifest)),
        "sample_jsonl_path": str(sample_jsonl_path),
        "sample_count": len(samples),
        "sample_id_mode": "view_prefixed",
        "source_summary": source_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sample_jsonl_path": str(sample_jsonl_path),
                "summary_path": str(summary_path),
                "sample_count": len(samples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _load_json(path_like: str) -> dict[str, object]:
    return json.loads(Path(path_like).read_text(encoding="utf-8"))


def _resolve_window_config(run_manifest: Mapping[str, object]) -> WindowConfig:
    config = run_manifest.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("run_manifest.config must be a mapping.")
    return WindowConfig(
        duration_ms=int(config.get("window_duration_ms", 5_000)),
        stride_ms=int(config.get("window_stride_ms", 5_000)),
    )


def _resolve_scope_overrides(
    run_manifest: Mapping[str, object],
) -> dict[str, tuple[datetime, datetime]]:
    config = run_manifest.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("run_manifest.config must be a mapping.")
    raw_overrides = config.get("export_scope_overrides_utc", {})
    if not isinstance(raw_overrides, Mapping):
        raise ValueError("export_scope_overrides_utc must be a mapping when provided.")
    resolved: dict[str, tuple[datetime, datetime]] = {}
    for sortie_id, bounds in raw_overrides.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"invalid export scope override for sortie {sortie_id}")
        resolved[str(sortie_id)] = (
            datetime.fromisoformat(str(bounds[0]).replace("Z", "+00:00")),
            datetime.fromisoformat(str(bounds[1]).replace("Z", "+00:00")),
        )
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
