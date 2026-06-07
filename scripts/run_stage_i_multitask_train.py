"""Run Stage I multitask joint training on real Stage H all-window assets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

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
from chronaris.pipelines import (  # noqa: E402
    StageIMultitaskTrainConfig,
    collect_stage_i_multitask_samples,
    run_stage_i_multitask_train,
)
from chronaris.pipelines.stage_i.stage_i_private_benchmark_data import (  # noqa: E402
    load_aligned_private_records,
    validate_private_stage_h_run_contract,
)
from chronaris.schema.models import WindowConfig  # noqa: E402


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-multitask")


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
    parser.add_argument("--output-root", default="docs/artifacts/assets/stage_i_multitask")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--mysql-database", default="rjgx_backend")
    parser.add_argument("--mysql-binary", default="mysql")
    parser.add_argument("--influx-binary", default="influx")
    parser.add_argument("--epoch-count", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--task-head-hidden-dim", type=int, default=32)
    parser.add_argument("--retrieval-embedding-dim", type=int, default=16)
    parser.add_argument("--task-loss-weight", type=float, default=1.0)
    parser.add_argument("--causal-weight", type=float, default=0.1)
    parser.add_argument("--causal-attention-temperature", type=float, default=1.0)
    parser.add_argument("--causal-event-bias-weight", type=float, default=0.25)
    parser.add_argument("--causal-lag-window-points", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    e_manifest = _load_json(args.e_run_manifest)
    f_manifest = _load_json(args.f_run_manifest)
    validate_private_stage_h_run_contract(e_manifest, stage_name="E")
    validate_private_stage_h_run_contract(f_manifest, stage_name="F")
    _validate_matched_stage_h_manifests(e_manifest, f_manifest)

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
    sortie_ids = tuple(str(sortie_id) for sortie_id in f_manifest["sortie_ids"])
    profiles = profile_resolver.resolve_many(sortie_ids)

    config = StageIMultitaskTrainConfig(
        run_id=args.run_id,
        output_root=args.output_root,
        task_head_hidden_dim=args.task_head_hidden_dim,
        retrieval_embedding_dim=args.retrieval_embedding_dim,
        task_loss_weight=args.task_loss_weight,
        causal_weight=args.causal_weight,
        causal_attention_temperature=args.causal_attention_temperature,
        causal_event_bias_weight=args.causal_event_bias_weight,
        causal_lag_window_points=args.causal_lag_window_points,
    )
    preview_config = replace(
        config.preview_config,
        device=args.device,
        epoch_count=args.epoch_count or config.preview_config.epoch_count,
        batch_size=args.batch_size or config.preview_config.batch_size,
        learning_rate=args.learning_rate or config.preview_config.learning_rate,
    )
    config = replace(config, preview_config=preview_config)

    samples, sample_source_summary = collect_stage_i_multitask_samples(
        profiles=profiles,
        window_config=_resolve_window_config(f_manifest),
        export_scope_overrides_utc=_resolve_scope_overrides(f_manifest),
        runner=influx_runner,
    )
    records = load_aligned_private_records(
        e_run_manifest_path=args.e_run_manifest,
        f_run_manifest_path=args.f_run_manifest,
    )
    task_payload = build_stage_i_real_task_payload(records)
    result = run_stage_i_multitask_train(
        config,
        samples=samples,
        task_entries=task_payload["entries"],
        source_summary={
            "source_type": "real_stage_h_multitask_train",
            "source_manifests": {
                "e_run_manifest_path": str(Path(args.e_run_manifest)),
                "f_run_manifest_path": str(Path(args.f_run_manifest)),
            },
            "sample_collection": sample_source_summary,
            "task_payload_summary": task_payload["summary"],
        },
    )
    report_path = _write_evidence_report(
        run_id=args.run_id,
        report_root=Path(args.report_root),
        summary=result.summary,
        summary_path=result.summary_path,
        task_payload_summary=task_payload["summary"],
        source_manifests={
            "e_run_manifest_path": str(Path(args.e_run_manifest)),
            "f_run_manifest_path": str(Path(args.f_run_manifest)),
        },
    )
    print(
        json.dumps(
            {
                "artifact_root": result.artifact_root,
                "checkpoint_path": result.checkpoint_path,
                "summary_path": result.summary_path,
                "task_manifest_path": result.task_manifest_path,
                "report_path": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _load_json(path_like: str) -> dict[str, object]:
    return json.loads(Path(path_like).read_text(encoding="utf-8"))


def _validate_matched_stage_h_manifests(
    e_manifest: Mapping[str, object],
    f_manifest: Mapping[str, object],
) -> None:
    if tuple(e_manifest.get("sortie_ids", ())) != tuple(f_manifest.get("sortie_ids", ())):
        raise ValueError("E/F Stage H manifests must use the same sortie_ids.")
    if tuple(e_manifest.get("generated_view_ids", ())) != tuple(f_manifest.get("generated_view_ids", ())):
        raise ValueError("E/F Stage H manifests must export the same generated_view_ids.")
    e_config = e_manifest.get("config", {})
    f_config = f_manifest.get("config", {})
    if not isinstance(e_config, Mapping) or not isinstance(f_config, Mapping):
        raise ValueError("E/F Stage H manifests must include config mappings.")
    fields_to_match = (
        "export_profile",
        "window_duration_ms",
        "window_stride_ms",
        "export_scope_overrides_utc",
        "physiology_point_limit_per_measurement",
        "vehicle_point_limit_per_measurement",
    )
    for field_name in fields_to_match:
        if e_config.get(field_name) != f_config.get(field_name):
            raise ValueError(f"E/F Stage H manifest config mismatch at {field_name}.")


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


def _write_evidence_report(
    *,
    run_id: str,
    report_root: Path,
    summary: Mapping[str, object],
    summary_path: str,
    task_payload_summary: Mapping[str, object],
    source_manifests: Mapping[str, str],
) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"thesis-weak-label-evidence-{run_id}.md"
    test_metrics = summary["test_metrics"]
    task_components = test_metrics.get("task_components", {})
    lines = [
        f"# Stage I Thesis Weak-Label Evidence - {run_id}",
        "",
        "- evidence_type: `thesis weak-label evidence`",
        "- interpretation: `mainline closure evidence`",
        "- boundary: 当前结果证明论文主线联合训练已接通，不等价于人工真值任务最优结果。",
        f"- checkpoint_path: `{summary['checkpoint_path']}`",
        f"- multitask_summary_path: `{summary_path}`",
        f"- thesis_task_manifest_path: `{summary['task_manifest_path']}`",
        "",
        "## Source Contract",
        "",
        f"- e_run_manifest_path: `{source_manifests['e_run_manifest_path']}`",
        f"- f_run_manifest_path: `{source_manifests['f_run_manifest_path']}`",
        "- sample_id contract: `view_id::raw_window_sample_id`",
        f"- benchmark_role: `{task_payload_summary['benchmark_role']}`",
        f"- task_role: `{task_payload_summary['task_role']}`",
        f"- thesis_task_boundary: `{task_payload_summary['thesis_task_boundary']}`",
        "",
        "## Task Coverage",
        "",
        "| task | total_count | valid_label_count |",
        "| --- | ---: | ---: |",
    ]
    coverage = task_payload_summary.get("coverage", {})
    task_counts = task_payload_summary.get("task_counts", {})
    for task_name in task_counts:
        task_coverage = coverage.get(task_name, {})
        lines.append(
            f"| `{task_name}` | {int(task_counts[task_name])} | "
            f"{int(task_coverage.get('valid_label_count', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Test Metrics",
            "",
            f"- sample_count: `{test_metrics['sample_count']}`",
            f"- reconstruction_total: `{test_metrics['reconstruction_total']:.6f}`",
            f"- alignment: `{test_metrics['alignment']:.6f}`",
            f"- physics_total: `{test_metrics['physics_total']:.6f}`",
            f"- causal_total: `{test_metrics['causal_total']:.6f}`",
            f"- task_total: `{test_metrics['task_total']:.6f}`",
            f"- total: `{test_metrics['total']:.6f}`",
            "",
            "## Task Components",
            "",
            "| task | weighted_loss |",
            "| --- | ---: |",
        ]
    )
    for task_name, value in task_components.items():
        lines.append(f"| `{task_name}` | {float(value):.6f} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


if __name__ == "__main__":
    raise SystemExit(main())
