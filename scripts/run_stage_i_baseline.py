"""Run Stage I baseline suites from prepared artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i_baseline import (
    render_stage_i_baseline_report,
    run_stage_i_baselines,
    write_baseline_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--dataset", choices=("uab", "nasa_csm"), default=None)
    parser.add_argument("--profile", choices=("session_v1", "window_v2"), default=None)
    parser.add_argument("--task-family", choices=("workload", "attention_state"), default=None)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = REPO_ROOT / args.artifact_root
    feature_table = _read_parquet(artifact_root / "feature_table.parquet")
    dataset_summary = json.loads((artifact_root / "dataset_summary.json").read_text(encoding="utf-8"))

    dataset_id = _resolve_dataset_id(args.dataset, str(dataset_summary["dataset_id"]))
    profile = args.profile or _infer_profile(feature_table)
    task_family = args.task_family or _infer_task_family(feature_table)

    artifacts = run_stage_i_baselines(
        feature_table,
        dataset_id=dataset_id,
        profile=profile,
        task_family=task_family,
        artifact_root=artifact_root,
    )
    write_baseline_artifacts(artifacts, artifact_root=artifact_root)

    report_path = REPO_ROOT / (
        args.report_path
        or f"docs/reports/stage-i-{args.dataset or dataset_id}-{profile}-{datetime.now().date().isoformat()}.md"
    )
    report_markdown = render_stage_i_baseline_report(
        artifact_root=artifact_root,
        dataset_summary=dataset_summary,
        objective_metrics=artifacts.objective_metrics,
        subjective_metrics=artifacts.subjective_metrics,
        plot_paths=artifacts.plot_paths,
    )
    (artifact_root / "baseline_report.md").write_text(report_markdown + "\n", encoding="utf-8")
    report_path.write_text(report_markdown + "\n", encoding="utf-8")

    payload = {
        "artifact_root": str(artifact_root),
        "baseline_report_path": str(artifact_root / "baseline_report.md"),
        "report_path": str(report_path),
        "dataset_id": dataset_id,
        "profile": profile,
        "task_family": task_family,
    }
    if artifacts.objective_metrics is not None:
        payload["objective_best_models"] = {
            group_name: value["best_model_name"]
            for group_name, value in artifacts.objective_metrics["primary_results"].items()
        }
    if artifacts.subjective_metrics is not None:
        payload["subjective_best_models"] = {
            group_name: value["best_model_name"]
            for group_name, value in artifacts.subjective_metrics["primary_results"].items()
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _resolve_dataset_id(cli_value: str | None, summary_dataset_id: str) -> str:
    if cli_value == "uab":
        return "uab_workload_dataset"
    if cli_value == "nasa_csm":
        return "nasa_csm"
    return summary_dataset_id


def _infer_profile(feature_table):
    granularities = sorted(set(feature_table["sample_granularity"].dropna().astype(str)))
    return "session_v1" if granularities == ["session"] else "window_v2"


def _infer_task_family(feature_table):
    families = sorted(set(feature_table["task_family"].dropna().astype(str)))
    if len(families) != 1:
        raise ValueError(f"cannot infer unique task_family from feature table: {families}")
    return families[0]


def _read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


if __name__ == "__main__":
    raise SystemExit(main())
