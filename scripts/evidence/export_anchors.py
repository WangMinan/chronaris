"""Export thesis-facing task evaluation key-condition anchors from frozen feature export assets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evidence.anchors import (  # noqa: E402
    StageIAnchorConfig,
    run_task_eval_anchor,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-task-eval-anchor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-export-run-manifest",
        default="docs/artifacts/runs/2026-04-27_feature-export-closure/run_manifest.json",
    )
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--output-root", default="docs/artifacts/runs")
    parser.add_argument("--report-root", default="docs/artifacts/runs")
    parser.add_argument(
        "--private-benchmark-summary-path",
        default=(
            "docs/artifacts/runs/"
            "20260607T-task-eval-private-opt-package-r2/private_benchmark_summary.json"
        ),
    )
    parser.add_argument("--top-k-windows", type=int, default=5)
    parser.add_argument(
        "--view-verdict-filter",
        choices=("all", "warn_only", "pass_only"),
        default="all",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    result = run_task_eval_anchor(
        StageIAnchorConfig(
            run_id=args.run_id,
            feature_export_run_manifest_path=_resolve_path(args.feature_export_run_manifest),
            output_root=_resolve_path(args.output_root),
            report_root=_resolve_path(args.report_root),
            private_benchmark_summary_path=_resolve_path(
                args.private_benchmark_summary_path
            ),
            top_k_windows=args.top_k_windows,
            view_verdict_filter=args.view_verdict_filter,
            device=args.device,
        )
    )
    print(
        json.dumps(
            {
                "anchor_manifest_path": result.anchor_manifest_path,
                "anchor_windows_csv_path": result.anchor_windows_csv_path,
                "anchor_report_path": result.report_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _resolve_path(path_like: str) -> str:
    path = Path(path_like)
    return str(path if path.is_absolute() else (REPO_ROOT / path))


if __name__ == "__main__":
    raise SystemExit(main())
