"""Run the minimal thesis-facing Stage I runtime/demo entry."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.serving import (  # noqa: E402
    StageIRuntimeDemoConfig,
    run_stage_i_runtime_demo,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-runtime-demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", required=True)
    parser.add_argument(
        "--source-type",
        choices=("auto", "stage_h_run_manifest", "optimized_candidate_package"),
        default="auto",
    )
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--artifact-root", default="docs/reports/assets/stage_i_runtime_demo")
    parser.add_argument("--report-root", default="docs/reports/stage_i")
    parser.add_argument("--no-window-csv", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    result = run_stage_i_runtime_demo(
        StageIRuntimeDemoConfig(
            run_id=args.run_id,
            source_path=_resolve_path(args.source_path),
            artifact_root=_resolve_path(args.artifact_root),
            report_root=_resolve_path(args.report_root),
            source_type=args.source_type,
            export_window_csv=not args.no_window_csv,
        )
    )
    print(
        json.dumps(
            {
                "runtime_demo_summary_path": result.summary_path,
                "runtime_demo_report_path": result.report_path,
                "runtime_demo_window_csv_path": result.window_csv_path,
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
