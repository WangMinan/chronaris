"""Build Stage I public transfer-boundary report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i.stage_i_public_transfer_boundary import (  # noqa: E402
    DEFAULT_CALIBRATION_SUMMARY_PATH,
    DEFAULT_MULTITASK_SUMMARY_PATH,
    DEFAULT_PRIVATE_SUMMARY_PATH,
    DEFAULT_PUBLIC_MAINLINE_SUMMARY_PATH,
    StageIPublicTransferBoundaryConfig,
    run_stage_i_public_transfer_boundary,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-public-transfer-boundary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--calibration-summary-path", default=DEFAULT_CALIBRATION_SUMMARY_PATH)
    parser.add_argument("--public-mainline-summary-path", default=DEFAULT_PUBLIC_MAINLINE_SUMMARY_PATH)
    parser.add_argument("--multitask-summary-path", default=DEFAULT_MULTITASK_SUMMARY_PATH)
    parser.add_argument("--private-summary-path", default=DEFAULT_PRIVATE_SUMMARY_PATH)
    parser.add_argument("--output-root", default="docs/artifacts/assets/stage_i_public_transfer_boundary")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_stage_i_public_transfer_boundary(
        StageIPublicTransferBoundaryConfig(
            run_id=args.run_id,
            calibration_summary_path=args.calibration_summary_path,
            public_mainline_summary_path=args.public_mainline_summary_path,
            multitask_summary_path=args.multitask_summary_path,
            private_summary_path=args.private_summary_path,
            output_root=args.output_root,
            report_root=args.report_root,
        )
    )
    print(
        json.dumps(
            {
                "artifact_root": result.artifact_root,
                "summary_path": result.summary_path,
                "report_path": result.report_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
