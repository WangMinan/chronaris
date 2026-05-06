"""Build the unified Stage I public-mainline report from existing artifacts."""

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

from chronaris.pipelines import (  # noqa: E402
    StageIPublicMainlineReportConfig,
    run_stage_i_public_mainline_report,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-public-mainline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--uab-summary", required=True)
    parser.add_argument("--nasa-summary", required=True)
    parser.add_argument("--deep-comparison-summary", required=True)
    parser.add_argument("--artifact-root", default="docs/reports/assets/stage_i_public_mainline")
    parser.add_argument("--report-root", default="docs/reports")
    parser.add_argument("--public-fusion-screen-summary")
    parser.add_argument("--public-fusion-nasa-confirm-summary")
    parser.add_argument("--public-fusion-uab-confirm-summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_stage_i_public_mainline_report(
        StageIPublicMainlineReportConfig(
            run_id=args.run_id,
            uab_summary_path=_resolve_path(args.uab_summary),
            nasa_summary_path=_resolve_path(args.nasa_summary),
            deep_comparison_summary_path=_resolve_path(args.deep_comparison_summary),
            artifact_root=_resolve_path(args.artifact_root),
            report_root=_resolve_path(args.report_root),
            public_fusion_screen_summary_path=(
                _resolve_path(args.public_fusion_screen_summary)
                if args.public_fusion_screen_summary
                else None
            ),
            public_fusion_nasa_confirm_summary_path=(
                _resolve_path(args.public_fusion_nasa_confirm_summary)
                if args.public_fusion_nasa_confirm_summary
                else None
            ),
            public_fusion_uab_confirm_summary_path=(
                _resolve_path(args.public_fusion_uab_confirm_summary)
                if args.public_fusion_uab_confirm_summary
                else None
            ),
        )
    )
    print(
        json.dumps(
            {
                "public_mainline_summary_path": result.summary_path,
                "public_mainline_report_path": result.report_path,
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
