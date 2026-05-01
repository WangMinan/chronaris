"""Run the Stage I Phase 2 case study over frozen Stage H assets."""

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

from chronaris.pipelines.stage_i_case_study import (
    StageICaseStudyConfig,
    run_stage_i_case_study,
    write_stage_i_case_study_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-h-run-manifest",
        default="docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/run_manifest.json",
    )
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--output-root", default="docs/reports/assets/stage_i")
    parser.add_argument(
        "--report-path",
        default=f"docs/reports/stage-i-case-study-phase2-{datetime.now().date().isoformat()}.md",
    )
    parser.add_argument("--top-k-windows", type=int, default=5)
    return parser.parse_args()


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-phase2-case-study")


def main() -> int:
    args = parse_args()
    config = StageICaseStudyConfig(
        run_id=args.run_id,
        stage_h_run_manifest_path=args.stage_h_run_manifest,
        output_root=args.output_root,
        report_path=args.report_path,
        top_k_windows=args.top_k_windows,
    )
    result = run_stage_i_case_study(config)
    report_path = write_stage_i_case_study_report(result)
    print(
        json.dumps(
            {
                "artifact_root": result.artifact_root,
                "report_path": report_path,
                "view_count": len(result.view_results),
                "warn_view_ids": [
                    item.view_summary.view_id
                    for item in result.view_results
                    if item.view_summary.verdict != "PASS"
                ],
                "pilot_comparison_count": len(result.pilot_comparisons),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
