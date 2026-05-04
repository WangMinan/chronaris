"""Run the minimal Stage I public-opt UAB subjective regression path."""

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
    StageIPublicOptConfig,
    run_stage_i_public_opt,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-public-opt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--prepared-artifact-root", required=True)
    parser.add_argument("--artifact-root", default="docs/reports/assets/stage_i_public_opt")
    parser.add_argument("--report-root", default="docs/reports")
    parser.add_argument("--dataset-id", default="uab_workload_dataset")
    parser.add_argument("--profile", default="window_v2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_stage_i_public_opt(
        StageIPublicOptConfig(
            run_id=args.run_id,
            prepared_artifact_root=str(REPO_ROOT / args.prepared_artifact_root),
            artifact_root=str(REPO_ROOT / args.artifact_root),
            report_root=str(REPO_ROOT / args.report_root),
            dataset_id=args.dataset_id,
            profile=args.profile,
            seed=args.seed,
        )
    )
    print(
        json.dumps(
            {
                "public_opt_feature_frame_path": result.feature_frame_path,
                "public_opt_predictions_path": result.predictions_path,
                "public_opt_summary_path": result.summary_path,
                "public_opt_report_path": result.report_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
