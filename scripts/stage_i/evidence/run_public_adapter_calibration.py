"""Build Stage I public adapter calibration summary from existing public outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i.evidence.public_adapter_calibration import (  # noqa: E402
    DEFAULT_NASA_SUMMARY_PATHS,
    DEFAULT_UAB_SKLEARN_SUMMARY_PATHS,
    DEFAULT_UAB_TORCH_SUMMARY_PATHS,
    StageIPublicAdapterCalibrationConfig,
    resolve_git_commit,
    run_stage_i_public_adapter_calibration,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-public-adapter-calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--output-root", default="docs/artifacts/assets/stage_i_public_adapter_calibration")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--uab-sklearn-summary-path", action="append", dest="uab_sklearn_summary_paths", default=[])
    parser.add_argument("--nasa-summary-path", action="append", dest="nasa_summary_paths", default=[])
    parser.add_argument("--uab-torch-summary-path", action="append", dest="uab_torch_summary_paths", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_stage_i_public_adapter_calibration(
        StageIPublicAdapterCalibrationConfig(
            run_id=args.run_id,
            output_root=args.output_root,
            report_root=args.report_root,
            uab_sklearn_summary_paths=tuple(args.uab_sklearn_summary_paths)
            or DEFAULT_UAB_SKLEARN_SUMMARY_PATHS,
            nasa_summary_paths=tuple(args.nasa_summary_paths)
            or DEFAULT_NASA_SUMMARY_PATHS,
            uab_torch_summary_paths=tuple(args.uab_torch_summary_paths)
            or DEFAULT_UAB_TORCH_SUMMARY_PATHS,
            git_commit=resolve_git_commit(cwd=REPO_ROOT),
        )
    )
    print(
        json.dumps(
            {
                "artifact_root": result.artifact_root,
                "summary_path": result.summary_path,
                "table_path": result.table_path,
                "report_path": result.report_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
