"""Build the task evaluation midterm evidence pack from verified task evaluation artifacts."""

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

from chronaris.evidence.midterm_pack import (  # noqa: E402
    StageIMidtermEvidenceConfig,
    run_task_eval_midterm_evidence,
)
from chronaris.evidence.midterm_pack import (  # noqa: E402
    DEFAULT_ANCHOR_MANIFEST_PATH,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DEEP_COMPARISON_SUMMARY_PATH,
    DEFAULT_NASA_PUBLIC_OPT_SUMMARY_PATH,
    DEFAULT_PRIVATE_SUMMARY_PATH,
    DEFAULT_PUBLIC_FUSION_SCREEN_SUMMARY_PATH,
    DEFAULT_PUBLIC_MAINLINE_SUMMARY_PATH,
    DEFAULT_REPORT_ROOT,
    DEFAULT_SUPPORT_SUMMARY_PATH,
    DEFAULT_UAB_PUBLIC_OPT_SUMMARY_PATH,
)
from chronaris.modeling.common.run_observer import (  # noqa: E402
    configure_task_eval_cli_logging,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-task-eval-midterm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--workspace-root", default=str(REPO_ROOT))
    parser.add_argument("--private-summary-path", default=DEFAULT_PRIVATE_SUMMARY_PATH)
    parser.add_argument("--support-summary-path", default=DEFAULT_SUPPORT_SUMMARY_PATH)
    parser.add_argument("--public-mainline-summary-path", default=DEFAULT_PUBLIC_MAINLINE_SUMMARY_PATH)
    parser.add_argument("--anchor-manifest-path", default=DEFAULT_ANCHOR_MANIFEST_PATH)
    parser.add_argument("--deep-comparison-summary-path", default=DEFAULT_DEEP_COMPARISON_SUMMARY_PATH)
    parser.add_argument("--public-fusion-screen-summary-path", default=DEFAULT_PUBLIC_FUSION_SCREEN_SUMMARY_PATH)
    parser.add_argument("--nasa-public-opt-summary-path", default=DEFAULT_NASA_PUBLIC_OPT_SUMMARY_PATH)
    parser.add_argument("--uab-public-opt-summary-path", default=DEFAULT_UAB_PUBLIC_OPT_SUMMARY_PATH)
    parser.add_argument("--uab-fairness-summary-path")
    parser.add_argument("--nasa-fusion-confirm-summary-path")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    configure_task_eval_cli_logging(sys.stdout)
    args = parse_args()
    result = run_task_eval_midterm_evidence(
        StageIMidtermEvidenceConfig(
            run_id=args.run_id,
            artifact_root=_resolve_path(args.artifact_root),
            report_root=_resolve_path(args.report_root),
            workspace_root=_resolve_path(args.workspace_root),
            private_summary_path=_resolve_path(args.private_summary_path),
            support_summary_path=_resolve_path(args.support_summary_path),
            public_mainline_summary_path=_resolve_path(args.public_mainline_summary_path),
            anchor_manifest_path=_resolve_path(args.anchor_manifest_path),
            deep_comparison_summary_path=_resolve_path(args.deep_comparison_summary_path),
            public_fusion_screen_summary_path=_resolve_path(args.public_fusion_screen_summary_path),
            nasa_public_opt_summary_path=_resolve_path(args.nasa_public_opt_summary_path),
            uab_public_opt_summary_path=_resolve_path(args.uab_public_opt_summary_path),
            uab_fairness_summary_path=(
                _resolve_path(args.uab_fairness_summary_path)
                if args.uab_fairness_summary_path
                else None
            ),
            nasa_fusion_confirm_summary_path=(
                _resolve_path(args.nasa_fusion_confirm_summary_path)
                if args.nasa_fusion_confirm_summary_path
                else None
            ),
        )
    )
    print(
        json.dumps(
            {
                "midterm_manifest_path": result.manifest_path,
                "midterm_metrics_path": result.metrics_path,
                "midterm_figure_index_path": result.figure_index_path,
                "cleanup_audit_path": result.cleanup_audit_path,
                "midterm_report_path": result.report_path,
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
