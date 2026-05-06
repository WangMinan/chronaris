"""Aggregate Stage I support evidence into alignment and causal reports."""

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
    StageISupportConfig,
    run_stage_i_support,
)
from chronaris.pipelines.stage_i.stage_i_support import (  # noqa: E402
    DEFAULT_ALIGNMENT_E_SUMMARY_PATH,
    DEFAULT_ALIGNMENT_F_SUMMARY_PATH,
    DEFAULT_CAUSAL_G_SUMMARY_PATH,
    DEFAULT_CASE_STUDY_ABLATION_CSV_PATH,
    DEFAULT_CASE_STUDY_SUMMARY_PATH,
    DEFAULT_DEEP_COMPARISON_SUMMARY_PATH,
    DEFAULT_PRIVATE_BENCHMARK_SUMMARY_PATH,
    DEFAULT_STAGE_H_RUN_MANIFEST_PATH,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-support")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--artifact-root", default="docs/reports/assets/stage_i_support")
    parser.add_argument("--report-root", default="docs/reports")
    parser.add_argument("--alignment-e-summary-path", default=DEFAULT_ALIGNMENT_E_SUMMARY_PATH)
    parser.add_argument("--alignment-f-summary-path", default=DEFAULT_ALIGNMENT_F_SUMMARY_PATH)
    parser.add_argument("--causal-g-summary-path", default=DEFAULT_CAUSAL_G_SUMMARY_PATH)
    parser.add_argument("--stage-h-run-manifest-path", default=DEFAULT_STAGE_H_RUN_MANIFEST_PATH)
    parser.add_argument("--case-study-summary-path", default=DEFAULT_CASE_STUDY_SUMMARY_PATH)
    parser.add_argument("--case-study-ablation-csv-path", default=DEFAULT_CASE_STUDY_ABLATION_CSV_PATH)
    parser.add_argument("--private-benchmark-summary-path", default=DEFAULT_PRIVATE_BENCHMARK_SUMMARY_PATH)
    parser.add_argument("--deep-comparison-summary-path", default=DEFAULT_DEEP_COMPARISON_SUMMARY_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_stage_i_support(
        StageISupportConfig(
            run_id=args.run_id,
            artifact_root=str(REPO_ROOT / args.artifact_root),
            report_root=str(REPO_ROOT / args.report_root),
            alignment_e_summary_path=str(REPO_ROOT / args.alignment_e_summary_path),
            alignment_f_summary_path=str(REPO_ROOT / args.alignment_f_summary_path),
            causal_g_summary_path=str(REPO_ROOT / args.causal_g_summary_path),
            stage_h_run_manifest_path=str(REPO_ROOT / args.stage_h_run_manifest_path),
            case_study_summary_path=str(REPO_ROOT / args.case_study_summary_path),
            case_study_ablation_csv_path=str(REPO_ROOT / args.case_study_ablation_csv_path),
            private_benchmark_summary_path=str(REPO_ROOT / args.private_benchmark_summary_path),
            deep_comparison_summary_path=(
                str(REPO_ROOT / args.deep_comparison_summary_path)
                if args.deep_comparison_summary_path
                else None
            ),
        )
    )
    print(
        json.dumps(
            {
                "support_summary_path": result.summary_path,
                "support_matrix_path": result.matrix_path,
                "main_ablation_matrix_path": result.main_matrix_path,
                "alignment_report_path": result.alignment_report_path,
                "causal_report_path": result.causal_report_path,
                "ablation_report_path": result.ablation_report_path,
                "overview_plot_path": result.overview_plot_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
