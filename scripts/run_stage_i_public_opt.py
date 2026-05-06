"""Run one Stage I public-opt path over prepared UAB or NASA assets."""

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
    parser.add_argument(
        "--feature-profile",
        default="full",
        choices=("full", "physiology_only", "context_only", "residual_only"),
    )
    parser.add_argument(
        "--head-catalog",
        default="expanded",
        choices=("minimal", "expanded"),
    )
    parser.add_argument(
        "--train-balance-policy",
        default="class_weight_balanced",
        choices=("none", "class_weight_balanced"),
    )
    parser.add_argument(
        "--ensemble-policy",
        default="none",
        choices=("none", "mean_top2", "vote_top2"),
    )
    parser.add_argument(
        "--winner-margin-policy",
        default="paper_gate",
        choices=("paper_gate", "none"),
    )
    parser.add_argument("--reference-phase3-closure-summary")
    parser.add_argument("--reference-deep-comparison-summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_stage_i_public_opt(
        StageIPublicOptConfig(
            run_id=args.run_id,
            prepared_artifact_root=_resolve_path(args.prepared_artifact_root),
            artifact_root=_resolve_path(args.artifact_root),
            report_root=_resolve_path(args.report_root),
            dataset_id=args.dataset_id,
            profile=args.profile,
            seed=args.seed,
            feature_profile=args.feature_profile,
            head_catalog=args.head_catalog,
            train_balance_policy=args.train_balance_policy,
            ensemble_policy=args.ensemble_policy,
            winner_margin_policy=args.winner_margin_policy,
            reference_phase3_closure_summary_path=(
                _resolve_path(args.reference_phase3_closure_summary)
                if args.reference_phase3_closure_summary
                else None
            ),
            reference_deep_comparison_summary_path=(
                _resolve_path(args.reference_deep_comparison_summary)
                if args.reference_deep_comparison_summary
                else None
            ),
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


def _resolve_path(path_like: str) -> str:
    path = Path(path_like)
    return str(path if path.is_absolute() else (REPO_ROOT / path))


if __name__ == "__main__":
    raise SystemExit(main())
