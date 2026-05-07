"""Run the GPU-first torch-native UAB regression branch for Stage I public opt."""

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

from chronaris.pipelines import (  # noqa: E402
    StageIPublicOptTorchUABConfig,
    run_stage_i_public_opt_torch_uab,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-public-opt-uab-torch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--prepared-artifact-root", required=True)
    parser.add_argument("--artifact-root", default="docs/reports/assets/stage_i_public_opt_torch")
    parser.add_argument("--report-root", default="docs/reports")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--screen-max-folds", type=int, default=2)
    parser.add_argument("--full-max-folds", type=int, default=None)
    parser.add_argument("--full-candidate-limit", type=int, default=2)
    parser.add_argument("--skip-full-loso", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-profile",
        action="append",
        dest="feature_profiles",
        choices=("full", "residual_only"),
        default=[],
    )
    parser.add_argument(
        "--learning-rate",
        action="append",
        dest="learning_rates",
        type=float,
        default=[],
    )
    parser.add_argument(
        "--weight-decay",
        action="append",
        dest="weight_decays",
        type=float,
        default=[],
    )
    parser.add_argument(
        "--ensemble-policy",
        choices=("none", "mean_top2"),
        default="none",
    )
    parser.add_argument("--reference-public-opt-summary")
    parser.add_argument("--reference-deep-comparison-summary")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    result = run_stage_i_public_opt_torch_uab(
        StageIPublicOptTorchUABConfig(
            run_id=args.run_id,
            prepared_artifact_root=_resolve_path(args.prepared_artifact_root),
            artifact_root=_resolve_path(args.artifact_root),
            report_root=_resolve_path(args.report_root),
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            screen_max_folds=args.screen_max_folds,
            full_max_folds=args.full_max_folds,
            run_full_loso=not args.skip_full_loso,
            full_candidate_limit=args.full_candidate_limit,
            ensemble_policy=args.ensemble_policy,
            seed=args.seed,
            feature_profiles=tuple(args.feature_profiles) or ("full", "residual_only"),
            learning_rates=tuple(args.learning_rates) or (1e-3, 3e-4),
            weight_decays=tuple(args.weight_decays) or (1e-4, 1e-3),
            reference_public_opt_summary_path=(
                _resolve_path(args.reference_public_opt_summary)
                if args.reference_public_opt_summary
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
                "public_opt_torch_feature_frame_path": result.feature_frame_path,
                "public_opt_torch_predictions_path": result.predictions_path,
                "public_opt_torch_summary_path": result.summary_path,
                "public_opt_torch_report_path": result.report_path,
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
