"""Run GPU-preferred chronaris_public_fusion screening over prepared public sequence assets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines import (  # noqa: E402
    run_stage_i_public_fusion_screen,
    StageIPublicFusionScreenConfig,
)
from chronaris.pipelines.stage_i.common.run_observer import (  # noqa: E402
    configure_stage_i_cli_logging,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-public-fusion-screen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--artifact-root", default="docs/artifacts/assets/stage_i_public_fusion_screen")
    parser.add_argument("--report-root", default="docs/artifacts")
    parser.add_argument("--uab-root", default=None)
    parser.add_argument("--nasa-root", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-cpu-debug",
        action="store_true",
        help="allow CPU fallback for debugging only; paper-facing runs require CUDA",
    )
    parser.add_argument(
        "--train-sampling-policy",
        choices=("none", "balanced_class"),
        default="none",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    configure_stage_i_cli_logging(sys.stdout)
    args = parse_args()
    dataset_prepared_roots = {}
    if args.uab_root:
        dataset_prepared_roots["uab_workload_dataset"] = _resolve_path(args.uab_root)
    if args.nasa_root:
        dataset_prepared_roots["nasa_csm"] = _resolve_path(args.nasa_root)
    if not dataset_prepared_roots:
        raise ValueError("at least one of --uab-root or --nasa-root is required")

    result = run_stage_i_public_fusion_screen(
        StageIPublicFusionScreenConfig(
            run_id=args.run_id,
            dataset_prepared_roots=dataset_prepared_roots,
            artifact_root=_resolve_path(args.artifact_root),
            report_root=_resolve_path(args.report_root),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_folds=args.max_folds,
            seed=args.seed,
            device=args.device,
            require_cuda=not args.allow_cpu_debug,
            train_sampling_policy=args.train_sampling_policy,
        )
    )
    print(
        json.dumps(
            {
                "fusion_screen_summary_path": result.summary_path,
                "fusion_screen_leaderboard_csv_path": result.leaderboard_csv_path,
                "fusion_screen_report_path": result.report_path,
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
