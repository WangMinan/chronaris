"""Run P28 chronaris_public_fusion refresh over public sequence datasets."""

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

from chronaris.pipelines.stage_i.common.run_observer import (  # noqa: E402
    configure_stage_i_cli_logging,
)
from chronaris.pipelines.stage_i.public.fusion_refresh import (  # noqa: E402
    StageIPublicFusionRefreshConfig,
    run_stage_i_public_fusion_refresh,
)
from chronaris.pipelines.stage_i.public.sequence_preparation import (  # noqa: E402
    StageISequencePreparationConfig,
    run_stage_i_sequence_preparation,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-public-fusion-refresh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument(
        "--artifact-root",
        default="docs/artifacts/assets/stage_i_public_fusion_refresh",
    )
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("nasa_csm", "uab_workload_dataset"),
        choices=("nasa_csm", "uab_workload_dataset"),
    )
    parser.add_argument("--nasa-root", default=None, help="prepared NASA sequence root")
    parser.add_argument("--uab-root", default=None, help="prepared UAB sequence root")
    parser.add_argument(
        "--prepare-sequences",
        action="store_true",
        help="prepare missing public sequence roots before running refresh",
    )
    parser.add_argument("--dataset-root", default="/home/wangminan/dataset/chronaris")
    parser.add_argument(
        "--prepared-root-base",
        default="/tmp/chronaris_stage_i_public_fusion_refresh",
    )
    parser.add_argument("--target-steps", type=int, default=64)
    parser.add_argument("--screen-epochs", type=int, default=5)
    parser.add_argument("--confirm-epochs", type=int, default=20)
    parser.add_argument("--screen-max-folds", type=int, default=2)
    parser.add_argument("--confirm-max-folds", default=None)
    parser.add_argument("--screen-candidate-limit", type=int, default=8)
    parser.add_argument("--confirm-top-k", "--top-k", dest="confirm_top_k", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--train-sampling-policy",
        choices=("none", "balanced_class"),
        default=None,
    )
    parser.add_argument("--screen-seed", type=int, default=42)
    parser.add_argument("--confirm-seeds", nargs="+", type=int, default=(42,))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        default=True,
        help="require CUDA for paper-facing runs (default)",
    )
    parser.add_argument(
        "--allow-cpu-debug",
        action="store_true",
        help="allow CPU fallback for debugging only; paper-facing runs require CUDA",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-run-id", default=None)
    parser.add_argument("--resume-from-progress", default=None)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--batch-log-interval", type=int, default=20)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--confirm-only", action="store_true")
    parser.add_argument("--candidate-filter", nargs="+", default=())
    parser.add_argument(
        "--dataset-filter",
        nargs="+",
        choices=("nasa_csm", "uab_workload_dataset"),
        default=(),
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
    if args.resume_run_id:
        args.run_id = args.resume_run_id
        args.resume = True
    prepared_roots = _collect_prepared_roots(args)
    result = run_stage_i_public_fusion_refresh(
        StageIPublicFusionRefreshConfig(
            run_id=args.run_id,
            dataset_prepared_roots=prepared_roots,
            artifact_root=_resolve_path(args.artifact_root),
            report_root=_resolve_path(args.report_root),
            datasets=tuple(args.datasets),
            screen_epochs=args.screen_epochs,
            confirm_epochs=args.confirm_epochs,
            screen_max_folds=args.screen_max_folds,
            confirm_max_folds=_parse_optional_int(args.confirm_max_folds),
            screen_candidate_limit=args.screen_candidate_limit,
            confirm_top_k=args.confirm_top_k,
            screen_seed=args.screen_seed,
            confirm_seeds=tuple(args.confirm_seeds),
            device=args.device,
            require_cuda=args.require_cuda and not args.allow_cpu_debug,
            resume=args.resume,
            resume_run_id=args.resume_run_id,
            resume_from_progress=args.resume_from_progress,
            skip_completed=args.skip_completed,
            allow_partial=args.allow_partial,
            screen_only=args.screen_only,
            confirm_only=args.confirm_only,
            candidate_filter=tuple(args.candidate_filter),
            dataset_filter=tuple(args.dataset_filter),
            heartbeat_seconds=args.heartbeat_seconds,
            batch_log_interval=args.batch_log_interval,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_sampling_policy=args.train_sampling_policy,
        )
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _collect_prepared_roots(args: argparse.Namespace) -> dict[str, str]:
    roots = {}
    if args.nasa_root:
        roots["nasa_csm"] = _resolve_path(args.nasa_root)
    if args.uab_root:
        roots["uab_workload_dataset"] = _resolve_path(args.uab_root)
    missing = [dataset for dataset in args.datasets if dataset not in roots]
    if missing and not args.prepare_sequences:
        raise ValueError(
            "missing prepared roots for "
            + ", ".join(missing)
            + "; pass --prepare-sequences or explicit --nasa-root/--uab-root."
        )
    for dataset_id in missing:
        output_root = Path(_resolve_path(args.prepared_root_base)) / args.run_id / dataset_id
        run_stage_i_sequence_preparation(
            StageISequencePreparationConfig(
                dataset_id=dataset_id,
                artifact_root=str(output_root),
                dataset_root=_resolve_path(args.dataset_root),
                profile="window_v2",
                target_steps=args.target_steps,
            )
        )
        roots[dataset_id] = str(output_root)
    return roots


def _resolve_path(path_like: str) -> str:
    path = Path(path_like)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _parse_optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = value.strip().lower()
    if normalized in {"none", "null", ""}:
        return None
    return int(normalized)


if __name__ == "__main__":
    raise SystemExit(main())
