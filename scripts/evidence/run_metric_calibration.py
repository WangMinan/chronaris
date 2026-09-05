#!/usr/bin/env python
"""Run P37 optimized Chronaris metric calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evidence.metric_calibration import (  # noqa: E402
    StageIMetricCalibrationConfig,
    run_task_eval_metric_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--tensor-cache", default="auto")
    parser.add_argument("--max-cache-gb", type=float, default=18.0)
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument("--batch-size-candidates", nargs="+", type=int, default=[2048, 1024, 512, 256, 128])
    parser.add_argument("--amp", default="bf16")
    parser.add_argument("--torch-compile", default="default")
    parser.add_argument("--cpu-workers", type=int, default=24)
    parser.add_argument("--parallel-fold-prep", type=int, default=8)
    parser.add_argument("--parallel-candidates", type=int, default=1)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--batch-log-interval", type=int, default=20)
    parser.add_argument("--profile-gpu", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--checkpoint-policy", default="last")
    parser.add_argument("--no-private-screen", action="store_true")
    parser.add_argument("--no-private-confirm", action="store_true")
    parser.add_argument("--no-public-screen", action="store_true")
    parser.add_argument("--no-public-confirm", action="store_true")
    parser.add_argument("--public-confirm-extra-seed17", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_task_eval_metric_calibration(
        StageIMetricCalibrationConfig(
            run_id=args.run_id,
            run_private_screen=not args.no_private_screen,
            run_private_confirm=not args.no_private_confirm,
            run_public_screen=not args.no_public_screen,
            run_public_confirm=not args.no_public_confirm,
            public_confirm_extra_seed17=args.public_confirm_extra_seed17,
            device=args.device,
            require_cuda=args.require_cuda,
            tensor_cache=args.tensor_cache,
            max_cache_gb=args.max_cache_gb,
            auto_batch_size=args.auto_batch_size,
            batch_size_candidates=tuple(args.batch_size_candidates),
            amp=args.amp,
            torch_compile=args.torch_compile,
            cpu_workers=args.cpu_workers,
            parallel_fold_prep=args.parallel_fold_prep,
            parallel_candidates=args.parallel_candidates,
            heartbeat_seconds=args.heartbeat_seconds,
            batch_log_interval=args.batch_log_interval,
            profile_gpu=args.profile_gpu,
            allow_partial=args.allow_partial,
            skip_completed=args.skip_completed or args.resume,
            checkpoint_policy=args.checkpoint_policy,
        )
    )
    print(result.summary_path)
    print(result.report_path)


if __name__ == "__main__":
    main()
