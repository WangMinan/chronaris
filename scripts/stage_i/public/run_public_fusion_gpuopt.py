"""Run P28 chronaris_public_fusion GPU optimization profiling."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i.common.run_observer import (  # noqa: E402
    configure_stage_i_cli_logging,
)
from chronaris.pipelines.stage_i.public.fusion_gpuopt import (  # noqa: E402
    StageIPublicFusionGPUOptConfig,
    run_stage_i_public_fusion_gpuopt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="20260701T-stage-i-public-fusion-refresh-r1-gpuopt-r1",
    )
    parser.add_argument("--base-p28-run-id", default="20260701T-stage-i-public-fusion-refresh-r1")
    parser.add_argument(
        "--base-p28-root",
        default=(
            "docs/artifacts/assets/stage_i_public_fusion_refresh/"
            "20260701T-stage-i-public-fusion-refresh-r1"
        ),
    )
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu-debug", action="store_true")
    parser.add_argument("--profile-epochs", type=int, default=1)
    parser.add_argument("--profile-batches", type=int, default=20)
    parser.add_argument("--tensor-cache", choices=("auto", "cuda", "cpu", "off"), default="auto")
    parser.add_argument("--baseline-tensor-cache", choices=("auto", "cuda", "cpu", "off"), default="off")
    parser.add_argument("--max-cache-gb", type=float, default=18.0)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--no-non-blocking-copy", dest="non_blocking_copy", action="store_false")
    parser.add_argument("--auto-batch-size", action="store_true", default=True)
    parser.add_argument("--no-auto-batch-size", dest="auto_batch_size", action="store_false")
    parser.add_argument(
        "--batch-size-candidates",
        nargs="+",
        type=int,
        default=(2048, 1024, 512, 256, 128, 64),
    )
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--grad-scaler", action="store_true", default=True)
    parser.add_argument("--no-grad-scaler", dest="grad_scaler", action="store_false")
    parser.add_argument("--amp-eval", action="store_true", default=True)
    parser.add_argument("--no-amp-eval", dest="amp_eval", action="store_false")
    parser.add_argument(
        "--torch-compile",
        choices=("off", "default", "reduce-overhead", "max-autotune"),
        default="off",
    )
    parser.add_argument("--compile-warmup-steps", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--batch-log-interval", type=int, default=20)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    configure_stage_i_cli_logging(sys.stdout)
    args = parse_args()
    result = run_stage_i_public_fusion_gpuopt(
        StageIPublicFusionGPUOptConfig(
            run_id=args.run_id,
            base_p28_run_id=args.base_p28_run_id,
            base_p28_root=_resolve_path(args.base_p28_root),
            artifact_root=_resolve_path(args.artifact_root) if args.artifact_root else None,
            report_root=_resolve_path(args.report_root),
            device=args.device,
            require_cuda=args.require_cuda and not args.allow_cpu_debug,
            profile_epochs=args.profile_epochs,
            profile_batches=args.profile_batches,
            baseline_tensor_cache=args.baseline_tensor_cache,
            tensor_cache=args.tensor_cache,
            max_cache_gb=args.max_cache_gb,
            pin_memory=args.pin_memory,
            non_blocking_copy=args.non_blocking_copy,
            auto_batch_size=args.auto_batch_size,
            batch_size_candidates=tuple(args.batch_size_candidates),
            amp=args.amp,
            grad_scaler=args.grad_scaler,
            amp_eval=args.amp_eval,
            torch_compile=args.torch_compile,
            compile_warmup_steps=args.compile_warmup_steps,
            eval_batch_size=args.eval_batch_size,
            heartbeat_seconds=args.heartbeat_seconds,
            batch_log_interval=args.batch_log_interval,
            skip_completed=args.skip_completed,
        )
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _resolve_path(path_like: str | None) -> str | None:
    if path_like is None:
        return None
    path = Path(path_like)
    return str(path if path.is_absolute() else REPO_ROOT / path)


if __name__ == "__main__":
    raise SystemExit(main())
