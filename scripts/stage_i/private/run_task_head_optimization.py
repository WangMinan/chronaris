"""Run P34 task-aware head optimization for private Stage H proxy tasks."""

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

from chronaris.pipelines.stage_i.common.run_observer import configure_stage_i_cli_logging  # noqa: E402
from chronaris.pipelines.stage_i.private.task_head_optimization import (  # noqa: E402
    P34_MODELS,
    StageITaskHeadOptimizationConfig,
    run_stage_i_task_head_optimization,
)

DEFAULT_E_MANIFEST = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json"
)
DEFAULT_F_MANIFEST = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_h/20260502T-stage-h-f-allwindow-clean/run_manifest.json"
)
if not DEFAULT_F_MANIFEST.exists():
    DEFAULT_F_MANIFEST = (
        REPO_ROOT
        / "docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json"
    )


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-task-heads-optimization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-run-id", default=None)
    parser.add_argument("--e-run-manifest", default=str(DEFAULT_E_MANIFEST))
    parser.add_argument("--f-run-manifest", default=str(DEFAULT_F_MANIFEST))
    parser.add_argument("--p30-root", default=None)
    parser.add_argument("--output-root", default="docs/artifacts/assets/stage_i_task_heads_optimization")
    parser.add_argument("--artifact-root", dest="output_root", default=argparse.SUPPRESS)
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--tasks", nargs="+", default=())
    parser.add_argument("--models", nargs="+", default=P34_MODELS)
    parser.add_argument("--candidate-filter", nargs="+", dest="models", default=argparse.SUPPRESS)
    parser.add_argument("--split-strategies", nargs="+", default=("leave_one_view_out",))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42,))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu-debug", action="store_true")
    parser.add_argument("--tensor-cache", choices=("auto", "cuda", "cpu", "off"), default="auto")
    parser.add_argument("--max-cache-gb", type=float, default=18.0)
    parser.add_argument("--auto-batch-size", action="store_true", default=True)
    parser.add_argument("--batch-size-candidates", nargs="+", type=int, default=(2048, 1024, 512, 256, 128))
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--torch-compile", choices=("off", "default", "reduce-overhead", "max-autotune"), default="off")
    parser.add_argument("--cpu-workers", type=int, default=24)
    parser.add_argument("--parallel-fold-prep", type=int, default=8)
    parser.add_argument("--parallel-candidates", type=int, default=1)
    parser.add_argument("--parallel-mode", default="screen_only")
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--batch-log-interval", type=int, default=20)
    parser.add_argument("--profile-gpu", action="store_true", default=True)
    parser.add_argument("--checkpoint-policy", choices=("off", "last", "epoch_and_fold"), default="last")
    parser.add_argument("--allow-partial", action="store_true", default=True)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--screen-only", action="store_true", default=True)
    parser.add_argument("--confirm-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    configure_stage_i_cli_logging(sys.stdout)
    args = parse_args()
    if args.resume_run_id:
        args.run_id = args.resume_run_id
        args.resume = True
    kwargs = {}
    if args.p30_root:
        kwargs["p30_root"] = _resolve_path(args.p30_root)
    result = run_stage_i_task_head_optimization(
        StageITaskHeadOptimizationConfig(
            run_id=args.run_id,
            e_run_manifest_path=_resolve_path(args.e_run_manifest),
            f_run_manifest_path=_resolve_path(args.f_run_manifest),
            output_root=_resolve_path(args.output_root),
            report_root=_resolve_path(args.report_root),
            models=tuple(args.models),
            seeds=tuple(args.seeds),
            split_strategies=tuple(args.split_strategies),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            layers=args.layers,
            dropout=args.dropout,
            device=args.device,
            require_cuda=args.require_cuda and not args.allow_cpu_debug,
            tensor_cache=args.tensor_cache,
            max_cache_gb=args.max_cache_gb,
            auto_batch_size=args.auto_batch_size,
            batch_size_candidates=tuple(args.batch_size_candidates),
            amp=args.amp,
            torch_compile=args.torch_compile,
            cpu_workers=args.cpu_workers,
            parallel_fold_prep=args.parallel_fold_prep,
            heartbeat_seconds=args.heartbeat_seconds,
            batch_log_interval=args.batch_log_interval,
            profile_gpu=args.profile_gpu,
            checkpoint_policy=args.checkpoint_policy,
            allow_partial=args.allow_partial,
            skip_completed=args.skip_completed,
            screen_only=args.screen_only and not args.confirm_only,
            **kwargs,
        )
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _resolve_path(path_like: str) -> str:
    path = Path(path_like)
    return str(path if path.is_absolute() else REPO_ROOT / path)


if __name__ == "__main__":
    raise SystemExit(main())
