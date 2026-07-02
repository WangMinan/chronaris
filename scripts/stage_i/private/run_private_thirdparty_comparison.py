"""Run P30 private Stage H Chronaris vs MulT/ContiFormer comparison."""

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
from chronaris.pipelines.stage_i.private.thirdparty_comparison import (  # noqa: E402
    MODEL_ORDER,
    StageIPrivateThirdPartyComparisonConfig,
    run_stage_i_private_thirdparty_comparison,
)

DEFAULT_E_MANIFEST = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json"
)
DEFAULT_F_MANIFEST = (
    REPO_ROOT
    / "docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json"
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-private-thirdparty-comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--e-run-manifest", default=str(DEFAULT_E_MANIFEST))
    parser.add_argument("--f-run-manifest", default=str(DEFAULT_F_MANIFEST))
    parser.add_argument("--output-root", default="docs/artifacts/assets/stage_i_private_thirdparty_comparison")
    parser.add_argument("--artifact-root", dest="output_root", default=argparse.SUPPRESS)
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--candidate-filter", nargs="+", dest="models", default=argparse.SUPPRESS)
    parser.add_argument("--dataset-filter", nargs="+", default=None)
    parser.add_argument("--fold-filter", nargs="+", default=())
    parser.add_argument("--seeds", nargs="+", type=int, default=(17, 29, 43, 71, 97))
    parser.add_argument(
        "--split-strategy",
        nargs="+",
        default=("leave_one_view_out", "leave_one_sortie_out"),
        choices=("leave_one_view_out", "leave_one_sortie_out"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--screen-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-cuda", action="store_true", default=True)
    parser.add_argument("--allow-cpu-debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-run-id", default=None)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--confirm-only", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--batch-log-interval", type=int, default=20)
    parser.add_argument("--tensor-cache", choices=("auto", "cuda", "cpu", "off"), default="auto")
    parser.add_argument("--max-cache-gb", type=float, default=18.0)
    parser.add_argument("--pin-memory", choices=("true", "false"), default="true")
    parser.add_argument("--non-blocking-copy", choices=("true", "false"), default="true")
    parser.add_argument("--auto-batch-size", action="store_true", default=True)
    parser.add_argument("--no-auto-batch-size", dest="auto_batch_size", action="store_false")
    parser.add_argument(
        "--batch-size-candidates",
        nargs="+",
        type=int,
        default=(24576, 16384, 8192, 4096, 2048, 1024, 512, 256, 128),
    )
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--amp-eval", choices=("true", "false"), default="true")
    parser.add_argument("--torch-compile", choices=("off", "default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--eval-batch-size", default=None)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--parallel-fold-prep", type=int, default=8)
    parser.add_argument("--profile-gpu", action="store_true", default=True)
    parser.add_argument("--no-profile-gpu", dest="profile_gpu", action="store_false")
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
    result = run_stage_i_private_thirdparty_comparison(
        StageIPrivateThirdPartyComparisonConfig(
            run_id=args.run_id,
            e_run_manifest_path=_resolve_path(args.e_run_manifest),
            f_run_manifest_path=_resolve_path(args.f_run_manifest),
            output_root=_resolve_path(args.output_root),
            report_root=_resolve_path(args.report_root),
            models=tuple(args.models),
            seeds=tuple(args.seeds),
            split_strategy=tuple(args.split_strategy),
            epochs=args.epochs,
            screen_epochs=args.screen_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            layers=args.layers,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            device=args.device,
            require_cuda=args.require_cuda and not args.allow_cpu_debug,
            resume=args.resume,
            skip_completed=args.skip_completed,
            allow_partial=args.allow_partial,
            screen_only=args.screen_only,
            confirm_only=args.confirm_only,
            heartbeat_seconds=args.heartbeat_seconds,
            batch_log_interval=args.batch_log_interval,
            tensor_cache=args.tensor_cache,
            max_cache_gb=args.max_cache_gb,
            pin_memory=_parse_bool(args.pin_memory),
            non_blocking_copy=_parse_bool(args.non_blocking_copy),
            auto_batch_size=args.auto_batch_size,
            batch_size_candidates=tuple(args.batch_size_candidates),
            amp=args.amp,
            amp_eval=_parse_bool(args.amp_eval),
            torch_compile=args.torch_compile,
            profile_gpu=args.profile_gpu,
            eval_batch_size=_parse_optional_int(args.eval_batch_size),
            num_workers=args.num_workers,
            parallel_fold_prep=args.parallel_fold_prep,
        )
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _resolve_path(path_like: str) -> str:
    path = Path(path_like)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _parse_optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(normalized)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected boolean value, got {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
