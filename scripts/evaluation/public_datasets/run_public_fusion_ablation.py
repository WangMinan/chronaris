"""Run P31 chronaris_public_fusion ablation over NASA/UAB public sequences."""

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

from chronaris.modeling.common.run_observer import (  # noqa: E402
    configure_task_eval_cli_logging,
)
from chronaris.evaluation.public_datasets.pipelines.fusion_ablation import (  # noqa: E402
    P28_SOURCE_ROOT,
    StageIPublicFusionAblationConfig,
    load_p28_prepared_roots,
    run_task_eval_public_fusion_ablation,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-task-eval-public-fusion-ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--artifact-root", default="docs/artifacts/runs")
    parser.add_argument("--report-root", default="docs/artifacts/runs")
    parser.add_argument("--p28-source-root", default=str(P28_SOURCE_ROOT))
    parser.add_argument("--base-refresh-run-id", default=None)
    parser.add_argument("--base-run-id", default=None)
    parser.add_argument("--nasa-root", default=None)
    parser.add_argument("--uab-root", default=None)
    parser.add_argument("--datasets", nargs="+", default=("nasa_csm", "uab_workload_dataset"))
    parser.add_argument("--dataset-filter", nargs="+", dest="datasets", default=argparse.SUPPRESS)
    parser.add_argument("--variant-filter", nargs="+", default=())
    parser.add_argument("--variants", nargs="+", dest="variant_filter", default=argparse.SUPPRESS)
    parser.add_argument("--candidate-filter", nargs="+", dest="variant_filter", default=argparse.SUPPRESS)
    parser.add_argument("--fold-filter", nargs="+", default=())
    parser.add_argument("--confirm-variants", nargs="+", default=None)
    parser.add_argument("--screen-epochs", type=int, default=5)
    parser.add_argument("--confirm-epochs", type=int, default=20)
    parser.add_argument("--screen-max-folds", default="2")
    parser.add_argument("--confirm-max-folds", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-confirm-seeds", nargs="+", type=int, default=())
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
    parser.add_argument("--cache-full-dataset", choices=("true", "false"), default="true")
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
    parser.add_argument("--parallel-candidates", type=int, default=1)
    parser.add_argument("--profile-gpu", action="store_true", default=True)
    parser.add_argument("--no-profile-gpu", dest="profile_gpu", action="store_false")
    parser.add_argument("--checkpoint-policy", choices=("off", "last", "epoch_and_fold"), default="last")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    configure_task_eval_cli_logging(sys.stdout)
    args = parse_args()
    if args.resume_run_id:
        args.run_id = args.resume_run_id
        args.resume = True
    if args.confirm_variants:
        args.variant_filter = args.confirm_variants
    prepared_roots = load_p28_prepared_roots(args.p28_source_root)
    if args.nasa_root:
        prepared_roots["nasa_csm"] = _resolve_path(args.nasa_root)
    if args.uab_root:
        prepared_roots["uab_workload_dataset"] = _resolve_path(args.uab_root)
    result = run_task_eval_public_fusion_ablation(
        StageIPublicFusionAblationConfig(
            run_id=args.run_id,
            dataset_prepared_roots=prepared_roots,
            artifact_root=_resolve_path(args.artifact_root),
            report_root=_resolve_path(args.report_root),
            datasets=tuple(args.datasets),
            variants=tuple(args.variant_filter),
            screen_epochs=args.screen_epochs,
            confirm_epochs=args.confirm_epochs,
            screen_max_folds=_parse_optional_int(args.screen_max_folds),
            confirm_max_folds=_parse_optional_int(args.confirm_max_folds),
            seed=args.seed,
            extra_confirm_seeds=tuple(args.extra_confirm_seeds),
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
            parallel_candidates=args.parallel_candidates,
            checkpoint_policy=args.checkpoint_policy,
            base_run_id=args.base_run_id,
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
