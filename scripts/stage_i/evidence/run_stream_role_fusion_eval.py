"""Run P35 stream-role-aware fusion routing evaluation."""

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

from chronaris.pipelines.stage_i.public.stream_role_fusion_eval import (  # noqa: E402
    StageIStreamRoleFusionEvalConfig,
    run_stage_i_stream_role_fusion_eval,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-stream-role-fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-run-id", default=None)
    parser.add_argument("--p31-root", default=None)
    parser.add_argument("--p34-root", default=None)
    parser.add_argument("--artifact-root", default="docs/artifacts/assets/stage_i_stream_role_fusion")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--private-tasks", nargs="+", default=())
    parser.add_argument("--public-datasets", nargs="+", default=("nasa_csm", "uab_workload_dataset"))
    parser.add_argument("--models", nargs="+", default=())
    parser.add_argument("--private-models", nargs="+", default=())
    parser.add_argument("--public-variants", nargs="+", default=())
    parser.add_argument("--run-private-confirm", action="store_true")
    parser.add_argument("--run-public-confirm", action="store_true")
    parser.add_argument("--confirm-epochs", type=int, default=20)
    parser.add_argument("--confirm-max-folds", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-confirm-seeds", nargs="+", type=int, default=())
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
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--batch-log-interval", type=int, default=20)
    parser.add_argument("--profile-gpu", action="store_true", default=True)
    parser.add_argument("--checkpoint-policy", choices=("off", "last", "epoch_and_fold"), default="last")
    parser.add_argument("--allow-partial", action="store_true", default=True)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.resume_run_id:
        args.run_id = args.resume_run_id
        args.resume = True
    kwargs = {}
    if args.p31_root:
        kwargs["p31_root"] = _resolve_path(args.p31_root)
    if args.p34_root:
        kwargs["p34_root"] = _resolve_path(args.p34_root)
    result = run_stage_i_stream_role_fusion_eval(
        StageIStreamRoleFusionEvalConfig(
            run_id=args.run_id,
            artifact_root=_resolve_path(args.artifact_root),
            report_root=_resolve_path(args.report_root),
            run_private_confirm=args.run_private_confirm,
            run_public_confirm=args.run_public_confirm,
            private_models=tuple(args.private_models or args.models),
            public_variants=tuple(args.public_variants or _public_variants_from_models(args.models)),
            public_datasets=tuple(args.public_datasets),
            confirm_epochs=args.confirm_epochs,
            confirm_max_folds=_parse_optional_int(args.confirm_max_folds),
            seed=args.seed,
            extra_confirm_seeds=tuple(args.extra_confirm_seeds),
            resume=args.resume,
            skip_completed=args.skip_completed,
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
            parallel_candidates=args.parallel_candidates,
            heartbeat_seconds=args.heartbeat_seconds,
            batch_log_interval=args.batch_log_interval,
            profile_gpu=args.profile_gpu,
            checkpoint_policy=args.checkpoint_policy,
            allow_partial=args.allow_partial,
            **kwargs,
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


def _public_variants_from_models(models: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    mapping = {
        "chronaris_v3_stream_role": "v3_stream_role",
        "chronaris_v3_stream_role_fusion": "v3_stream_role",
        "v3_stream_role": "v3_stream_role",
        "v3_stream_role_fusion": "v3_stream_role",
        "v3_stream_role_adaptive": "v3_stream_role",
        "v3_no_role_gate": "v3_no_role_gate",
        "v3_fixed_causal_lag": "v3_force_private_causal",
        "v3_force_private_causal": "v3_force_private_causal",
        "v3_context_adapter_only": "v3_context_adapter_only",
    }
    selected: list[str] = []
    for model in models:
        mapped = mapping.get(str(model).strip())
        if mapped and mapped not in selected:
            selected.append(mapped)
    return tuple(selected)


if __name__ == "__main__":
    raise SystemExit(main())
