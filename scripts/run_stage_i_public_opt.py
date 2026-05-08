"""Run one Stage I public-opt path over prepared UAB or NASA assets."""

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
    StageIPublicOptConfig,
    StageIPublicOptTorchUABConfig,
    run_stage_i_public_opt,
    run_stage_i_public_opt_torch_uab,
)
from chronaris.pipelines.stage_i.stage_i_public_opt_data import (  # noqa: E402
    normalize_public_opt_dataset_id,
)
from chronaris.pipelines.stage_i.stage_i_run_observer import (  # noqa: E402
    configure_stage_i_cli_logging,
)

DEFAULT_SKLEARN_ARTIFACT_ROOT = "docs/reports/assets/stage_i_public_opt"
DEFAULT_TORCH_ARTIFACT_ROOT = "docs/reports/assets/stage_i_public_opt_torch"


def _default_run_id(*, dataset_id: str, backend: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if backend == "torch" and dataset_id == "uab_workload_dataset":
        return f"{timestamp}-stage-i-public-opt-uab-torch"
    return f"{timestamp}-stage-i-public-opt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--prepared-artifact-root", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--report-root", default="docs/reports")
    parser.add_argument("--dataset-id", default="uab_workload_dataset")
    parser.add_argument("--profile", default="window_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "sklearn", "torch"),
    )
    parser.add_argument(
        "--feature-profile",
        default="full",
        choices=(
            "full",
            "physiology_only",
            "physiology_lowdim",
            "physiology_scalar_only",
            "context_only",
            "residual_only",
        ),
    )
    parser.add_argument(
        "--head-catalog",
        default="expanded",
        choices=("minimal", "expanded", "uab_hybrid"),
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
        "--prediction-aggregation-policy",
        default="none",
        choices=("none", "session_mean_broadcast", "session_median_broadcast"),
    )
    parser.add_argument(
        "--winner-margin-policy",
        default="paper_gate",
        choices=("paper_gate", "none"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--screen-max-folds", type=int, default=2)
    parser.add_argument("--full-max-folds", type=int)
    parser.add_argument("--full-candidate-limit", type=int, default=2)
    parser.add_argument("--full-group-winner-limit", type=int, default=1)
    parser.add_argument("--skip-full-loso", action="store_true")
    parser.add_argument(
        "--allow-cpu-debug",
        action="store_true",
        help="allow torch UAB to run on CPU for debugging; paper-facing runs require CUDA",
    )
    parser.add_argument(
        "--allow-cpu-heavy-sklearn",
        action="store_true",
        help="allow historical CPU-heavy UAB sklearn hybrid reproduction",
    )
    parser.add_argument(
        "--selected-subset",
        action="append",
        dest="selected_subsets",
        choices=("n_back", "heat_the_chair"),
        default=[],
    )
    parser.add_argument(
        "--torch-candidate-catalog",
        choices=("default", "heat_specialist"),
        default="heat_specialist",
    )
    parser.add_argument(
        "--supervision-granularity",
        choices=("window", "session_pooled_broadcast"),
        default="window",
    )
    parser.add_argument(
        "--torch-feature-profile",
        action="append",
        dest="torch_feature_profiles",
        choices=(
            "full",
            "residual_only",
            "physiology_only",
            "physiology_lowdim",
            "physiology_scalar_only",
        ),
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
    parser.add_argument("--reference-public-opt-summary")
    parser.add_argument("--reference-phase3-closure-summary")
    parser.add_argument("--reference-deep-comparison-summary")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    configure_stage_i_cli_logging(sys.stdout)
    args = parse_args()
    payload = _run_from_args(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_from_args(args: argparse.Namespace) -> dict[str, object]:
    dataset_id = normalize_public_opt_dataset_id(args.dataset_id)
    backend = _resolve_backend(
        requested_backend=args.backend,
        dataset_id=dataset_id,
    )
    run_id = args.run_id or _default_run_id(dataset_id=dataset_id, backend=backend)
    artifact_root = _resolve_artifact_root(
        artifact_root=args.artifact_root,
        backend=backend,
    )
    if backend == "torch":
        result = _run_torch_backend(
            args=args,
            dataset_id=dataset_id,
            run_id=run_id,
            artifact_root=artifact_root,
        )
        payload = {
            "backend": backend,
            "dataset_id": dataset_id,
            "runtime_device": result.summary.get("runtime_device"),
            "public_opt_feature_frame_path": result.feature_frame_path,
            "public_opt_predictions_path": result.predictions_path,
            "public_opt_summary_path": result.summary_path,
            "public_opt_report_path": result.report_path,
        }
        return payload
    _validate_sklearn_args(args=args, dataset_id=dataset_id)
    result = run_stage_i_public_opt(
        StageIPublicOptConfig(
            run_id=run_id,
            prepared_artifact_root=_resolve_path(args.prepared_artifact_root),
            artifact_root=_resolve_path(artifact_root),
            report_root=_resolve_path(args.report_root),
            dataset_id=dataset_id,
            profile=args.profile,
            seed=args.seed,
            feature_profile=args.feature_profile,
            head_catalog=args.head_catalog,
            train_balance_policy=args.train_balance_policy,
            ensemble_policy=args.ensemble_policy,
            prediction_aggregation_policy=args.prediction_aggregation_policy,
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
    return {
        "backend": backend,
        "dataset_id": dataset_id,
        "public_opt_feature_frame_path": result.feature_frame_path,
        "public_opt_predictions_path": result.predictions_path,
        "public_opt_summary_path": result.summary_path,
        "public_opt_report_path": result.report_path,
    }


def _run_torch_backend(
    *,
    args: argparse.Namespace,
    dataset_id: str,
    run_id: str,
    artifact_root: str,
):
    _validate_torch_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_stage_i_public_opt_torch_uab(
        StageIPublicOptTorchUABConfig(
            run_id=run_id,
            prepared_artifact_root=_resolve_path(args.prepared_artifact_root),
            artifact_root=_resolve_path(artifact_root),
            report_root=_resolve_path(args.report_root),
            dataset_id=dataset_id,
            profile=args.profile,
            device=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            screen_max_folds=args.screen_max_folds,
            full_max_folds=args.full_max_folds,
            run_full_loso=not args.skip_full_loso,
            full_candidate_limit=args.full_candidate_limit,
            full_group_winner_limit=args.full_group_winner_limit,
            ensemble_policy=args.ensemble_policy,
            prediction_aggregation_policy=args.prediction_aggregation_policy,
            supervision_granularity=args.supervision_granularity,
            require_cuda=not args.allow_cpu_debug,
            candidate_catalog=args.torch_candidate_catalog,
            selected_subsets=tuple(args.selected_subsets) or (
                ("heat_the_chair",)
                if args.torch_candidate_catalog == "heat_specialist"
                else ("n_back", "heat_the_chair")
            ),
            learning_rates=tuple(args.learning_rates) or (1e-3, 3e-4),
            weight_decays=tuple(args.weight_decays) or (1e-4, 1e-3),
            feature_profiles=tuple(args.torch_feature_profiles) or (
                ("physiology_lowdim",)
                if args.torch_candidate_catalog == "heat_specialist"
                else (
                    "full",
                    "residual_only",
                    "physiology_only",
                    "physiology_lowdim",
                    "physiology_scalar_only",
                )
            ),
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


def _resolve_backend(*, requested_backend: str, dataset_id: str) -> str:
    if requested_backend == "auto":
        return "torch" if dataset_id == "uab_workload_dataset" else "sklearn"
    return requested_backend


def _resolve_artifact_root(*, artifact_root: str | None, backend: str) -> str:
    if artifact_root:
        return artifact_root
    if backend == "torch":
        return DEFAULT_TORCH_ARTIFACT_ROOT
    return DEFAULT_SKLEARN_ARTIFACT_ROOT


def _validate_torch_args(args: argparse.Namespace) -> None:
    if args.feature_profile not in {
        "full",
        "residual_only",
        "physiology_only",
        "physiology_lowdim",
        "physiology_scalar_only",
    }:
        raise ValueError(
            "torch UAB route only supports full/residual_only/physiology_only/physiology_lowdim/physiology_scalar_only feature profiles. "
            "Use --torch-feature-profile or switch to --backend sklearn."
        )


def _validate_sklearn_args(
    *,
    args: argparse.Namespace,
    dataset_id: str,
) -> None:
    if (
        dataset_id == "uab_workload_dataset"
        and args.head_catalog == "uab_hybrid"
        and not args.allow_cpu_heavy_sklearn
    ):
        raise ValueError(
            "UAB --backend sklearn --head-catalog uab_hybrid is CPU-heavy historical reproduction. "
            "Use --backend torch --torch-candidate-catalog heat_specialist --selected-subset heat_the_chair "
            "--device cuda for the next optimization run, or pass --allow-cpu-heavy-sklearn to reproduce it explicitly."
        )


def _resolve_path(path_like: str) -> str:
    path = Path(path_like)
    return str(path if path.is_absolute() else (REPO_ROOT / path))


if __name__ == "__main__":
    raise SystemExit(main())
