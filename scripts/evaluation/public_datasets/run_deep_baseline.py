"""Run one task evaluation deep baseline over prepared sequence assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.public_datasets.pipelines.deep_baseline import (  # noqa: E402
    StageIDeepBaselineConfig,
    run_task_eval_deep_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=("mult", "contiformer", "chronaris_public_fusion"),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("feature_export_case", "uab_workload_dataset", "nasa_csm"),
    )
    parser.add_argument(
        "--prepared-root",
        required=True,
        help="directory created by scripts/task_eval/data/prepare_sequences.py",
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--reference-artifact-root", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fusion-event-bias-weight", type=float, default=0.25)
    parser.add_argument("--fusion-lag-window-points", type=int, default=None)
    parser.add_argument(
        "--fusion-normalize-states",
        choices=("true", "false"),
        default="true",
    )
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--train-sampling-policy",
        choices=("none", "balanced_class"),
        default="none",
    )
    parser.add_argument(
        "--regression-loss",
        choices=("mse", "smooth_l1", "huber"),
        default="mse",
    )
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--target-transform",
        choices=("none", "zscore_train", "robust_train"),
        default="none",
    )
    parser.add_argument("--gradient-clip-max-norm", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--batch-log-interval", type=int, default=20)
    parser.add_argument("--checkpoint-policy", choices=("off", "last", "epoch_and_fold"), default="last")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_task_eval_deep_baseline(
        StageIDeepBaselineConfig(
            model_name=args.model,
            dataset_id=args.dataset,
            profile=args.profile
            or ("real_sortie_v1" if args.dataset == "feature_export_case" else "window_v2"),
            prepared_artifact_root=_resolve_path(args.prepared_root),
            artifact_root=_resolve_path(args.artifact_root),
            reference_artifact_root=(
                _resolve_path(args.reference_artifact_root)
                if args.reference_artifact_root
                else None
            ),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            layers=args.layers,
            dropout=args.dropout,
            fusion_event_bias_weight=args.fusion_event_bias_weight,
            fusion_lag_window_points=args.fusion_lag_window_points,
            fusion_normalize_states=(args.fusion_normalize_states == "true"),
            max_folds=args.max_folds,
            seed=args.seed,
            device=args.device,
            train_sampling_policy=args.train_sampling_policy,
            regression_loss=args.regression_loss,
            huber_delta=args.huber_delta,
            target_transform=args.target_transform,
            gradient_clip_max_norm=args.gradient_clip_max_norm,
            weight_decay=args.weight_decay,
            heartbeat_seconds=args.heartbeat_seconds,
            batch_log_interval=args.batch_log_interval,
            checkpoint_policy=args.checkpoint_policy,
        ),
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _resolve_path(path_like: str) -> str:
    path = Path(path_like)
    return str(path if path.is_absolute() else (REPO_ROOT / path))


if __name__ == "__main__":
    raise SystemExit(main())
