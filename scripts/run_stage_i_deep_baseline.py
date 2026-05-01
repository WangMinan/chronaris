"""Run one Stage I deep baseline over prepared sequence assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i_deep_baseline import (  # noqa: E402
    StageIDeepBaselineConfig,
    run_stage_i_deep_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("mult", "contiformer"))
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("stage_h_case", "uab_workload_dataset", "nasa_csm"),
    )
    parser.add_argument(
        "--prepared-root",
        required=True,
        help="directory created by prepare_stage_i_sequences.py",
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
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_stage_i_deep_baseline(
        StageIDeepBaselineConfig(
            model_name=args.model,
            dataset_id=args.dataset,
            profile=args.profile
            or ("real_sortie_v1" if args.dataset == "stage_h_case" else "window_v2"),
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
            max_folds=args.max_folds,
            seed=args.seed,
        ),
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


def _resolve_path(path_like: str) -> str:
    path = Path(path_like)
    return str(path if path.is_absolute() else (REPO_ROOT / path))


if __name__ == "__main__":
    raise SystemExit(main())
