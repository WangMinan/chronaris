"""Run the fixed Stage I deep-comparison order: Stage H real sortie -> UAB -> NASA."""

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
    StageIDeepComparisonConfig,
    run_stage_i_deep_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=("mult", "contiformer"),
        choices=("mult", "contiformer"),
    )
    parser.add_argument("--stage-h-root", default=None)
    parser.add_argument("--uab-root", default=None)
    parser.add_argument("--nasa-root", default=None)
    parser.add_argument("--stage-h-reference-root", default=None)
    parser.add_argument(
        "--uab-reference-root",
        default="docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/uab_window",
    )
    parser.add_argument(
        "--nasa-reference-root",
        default="docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/nasa_attention",
    )
    parser.add_argument("--artifact-root", required=True)
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
    dataset_roots = {
        dataset_id: _resolve_path(path_like)
        for dataset_id, path_like in {
            "stage_h_case": args.stage_h_root,
            "uab_workload_dataset": args.uab_root,
            "nasa_csm": args.nasa_root,
        }.items()
        if path_like
    }
    reference_roots = {
        dataset_id: _resolve_path(path_like)
        for dataset_id, path_like in {
            "stage_h_case": args.stage_h_reference_root,
            "uab_workload_dataset": args.uab_reference_root,
            "nasa_csm": args.nasa_reference_root,
        }.items()
        if path_like
    }
    result = run_stage_i_deep_comparison(
        StageIDeepComparisonConfig(
            model_names=tuple(args.models),
            dataset_artifact_roots=dataset_roots,
            output_root=_resolve_path(args.artifact_root),
            reference_artifact_roots=reference_roots,
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
