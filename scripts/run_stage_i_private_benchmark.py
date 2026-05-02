"""Run the private Stage I benchmark over real Stage H all-window assets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines import (  # noqa: E402
    StageIPrivateBenchmarkConfig,
    run_stage_i_private_benchmark,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-private-benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--e-run-manifest", required=True)
    parser.add_argument("--f-run-manifest", required=True)
    parser.add_argument("--output-root", default="docs/reports/assets/stage_i_private")
    parser.add_argument("--report-root", default="docs/reports")
    parser.add_argument("--deep-epochs", type=int, default=1)
    parser.add_argument("--deep-learning-rate", type=float, default=1e-3)
    parser.add_argument("--deep-batch-size", type=int, default=32)
    parser.add_argument("--deep-hidden-dim", type=int, default=32)
    parser.add_argument("--deep-num-heads", type=int, default=2)
    parser.add_argument("--deep-layers", type=int, default=1)
    parser.add_argument("--deep-dropout", type=float, default=0.1)
    parser.add_argument("--max-deep-folds", type=int)
    parser.add_argument("--enable-optimized-chronaris", action="store_true")
    parser.add_argument("--target-variant-name", default="chronaris_opt")
    parser.add_argument("--lag-window-points", type=int, default=3)
    parser.add_argument("--residual-mode", default="raw_window_stats")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_stage_i_private_benchmark(
        StageIPrivateBenchmarkConfig(
            run_id=args.run_id,
            e_run_manifest_path=args.e_run_manifest,
            f_run_manifest_path=args.f_run_manifest,
            output_root=args.output_root,
            report_root=args.report_root,
            deep_epochs=args.deep_epochs,
            deep_learning_rate=args.deep_learning_rate,
            deep_batch_size=args.deep_batch_size,
            deep_hidden_dim=args.deep_hidden_dim,
            deep_num_heads=args.deep_num_heads,
            deep_layers=args.deep_layers,
            deep_dropout=args.deep_dropout,
            max_deep_folds=args.max_deep_folds,
            enable_optimized_chronaris=args.enable_optimized_chronaris,
            target_variant_name=args.target_variant_name,
            lag_window_points=args.lag_window_points,
            residual_mode=args.residual_mode,
        )
    )
    print(
        json.dumps(
            {
                "artifact_root": result.artifact_root,
                "task_manifest_path": result.task_manifest_path,
                "task_summary_path": result.task_summary_path,
                "benchmark_summary_path": result.benchmark_summary_path,
                "alignment_report_path": result.alignment_report_path,
                "causal_report_path": result.causal_report_path,
                "optimality_report_path": result.optimality_report_path,
                "optimization_report_path": result.optimization_report_path,
                "optimized_candidate_summary_path": result.optimized_candidate_summary_path,
                "optimized_candidate_metrics_path": result.optimized_candidate_metrics_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
