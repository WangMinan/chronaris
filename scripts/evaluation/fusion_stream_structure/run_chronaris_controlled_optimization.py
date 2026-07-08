"""Run controlled Chronaris candidate optimization after E3 review."""

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

from chronaris.evaluation.fusion_stream_structure.chronaris_controlled_optimization import (  # noqa: E402
    ControlledOptimizationConfig,
    run_controlled_optimization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-run-id", default="2026-07-08_chronaris-controlled-optimization-dev")
    parser.add_argument("--confirm-run-id", default="2026-07-08_chronaris-controlled-optimization-confirm")
    parser.add_argument("--representation-run-id", default="2026-07-08_chronaris-oof-representation-export")
    parser.add_argument("--output-root", default="docs/artifacts/runs")
    parser.add_argument("--e-run-manifest-path", default="docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json")
    parser.add_argument("--f-run-manifest-path", default="docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json")
    parser.add_argument("--old-baseline-input-path", default="docs/artifacts/runs/2026-07-07_fusion-stream-structure-dingxin-four-method-validation/combined_four_method_e3_input_long.csv")
    parser.add_argument("--old-consistency-table-path", default="docs/artifacts/runs/2026-07-08_e3-result-review/t1_t2_e3_consistency_table.csv")
    parser.add_argument("--old-e3-method-summary-path", default="docs/artifacts/runs/2026-07-08_e3-result-review/e3_method_metric_summary.csv")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--dev-epochs", type=int, default=2)
    parser.add_argument("--confirm-epochs", type=int, default=5)
    parser.add_argument("--smoke-max-folds", type=int, default=1)
    parser.add_argument("--dev-max-folds", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    parser.add_argument("--no-require-cuda", dest="require_cuda", action="store_false")
    parser.add_argument("--tensor-cache", default="auto")
    parser.add_argument("--amp", default="bf16")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-cache-gb", type=float, default=18.0)
    parser.add_argument("--no-confirmation", dest="run_confirmation", action="store_false", default=True)
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ControlledOptimizationConfig(
        dev_run_id=args.dev_run_id,
        confirm_run_id=args.confirm_run_id,
        representation_run_id=args.representation_run_id,
        output_root=args.output_root,
        e_run_manifest_path=args.e_run_manifest_path,
        f_run_manifest_path=args.f_run_manifest_path,
        old_baseline_input_path=args.old_baseline_input_path,
        old_consistency_table_path=args.old_consistency_table_path,
        old_e3_method_summary_path=args.old_e3_method_summary_path,
        seed=int(args.seed),
        smoke_epochs=int(args.smoke_epochs),
        dev_epochs=int(args.dev_epochs),
        confirm_epochs=int(args.confirm_epochs),
        smoke_max_folds=int(args.smoke_max_folds),
        dev_max_folds=args.dev_max_folds,
        device=args.device,
        require_cuda=bool(args.require_cuda),
        tensor_cache=args.tensor_cache,
        amp=args.amp,
        batch_size=int(args.batch_size),
        eval_batch_size=args.eval_batch_size,
        max_cache_gb=float(args.max_cache_gb),
        run_confirmation=bool(args.run_confirmation),
        max_candidates=args.max_candidates,
    )
    summary = run_controlled_optimization(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
