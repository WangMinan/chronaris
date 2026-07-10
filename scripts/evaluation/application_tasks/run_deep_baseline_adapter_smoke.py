"""Run task-head-free causal MulT and ContiFormer production adapter smoke."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.application_tasks import (  # noqa: E402
    DeepBaselineAdapterSmokeConfig,
    run_deep_baseline_adapter_smoke,
)
from chronaris.modeling.common.run_observer import configure_task_eval_cli_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="2026-07-11_deep-baseline-adapter-smoke",
    )
    parser.add_argument("--compact-output-root", default="docs/artifacts/runs")
    parser.add_argument("--heavy-output-root", default="artifacts/application_evaluation")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main() -> int:
    configure_task_eval_cli_logging(sys.stderr)
    args = parse_args()
    result = run_deep_baseline_adapter_smoke(
        DeepBaselineAdapterSmokeConfig(
            run_id=args.run_id,
            compact_output_root=args.compact_output_root,
            heavy_output_root=args.heavy_output_root,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
