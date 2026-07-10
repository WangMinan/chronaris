#!/usr/bin/env python3
"""Run the Chronaris continuous production adapter smoke."""

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
    ChronarisContinuousAdapterSmokeConfig,
    run_chronaris_continuous_adapter_smoke,
)
from chronaris.modeling.common.run_observer import (  # noqa: E402
    configure_task_eval_cli_logging,
)


def main() -> int:
    configure_task_eval_cli_logging(sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="2026-07-11_chronaris-continuous-adapter-smoke",
    )
    parser.add_argument("--compact-output-root", default="docs/artifacts/runs")
    parser.add_argument(
        "--heavy-output-root",
        default="artifacts/application_evaluation",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = run_chronaris_continuous_adapter_smoke(
        ChronarisContinuousAdapterSmokeConfig(
            run_id=args.run_id,
            compact_output_root=args.compact_output_root,
            heavy_output_root=args.heavy_output_root,
            seed=args.seed,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
