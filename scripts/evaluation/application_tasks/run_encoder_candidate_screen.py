#!/usr/bin/env python3
"""Run the seed-17 equal-budget encoder candidate screen."""

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
    EncoderCandidateScreenRunConfig,
    run_encoder_candidate_screen,
)
from chronaris.modeling.common.run_observer import (  # noqa: E402
    configure_task_eval_cli_logging,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="2026-07-11_encoder-candidate-screen-seed17")
    parser.add_argument("--compact-output-root", default="docs/artifacts/runs")
    parser.add_argument("--heavy-output-root", default="artifacts/application_evaluation")
    parser.add_argument("--simulation-root", default="artifacts/application_evaluation/2026-07-10_aviation-simulation-formal")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_encoder_candidate_screen(
        EncoderCandidateScreenRunConfig(
            run_id=args.run_id,
            compact_output_root=args.compact_output_root,
            heavy_output_root=args.heavy_output_root,
            simulation_root=args.simulation_root,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
