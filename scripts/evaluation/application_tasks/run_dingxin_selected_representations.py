#!/usr/bin/env python3
"""Export Dingxin six-method representations from selected checkpoints."""

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
    DingxinSelectedRepresentationConfig,
    run_dingxin_selected_representations,
)
from chronaris.modeling.common.run_observer import configure_task_eval_cli_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="2026-07-12_dingxin-selected-representations-seed17")
    parser.add_argument("--compact-output-root", default="docs/artifacts/runs")
    parser.add_argument("--heavy-output-root", default="artifacts/application_evaluation")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fold-id", action="append", default=[])
    parser.add_argument("--fit-batch-size", type=int, default=2)
    parser.add_argument("--export-batch-size", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_dingxin_selected_representations(
        DingxinSelectedRepresentationConfig(
            run_id=args.run_id,
            compact_output_root=args.compact_output_root,
            heavy_output_root=args.heavy_output_root,
            seed=args.seed,
            fold_ids=(
                tuple(args.fold_id)
                if args.fold_id
                else DingxinSelectedRepresentationConfig().fold_ids
            ),
            fit_batch_size=args.fit_batch_size,
            export_batch_size=args.export_batch_size,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
