"""Export complete six-method catalogs for simplified Dingxin tasks."""

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

from chronaris.evaluation.dingxin.simple_downstream_representation_run import (  # noqa: E402
    SimpleDownstreamRepresentationConfig,
    run_simple_downstream_representations,
)


def main() -> int:
    defaults = SimpleDownstreamRepresentationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=defaults.run_id)
    parser.add_argument("--pretraining-run-id", default=defaults.pretraining_run_id)
    parser.add_argument("--task-protocol-run-id", default=defaults.task_protocol_run_id)
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--fold-id", action="append", default=[])
    parser.add_argument("--fit-batch-size", type=int, default=defaults.fit_batch_size)
    parser.add_argument("--export-batch-size", type=int, default=defaults.export_batch_size)
    parser.add_argument("--baseline-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--chronaris-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = run_simple_downstream_representations(
        SimpleDownstreamRepresentationConfig(
            run_id=args.run_id,
            pretraining_run_id=args.pretraining_run_id,
            task_protocol_run_id=args.task_protocol_run_id,
            seeds=tuple(args.seed) or defaults.seeds,
            fold_ids=tuple(args.fold_id) or defaults.fold_ids,
            fit_batch_size=args.fit_batch_size,
            export_batch_size=args.export_batch_size,
            baseline_device=args.baseline_device,
            chronaris_device=args.chronaris_device,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
