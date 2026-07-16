"""Run fixed Logistic/Ridge consumers for simplified Dingxin tasks."""

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

from chronaris.evaluation.dingxin.simple_downstream_protocol import (  # noqa: E402
    SIMPLE_DOWNSTREAM_METHODS,
)
from chronaris.evaluation.dingxin.simple_downstream_run import (  # noqa: E402
    SimpleDownstreamRunConfig,
    run_simple_downstream_consumers,
)


def main() -> int:
    defaults = SimpleDownstreamRunConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=defaults.run_id)
    parser.add_argument("--representation-run-id", default=defaults.representation_run_id)
    parser.add_argument("--task-protocol-run-id", default=defaults.task_protocol_run_id)
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--fold-id", action="append", default=[])
    parser.add_argument("--method", action="append", choices=SIMPLE_DOWNSTREAM_METHODS, default=[])
    parser.add_argument(
        "--evaluation-scope",
        choices=("engineering_smoke", "formal_confirmation"),
        default=defaults.evaluation_scope,
    )
    args = parser.parse_args()
    result = run_simple_downstream_consumers(
        SimpleDownstreamRunConfig(
            run_id=args.run_id,
            representation_run_id=args.representation_run_id,
            task_protocol_run_id=args.task_protocol_run_id,
            seeds=tuple(args.seed) or defaults.seeds,
            fold_ids=tuple(args.fold_id) or defaults.fold_ids,
            methods=tuple(args.method) or defaults.methods,
            evaluation_scope=args.evaluation_scope,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
