#!/usr/bin/env python3
"""Revalidate and aggregate all five Dingxin pretraining fold runs."""

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
    DingxinPretrainingAggregateConfig,
    run_dingxin_pretraining_aggregate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="2026-07-11_dingxin-five-fold-pretraining",
    )
    parser.add_argument("--output-root", default="docs/artifacts/runs")
    args = parser.parse_args()
    result = run_dingxin_pretraining_aggregate(
        DingxinPretrainingAggregateConfig(
            run_id=args.run_id,
            output_root=args.output_root,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
