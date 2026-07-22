#!/usr/bin/env python3
"""Build the bounded thesis report for the simplified Dingxin evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from chronaris.evaluation.dingxin.simple_downstream_reporting import (
    SimpleDownstreamReportingConfig,
    build_simple_downstream_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="2026-07-16_simple-downstream-thesis-evidence",
    )
    args = parser.parse_args()
    result = build_simple_downstream_report(
        SimpleDownstreamReportingConfig(run_id=args.run_id)
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
