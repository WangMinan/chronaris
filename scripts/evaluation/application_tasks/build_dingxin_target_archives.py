#!/usr/bin/env python3
"""Build independent weak-supervision targets for the fixed Dingxin data."""

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
    DingxinTargetArchiveConfig,
    run_dingxin_target_archive,
)
from chronaris.modeling.common.run_observer import (  # noqa: E402
    configure_task_eval_cli_logging,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="2026-07-11_dingxin-application-targets",
    )
    parser.add_argument("--compact-output-root", default="docs/artifacts/runs")
    parser.add_argument(
        "--heavy-output-root",
        default="artifacts/application_evaluation",
    )
    parser.add_argument(
        "--fixed-audit-root",
        default="docs/artifacts/runs/2026-07-10_fixed-data-audit",
    )
    parser.add_argument(
        "--snapshot-root",
        default="artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_dingxin_target_archive(
        DingxinTargetArchiveConfig(
            run_id=args.run_id,
            compact_output_root=args.compact_output_root,
            heavy_output_root=args.heavy_output_root,
            fixed_audit_root=args.fixed_audit_root,
            snapshot_root=args.snapshot_root,
            resume=args.resume,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
