#!/usr/bin/env python3
"""Audit lazy Dingxin raw contexts against independent target archives."""

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
    DingxinContextBindingConfig,
    run_dingxin_context_binding_audit,
)
from chronaris.modeling.common.run_observer import (  # noqa: E402
    configure_task_eval_cli_logging,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="2026-07-11_dingxin-context-bindings",
    )
    parser.add_argument("--output-root", default="docs/artifacts/runs")
    parser.add_argument(
        "--snapshot-root",
        default="artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot",
    )
    parser.add_argument(
        "--fixed-audit-root",
        default="docs/artifacts/runs/2026-07-10_fixed-data-audit",
    )
    parser.add_argument(
        "--target-run-root",
        default="docs/artifacts/runs/2026-07-11_dingxin-application-targets",
    )
    args = parser.parse_args()
    configure_task_eval_cli_logging(sys.stderr)
    result = run_dingxin_context_binding_audit(
        DingxinContextBindingConfig(
            run_id=args.run_id,
            output_root=args.output_root,
            snapshot_root=args.snapshot_root,
            fixed_audit_root=args.fixed_audit_root,
            target_run_root=args.target_run_root,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
