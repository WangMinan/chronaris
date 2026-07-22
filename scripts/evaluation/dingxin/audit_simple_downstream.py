"""Audit and materialize the frozen simplified Dingxin downstream protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.dingxin.simple_downstream_audit import (  # noqa: E402
    SimpleDownstreamAuditConfig,
    run_simple_downstream_audit,
)


def parse_args() -> argparse.Namespace:
    defaults = SimpleDownstreamAuditConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=defaults.run_id)
    parser.add_argument("--compact-output-root", default=defaults.compact_output_root)
    parser.add_argument("--heavy-output-root", default=defaults.heavy_output_root)
    parser.add_argument("--fixed-audit-root", default=defaults.fixed_audit_root)
    parser.add_argument("--snapshot-root", default=defaults.snapshot_root)
    parser.add_argument(
        "--historical-representation-root",
        default=defaults.historical_representation_root,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_simple_downstream_audit(
        SimpleDownstreamAuditConfig(
            run_id=args.run_id,
            compact_output_root=args.compact_output_root,
            heavy_output_root=args.heavy_output_root,
            fixed_audit_root=args.fixed_audit_root,
            snapshot_root=args.snapshot_root,
            historical_representation_root=args.historical_representation_root,
            repository_root=str(REPO_ROOT),
        )
    )
    print(json.dumps(asdict_result(result), ensure_ascii=False, indent=2))
    return 0


def asdict_result(result) -> dict[str, object]:
    return {
        "compact_root": result.compact_root,
        "heavy_root": result.heavy_root,
        "protocol_path": result.protocol_path,
        "compatibility_report_path": result.compatibility_report_path,
        "task_summary_path": result.task_summary_path,
        "report_path": result.report_path,
        "evidence_manifest_path": result.evidence_manifest_path,
    }


if __name__ == "__main__":
    raise SystemExit(main())
