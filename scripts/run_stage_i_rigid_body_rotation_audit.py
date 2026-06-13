"""Run Stage I rigid-body rotation audit."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i.stage_i_rigid_body_rotation_audit import (  # noqa: E402
    DEFAULT_RIGID_BODY_SUMMARY_PATH,
    StageIRigidBodyRotationAuditConfig,
    run_stage_i_rigid_body_rotation_audit,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stage-i-rigid-body-rotation-audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--rigid-body-summary-path", default=DEFAULT_RIGID_BODY_SUMMARY_PATH)
    parser.add_argument("--output-root", default="docs/artifacts/assets/stage_i_rotation_audit")
    parser.add_argument("--report-root", default="docs/artifacts/stage_i")
    parser.add_argument("--sortie-id", default="20251005_四01_ACT-4_云_J20_22#01")
    parser.add_argument("--mysql-database", default="rjgx_backend")
    parser.add_argument("--strict-mysql-field-labels", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_stage_i_rigid_body_rotation_audit(
        StageIRigidBodyRotationAuditConfig(
            run_id=args.run_id,
            rigid_body_summary_path=args.rigid_body_summary_path,
            output_root=args.output_root,
            report_root=args.report_root,
            sortie_id=args.sortie_id,
            mysql_database=args.mysql_database,
            strict_mysql_field_labels=args.strict_mysql_field_labels,
        )
    )
    print(
        json.dumps(
            {
                "artifact_root": result.artifact_root,
                "summary_path": result.summary_path,
                "report_path": result.report_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
