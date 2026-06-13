"""Tests for Stage I rigid-body rotation audit."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i.stage_i_rigid_body_rotation_audit import (  # noqa: E402
    StageIRigidBodyRotationAuditConfig,
    run_stage_i_rigid_body_rotation_audit,
)


class StageIRotationAuditTest(unittest.TestCase):
    def test_rotation_audit_refreshes_yaw_mapping_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path = root / "rigid_body_ablation_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "families": {
                            "rigid_body": {
                                "rigid_body_mapping_diagnostics": {
                                    "feature_labels": {
                                        "BUS.code1030": "[TSPI数据][载机俯仰角][_角度_毫弧度]",
                                        "BUS.code1031": "[TSPI数据][载机真航向][_角度_毫弧度]",
                                        "BUS.code1032": "[TSPI数据][载机横滚角][_角度_毫弧度]",
                                    }
                                }
                            }
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "chronaris.pipelines.stage_i.stage_i_rigid_body_rotation_audit._resolve_vehicle_field_labels",
                return_value=(
                    {
                        "BUS.code1031": "[TSPI数据][载机真航向][_角度_毫弧度]",
                    },
                    {"status": "loaded", "field_count": 1, "error": None},
                ),
            ):
                result = run_stage_i_rigid_body_rotation_audit(
                    StageIRigidBodyRotationAuditConfig(
                        run_id="rotation-audit-smoke",
                        rigid_body_summary_path=str(summary_path),
                        output_root=str(root / "artifacts"),
                        report_root=str(root / "reports"),
                        strict_mysql_field_labels=False,
                    )
                )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["rotation_status"], "disabled")
            self.assertTrue(summary["stage_h_feature_rotation_groups"]["yaw"])
            self.assertTrue(Path(result.report_path).exists())


if __name__ == "__main__":
    unittest.main()
