"""Tests for Stage I unified evidence runner."""

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

from chronaris.pipelines.stage_i.stage_i_evidence_runner import (  # noqa: E402
    StageIEvidenceRunnerConfig,
    run_stage_i_evidence_closure,
)


class StageIEvidenceRunnerTest(unittest.TestCase):
    def test_evidence_runner_records_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rigid_summary = root / "rigid_summary.json"
            rigid_report = root / "rigid_report.md"
            semantic_summary = root / "semantic_summary.json"
            semantic_report = root / "semantic_report.md"
            for path in (rigid_summary, rigid_report, semantic_summary, semantic_report):
                path.write_text("{}\n", encoding="utf-8")

            result = run_stage_i_evidence_closure(
                StageIEvidenceRunnerConfig(
                    run_id="evidence-runner-smoke",
                    output_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    only=("rigid_body", "semantic"),
                    existing_outputs={
                        "rigid_body": {
                            "summary_path": str(rigid_summary),
                            "report_path": str(rigid_report),
                            "evidence_layer": "rigid_body_support",
                        },
                        "semantic": {
                            "summary_path": str(semantic_summary),
                            "report_path": str(semantic_report),
                            "evidence_layer": "semantic_support",
                        },
                    },
                )
            )

            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(manifest["tasks"]["rigid_body"]["reused_existing"])
            self.assertTrue(manifest["tasks"]["semantic"]["reused_existing"])

    def test_evidence_runner_keeps_partial_manifest_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def _side_effect(*, task_name, config, run_root):
                if task_name == "semantic":
                    raise RuntimeError("synthetic failure")
                return {
                    "status": "completed",
                    "evidence_layer": "rigid_body_support",
                    "reused_existing": True,
                    "commands": [],
                    "outputs": {"summary_path": "ok"},
                }

            with patch(
                "chronaris.pipelines.stage_i.stage_i_evidence_runner._run_one_task",
                side_effect=_side_effect,
            ):
                result = run_stage_i_evidence_closure(
                    StageIEvidenceRunnerConfig(
                        run_id="evidence-runner-failure",
                        output_root=str(root / "artifacts"),
                        report_root=str(root / "reports"),
                        only=("rigid_body", "semantic"),
                    )
                )

            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "partial_failed")
            self.assertEqual(manifest["tasks"]["rigid_body"]["status"], "completed")
            self.assertEqual(manifest["tasks"]["semantic"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
