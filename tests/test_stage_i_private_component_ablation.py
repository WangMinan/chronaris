"""Tests for Stage I private proxy component ablation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i.stage_i_private_component_ablation import (  # noqa: E402
    StageIPrivateComponentAblationConfig,
    run_stage_i_private_component_ablation,
)

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "stage_i_private_component_helpers",
    Path(__file__).resolve().with_name("test_stage_i_deep_pipeline.py"),
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError("failed to load private component helpers")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_write_private_stage_h_run = _HELPER_MODULE._write_private_stage_h_run


class StageIPrivateComponentAblationTest(unittest.TestCase):
    def test_private_component_ablation_writes_summary_and_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest = _write_private_stage_h_run(
                root,
                run_name="stage-h-e",
                amplitude_scale=0.8,
                physics_enabled=False,
            )
            f_manifest = _write_private_stage_h_run(
                root,
                run_name="stage-h-f",
                amplitude_scale=1.4,
                physics_enabled=True,
            )

            result = run_stage_i_private_component_ablation(
                StageIPrivateComponentAblationConfig(
                    run_id="private-component-smoke",
                    e_run_manifest_path=str(e_manifest),
                    f_run_manifest_path=str(f_manifest),
                    output_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                )
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["evidence_layer"], "private_proxy")
            self.assertTrue(Path(result.table_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            variants = summary["tasks"]["T1_maneuver_intensity_class"]["variants"]
            self.assertIn("chronaris_opt", variants)
            self.assertIn("chronaris_opt_no_time_residual", variants)
            self.assertIn("chronaris_opt_no_task_head", variants)
            rows = summary["rows"]
            self.assertTrue(any(row["component"] == "remove_causal_mask" for row in rows))
            self.assertTrue(any(row["component"] == "remove_time_residual" for row in rows))
            self.assertTrue(any(row["component"] == "remove_task_aware_head" for row in rows))


if __name__ == "__main__":
    unittest.main()
