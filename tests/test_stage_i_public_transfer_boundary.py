"""Tests for public adapter calibration and transfer-boundary builders."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i.stage_i_public_adapter_calibration import (  # noqa: E402
    StageIPublicAdapterCalibrationConfig,
    run_stage_i_public_adapter_calibration,
)
from chronaris.pipelines.stage_i.stage_i_public_transfer_boundary import (  # noqa: E402
    StageIPublicTransferBoundaryConfig,
    run_stage_i_public_transfer_boundary,
)


class StageIPublicTransferBoundaryTest(unittest.TestCase):
    def test_public_adapter_calibration_and_transfer_boundary_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uab_summary_path = root / "uab_public_opt_summary.json"
            nasa_summary_path = root / "nasa_public_opt_summary.json"
            torch_summary_path = root / "uab_public_opt_torch_summary.json"
            public_mainline_summary_path = root / "public_mainline_summary.json"
            multitask_summary_path = root / "multitask_summary.json"
            private_summary_path = root / "private_benchmark_summary.json"

            uab_summary_path.write_text(
                json.dumps(
                    {
                        "run_id": "uab-sklearn",
                        "dataset_id": "uab_workload_dataset",
                        "subset_results": {
                            "heat_the_chair": {
                                "heads": {
                                    "target_prior_median": {"rmse": 1.2, "mae": 1.0},
                                    "ridge_residual_cv": {"rmse": 1.4, "mae": 1.1},
                                }
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            nasa_summary_path.write_text(
                json.dumps(
                    {
                        "run_id": "nasa-sklearn",
                        "dataset_id": "nasa_csm",
                        "subset_results": {
                            "combined": {
                                "heads": {
                                    "balanced_logistic_context": {
                                        "macro_f1": 0.7,
                                        "balanced_accuracy": 0.72,
                                    }
                                }
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            torch_summary_path.write_text(
                json.dumps(
                    {
                        "run_id": "uab-torch",
                        "dataset_id": "uab_workload_dataset",
                        "screen_config": {"selected_subsets": ["heat_the_chair"]},
                        "winning_candidate": {"candidate_id": "heat-specialist"},
                        "final_result": {
                            "groups": {
                                "heat_the_chair": {
                                    "rmse": 1.3,
                                    "mae": 1.1,
                                }
                            },
                            "group_selections": {
                                "heat_the_chair": {"selected_source_id": "heat-specialist"}
                            },
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            public_mainline_summary_path.write_text(
                json.dumps(
                    {
                        "public_mainline_status": "NASA closed, UAB partial",
                        "source_paths": {"uab_summary_path": str(uab_summary_path)},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            multitask_summary_path.write_text(
                json.dumps({"run_id": "multitask-real-closure"}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            private_summary_path.write_text(
                json.dumps(
                    {
                        "run_id": "private-opt",
                        "evidence_layers": {
                            "thesis_task_evidence": {
                                "summary": {
                                    "thesis_task_boundary": "thesis_weak_label_not_human_truth"
                                }
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            calibration_result = run_stage_i_public_adapter_calibration(
                StageIPublicAdapterCalibrationConfig(
                    run_id="public-calibration-smoke",
                    output_root=str(root / "calibration"),
                    report_root=str(root / "reports"),
                    uab_sklearn_summary_paths=(str(uab_summary_path),),
                    nasa_summary_paths=(str(nasa_summary_path),),
                    uab_torch_summary_paths=(str(torch_summary_path),),
                )
            )
            calibration_summary = json.loads(
                Path(calibration_result.summary_path).read_text(encoding="utf-8")
            )
            self.assertIn("calibration_baseline", calibration_summary["best_by_category"])
            self.assertIn("legacy_public_opt", calibration_summary["best_by_category"])
            self.assertIn("torch_uab", calibration_summary["best_by_category"])

            boundary_result = run_stage_i_public_transfer_boundary(
                StageIPublicTransferBoundaryConfig(
                    run_id="public-transfer-smoke",
                    calibration_summary_path=calibration_result.summary_path,
                    public_mainline_summary_path=str(public_mainline_summary_path),
                    multitask_summary_path=str(multitask_summary_path),
                    private_summary_path=str(private_summary_path),
                    output_root=str(root / "boundary"),
                    report_root=str(root / "reports"),
                )
            )
            boundary_summary = json.loads(
                Path(boundary_result.summary_path).read_text(encoding="utf-8")
            )
            self.assertEqual(boundary_summary["evidence_layer"], "transfer_boundary")
            self.assertEqual(boundary_summary["public_mainline_status"], "NASA closed, UAB partial")
            self.assertEqual(len(boundary_summary["data_boundary_rows"]), 3)
            self.assertTrue(Path(boundary_result.report_path).exists())


if __name__ == "__main__":
    unittest.main()
