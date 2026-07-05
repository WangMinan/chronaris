"""Tests for optimized Chronaris private benchmark candidates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest

import torch

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.dingxin.pipelines.benchmark import (  # noqa: E402
    StageIPrivateBenchmarkConfig,
    run_task_eval_private_benchmark,
)
from chronaris.evaluation.dingxin.pipelines.benchmark_data import (  # noqa: E402
    TASK_MANEUVER,
    TASK_RESPONSE,
    TASK_RETRIEVAL,
    build_variant_feature_frames,
    load_aligned_private_records,
)

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "task_eval_deep_pipeline_helpers",
    SRC.parent / "tests" / "modeling" / "test_deep_pipeline.py",
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError("failed to load private feature export synthetic helper")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_write_private_feature_export_run = _HELPER_MODULE._write_private_feature_export_run


class StageIPrivateOptimizationTest(unittest.TestCase):
    def test_optimized_candidate_generates_frames_without_label_residuals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest, f_manifest = _write_pair(root)
            records = load_aligned_private_records(
                e_run_manifest_path=str(e_manifest),
                f_run_manifest_path=str(f_manifest),
            )

            frames, diagnostics = build_variant_feature_frames(
                records,
                enable_optimized_chronaris=True,
                target_variant_name="chronaris_opt",
                lag_window_points=3,
                residual_mode="raw_window_stats",
            )

            self.assertIn("chronaris_opt", frames)
            self.assertIn("chronaris_opt_no_causal_mask", frames)
            self.assertIn("chronaris_opt", diagnostics)
            self.assertIn("chronaris_opt_no_causal_mask", diagnostics)
            feature_keys = set(frames["chronaris_opt"]["feature_values"].iloc[0])
            self.assertTrue(any(key.startswith("residual__") for key in feature_keys))
            self.assertFalse(any("label" in key.lower() for key in feature_keys))
            no_mask_values = frames["chronaris_opt_no_causal_mask"]["feature_values"].iloc[0]
            self.assertEqual(no_mask_values["diag__causal_residual_gate"], 0.0)

    def test_optimized_private_benchmark_supports_target_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest, f_manifest = _write_pair(root)
            _reverse_second_pilot_raw_summary(f_manifest)

            result = run_task_eval_private_benchmark(
                StageIPrivateBenchmarkConfig(
                    run_id="private-optimized-smoke",
                    e_run_manifest_path=str(e_manifest),
                    f_run_manifest_path=str(f_manifest),
                    output_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    enable_optimized_chronaris=True,
                    export_optimized_package=True,
                    target_variant_name="chronaris_opt",
                    lag_window_points=3,
                    residual_mode="raw_window_stats",
                    max_deep_folds=1,
                    deep_epochs=1,
                    deep_batch_size=4,
                )
            )

            summary = json.loads(Path(result.benchmark_summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["target_variant_name"], "chronaris_opt")
            self.assertIn("chronaris_opt", summary["variant_order"])
            self.assertIn("chronaris_opt_no_causal_mask", summary["variant_order"])
            self.assertTrue(Path(result.optimization_report_path).exists())
            self.assertTrue(Path(result.optimized_candidate_summary_path).exists())
            self.assertTrue(Path(result.optimized_candidate_metrics_path).exists())
            self.assertTrue(Path(result.optimized_candidate_package_path).exists())
            self.assertTrue(Path(result.optimized_candidate_package_report_path).exists())
            self.assertTrue(Path(result.thesis_task_manifest_path).exists())
            self.assertTrue(Path(result.thesis_task_summary_path).exists())
            self.assertIn(
                "t1_chronaris_opt_beats_chronaris_opt_no_causal_mask",
                summary["criterion_details"],
            )
            self.assertIn("evidence_layers", summary)
            self.assertIn("proxy_evidence", summary["evidence_layers"])
            self.assertIn("thesis_task_evidence", summary["evidence_layers"])
            self.assertEqual(
                summary["evidence_layers"]["proxy_evidence"]["summary"]["evidence_layer"],
                "proxy_evidence",
            )
            self.assertEqual(
                summary["evidence_layers"]["thesis_task_evidence"]["summary"]["benchmark_role"],
                "thesis_task_weak_label_benchmark",
            )
            package = json.loads(Path(result.optimized_candidate_package_path).read_text(encoding="utf-8"))
            self.assertEqual(package["target_variant_name"], "chronaris_opt")
            self.assertEqual(package["tasks"][TASK_MANEUVER]["head_family"], "class_balanced_threshold")
            recommended_head = package["tasks"][TASK_RESPONSE]["recommended_head"]
            self.assertIn(recommended_head, package["tasks"][TASK_RESPONSE]["available_heads"])
            self.assertEqual(
                package["tasks"][TASK_RETRIEVAL]["head_family"],
                "chronaris_time_residual_retrieval",
            )

            t1 = summary["tasks"][TASK_MANEUVER]["variants"]
            self.assertGreater(
                t1["chronaris_opt"]["best_metrics"]["macro_f1"],
                t1["chronaris_opt_no_causal_mask"]["best_metrics"]["macro_f1"],
            )
            t3 = summary["tasks"][TASK_RETRIEVAL]["variants"]
            self.assertGreater(
                t3["chronaris_opt"]["top1_accuracy"],
                t3["chronaris_opt_no_causal_mask"]["top1_accuracy"],
            )
            self.assertGreater(
                t3["chronaris_opt"]["top1_accuracy"],
                t3["naive_sync"]["top1_accuracy"],
            )

    def test_custom_target_variant_name_replaces_g_min_conclusion_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest, f_manifest = _write_pair(root)

            result = run_task_eval_private_benchmark(
                StageIPrivateBenchmarkConfig(
                    run_id="private-custom-optimized-smoke",
                    e_run_manifest_path=str(e_manifest),
                    f_run_manifest_path=str(f_manifest),
                    output_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    enable_optimized_chronaris=True,
                    target_variant_name="chronaris_opt_custom",
                    max_deep_folds=1,
                    deep_epochs=1,
                    deep_batch_size=4,
                )
            )

            summary = json.loads(Path(result.benchmark_summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["conclusion"]["target_variant_name"], "chronaris_opt_custom")
            self.assertIn("chronaris_opt_custom", summary["tasks"][TASK_MANEUVER]["variants"])
            self.assertIn(
                "t1_chronaris_opt_custom_beats_module_baselines",
                summary["conclusion"]["criterion_details"],
            )
            self.assertNotIn(
                "t1_g_min_beats_module_baselines",
                summary["conclusion"]["criterion_details"],
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_private_benchmark_deep_models_support_cuda_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest, f_manifest = _write_pair(root)

            result = run_task_eval_private_benchmark(
                StageIPrivateBenchmarkConfig(
                    run_id="private-cuda-smoke",
                    e_run_manifest_path=str(e_manifest),
                    f_run_manifest_path=str(f_manifest),
                    output_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    deep_model_names=("mult",),
                    max_deep_folds=1,
                    deep_epochs=1,
                    deep_batch_size=4,
                    device="cuda",
                )
            )

            summary = json.loads(Path(result.benchmark_summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["deep_runtime_device"], "cuda")
            self.assertEqual(
                summary["tasks"][TASK_MANEUVER]["deep_models"]["mult"]["status"],
                "completed",
            )


def _write_pair(root: Path) -> tuple[Path, Path]:
    e_manifest = _write_private_feature_export_run(
        root,
        run_name="feature-export-e",
        amplitude_scale=0.8,
        physics_enabled=False,
    )
    f_manifest = _write_private_feature_export_run(
        root,
        run_name="feature-export-f",
        amplitude_scale=1.4,
        physics_enabled=True,
    )
    return e_manifest, f_manifest


def _reverse_second_pilot_raw_summary(run_manifest_path: Path) -> None:
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    for sortie_manifest_path in run_manifest["sortie_manifest_paths"].values():
        sortie_manifest = json.loads(Path(sortie_manifest_path).read_text(encoding="utf-8"))
        for view_id, view_manifest_path in sortie_manifest["view_manifest_paths"].items():
            if not view_id.endswith("pilot_10003"):
                continue
            view_manifest = json.loads(Path(view_manifest_path).read_text(encoding="utf-8"))
            raw_path = Path(view_manifest["artifact_paths"]["raw_window_summary_jsonl"])
            rows = [
                json.loads(line)
                for line in raw_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            reversed_stats = list(reversed(rows))
            rewritten = []
            for row, source in zip(rows, reversed_stats, strict=True):
                updated = dict(row)
                updated["physiology_feature_stats"] = source["physiology_feature_stats"]
                updated["vehicle_feature_stats"] = source["vehicle_feature_stats"]
                rewritten.append(updated)
            raw_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rewritten),
                encoding="utf-8",
            )


if __name__ == "__main__":
    unittest.main()
