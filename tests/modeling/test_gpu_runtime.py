"""Tests for task evaluation public fusion GPU runtime helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import torch

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.modeling.common.gpu_runtime import (  # noqa: E402
    choose_auto_batch_size,
    completed_profile_key,
    compute_fold_normalization_stats,
    get_train_batch,
    make_grad_scaler,
    prepare_fold_tensors,
    remaining_profile_keys,
    resolve_amp_runtime,
    validate_optimization_summary_schema,
)


class StageIGPURuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        values = np.asarray(
            [
                [[0.0, 10.0], [1.0, 11.0]],
                [[2.0, 12.0], [3.0, 13.0]],
                [[100.0, 110.0], [101.0, 111.0]],
            ],
            dtype=np.float32,
        )
        self.modality_arrays = {"pilot": values, "vehicle": values + 1.0}
        self.modality_masks = {
            "pilot": np.ones((3, 2), dtype=np.float32),
            "vehicle": np.ones((3, 2), dtype=np.float32),
        }
        self.time_axis = np.zeros((3, 2), dtype=np.float32)
        self.targets = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        self.train_indices = np.asarray([0, 1], dtype=int)

    def test_tensor_cache_stats_use_train_indices_only(self) -> None:
        stats = compute_fold_normalization_stats(
            modality_arrays=self.modality_arrays,
            modality_masks=self.modality_masks,
            ordered_modalities=("pilot", "vehicle"),
            train_indices=self.train_indices,
        )
        self.assertAlmostEqual(float(stats.modality_mean["pilot"][0]), 1.5)
        self.assertLess(float(stats.modality_mean["pilot"][0]), 10.0)
        self.assertEqual(stats.train_indices.tolist(), [0, 1])

    def test_tensor_cache_modes_preserve_batch_shapes(self) -> None:
        off = prepare_fold_tensors(
            modality_arrays=self.modality_arrays,
            modality_masks=self.modality_masks,
            time_axis=self.time_axis,
            ordered_modalities=("pilot", "vehicle"),
            train_indices=self.train_indices,
            targets=self.targets,
            requested_mode="off",
            device="cpu",
        )
        cpu = prepare_fold_tensors(
            modality_arrays=self.modality_arrays,
            modality_masks=self.modality_masks,
            time_axis=self.time_axis,
            ordered_modalities=("pilot", "vehicle"),
            train_indices=self.train_indices,
            targets=self.targets,
            requested_mode="cpu",
            device="cpu",
        )
        off_batch = get_train_batch(off, np.asarray([0, 2]), device="cpu")
        cpu_batch = get_train_batch(cpu, np.asarray([0, 2]), device="cpu")
        self.assertEqual(off_batch[0]["pilot"].shape, cpu_batch[0]["pilot"].shape)
        self.assertEqual(off_batch[2].shape, cpu_batch[2].shape)
        self.assertEqual(off.tensor_cache_mode, "off")
        self.assertEqual(cpu.tensor_cache_mode, "cpu")

    def test_auto_batch_size_falls_back_on_simulated_oom(self) -> None:
        def _try(batch_size: int) -> None:
            if batch_size > 128:
                raise RuntimeError("CUDA out of memory")

        selected, attempts = choose_auto_batch_size(
            candidates=(512, 256, 128),
            try_batch=_try,
            fallback=64,
        )
        self.assertEqual(selected, 128)
        self.assertEqual([row["status"] for row in attempts], ["oom", "oom", "ok"])

    def test_resume_completed_fold_filter(self) -> None:
        key = completed_profile_key(
            dataset_id="nasa_csm",
            candidate_id="candidate",
            seed=42,
            track="objective",
            evaluation_group="combined",
            split_group="subject_10",
        )
        self.assertEqual(remaining_profile_keys([key, "other"], {key}, skip_completed=True), ["other"])
        self.assertEqual(remaining_profile_keys([key], {key}, skip_completed=False), [key])

    def test_optimization_summary_schema_validation(self) -> None:
        payload = {
            "run_id": "r",
            "base_p28_run_id": "base",
            "status": "completed",
            "generated_at_utc": "2026-07-01T00:00:00Z",
            "gpu_name": "gpu",
            "torch_version": "x",
            "cuda_available": False,
            "cuda_version": None,
            "optimization_enabled": {},
            "throughput": {},
            "timing": {},
            "memory": {},
            "utilization": {},
            "fallbacks": [],
            "resume_command": "cmd",
        }
        ok, missing = validate_optimization_summary_schema(payload)
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_prepare_fold_tensors_does_not_mutate_split_ids(self) -> None:
        train_copy = self.train_indices.copy()
        _prepared = prepare_fold_tensors(
            modality_arrays=self.modality_arrays,
            modality_masks=self.modality_masks,
            time_axis=self.time_axis,
            ordered_modalities=("pilot", "vehicle"),
            train_indices=self.train_indices,
            targets=self.targets,
            requested_mode="cpu",
            device="cpu",
        )
        np.testing.assert_array_equal(self.train_indices, train_copy)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_amp_smoke_forward_backward_on_cuda(self) -> None:
        amp = resolve_amp_runtime(requested_mode="bf16", device="cuda", grad_scaler=True)
        scaler = make_grad_scaler(amp, device="cuda")
        model = torch.nn.Linear(4, 2).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.randn(8, 4, device="cuda")
        targets = torch.randint(0, 2, (8,), device="cuda")
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device="cuda"):
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        self.assertTrue(torch.isfinite(loss.detach()).item())


if __name__ == "__main__":
    unittest.main()
