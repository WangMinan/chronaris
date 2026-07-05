"""Tests for task evaluation optimized task-aware heads."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.models.alignment.task_heads_v2 import (  # noqa: E402
    PhysiologyResponseResidualRegressionHead,
    VehicleDominantAuxiliaryClassificationHead,
    class_balanced_focal_loss,
    gate_regularization_loss,
)
from chronaris.models.alignment.task_losses_v2 import TrainFoldTargetTransform  # noqa: E402


class StageITaskHeadsV2Test(unittest.TestCase):
    def test_t1_vehicle_auxiliary_head_shape_and_gate_summary(self) -> None:
        head = VehicleDominantAuxiliaryClassificationHead(
            vehicle_dim=4,
            fused_dim=12,
            output_dim=3,
            hidden_dim=8,
        )
        output = head(
            vehicle_states=torch.randn(5, 7, 4),
            fused_states=torch.randn(5, 7, 12),
        )
        self.assertEqual(tuple(output.logits.shape), (5, 3))
        self.assertEqual(tuple(output.gate.shape), (5, 1))
        summary = output.contribution_summary().to_jsonable()
        self.assertIn("gate_mean", summary)
        self.assertTrue(torch.isfinite(gate_regularization_loss(output.gate)).item())

    def test_t1_class_balanced_focal_loss_is_finite(self) -> None:
        logits = torch.randn(6, 3)
        targets = torch.tensor([0, 0, 0, 1, 1, 2])
        weights = torch.tensor([0.5, 1.0, 2.0])
        loss = class_balanced_focal_loss(
            logits,
            targets,
            class_weights=weights,
            gamma=2.0,
            label_smoothing=0.05,
        )
        self.assertTrue(torch.isfinite(loss).item())

    def test_t2_residual_head_decomposes_prediction(self) -> None:
        head = PhysiologyResponseResidualRegressionHead(
            physiology_dim=4,
            vehicle_dim=4,
            fused_dim=12,
            output_dim=1,
            hidden_dim=8,
            predict_uncertainty=True,
        )
        output = head(
            physiology_states=torch.randn(6, 5, 4),
            vehicle_states=torch.randn(6, 5, 4),
            fused_states=torch.randn(6, 5, 12),
        )
        expected = output.persistence + output.vehicle_excitation + output.interaction
        self.assertTrue(torch.allclose(output.prediction, expected))
        self.assertIsNotNone(output.uncertainty)
        self.assertIn("interaction_abs_mean", output.decomposition_summary())

    def test_target_transform_uses_supplied_train_targets_only(self) -> None:
        train_targets = torch.tensor([1.0, 2.0, 3.0])
        test_targets = torch.tensor([1000.0])
        transform = TrainFoldTargetTransform.from_train_targets(train_targets)
        self.assertLess(float(transform.mean), float(test_targets.item()))
        recovered = transform.inverse(transform.transform(train_targets))
        self.assertTrue(torch.allclose(recovered, train_targets, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
