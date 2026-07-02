"""Tests for Stage I T3 contrastive retrieval helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.models.alignment.contrastive import (  # noqa: E402
    RetrievalCandidate,
    build_positive_index_tensor,
    info_nce_loss,
    retrieval_metrics_from_scores,
    validate_same_sortie_cross_pilot_policy,
)


class StageIContrastiveRetrievalTest(unittest.TestCase):
    def test_info_nce_loss_is_finite(self) -> None:
        anchors = torch.randn(4, 8)
        candidates = anchors + 0.01 * torch.randn(4, 8)
        positive = torch.arange(4)
        loss = info_nce_loss(anchors, candidates, positive, temperature=0.1)
        self.assertTrue(torch.isfinite(loss).item())

    def test_candidate_policy_requires_same_sortie_cross_pilot_positive(self) -> None:
        validate_same_sortie_cross_pilot_policy(
            (
                RetrievalCandidate("a", "b", same_sortie_cross_pilot=True, is_positive=True),
                RetrievalCandidate("a", "c", same_sortie_cross_pilot=False, is_positive=False),
            )
        )
        with self.assertRaises(ValueError):
            validate_same_sortie_cross_pilot_policy(
                (RetrievalCandidate("a", "b", same_sortie_cross_pilot=False, is_positive=True),)
            )

    def test_retrieval_metrics_and_positive_index_mapping(self) -> None:
        positive = build_positive_index_tensor(
            ("a", "b"),
            ("x", "b_pos", "a_pos"),
            {"a": "a_pos", "b": "b_pos"},
        )
        self.assertEqual(positive.tolist(), [2, 1])
        scores = torch.tensor([[0.1, 0.2, 0.9], [0.1, 0.8, 0.3]])
        metrics = retrieval_metrics_from_scores(scores, positive)
        self.assertEqual(metrics["top1"], 1.0)
        self.assertGreater(metrics["positive_negative_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
