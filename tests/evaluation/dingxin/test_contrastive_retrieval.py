"""Tests for task evaluation T3 contrastive retrieval helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.models.alignment.contrastive import (  # noqa: E402
    HardNegativeSamplerConfig,
    RetrievalCandidate,
    build_positive_index_tensor,
    info_nce_loss,
    retrieval_metrics_from_scores,
    stratified_hard_negative_samples,
    supervised_contrastive_margin_loss,
    validate_same_sortie_cross_pilot_policy,
)


class StageIContrastiveRetrievalTest(unittest.TestCase):
    def test_info_nce_loss_is_finite(self) -> None:
        anchors = torch.randn(4, 8)
        candidates = anchors + 0.01 * torch.randn(4, 8)
        positive = torch.arange(4)
        weights = torch.ones(4, 4)
        weights[:, 1:] = 2.0
        loss = info_nce_loss(anchors, candidates, positive, temperature=0.1, candidate_weights=weights)
        self.assertTrue(torch.isfinite(loss).item())
        scores = anchors @ candidates.T
        margin = supervised_contrastive_margin_loss(scores, positive, margin=0.1, negative_weights=weights)
        self.assertTrue(torch.isfinite(margin).item())

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

    def test_hard_negative_sampler_respects_same_sortie_cross_pilot_policy(self) -> None:
        rows = (
            {"sample_id": "a", "sortie_id": "s1", "pilot_id": "p1", "window_index": 10},
            {"sample_id": "b", "sortie_id": "s1", "pilot_id": "p2", "window_index": 10},
            {"sample_id": "c", "sortie_id": "s1", "pilot_id": "p2", "window_index": 11},
            {"sample_id": "d", "sortie_id": "s1", "pilot_id": "p2", "window_index": 30},
            {"sample_id": "e", "sortie_id": "s2", "pilot_id": "p3", "window_index": 1},
        )
        samples = stratified_hard_negative_samples(
            rows,
            anchor_index=0,
            positive_index=1,
            pool_indices=range(len(rows)),
            config=HardNegativeSamplerConfig(near_window_radius=2, max_per_kind=1),
        )
        kinds = {sample.negative_kind for sample in samples}
        self.assertIn("hard_negative_same_sortie_near_window", kinds)
        self.assertIn("hard_negative_same_sortie_far_window", kinds)
        self.assertIn("easy_negative_different_sortie", kinds)
        self.assertNotIn(1, {sample.candidate_index for sample in samples})


if __name__ == "__main__":
    unittest.main()
