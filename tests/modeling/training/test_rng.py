from __future__ import annotations

import random

import numpy as np
import torch

from chronaris.modeling.training.rng import (
    canonical_training_state_sha256,
    capture_rng_state,
    isolated_training_rng,
)


def test_isolated_training_rng_reproduces_and_restores_callers() -> None:
    torch.manual_seed(91)
    np.random.seed(91)
    random.seed(91)
    outer = capture_rng_state()

    def draw():
        with isolated_training_rng(17):
            return torch.rand(4), np.random.random(4), tuple(random.random() for _ in range(4))

    first = draw()
    torch.rand(7)
    np.random.random(7)
    random.random()
    second = draw()

    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]
    restored = capture_rng_state()
    assert canonical_training_state_sha256(outer) != canonical_training_state_sha256(restored)


def test_canonical_training_state_hash_ignores_mapping_order() -> None:
    left = {"weight": torch.arange(4), "step": 2}
    right = {"step": 2, "weight": torch.arange(4)}
    assert canonical_training_state_sha256(left) == canonical_training_state_sha256(right)
