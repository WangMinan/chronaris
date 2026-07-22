from __future__ import annotations

import numpy as np

from chronaris.evaluation.representation_diagnostics import (
    compute_representation_geometry,
    geometry_rows,
)


def test_effective_rank_is_full_for_isotropic_and_one_for_rank_collapse() -> None:
    rng = np.random.default_rng(17)
    isotropic = rng.standard_normal((512, 32))
    geometry_iso = compute_representation_geometry(isotropic)
    # Isotropic Gaussian: effective rank should be close to D=32.
    assert geometry_iso.effective_rank > 28.0
    assert geometry_iso.dimension_utilization > 0.9

    collapsed = rng.standard_normal((512, 1)) * np.ones((1, 32))
    geometry_col = compute_representation_geometry(collapsed)
    # Rank-1 collapse: effective rank ~1, only one dimension utilized.
    assert geometry_col.effective_rank < 1.2
    assert geometry_col.dimension_utilization <= 1.0 / 32.0 + 1e-6


def test_geometry_rows_round_trip() -> None:
    rng = np.random.default_rng(3)
    geom = compute_representation_geometry(rng.standard_normal((64, 16)), seed=3)
    rows = geometry_rows([("chronaris_safe_lag", geom)])
    assert len(rows) == 1
    assert rows[0]["label"] == "chronaris_safe_lag"
    assert rows[0]["feature_dim"] == 16
    assert rows[0]["sample_count"] == 64
    assert rows[0]["effective_rank"] > 0.0
