import numpy as np
import pytest

from chronaris.dataset.cogpilot_performance import history_queries, interval_mean


def test_target_and_persistence_have_disjoint_time_support():
    t = np.arange(-6., 7., .25)
    y = np.where(t > 0, 8., 3.)
    y[t == 0] = 999.
    past, pt, _ = interval_mean(t, y, -5., 0., include_end=False)
    future, ft, _ = interval_mean(t, y, 0., 5., include_end=True)
    assert past == 3 and future == 8
    assert np.all(pt < 0) and np.all(ft > 0) and ft.max() <= 5


def test_nonuniform_cells_and_missing_support():
    result, _, _ = interval_mean(np.array([.2, .6, 1.]), np.array([1., 3., 5.]), 0., 1., include_end=True)
    assert result == pytest.approx(.4 + 1.2 + 1.)
    with pytest.raises(ValueError, match='target_gap'):
        interval_mean(np.array([.2, 4., 5.]), np.ones(3), 0., 5., include_end=True)


def test_future_values_and_deletion_leave_queries_unchanged():
    t = np.arange(0., 35., .1)
    x = np.sin(t)[:, None]
    before = history_queries(t, x, 0., 30.)
    changed = x.copy(); changed[t >= 30] = 1e6
    modified = history_queries(t, changed, 0., 30.)
    deleted = history_queries(t[t < 30], x[t < 30], 0., 30.)
    for expected, a, b in zip(before, modified, deleted):
        np.testing.assert_array_equal(expected, a)
        np.testing.assert_array_equal(expected, b)
    assert np.nanmax(before[1]) < 30


def test_reversed_input_is_not_silently_reordered():
    with pytest.raises(ValueError, match='input_time_order'):
        history_queries(np.array([0., 2., 1., 3.]), np.ones((4, 1)), 0., 30.)
