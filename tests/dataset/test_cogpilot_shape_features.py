import numpy as np
import pytest
from chronaris.dataset.cogpilot_shape_features import shape_native_features, shape_scale_features, shape_scale_floors
from chronaris.dataset.cogpilot_history_features import QUERY_SECONDS


def test_step_preserves_oscillation_prefix_and_positive_gain():
    t = np.arange(0, 30, 1/256); x = np.sin(2*np.pi*t); x[t > 10] += 100
    a = shape_native_features(t, x, 0, QUERY_SECONDS, (.5, 8))
    assert a[0][-1, 1] > .5
    modified = x.copy(); modified[t >= 15] = 1e6
    b = shape_native_features(t, modified, 0, QUERY_SECONDS, (.5, 8))
    np.testing.assert_array_equal(a[0][:48], b[0][:48])
    gain = shape_native_features(t, x*3+7, 0, QUERY_SECONDS, (.5, 8))
    np.testing.assert_allclose(gain[0][:, :3], a[0][:, :3]*3, atol=1e-10)
    values = np.tile(a[0], (1, 1, 4)); masks = np.tile(a[1], (1, 1, 4))
    out = shape_scale_features(values, masks, np.ones((1, 4)), np.ones(4))
    assert np.isfinite(out).all() and out[0, -1, 1] > 0
    # A late gap stays unavailable, rather than being silently filled.
    short = shape_native_features(t[t < 29.7], x[t < 29.7], 0, QUERY_SECONDS, (.5, 8))
    assert not short[1][-1].any()


def test_floor_is_fit_only_and_rejects_degenerate_fit():
    scales = np.array([[1,2,3,4], [2,3,4,5], [1e9]*4], float)
    expected = shape_scale_floors(scales, [0,1])
    scales[2] *= 10
    assert expected == shape_scale_floors(scales, [0,1])
    with pytest.raises(ValueError):
        shape_scale_floors(np.full((3,4), 1e-6), [0,1])
