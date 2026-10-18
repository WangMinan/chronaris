import numpy as np
import pytest
from chronaris.dataset.cogpilot_history_features import (
    QUERY_SECONDS, geodetic_ned, native_dynamic_features, strict_history_queries, performance_row_identity,
)


def test_geodetic_origin_and_vertical_sign():
    np.testing.assert_allclose(geodetic_ned(42., -71., 30., (-71., 42., 30.)), 0, atol=1e-9)
    np.testing.assert_allclose(geodetic_ned(42., -71., 130., (-71., 42., 30.)), [0, 0, -100], atol=1e-8)


def test_serialization_identity_is_exact_and_rejects_wrong_rows():
    source = np.array([738193.7966481076, 738193.7966500])
    np.testing.assert_array_equal(performance_row_identity(source, np.array([738193.796648108])), [0])
    with pytest.raises(ValueError, match='does not match'):
        performance_row_identity(source, np.array([738193.796648109]))
    with pytest.raises(ValueError, match='ambiguous'):
        performance_row_identity(np.array([source[0], np.nextafter(source[0], np.inf)]), np.array([738193.796648108]))


def test_strict_query_never_reads_cutoff():
    t = np.arange(0., 31., .25); x = t[:, None].copy(); x[t >= 30] = 999
    values, mask, age = strict_history_queries(t, x, 0., QUERY_SECONDS)
    assert values[-1, 0] == 29.75 and age[-1, 0] == .25 and mask[-1, 0]


def test_native_dynamics_prefix_and_baseline_gain_invariance():
    t = np.arange(0., 31., 1/128); x = np.sin(t*2*np.pi*2)+.2*np.sin(t*2*np.pi*7)
    base = native_dynamic_features(t, x, 0., QUERY_SECONDS, (.5, 8.))
    shifted = native_dynamic_features(t, x*3+400, 0., QUERY_SECONDS, (.5, 8.))
    np.testing.assert_allclose(base[0], shifted[0], atol=1e-5, rtol=1e-5)
    changed = x.copy(); changed[t >= 15] = 1e5
    future = native_dynamic_features(t, changed, 0., QUERY_SECONDS, (.5, 8.))
    deleted = native_dynamic_features(t[t < 15], x[t < 15], 0., QUERY_SECONDS, (.5, 8.))
    for other in (future, deleted):
        np.testing.assert_array_equal(base[0][QUERY_SECONDS <= 15], other[0][QUERY_SECONDS <= 15])
    assert not base[1][QUERY_SECONDS < 2].any()
    assert base[1].mean() >= .9


def test_constant_and_frequency_response():
    t = np.arange(0., 31., 1/128)
    constant = native_dynamic_features(t, np.full(len(t), 4.), 0., QUERY_SECONDS, (5., 40.))
    assert np.isfinite(constant[0]).all() and np.max(np.abs(constant[0])) == 0
    low = native_dynamic_features(t, np.sin(2*np.pi*t), 0., QUERY_SECONDS, (5., 40.))[0]
    high = native_dynamic_features(t, np.sin(2*np.pi*15*t), 0., QUERY_SECONDS, (5., 40.))[0]
    assert high[-16:, 2].mean() > 5*low[-16:, 2].mean()
