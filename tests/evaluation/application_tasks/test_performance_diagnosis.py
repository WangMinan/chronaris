import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chronaris.evaluation.application_tasks.performance_pilot_diagnosis import affine_readout, error_stats


def test_affine_decomposition_includes_scaling_and_intercept():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(15, 4))*np.array([.1, 2., 10., 30.])+5
    y = x @ np.array([1., -2., .3, .1])+12
    fitted = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(x, y)
    weight, bias = affine_readout(fitted)
    np.testing.assert_allclose(x @ weight+bias, fitted.predict(x), atol=1e-12)


def test_error_bias_variance_identity():
    y = np.arange(10.)
    p = 2*y+3
    stats = error_stats(y, p)
    np.testing.assert_allclose(stats['rmse']**2, stats['bias']**2+stats['centered_rmse']**2)
