"""Tests for Big Data nested subsampling / partitioning in evolutionary_ml."""

import time
import numpy as np
import pytest

from evolutionary_ml.robust import EvolutionaryLTSRegressor, EvolutionaryRobustRegressor


def test_subsampling_validation_errors():
    with pytest.raises(ValueError, match="subsample_threshold must be >= 1"):
        reg = EvolutionaryLTSRegressor(subsample_threshold=0)
        reg.fit(np.ones((20, 2)), np.ones(20))

    with pytest.raises(ValueError, match="subsample_size must be >= 1"):
        reg = EvolutionaryLTSRegressor(subsample_size=0)
        reg.fit(np.ones((20, 2)), np.ones(20))

    with pytest.raises(ValueError, match="n_subsamples must be >= 1"):
        reg = EvolutionaryLTSRegressor(n_subsamples=0)
        reg.fit(np.ones((20, 2)), np.ones(20))


def test_nested_subsampling_recovery():
    """Verify that nested subsampling on n=1500 recovers true coefficients."""
    rng = np.random.RandomState(42)
    n = 1500
    p = 4
    X = rng.randn(n, p)
    true_beta = np.array([2.0, -1.5, 0.8, 3.2])
    true_intercept = 1.0
    y = true_intercept + X @ true_beta + rng.randn(n) * 0.1

    # Contaminate with 30% bad leverage points
    n_out = int(0.30 * n)
    out_idx = rng.choice(n, size=n_out, replace=False)
    X[out_idx, :] += rng.uniform(20.0, 40.0, size=(n_out, p))
    y[out_idx] += rng.uniform(-100.0, -50.0, size=n_out)

    reg = EvolutionaryLTSRegressor(
        subsample_threshold=400,
        subsample_size=300,
        n_subsamples=4,
        max_iter=60,
        c_steps=2,
        reweight=True,
        random_state=42,
    )
    reg.fit(X, y)

    coef_err = np.linalg.norm(reg.coef_ - true_beta)
    intercept_err = abs(reg.intercept_ - true_intercept)

    assert coef_err < 0.15, f"High coefficient error: {coef_err}"
    assert intercept_err < 0.15, f"High intercept error: {intercept_err}"

    # Verify outlier detection precision: bad points should not be inliers
    bad_inliers = np.sum(reg.inlier_mask_[out_idx])
    assert bad_inliers == 0, f"Expected 0 outliers as inliers, got {bad_inliers}"


def test_nested_subsampling_speed_on_large_n():
    """Verify that partitioned subsampling scales efficiently on n=5000."""
    rng = np.random.RandomState(42)
    n = 5000
    p = 3
    X = rng.randn(n, p)
    beta = np.array([1.0, 2.0, -1.0])
    y = X @ beta + rng.randn(n) * 0.1

    # 20% outliers
    n_out = int(0.20 * n)
    out_idx = rng.choice(n, size=n_out, replace=False)
    y[out_idx] += 30.0

    t0 = time.perf_counter()
    reg = EvolutionaryLTSRegressor(
        subsample_threshold=500,
        subsample_size=400,
        n_subsamples=4,
        max_iter=40,
        random_state=42,
    )
    reg.fit(X, y)
    duration = time.perf_counter() - t0

    assert duration < 2.0, f"Nested subsampling took too long: {duration:.2f}s"
    assert np.linalg.norm(reg.coef_ - beta) < 0.15
