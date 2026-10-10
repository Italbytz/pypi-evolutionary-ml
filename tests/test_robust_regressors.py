"""Unit tests for evolutionary robust regressors."""

import numpy as np
import pytest
from sklearn.base import clone

from evolutionary_ml.robust import (
    EvolutionaryRobustRegressor,
    EvolutionaryLTSRegressor,
    EvolutionaryLMSRegressor,
    EvolutionaryLTARegressor,
    EvolutionaryLQDRegressor,
)


@pytest.fixture
def synthetic_outlier_data():
    """Create clean synthetic linear data with contaminated outliers."""
    rng = np.random.RandomState(42)
    n = 60
    X = rng.uniform(-2, 2, size=(n, 2))
    # True relation: y = 3.0 + 1.5 * x0 - 2.0 * x1 + noise
    y = 3.0 + 1.5 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, 0.05, size=n)

    # Contaminate 20% of data with heavy leverage outliers
    outlier_idx = rng.choice(n, size=12, replace=False)
    y[outlier_idx] += rng.uniform(20, 50, size=len(outlier_idx))

    return X, y, outlier_idx


def test_lts_recovers_ground_truth(synthetic_outlier_data):
    X, y, outlier_idx = synthetic_outlier_data
    model = EvolutionaryLTSRegressor(
        max_iter=1000,
        c_steps=2,
        random_state=42,
    )
    model.fit(X, y)

    # Intercept ~ 3.0, coef ~ [1.5, -2.0]
    np.testing.assert_allclose(model.intercept_, 3.0, atol=0.25)
    np.testing.assert_allclose(model.coef_, [1.5, -2.0], atol=0.25)

    # Inliers should not include the heavy outliers
    inliers = np.where(model.inlier_mask_)[0]
    overlap = np.intersect1d(inliers, outlier_idx)
    assert len(overlap) <= 1


def test_determinism(synthetic_outlier_data):
    X, y, _ = synthetic_outlier_data
    m1 = EvolutionaryLTSRegressor(max_iter=200, random_state=123)
    m2 = EvolutionaryLTSRegressor(max_iter=200, random_state=123)
    m1.fit(X, y)
    m2.fit(X, y)

    np.testing.assert_array_equal(m1.coef_, m2.coef_)
    assert m1.intercept_ == m2.intercept_
    assert m1.best_criterion_ == m2.best_criterion_


def test_no_intercept(synthetic_outlier_data):
    X, y, _ = synthetic_outlier_data
    model = EvolutionaryLTSRegressor(
        fit_intercept=False,
        max_iter=200,
        random_state=42,
    )
    model.fit(X, y)

    assert model.intercept_ == 0.0
    assert model.coef_.shape == (2,)


def test_predict_and_score(synthetic_outlier_data):
    X, y, _ = synthetic_outlier_data
    model = EvolutionaryLTSRegressor(max_iter=200, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)

    r2 = model.score(X, y)
    assert isinstance(r2, float)


def test_clone_compatibility():
    model = EvolutionaryLTSRegressor(max_iter=500, quantile=0.75, c_steps=1)
    cloned = clone(model)
    assert cloned.max_iter == 500
    assert cloned.quantile == 0.75
    assert cloned.c_steps == 1


def test_invalid_parameters():
    with pytest.raises(ValueError, match="max_iter"):
        EvolutionaryLTSRegressor(max_iter=0).fit(np.ones((10, 2)), np.ones(10))

    with pytest.raises(ValueError, match="c_steps"):
        EvolutionaryLTSRegressor(c_steps=-1).fit(np.ones((10, 2)), np.ones(10))

    with pytest.raises(ValueError, match="prob_move"):
        EvolutionaryLTSRegressor(prob_move=1.5).fit(np.ones((10, 2)), np.ones(10))

    with pytest.raises(ValueError, match="quantile"):
        EvolutionaryLTSRegressor(quantile=0.2).fit(np.ones((10, 2)), np.ones(10))


def test_unified_regressor():
    X = np.random.randn(20, 2)
    y = np.random.randn(20)

    for loss in ["lts", "lms", "lta", "lqd"]:
        reg = EvolutionaryRobustRegressor(loss=loss, max_iter=50, random_state=42)
        reg.fit(X, y)
        assert hasattr(reg, "coef_")
        assert reg.coef_.shape == (2,)
