"""Benchmark test on Brownlee's stackloss dataset."""

import numpy as np
import pytest

from evolutionary_ml.robust import (
    EvolutionaryLTSRegressor,
    EvolutionaryLMSRegressor,
    EvolutionaryLTARegressor,
    EvolutionaryLQDRegressor,
)

# Standard stackloss data (Brownlee 1965)
# Columns 1-3: Air flow, Water temp, Acid conc. Column 4: Stack loss
STACKLOSS_DATA = np.array([
    [80.0, 27.0, 89.0, 42.0],
    [80.0, 27.0, 88.0, 37.0],
    [75.0, 25.0, 90.0, 37.0],
    [62.0, 24.0, 87.0, 28.0],
    [62.0, 22.0, 87.0, 18.0],
    [62.0, 23.0, 87.0, 18.0],
    [62.0, 24.0, 93.0, 19.0],
    [62.0, 24.0, 93.0, 20.0],
    [58.0, 23.0, 87.0, 15.0],
    [58.0, 18.0, 80.0, 14.0],
    [58.0, 18.0, 89.0, 14.0],
    [58.0, 17.0, 88.0, 13.0],
    [58.0, 18.0, 82.0, 11.0],
    [58.0, 19.0, 93.0, 12.0],
    [50.0, 18.0, 89.0, 8.0],
    [50.0, 18.0, 86.0, 7.0],
    [50.0, 19.0, 72.0, 8.0],
    [50.0, 19.0, 79.0, 8.0],
    [50.0, 20.0, 80.0, 9.0],
    [56.0, 20.0, 82.0, 15.0],
    [70.0, 20.0, 91.0, 15.0],
], dtype=np.float64)

X_stackloss = STACKLOSS_DATA[:, :3]
y_stackloss = STACKLOSS_DATA[:, 3]


def test_stackloss_lts():
    """Test EvolutionaryLTSRegressor on stackloss data."""
    # Classical outliers are 1, 3, 4, 21 (1-based), i.e., 0, 2, 3, 20 (0-based)
    # Pure trimming (reweight=False): exactly h=13 inliers
    model_pure = EvolutionaryLTSRegressor(
        max_iter=1500,
        c_steps=2,
        reweight=False,
        random_state=42,
    )
    model_pure.fit(X_stackloss, y_stackloss)
    assert np.sum(model_pure.inlier_mask_) == 13
    assert model_pure.best_criterion_ < 10.0

    # Reweighted (reweight=True, default): adaptive inlier recovery
    model_rw = EvolutionaryLTSRegressor(
        max_iter=1500,
        c_steps=2,
        reweight=True,
        random_state=42,
    )
    model_rw.fit(X_stackloss, y_stackloss)
    assert np.sum(model_rw.inlier_mask_) >= 15
    assert hasattr(model_rw, "coef_")
    assert hasattr(model_rw, "intercept_")


def test_stackloss_lms():
    """Test EvolutionaryLMSRegressor on stackloss data."""
    model = EvolutionaryLMSRegressor(
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_stackloss, y_stackloss)

    assert model.coef_.shape == (3,)
    assert model.best_criterion_ < 2.0


def test_stackloss_lta():
    """Test EvolutionaryLTARegressor on stackloss data."""
    model = EvolutionaryLTARegressor(
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_stackloss, y_stackloss)

    assert model.coef_.shape == (3,)
    assert model.best_criterion_ < 10.0


def test_stackloss_lqd():
    """Test EvolutionaryLQDRegressor on stackloss data."""
    model = EvolutionaryLQDRegressor(
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_stackloss, y_stackloss)

    assert model.coef_.shape == (3,)
    assert model.best_criterion_ < 5.0
