"""Contract tests for scikit-learn estimator compliance."""

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from evolutionary_ml.robust import (
    EvolutionaryLTSRegressor,
    EvolutionaryLMSRegressor,
    EvolutionaryLTARegressor,
    EvolutionaryLQDRegressor,
    EvolutionaryRobustRegressor,
)


@parametrize_with_checks([
    EvolutionaryLTSRegressor(max_iter=15, random_state=42),
    EvolutionaryLMSRegressor(max_iter=15, random_state=42),
    EvolutionaryLTARegressor(max_iter=15, random_state=42),
    EvolutionaryLQDRegressor(max_iter=15, random_state=42),
    EvolutionaryRobustRegressor(loss="lts", max_iter=15, random_state=42),
])
def test_sklearn_contracts(estimator, check):
    """Run scikit-learn's standard estimator validation checks on all robust estimators."""
    check(estimator)
