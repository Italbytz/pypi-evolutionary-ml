"""EvolutionaryML: Scikit-learn compatible evolutionary machine learning toolkit.

Provides high-breakdown robust regression estimators (LTS, LMS, LTA, LQD) and
evolutionary machine learning algorithms.
"""

from evolutionary_ml._version import __version__
from evolutionary_ml.robust import (
    BaseEvolutionaryRobustRegressor,
    EvolutionaryRobustRegressor,
    EvolutionaryLTSRegressor,
    EvolutionaryLMSRegressor,
    EvolutionaryLTARegressor,
    EvolutionaryLQDRegressor,
)

__all__ = [
    "__version__",
    "BaseEvolutionaryRobustRegressor",
    "EvolutionaryRobustRegressor",
    "EvolutionaryLTSRegressor",
    "EvolutionaryLMSRegressor",
    "EvolutionaryLTARegressor",
    "EvolutionaryLQDRegressor",
]
