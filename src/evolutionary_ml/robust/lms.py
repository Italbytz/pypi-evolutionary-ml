"""Least Median of Squares (LMS / LQS) Robust Regressor."""

from __future__ import annotations

from typing import Literal
import numpy as np

from evolutionary_ml.robust._base import BaseEvolutionaryRobustRegressor


class EvolutionaryLMSRegressor(BaseEvolutionaryRobustRegressor):
    """Evolutionary Least Median of Squares (LMS / LQS) Regressor.

    Minimizes the h-th order statistic of squared residuals:
        r_{(h)}^2

    Based on Rousseeuw (1984) and extended to a (mu + lambda) population EA
    with uniform subset crossover and Hawkins/Move mutations.

    Parameters
    ----------
    quantile : int, float or None, default=None
        Trimming quantile or exact number of observations to include.
    max_iter : int, default=1000
        Maximum number of evolutionary generations (iterations).
    population_size : int, default=10
        Number of parent individuals (mu) in the population.
    n_offspring : int, default=10
        Number of offspring (lambda) produced per generation.
    crossover_rate : float, default=0.5
        Probability of applying uniform subset crossover.
    n_iter_no_change : int or None, default=None
        Early stopping patience.
    c_steps : int, default=0
        Number of concentration steps (C-steps) per evaluation.
    prob_move : float, default=0.2
        Probability of applying the residual-window Move operator.
    reweight : bool, default=True
        Whether to perform a second-stage Reweighted Least Squares (RLS) fit.
    init_method : {"random", "pca"}, default="random"
        Initial subset selection method.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the evolutionary search.
    """

    def __init__(
        self,
        quantile: int | float | None = None,
        max_iter: int = 1000,
        population_size: int = 10,
        n_offspring: int = 10,
        crossover_rate: float = 0.5,
        n_iter_no_change: int | None = None,
        c_steps: int = 0,
        prob_move: float = 0.2,
        reweight: bool = True,
        init_method: Literal["random", "pca"] = "random",
        fit_intercept: bool = True,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(
            quantile=quantile,
            max_iter=max_iter,
            population_size=population_size,
            n_offspring=n_offspring,
            crossover_rate=crossover_rate,
            n_iter_no_change=n_iter_no_change,
            c_steps=c_steps,
            prob_move=prob_move,
            reweight=reweight,
            init_method=init_method,
            fit_intercept=fit_intercept,
            random_state=random_state,
        )

    @property
    def loss_name(self) -> str:
        return "lms"

    def _compute_criterion(
        self, residuals: np.ndarray, h: int
    ) -> tuple[float, np.ndarray]:
        sq_res = residuals * residuals
        h_indices = np.argsort(sq_res)[:h]
        crit = float(sq_res[h_indices[-1]])
        return crit, h_indices
