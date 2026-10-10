"""Least Trimmed Absolute (LTA) Robust Regressor."""

from __future__ import annotations

import numpy as np

from evolutionary_ml.robust._base import BaseEvolutionaryRobustRegressor


class EvolutionaryLTARegressor(BaseEvolutionaryRobustRegressor):
    """Evolutionary Least Trimmed Absolute (LTA) Regressor.

    Minimizes the sum of the h smallest absolute residuals:
        sum_{i=1}^h |r|_{(i)}

    Based on Tableman (1994) and the evolutionary framework of
    Nunkesser & Morell (2010), Computational Statistics & Data Analysis.

    Parameters
    ----------
    quantile : int, float or None, default=None
        Trimming quantile or exact number of observations to include.
        If None, defaults to floor((n + p + 1) / 2).
        If float in [0.5, 1.0], h = int(floor(quantile * n)).
        If int, h = quantile.
    max_iter : int, default=1000
        Maximum number of evolutionary generations (iterations).
    c_steps : int, default=0
        Number of concentration steps (C-steps) per evaluation.
    prob_move : float, default=0.2
        Probability of applying the residual-window Move operator.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the evolutionary search.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Independent term in the linear model.
    inlier_mask_ : np.ndarray of shape (n_samples,)
        Boolean mask of inliers (the h points with lowest absolute residuals).
    residuals_ : np.ndarray of shape (n_samples,)
        Residuals of the final fitted model.
    best_criterion_ : float
        Sum of the h smallest absolute residuals achieved by the model.
    n_iter_ : int
        Number of iterations run.
    n_features_in_ : int
        Number of features seen during fit.
    """

    @property
    def loss_name(self) -> str:
        return "lta"

    def _compute_criterion(
        self, residuals: np.ndarray, h: int
    ) -> tuple[float, np.ndarray]:
        abs_res = np.abs(residuals)
        h_indices = np.argsort(abs_res)[:h]
        crit = float(np.sum(abs_res[h_indices]))
        return crit, h_indices
