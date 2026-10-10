"""Least Quartile Difference (LQD) Robust Regressor."""

from __future__ import annotations

import math
import numpy as np

from evolutionary_ml.robust._base import BaseEvolutionaryRobustRegressor


class EvolutionaryLQDRegressor(BaseEvolutionaryRobustRegressor):
    """Evolutionary Least Quartile Difference (LQD) Regressor.

    Minimizes the Q_n-type order statistic of residual differences:
        {|r_i - r_j|; i < j}_{(k_h)}
    where k_h = binom(h, 2).

    Based on Rousseeuw & Croux (1993), Croux et al. (1994), and the
    evolutionary framework of Nunkesser & Morell (2010).

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
        Boolean mask of inliers.
    residuals_ : np.ndarray of shape (n_samples,)
        Residuals of the final fitted model.
    best_criterion_ : float
        LQD criterion value achieved by the model.
    n_iter_ : int
        Number of iterations run.
    n_features_in_ : int
        Number of features seen during fit.
    """

    @property
    def loss_name(self) -> str:
        return "lqd"

    def _compute_criterion(
        self, residuals: np.ndarray, h: int
    ) -> tuple[float, np.ndarray]:
        n = len(residuals)
        # Identify inliers as points closest to the median residual
        med = np.median(residuals)
        dev = np.abs(residuals - med)
        h_indices = np.argsort(dev)[:h]

        if h < 2:
            return 0.0, h_indices

        k_h = math.comb(h, 2)

        # Compute pairwise differences of all n points
        i_idx, j_idx = np.triu_indices(n, k=1)
        diffs = np.abs(residuals[i_idx] - residuals[j_idx])

        # Target order statistic index (0-based)
        k_target = min(k_h - 1, len(diffs) - 1)
        if k_target < 0:
            crit = 0.0
        else:
            # np.partition is O(n^2) and much faster than full sort
            crit = float(np.partition(diffs, k_target)[k_target])

        return crit, h_indices
