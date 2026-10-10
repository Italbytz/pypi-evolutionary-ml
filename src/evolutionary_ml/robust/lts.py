"""Least Trimmed Squares (LTS) Robust Regressor."""

from __future__ import annotations

from typing import Literal
import numpy as np

from evolutionary_ml.robust._base import BaseEvolutionaryRobustRegressor


class EvolutionaryLTSRegressor(BaseEvolutionaryRobustRegressor):
    """Evolutionary Least Trimmed Squares (LTS) Regressor.

    Minimizes the sum of the h smallest squared residuals:
        sum_{i=1}^h r_{(i)}^2

    Based on the evolutionary algorithm for robust regression by
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
        0 means pure (1+1)-EA as in Nunkesser & Morell (2010).
        >0 enables memetic local improvement (FAST-LTS style).
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
        Sum of the h smallest squared residuals achieved by the model.
    n_iter_ : int
        Number of iterations run.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        quantile: int | float | None = None,
        max_iter: int = 1000,
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
            c_steps=c_steps,
            prob_move=prob_move,
            reweight=reweight,
            init_method=init_method,
            fit_intercept=fit_intercept,
            random_state=random_state,
        )

    @property
    def loss_name(self) -> str:
        return "lts"

    def _compute_criterion(
        self, residuals: np.ndarray, h: int
    ) -> tuple[float, np.ndarray]:
        sq_res = residuals * residuals
        h_indices = np.argsort(sq_res)[:h]
        crit = float(np.sum(sq_res[h_indices]))
        return crit, h_indices
