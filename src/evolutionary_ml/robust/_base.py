"""Base classes and utilities for evolutionary robust regression."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data


def _compute_default_h(n: int, p: int) -> int:
    """Compute default trimming parameter h according to Rousseeuw & Leroy (1987).

    h = floor((n + p + 1) / 2)
    """
    h = int(math.floor((n + p + 1) / 2.0))
    return max(p, min(h, n))


def _compute_residuals(X_tilde: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute residuals r_i = y_i - X_tilde_i @ beta."""
    return y - np.dot(X_tilde, beta)


def _solve_elemental_subset(X_subset: np.ndarray, y_subset: np.ndarray) -> np.ndarray:
    """Solve linear system for elemental subset using least squares."""
    coef, _, _, _ = np.linalg.lstsq(X_subset, y_subset, rcond=None)
    return coef


class BaseEvolutionaryRobustRegressor(RegressorMixin, BaseEstimator, ABC):
    """Abstract base class for evolutionary robust regression estimators.

    Parameters
    ----------
    quantile : int, float or None, default=None
        Trimming quantile or exact number of observations to include in the criterion.
        If None, defaults to floor((n + p + 1) / 2).
        If float in (0.5, 1.0], h = int(floor(quantile * n)).
        If int, h = quantile.
    max_iter : int, default=1000
        Maximum number of evolutionary generations (iterations).
    c_steps : int, default=0
        Number of concentration steps (C-steps) applied to refine solutions.
        0 means pure (1+1)-evolutionary algorithm as in Nunkesser & Morell (2010).
        >0 enables memetic local improvement (FAST-LTS style).
    prob_move : float, default=0.2
        Probability of applying the residual-window Move operator instead of the Hawkins point swap.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the evolutionary search.
    """

    def __init__(
        self,
        quantile: int | float | None = None,
        max_iter: int = 1000,
        c_steps: int = 0,
        prob_move: float = 0.2,
        fit_intercept: bool = True,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.quantile = quantile
        self.max_iter = max_iter
        self.c_steps = c_steps
        self.prob_move = prob_move
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    @property
    @abstractmethod
    def loss_name(self) -> str:
        """Name of the loss criterion."""
        pass

    @abstractmethod
    def _compute_criterion(
        self, residuals: np.ndarray, h: int
    ) -> tuple[float, np.ndarray]:
        """Compute the robust criterion and return (criterion_value, inlier_indices).

        Parameters
        ----------
        residuals : np.ndarray of shape (n_samples,)
            Full residual vector.
        h : int
            Number of points to trim / consider.

        Returns
        -------
        criterion : float
            Value of the criterion to minimize.
        h_indices : np.ndarray of shape (h,)
            Indices of the h points defining the robust criterion.
        """
        pass

    def _resolve_h(self, n_samples: int, p_params: int) -> int:
        """Resolve the effective trimming value h."""
        if self.quantile is None:
            return _compute_default_h(n_samples, p_params)

        if isinstance(self.quantile, (int, np.integer)):
            h = int(self.quantile)
        elif isinstance(self.quantile, (float, np.floating)):
            if not (0.5 <= self.quantile <= 1.0):
                raise ValueError(
                    f"quantile as float must be in [0.5, 1.0], got {self.quantile}"
                )
            h = int(math.floor(self.quantile * n_samples))
        else:
            raise TypeError(
                f"quantile must be None, int, or float, got {type(self.quantile)}"
            )

        if h < p_params:
            h = min(p_params, n_samples)
        if h > n_samples:
            h = n_samples
        return h

    def _c_step(
        self, X_tilde: np.ndarray, y: np.ndarray, h_indices: np.ndarray, h: int
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Perform one C-step (concentration step) using OLS on best h points."""
        beta_c = _solve_elemental_subset(X_tilde[h_indices], y[h_indices])
        residuals_c = _compute_residuals(X_tilde, y, beta_c)
        crit_c, new_h_indices = self._compute_criterion(residuals_c, h)
        return beta_c, residuals_c, crit_c, new_h_indices

    def fit(self, X: Any, y: Any) -> BaseEvolutionaryRobustRegressor:
        """Fit the robust regression model using the evolutionary algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(self, X=X, y=y, accept_sparse=False, y_numeric=True)
        y = y.astype(np.float64)
        X = X.astype(np.float64)

        n_samples, n_features = X.shape

        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.c_steps < 0:
            raise ValueError(f"c_steps must be >= 0, got {self.c_steps}")
        if not (0.0 <= self.prob_move <= 1.0):
            raise ValueError(f"prob_move must be in [0, 1], got {self.prob_move}")

        rng = check_random_state(self.random_state)

        # Build augmented matrix X_tilde
        if self.fit_intercept:
            X_tilde = np.hstack([np.ones((n_samples, 1), dtype=np.float64), X])
            p_params = n_features + 1
        else:
            X_tilde = X
            p_params = n_features

        h = self._resolve_h(n_samples, p_params)

        # If sample size is very small (n_samples <= p_params), standard least squares solves exactly
        if n_samples <= p_params:
            beta = _solve_elemental_subset(X_tilde, y)
            residuals = _compute_residuals(X_tilde, y, beta)
            crit, h_indices = self._compute_criterion(residuals, min(h, n_samples))
            self._finalize(beta, residuals, crit, h_indices, n_samples)
            self.n_iter_ = 0
            return self

        # 1. Initialize elemental subset of size p_params
        active_indices = np.sort(rng.choice(n_samples, size=p_params, replace=False))
        beta = _solve_elemental_subset(X_tilde[active_indices], y[active_indices])
        residuals = _compute_residuals(X_tilde, y, beta)
        best_crit, best_h_indices = self._compute_criterion(residuals, h)

        if self.c_steps > 0:
            for _ in range(self.c_steps):
                beta_c, residuals_c, crit_c, new_h_indices = self._c_step(
                    X_tilde, y, best_h_indices, h
                )
                if crit_c <= best_crit:
                    beta, residuals, best_crit, best_h_indices = (
                        beta_c,
                        residuals_c,
                        crit_c,
                        new_h_indices,
                    )

        best_beta = beta.copy()
        best_residuals = residuals.copy()

        # 2. (1+1)-Evolutionary Algorithm
        for iteration in range(1, self.max_iter + 1):
            use_move = (rng.uniform(0.0, 1.0) < self.prob_move)

            if use_move:
                # Move operator: pick k in [p_params, n_samples]
                k = rng.randint(p_params, n_samples + 1)
                # Sort absolute residuals to find window of points with similar residuals
                order = np.argsort(np.abs(residuals))
                start = max(0, k - p_params)
                candidate_indices = order[start : start + p_params]
                if len(candidate_indices) < p_params:
                    # Fallback to swap if window is too small
                    candidate_indices = active_indices.copy()
            else:
                # Point swap: Hawkins (1993)
                candidate_indices = active_indices.copy()
                inactive = np.setdiff1d(np.arange(n_samples), candidate_indices)
                if len(inactive) > 0:
                    out_idx = rng.randint(0, p_params)
                    in_elem = rng.choice(inactive)
                    candidate_indices[out_idx] = in_elem
                    candidate_indices = np.sort(candidate_indices)

            # Solve elemental candidate
            try:
                candidate_beta = _solve_elemental_subset(
                    X_tilde[candidate_indices], y[candidate_indices]
                )
            except Exception:
                continue

            candidate_residuals = _compute_residuals(X_tilde, y, candidate_beta)
            candidate_crit, candidate_h_indices = self._compute_criterion(
                candidate_residuals, h
            )

            # Optional C-step refinement
            if self.c_steps > 0:
                for _ in range(self.c_steps):
                    beta_c, residuals_c, crit_c, new_h_indices = self._c_step(
                        X_tilde, y, candidate_h_indices, h
                    )
                    if crit_c <= candidate_crit:
                        candidate_beta = beta_c
                        candidate_residuals = residuals_c
                        candidate_crit = crit_c
                        candidate_h_indices = new_h_indices

            # Selection: (1+1) elitist update
            if candidate_crit <= best_crit:
                best_crit = candidate_crit
                best_beta = candidate_beta.copy()
                best_residuals = candidate_residuals.copy()
                best_h_indices = candidate_h_indices.copy()
                active_indices = candidate_indices.copy()
                residuals = candidate_residuals

        self.n_iter_ = self.max_iter
        self._finalize(best_beta, best_residuals, best_crit, best_h_indices, n_samples)
        return self

    def _finalize(
        self,
        beta: np.ndarray,
        residuals: np.ndarray,
        best_crit: float,
        best_h_indices: np.ndarray,
        n_samples: int,
    ) -> None:
        """Extract coefficients, intercept, residuals, and inlier mask."""
        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = np.asarray(beta[1:], dtype=np.float64)
        else:
            self.intercept_ = 0.0
            self.coef_ = np.asarray(beta, dtype=np.float64)

        self.residuals_ = residuals
        self.best_criterion_ = float(best_crit)

        self.inlier_mask_ = np.zeros(n_samples, dtype=bool)
        self.inlier_mask_[best_h_indices] = True

    def predict(self, X: Any) -> np.ndarray:
        """Predict using the robust linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X = validate_data(self, X=X, accept_sparse=False, reset=False)
        return np.dot(X, self.coef_) + self.intercept_

    def __sklearn_tags__(self):
        """Declare scikit-learn estimator tags."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.target_tags.required = True
        return tags
