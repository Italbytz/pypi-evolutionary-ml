"""Base classes and utilities for evolutionary robust regression."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Literal, NamedTuple

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


def _crossover_subsets(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    p_params: int,
    n_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Perform uniform subset crossover between two parent elemental subsets."""
    common = np.intersect1d(parent_a, parent_b)
    disjoint = np.setdiff1d(np.union1d(parent_a, parent_b), common)

    n_needed = p_params - len(common)
    if n_needed <= 0:
        return np.sort(common[:p_params])

    if len(disjoint) >= n_needed:
        selected = rng.choice(disjoint, size=n_needed, replace=False)
        child = np.concatenate([common, selected])
    else:
        used = np.union1d(parent_a, parent_b)
        unused = np.setdiff1d(np.arange(n_samples), used)
        if len(unused) > 0:
            extra = rng.choice(
                unused, size=min(n_needed - len(disjoint), len(unused)), replace=False
            )
            child = np.concatenate([common, disjoint, extra])
        else:
            child = np.union1d(parent_a, parent_b)

    return np.sort(child[:p_params])


def _mutate_subset(
    subset: np.ndarray,
    residuals: np.ndarray,
    p_params: int,
    n_samples: int,
    prob_move: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Mutate an elemental subset via either the Move operator or Hawkins swap."""
    use_move = rng.uniform(0.0, 1.0) < prob_move
    if use_move:
        k = rng.randint(p_params, n_samples + 1)
        order = np.argsort(np.abs(residuals))
        start = max(0, k - p_params)
        candidate = order[start : start + p_params]
        if len(candidate) == p_params:
            return np.sort(candidate)

    # Hawkins point swap: replace 1 element
    candidate = subset.copy()
    inactive = np.setdiff1d(np.arange(n_samples), candidate)
    if len(inactive) > 0:
        out_idx = rng.randint(0, len(candidate))
        in_elem = rng.choice(inactive)
        candidate[out_idx] = in_elem
    return np.sort(candidate)


class _Individual(NamedTuple):
    subset: np.ndarray
    beta: np.ndarray
    residuals: np.ndarray
    crit: float
    h_indices: np.ndarray


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
    population_size : int, default=10
        Number of parent individuals (mu) in the population.
        Setting to 1 reproduces the classic (1+1)-EA of Nunkesser & Morell (2010).
    n_offspring : int, default=10
        Number of offspring (lambda) produced per generation.
    crossover_rate : float, default=0.5
        Probability of applying uniform subset crossover to pair parents.
    n_iter_no_change : int or None, default=None
        Early stopping: terminate if no improvement occurs for this number of generations.
    c_steps : int, default=0
        Number of concentration steps (C-steps) applied to refine solutions.
        0 means pure evolutionary algorithm.
        >0 enables memetic local improvement (FAST-LTS style).
    prob_move : float, default=0.2
        Probability of applying the residual-window Move operator instead of the Hawkins point swap.
    reweight : bool, default=True
        Whether to perform a second-stage Reweighted Least Squares (RLS) fit.
        RLS achieves 95% Gaussian asymptotic efficiency while preserving the 50% breakdown point.
    init_method : {"random", "pca"}, default="random"
        Initial subset selection method. "pca" uses robust principal directions to seed initial points.
    subsample_threshold : int or None, default=None
        Threshold on the number of samples to trigger Big Data nested subsampling / partitioning.
        If n_samples > subsample_threshold, the dataset is partitioned into sub-blocks
        for accelerated evolutionary candidate search, followed by global concentration and
        refinement on the full dataset. If None, full dataset is evaluated directly.
    subsample_size : int, default=500
        Target size of each sub-block partition when nested subsampling is activated.
    n_subsamples : int, default=5
        Maximum number of disjoint partitions to divide large datasets into.
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
        subsample_threshold: int | None = None,
        subsample_size: int = 500,
        n_subsamples: int = 5,
        fit_intercept: bool = True,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.quantile = quantile
        self.max_iter = max_iter
        self.population_size = population_size
        self.n_offspring = n_offspring
        self.crossover_rate = crossover_rate
        self.n_iter_no_change = n_iter_no_change
        self.c_steps = c_steps
        self.prob_move = prob_move
        self.reweight = reweight
        self.init_method = init_method
        self.subsample_threshold = subsample_threshold
        self.subsample_size = subsample_size
        self.n_subsamples = n_subsamples
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

    def _evaluate_candidate(
        self,
        subset: np.ndarray,
        X_tilde: np.ndarray,
        y: np.ndarray,
        h: int,
    ) -> _Individual | None:
        """Evaluate an elemental subset candidate and apply optional C-steps."""
        try:
            beta = _solve_elemental_subset(X_tilde[subset], y[subset])
        except Exception:
            return None

        residuals = _compute_residuals(X_tilde, y, beta)
        crit, h_indices = self._compute_criterion(residuals, h)

        if self.c_steps > 0:
            for _ in range(self.c_steps):
                beta_c, residuals_c, crit_c, new_h_indices = self._c_step(
                    X_tilde, y, h_indices, h
                )
                if crit_c <= crit:
                    beta, residuals, crit, h_indices = (
                        beta_c,
                        residuals_c,
                        crit_c,
                        new_h_indices,
                    )

        return _Individual(
            subset=subset,
            beta=beta,
            residuals=residuals,
            crit=crit,
            h_indices=h_indices,
        )

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
        if self.population_size < 1:
            raise ValueError(f"population_size must be >= 1, got {self.population_size}")
        if self.n_offspring < 1:
            raise ValueError(f"n_offspring must be >= 1, got {self.n_offspring}")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError(f"crossover_rate must be in [0, 1], got {self.crossover_rate}")
        if self.n_iter_no_change is not None and self.n_iter_no_change < 1:
            raise ValueError(f"n_iter_no_change must be >= 1, got {self.n_iter_no_change}")
        if self.c_steps < 0:
            raise ValueError(f"c_steps must be >= 0, got {self.c_steps}")
        if not (0.0 <= self.prob_move <= 1.0):
            raise ValueError(f"prob_move must be in [0, 1], got {self.prob_move}")
        if self.init_method not in ("random", "pca"):
            raise ValueError(
                f"init_method must be 'random' or 'pca', got '{self.init_method}'"
            )
        if self.subsample_threshold is not None and self.subsample_threshold < 1:
            raise ValueError(
                f"subsample_threshold must be >= 1, got {self.subsample_threshold}"
            )
        if self.subsample_size < 1:
            raise ValueError(f"subsample_size must be >= 1, got {self.subsample_size}")
        if self.n_subsamples < 1:
            raise ValueError(f"n_subsamples must be >= 1, got {self.n_subsamples}")

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
            self._finalize(beta, residuals, crit, h_indices, n_samples, X_tilde, y)
            self.n_iter_ = 0
            return self

        # Big Data nested subsampling / partitioning
        if self.subsample_threshold is not None and n_samples > self.subsample_threshold:
            k_blocks = min(self.n_subsamples, max(2, n_samples // self.subsample_size))
            shuffled_indices = rng.permutation(n_samples)
            partitions = np.array_split(shuffled_indices, k_blocks)

            pooled_betas: list[np.ndarray] = []
            iter_per_block = max(10, self.max_iter // k_blocks)

            for part in partitions:
                if len(part) <= p_params:
                    continue
                X_part = X_tilde[part]
                y_part = y[part]
                X_raw_part = X[part] if n_features > 0 else None
                n_part = len(part)
                h_part = self._resolve_h(n_part, p_params)

                sub_pop, _ = self._run_evolutionary_search(
                    X_part, y_part, h_part, p_params, n_part, iter_per_block, rng, X_raw_part
                )
                elite_count = max(2, self.population_size // k_blocks)
                for ind in sub_pop[:elite_count]:
                    pooled_betas.append(ind.beta)

            # Optional PCA candidate on full sample
            if self.init_method == "pca" and n_features > 0:
                X_centered = X - np.median(X, axis=0)
                try:
                    _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
                    proj = np.abs(np.dot(X_centered, vt[0]))
                    order = np.argsort(proj)
                    pca_subset = np.sort(order[:p_params])
                    pooled_betas.append(
                        _solve_elemental_subset(X_tilde[pca_subset], y[pca_subset])
                    )
                except Exception:
                    pass

            # Full dataset refinement: evaluate pooled candidates on full dataset with C-steps
            candidates_on_full: list[_Individual] = []
            c_steps_to_apply = max(2, self.c_steps)
            for beta_cand in pooled_betas:
                res_full = _compute_residuals(X_tilde, y, beta_cand)
                crit_full, h_ind_full = self._compute_criterion(res_full, h)

                beta_curr = beta_cand
                crit_curr = crit_full
                res_curr = res_full
                h_ind_curr = h_ind_full
                for _ in range(c_steps_to_apply):
                    beta_c, res_c, crit_c, h_c = self._c_step(
                        X_tilde, y, h_ind_curr, h
                    )
                    if crit_c <= crit_curr:
                        beta_curr, res_curr, crit_curr, h_ind_curr = beta_c, res_c, crit_c, h_c
                    else:
                        break
                candidates_on_full.append(
                    _Individual(
                        subset=np.arange(p_params),
                        beta=beta_curr,
                        residuals=res_curr,
                        crit=crit_curr,
                        h_indices=h_ind_curr,
                    )
                )

            candidates_on_full.sort(key=lambda ind: ind.crit)
            best_ind = candidates_on_full[0]
            self.n_iter_ = iter_per_block
        else:
            # Standard full dataset evolutionary search
            population, actual_iterations = self._run_evolutionary_search(
                X_tilde, y, h, p_params, n_samples, self.max_iter, rng, X
            )
            best_ind = population[0]
            self.n_iter_ = actual_iterations

        self._finalize(
            best_ind.beta,
            best_ind.residuals,
            best_ind.crit,
            best_ind.h_indices,
            n_samples,
            X_tilde,
            y,
        )
        return self

    def _run_evolutionary_search(
        self,
        X_tilde: np.ndarray,
        y: np.ndarray,
        h: int,
        p_params: int,
        n_samples: int,
        max_iter: int,
        rng: np.random.RandomState,
        X_raw: np.ndarray | None = None,
    ) -> tuple[list[_Individual], int]:
        """Run (mu + lambda) evolutionary search on a given design matrix."""
        n_features = X_tilde.shape[1] - (1 if self.fit_intercept else 0)

        # 1. Initialize population of size mu = population_size
        population: list[_Individual] = []
        for i in range(self.population_size):
            if i == 0 and self.init_method == "pca" and n_features > 0 and X_raw is not None:
                X_centered = X_raw - np.median(X_raw, axis=0)
                try:
                    _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
                    proj = np.abs(np.dot(X_centered, vt[0]))
                    order = np.argsort(proj)
                    subset = np.sort(order[:p_params])
                except Exception:
                    subset = np.sort(rng.choice(n_samples, size=p_params, replace=False))
            else:
                subset = np.sort(rng.choice(n_samples, size=p_params, replace=False))

            ind = self._evaluate_candidate(subset, X_tilde, y, h)
            if ind is None:
                subset = np.sort(rng.choice(n_samples, size=p_params, replace=False))
                ind = self._evaluate_candidate(subset, X_tilde, y, h)
            if ind is not None:
                population.append(ind)

        # Ensure at least one individual is in population
        if len(population) == 0:
            subset = np.arange(p_params)
            ind = self._evaluate_candidate(subset, X_tilde, y, h)
            if ind is not None:
                population.append(ind)

        if len(population) == 0:
            raise ValueError("Failed to initialize any valid candidate subset.")

        # Sort population by criterion ascending
        population.sort(key=lambda ind: ind.crit)
        best_ind = population[0]

        # 2. (mu + lambda) Evolutionary Algorithm Loop
        generations_without_improvement = 0
        actual_iterations = 0

        for iteration in range(1, max_iter + 1):
            actual_iterations = iteration
            offspring_list: list[_Individual] = []

            for _ in range(self.n_offspring):
                # Parent 1 via binary tournament
                if len(population) > 1:
                    t_idx = rng.choice(len(population), size=2, replace=False)
                    p1 = population[min(t_idx[0], t_idx[1])]
                else:
                    p1 = population[0]

                # Crossover
                if len(population) > 1 and rng.uniform(0.0, 1.0) < self.crossover_rate:
                    t_idx2 = rng.choice(len(population), size=2, replace=False)
                    p2 = population[min(t_idx2[0], t_idx2[1])]
                    child_subset = _crossover_subsets(
                        p1.subset, p2.subset, p_params, n_samples, rng
                    )
                else:
                    child_subset = p1.subset.copy()

                # Mutation
                child_subset = _mutate_subset(
                    child_subset, p1.residuals, p_params, n_samples, self.prob_move, rng
                )

                # Evaluation
                child_ind = self._evaluate_candidate(child_subset, X_tilde, y, h)
                if child_ind is not None:
                    offspring_list.append(child_ind)

            # (mu + lambda) elitist selection
            combined = population + offspring_list
            combined.sort(key=lambda ind: ind.crit)
            population = combined[: self.population_size]

            # Check for improvement
            if population[0].crit < best_ind.crit:
                best_ind = population[0]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if (
                self.n_iter_no_change is not None
                and generations_without_improvement >= self.n_iter_no_change
            ):
                break

        return population, actual_iterations

    def _finalize(
        self,
        beta: np.ndarray,
        residuals: np.ndarray,
        best_crit: float,
        best_h_indices: np.ndarray,
        n_samples: int,
        X_tilde: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Extract coefficients, intercept, residuals, and inlier mask, with optional RLS."""
        final_beta = beta
        final_residuals = residuals
        final_inliers = best_h_indices

        # Stage 2: Reweighted Least Squares (RLS)
        if self.reweight and n_samples > len(beta):
            med_r = np.median(residuals)
            mad_r = float(np.median(np.abs(residuals - med_r)))
            sigma_0 = 1.4826 * mad_r if mad_r > 1e-12 else float(np.std(residuals))

            if sigma_0 > 1e-12:
                cutoff = 2.5 * sigma_0
                inlier_candidates = np.where(np.abs(residuals) <= cutoff)[0]
                if len(inlier_candidates) >= len(beta):
                    try:
                        beta_rw = _solve_elemental_subset(
                            X_tilde[inlier_candidates], y[inlier_candidates]
                        )
                        residuals_rw = _compute_residuals(X_tilde, y, beta_rw)
                        final_beta = beta_rw
                        final_residuals = residuals_rw
                        final_inliers = inlier_candidates
                    except Exception:
                        pass

        if self.fit_intercept:
            self.intercept_ = float(final_beta[0])
            self.coef_ = np.asarray(final_beta[1:], dtype=np.float64)
        else:
            self.intercept_ = 0.0
            self.coef_ = np.asarray(final_beta, dtype=np.float64)

        self.residuals_ = final_residuals
        self.best_criterion_ = float(best_crit)

        self.inlier_mask_ = np.zeros(n_samples, dtype=bool)
        self.inlier_mask_[final_inliers] = True

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
