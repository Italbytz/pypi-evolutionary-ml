"""Robust regression algorithms powered by evolutionary computation."""

from __future__ import annotations

from typing import Any, Literal
import numpy as np

from evolutionary_ml.robust._base import BaseEvolutionaryRobustRegressor
from evolutionary_ml.robust.lts import EvolutionaryLTSRegressor
from evolutionary_ml.robust.lms import EvolutionaryLMSRegressor
from evolutionary_ml.robust.lta import EvolutionaryLTARegressor
from evolutionary_ml.robust.lqd import EvolutionaryLQDRegressor


class EvolutionaryRobustRegressor(BaseEvolutionaryRobustRegressor):
    """Unified Evolutionary Robust Linear Regressor.

    Supports multiple high-breakdown robust loss criteria:
    - "lts": Least Trimmed Squares
    - "lms": Least Median of Squares (LQS)
    - "lta": Least Trimmed Absolute deviation
    - "lqd": Least Quartile Difference

    Parameters
    ----------
    loss : {"lts", "lms", "lta", "lqd"}, default="lts"
        Robust loss criterion to minimize.
    quantile : int, float or None, default=None
        Trimming quantile or exact number of observations to include.
    max_iter : int, default=1000
        Maximum number of evolutionary generations (iterations).
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
        loss: Literal["lts", "lms", "lta", "lqd"] = "lts",
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
        self.loss = loss

    @property
    def loss_name(self) -> str:
        return self.loss

    def _compute_criterion(
        self, residuals: np.ndarray, h: int
    ) -> tuple[float, np.ndarray]:
        if self.loss == "lts":
            sq_res = residuals * residuals
            h_indices = np.argsort(sq_res)[:h]
            return float(np.sum(sq_res[h_indices])), h_indices
        elif self.loss == "lms":
            sq_res = residuals * residuals
            h_indices = np.argsort(sq_res)[:h]
            return float(sq_res[h_indices[-1]]), h_indices
        elif self.loss == "lta":
            abs_res = np.abs(residuals)
            h_indices = np.argsort(abs_res)[:h]
            return float(np.sum(abs_res[h_indices])), h_indices
        elif self.loss == "lqd":
            import math
            n = len(residuals)
            med = np.median(residuals)
            dev = np.abs(residuals - med)
            h_indices = np.argsort(dev)[:h]
            if h < 2:
                return 0.0, h_indices
            k_h = math.comb(h, 2)
            i_idx, j_idx = np.triu_indices(n, k=1)
            diffs = np.abs(residuals[i_idx] - residuals[j_idx])
            k_target = min(k_h - 1, len(diffs) - 1)
            if k_target < 0:
                crit = 0.0
            else:
                crit = float(np.partition(diffs, k_target)[k_target])
            return crit, h_indices
        else:
            raise ValueError(f"Unknown loss '{self.loss}'. Supported: 'lts', 'lms', 'lta', 'lqd'")


__all__ = [
    "BaseEvolutionaryRobustRegressor",
    "EvolutionaryRobustRegressor",
    "EvolutionaryLTSRegressor",
    "EvolutionaryLMSRegressor",
    "EvolutionaryLTARegressor",
    "EvolutionaryLQDRegressor",
]
