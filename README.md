# EvolutionaryML (`evolutionary-ml`)

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/evolutionary-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**EvolutionaryML** is a scikit-learn compatible Python toolkit for evolutionary machine learning, featuring high-breakdown robust regression estimators.

It ports, modernizes, and extends the algorithms originally introduced in:
> **An evolutionary algorithm for robust regression**  
> Robin Nunkesser and Oliver Morell  
> *Computational Statistics & Data Analysis (CSDA)*, 54(12):3241–3248, 2010.  
> [DOI: 10.1016/j.csda.2010.03.018](https://doi.org/10.1016/j.csda.2010.03.018)

---

## Key Features

- **Strict Scikit-Learn Compatibility**: Fully adheres to scikit-learn estimator conventions (`fit`, `predict`, `score`, `inlier_mask_`, `coef_`, `intercept_`, and cloneable).
- **High Breakdown Point**: 50% breakdown estimators resistant to severe leverage points and gross contamination.
- **Implemented Estimators**:
  - `EvolutionaryLTSRegressor`: Least Trimmed Squares (minimizing sum of $h$ smallest squared residuals).
  - `EvolutionaryLMSRegressor`: Least Median of Squares (minimizing $h$-th squared residual).
  - `EvolutionaryLTARegressor`: Least Trimmed Absolute deviation (minimizing sum of $h$ smallest absolute residuals).
  - `EvolutionaryLQDRegressor`: Least Quartile Difference (minimizing $Q_n$-type scale of residual differences).
  - `EvolutionaryRobustRegressor`: Unified interface with selectable loss (`"lts"`, `"lms"`, `"lta"`, `"lqd"`).
- **Hybrid / Memetic Support**: Optional Rousseeuw concentration steps (`c_steps=2`) enabling combined evolutionary global search and fast local refinement.
- **Pure NumPy Vectorization**: Fast vector arithmetic with no heavy external metaheuristic dependencies.

---

## Installation

```bash
pip install evolutionary-ml
```

Or using `uv`:
```bash
uv add evolutionary-ml
```

---

## Quickstart

```python
import numpy as np
from evolutionary_ml.robust import EvolutionaryLTSRegressor

# Contaminated data with outliers
X = np.random.uniform(-3, 3, size=(100, 2))
y = 2.0 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + np.random.normal(0, 0.1, size=100)

# Add 20% severe leverage outliers
outliers = np.random.choice(100, size=20, replace=False)
y[outliers] += np.random.uniform(30, 60, size=20)

# Fit Evolutionary LTS Regressor
reg = EvolutionaryLTSRegressor(max_iter=1000, c_steps=2, random_state=42)
reg.fit(X, y)

print(f"Intercept: {reg.intercept_:.3f}")   # ~2.0
print(f"Slopes:    {reg.coef_}")           # ~[1.5, -0.8]
print(f"Inliers detected: {np.sum(reg.inlier_mask_)} / 100")
```

---

## Architecture & Portfolio Context

In the wider ML ecosystem within this repository:

1. **`pypi-scoredrulesets`**: Interpretable rule-based learning (genetic programming, learning classifier systems, subgroup discovery).
2. **`pypi-evolutionary-ml`**: Continuous & combinatorial evolutionary machine learning, robust statistics, and evolutionary estimators.
3. **`pypi-catgen`**: High-cardinality categorical data synthesis and benchmark generator.
4. **`pypi-benchmarkplots`**: Visualization toolkit for critical difference diagrams, win-tie-loss, and multi-algorithm evaluation.
5. **`transparent-ml`**: High-level integration and benchmark harness connecting interpretable and robust estimators.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
