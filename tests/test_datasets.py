"""Tests for bundled benchmark datasets in evolutionary_ml.datasets."""

import numpy as np
import pytest
from sklearn.utils import Bunch

from evolutionary_ml.datasets import (
    load_coleman,
    load_hbk,
    load_salinity,
    load_stackloss,
    load_stars,
    load_wood,
)
from evolutionary_ml.robust import EvolutionaryLTSRegressor


@pytest.mark.parametrize(
    "loader,expected_n,expected_p,expected_outliers",
    [
        (load_hbk, 75, 3, list(range(14))),
        (load_stars, 47, 1, [10, 19, 29, 33]),
        (load_stackloss, 21, 3, [0, 2, 3, 20]),
        (load_salinity, 28, 3, [4, 15, 22, 23]),
        (load_coleman, 20, 5, [2, 16, 17]),
        (load_wood, 20, 5, [3, 5, 7, 18]),
    ],
)
def test_dataset_loader_bunch(loader, expected_n, expected_p, expected_outliers):
    data = loader()
    assert isinstance(data, Bunch)
    assert data.data.shape == (expected_n, expected_p)
    assert data.target.shape == (expected_n,)
    assert len(data.feature_names) == expected_p
    assert len(data.target_names) == 1
    assert data.outlier_indices == expected_outliers
    assert len(data.DESCR) > 20
    assert data.data.dtype == np.float64
    assert data.target.dtype == np.float64


@pytest.mark.parametrize(
    "loader,expected_n,expected_p",
    [
        (load_hbk, 75, 3),
        (load_stars, 47, 1),
        (load_stackloss, 21, 3),
        (load_salinity, 28, 3),
        (load_coleman, 20, 5),
        (load_wood, 20, 5),
    ],
)
def test_dataset_loader_return_X_y(loader, expected_n, expected_p):
    X, y = loader(return_X_y=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (expected_n, expected_p)
    assert y.shape == (expected_n,)


def test_hbk_specific_attributes():
    hbk = load_hbk()
    assert hasattr(hbk, "bad_leverage_indices")
    assert hasattr(hbk, "good_leverage_indices")
    assert hbk.bad_leverage_indices == list(range(10))
    assert hbk.good_leverage_indices == list(range(10, 14))


def test_fit_evolts_on_hbk():
    """Verify that EvolutionaryLTSRegressor successfully identifies HBK outliers."""
    X, y = load_hbk(return_X_y=True)
    reg = EvolutionaryLTSRegressor(
        random_state=42,
        reweight=True,
        population_size=10,
        n_offspring=10,
        c_steps=2,
    )
    reg.fit(X, y)
    assert reg.inlier_mask_ is not None

    # Observations 0..9 are bad leverage points; they must not be inliers
    bad_inliers = [i for i in range(10) if reg.inlier_mask_[i]]
    assert len(bad_inliers) == 0, f"Bad leverage points identified as inliers: {bad_inliers}"

    # Clean observations (14..74) should overwhelmingly be recognized as inliers
    clean_inliers = [i for i in range(14, 75) if reg.inlier_mask_[i]]
    assert len(clean_inliers) >= 58  # At least 95% of clean points recognized


def test_fit_evolts_on_stars():
    """Verify that EvolutionaryLTSRegressor successfully fits stars dataset."""
    X, y = load_stars(return_X_y=True)
    reg = EvolutionaryLTSRegressor(
        random_state=42,
        reweight=True,
        c_steps=2,
    )
    reg.fit(X, y)
    assert reg.inlier_mask_ is not None
    # Check that giant stars (10, 19, 29, 33) are rejected
    giant_stars = [10, 19, 29, 33]
    giant_inliers = [i for i in giant_stars if reg.inlier_mask_[i]]
    assert len(giant_inliers) <= 1
