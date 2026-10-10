"""Benchmark datasets for robust statistics and machine learning."""

from evolutionary_ml.datasets._base import (
    load_coleman,
    load_hbk,
    load_salinity,
    load_stackloss,
    load_stars,
    load_wood,
)

__all__ = [
    "load_hbk",
    "load_stars",
    "load_stackloss",
    "load_salinity",
    "load_coleman",
    "load_wood",
]
