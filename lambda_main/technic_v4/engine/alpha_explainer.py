"""Predictive alpha attribution breakdown using linear regression."""

from __future__ import annotations

import statsmodels.api as sm
import pandas as pd


def decompose_predictive_alpha(returns: pd.Series, factors: pd.DataFrame):
    """Attribute returns to alpha sources (factor, sector, timing) via OLS."""
    X = sm.add_constant(factors)
    model = sm.OLS(returns, X).fit()
    return model.summary()


__all__ = ["decompose_predictive_alpha"]
