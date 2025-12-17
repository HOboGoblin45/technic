"""Strategy alpha decomposition utilities."""

from __future__ import annotations

import pandas as pd


def decompose_alpha(returns: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Break down alpha into active return and rolling alpha."""
    active = returns - benchmark
    rolling = active.rolling(10).mean()
    return pd.DataFrame(
        {
            "Active Return": active,
            "Rolling Alpha": rolling,
        }
    )


__all__ = ["decompose_alpha"]
