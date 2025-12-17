from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _sector_penalty(results_df: pd.DataFrame, sector_col: str = "Sector") -> pd.Series:
    if sector_col not in results_df.columns:
        return pd.Series(0.0, index=results_df.index)
    sectors = results_df[sector_col].fillna("Unknown").astype(str)
    counts = sectors.value_counts()
    penalties = sectors.map(lambda s: counts.get(s, 1))
    return penalties / penalties.max()


def rank_results(results_df: pd.DataFrame, max_positions: int = 50) -> pd.DataFrame:
    """
    Portfolio-aware ranking:
    - Combine TechRating and AlphaScore
    - Penalize sector concentration
    - Return top max_positions rows sorted by final score
    """
    if results_df is None or results_df.empty:
        return results_df

    df = results_df.copy()

    base = df.get("TechRating", pd.Series(0, index=df.index)).fillna(0)
    alpha = df.get("AlphaScore", pd.Series(0, index=df.index)).fillna(0)
    composite = base + alpha * 10  # scale alpha into TR-like points

    penalty = _sector_penalty(df, sector_col="Sector")
    df["CompositeScore"] = composite * (1 - 0.15 * penalty)

    df = df.sort_values("CompositeScore", ascending=False)
    if max_positions and len(df) > max_positions:
        df = df.head(max_positions)

    return df


__all__ = ["rank_results"]
