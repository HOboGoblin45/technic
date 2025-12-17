"""Overlay past outcomes of similar signal patterns."""

from __future__ import annotations

import pandas as pd
from typing import Iterable, List


def overlay_recall_patterns(
    df: pd.DataFrame,
    signal_dates: Iterable[pd.Timestamp],
    window: int = 10,
) -> List[pd.DataFrame]:
    """Collect slices around historical signal dates for pattern recall.

    Args:
        df: DataFrame indexed by datetime-like with price/feature columns.
        signal_dates: Iterable of dates to center the window on.
        window: Number of periods before/after to include.

    Returns:
        List of DataFrame slices for each signal date found in the index.
    """
    patterns: List[pd.DataFrame] = []
    for date in signal_dates:
        if date in df.index:
            patterns.append(df.loc[date - window : date + window].copy())
    return patterns


__all__ = ["overlay_recall_patterns"]
