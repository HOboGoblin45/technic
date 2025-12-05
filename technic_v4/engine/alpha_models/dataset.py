from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def build_cross_sectional_dataset(
    snapshots: List[pd.DataFrame],
    feature_columns: List[str],
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Simple cross-sectional dataset builder.

    Assumes snapshots is a list of DataFrames already containing feature_columns
    and target_column. Concatenates and returns X, y.
    """
    if not snapshots:
        return pd.DataFrame(), pd.Series(dtype=float)

    df = pd.concat(snapshots, axis=0, ignore_index=False)
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    return X, y
