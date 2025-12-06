"""Option chain structurer for moneyness and Greek sensitivity."""

from __future__ import annotations

import pandas as pd
from typing import Tuple


def build_chain(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Organize calls/puts by moneyness and delta."""
    required = {"strike", "underlying_price", "type", "delta"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    df = df.copy()
    df["moneyness"] = df["strike"] / df["underlying_price"]
    calls = df[df["type"] == "call"].sort_values("delta", ascending=False)
    puts = df[df["type"] == "put"].sort_values("delta")
    return calls.head(5), puts.head(5)


__all__ = ["build_chain"]
