# technic_v4/universe_loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

# Always look for ticker_universe.csv next to this file
UNIVERSE_FILE = Path(__file__).resolve().with_name("ticker_universe.csv")


@dataclass
class UniverseRow:
    symbol: str
    sector: Optional[str]
    industry: Optional[str]
    subindustry: Optional[str] = None


def _normalize_str(value) -> Optional[str]:
    """
    Return a stripped string or None.

    Keeps the original capitalization but removes leading/trailing whitespace
    and turns empty strings into None.
    """
    if isinstance(value, str):
        s = value.strip()
        return s or None
    return None


def load_universe(path: Union[str, Path, None] = None) -> List[UniverseRow]:
    """
    Load the universe CSV and return a list of UniverseRow objects.

    Expected columns in ticker_universe.csv (case-insensitive):
      - symbol  (required)  [symbol / ticker]
      - sector  (optional)  [sector]
      - industry (optional) [industry / sicdescription]

    Any missing sector/industry just comes through as None.
    """
    if path is None:
        path = UNIVERSE_FILE
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names to lower-case for flexibility
    lower_cols = {c.lower(): c for c in df.columns}

    # --- required: symbol / ticker ---
    symbol_col = lower_cols.get("symbol") or lower_cols.get("ticker")
    if not symbol_col:
        raise ValueError(
            f"Universe file {path} must have a 'symbol' or 'ticker' column "
            f"(found: {list(df.columns)})"
        )

    # --- optional: sector / industry / sicdescription ---
    sector_col = lower_cols.get("sector")
    industry_col = lower_cols.get("industry")
    sic_col = lower_cols.get("sicdescription")
    subindustry_col = lower_cols.get("subindustry")

    # Clean up symbol column: strip, drop blanks
    df[symbol_col] = df[symbol_col].astype(str).str.strip()
    df = df[df[symbol_col] != ""]

    rows: List[UniverseRow] = []

    for _, rec in df.iterrows():
        sym = str(rec[symbol_col]).strip()
        if not sym:
            continue

        sector_val = rec[sector_col] if sector_col else None
        industry_val = rec[industry_col] if industry_col else None

        # Fallback: use SIC description as industry if empty/missing
        if (industry_val is None or str(industry_val).strip() == "") and sic_col:
            industry_val = rec[sic_col]

        subindustry_val = rec[subindustry_col] if subindustry_col else None

        rows.append(
            UniverseRow(
                symbol=sym.upper(),
                sector=_normalize_str(sector_val),
                industry=_normalize_str(industry_val),
                subindustry=_normalize_str(subindustry_val),
            )
        )


    if not rows:
        raise ValueError(f"Universe file {path} produced zero usable symbols.")

    return rows
