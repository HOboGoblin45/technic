"""Persistent alpha history store."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

INDEX_PATH = Path("index/alpha_store_map.json")
STORE_DIR = Path("alpha_store")


def save_alpha(symbol: str, date: str, df: pd.DataFrame) -> Path:
    """Save daily alpha scores to compressed Parquet and update index."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_path = STORE_DIR / f"{symbol}_{date}.parquet"
    df.to_parquet(file_path, compression="gzip")

    index = {}
    if INDEX_PATH.exists():
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    index[f"{symbol}_{date}"] = str(file_path)
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return file_path


__all__ = ["save_alpha", "INDEX_PATH", "STORE_DIR"]
