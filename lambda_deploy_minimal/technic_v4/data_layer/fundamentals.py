"""
Lightweight fundamentals interface (Phase 1 scaffold).

- File-based cache under data_cache/fundamentals/{symbol}.json
- Graceful fallback (returns empty dict) when no data is present
- TTL-based memo cache to avoid repeated disk reads during a session

This can be replaced with a vendor-backed fetcher (Polygon, FinancialModelingPrep, etc.)
without changing the UI surface.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
FUNDAMENTALS_DIR = BASE_DIR / "data_cache" / "fundamentals"
FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)

# cache key: symbol -> (ts, payload)
_MEMO: Dict[str, tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 300.0


@dataclass
class FundamentalsSnapshot:
    raw: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def _now() -> float:
    return time.time()


def _load_from_disk(symbol: str) -> Dict[str, Any]:
    path = FUNDAMENTALS_DIR / f"{symbol.upper()}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_fundamentals(symbol: str) -> FundamentalsSnapshot:
    """
    Return fundamentals snapshot for `symbol`.
    - Tries memo cache first
    - Then disk cache (data_cache/fundamentals/{symbol}.json)
    - Returns empty payload if nothing is available
    """
    sym = symbol.upper()
    memo = _MEMO.get(sym)
    if memo and (_now() - memo[0] <= CACHE_TTL_SECONDS):
        return FundamentalsSnapshot(memo[1])

    payload = _load_from_disk(sym)
    _MEMO[sym] = (_now(), payload)
    return FundamentalsSnapshot(payload)


def save_fundamentals(symbol: str, data: Dict[str, Any]) -> None:
    """
    Save fundamentals payload to disk and memo cache.
    Useful for offline ingestion scripts.
    """
    sym = symbol.upper()
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        (FUNDAMENTALS_DIR / f"{sym}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    except Exception:
        return
    _MEMO[sym] = (_now(), data)
