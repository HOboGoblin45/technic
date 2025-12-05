"""
Optional Ray-powered scanning helpers.
"""

from __future__ import annotations

import os
from typing import List, Optional

try:
    import ray
except Exception:  # pragma: no cover
    ray = None


def init_ray_if_enabled() -> bool:
    """
    Initialize Ray if TECHNIC_USE_RAY is true. Returns True if Ray is ready.
    """
    use_ray = str(os.getenv("TECHNIC_USE_RAY", "false")).lower() in {"1", "true", "yes"}
    if not use_ray or ray is None:
        return False
    if ray.is_initialized():
        return True
    try:
        ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
        return True
    except Exception:
        return False


def run_ray_scans(symbols: List[str], scan_config, regime_tags: Optional[dict] = None):
    """
    Run scans via Ray if enabled; return list of Series rows or None if Ray disabled/unavailable.
    """
    if not init_ray_if_enabled() or ray is None:
        return None

    try:
        from technic_v4.scanner_core import _scan_symbol  # local import to avoid circular at module load
    except Exception:
        return None

    cfg_dict = {
        "lookback_days": scan_config.lookback_days,
        "trade_style": scan_config.trade_style,
    }

    @ray.remote
    def scan_symbol_remote(symbol: str):
        try:
            return _scan_symbol(
                symbol=symbol,
                lookback_days=cfg_dict["lookback_days"],
                trade_style=cfg_dict["trade_style"],
            )
        except Exception:
            return None

    futures = [scan_symbol_remote.remote(sym) for sym in symbols]
    try:
        results = ray.get(futures)
    except Exception:
        return None
    rows = [r for r in results if r is not None]
    return rows
