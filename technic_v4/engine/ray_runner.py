"""
Optional Ray-powered scanning helpers.
"""

from __future__ import annotations

from typing import List, Optional

try:
    import ray
except Exception:  # pragma: no cover
    ray = None

from technic_v4.config.settings import get_settings


def init_ray_if_enabled() -> bool:
    """
    Initialize Ray if TECHNIC_USE_RAY is true. Returns True if Ray is ready.
    """
    import os
    import warnings
    
    settings = get_settings()
    use_ray = settings.use_ray
    if not use_ray or ray is None:
        return False
    if ray.is_initialized():
        return True
    try:
        # Suppress Ray GPU warning when num_gpus=0
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        
        # Suppress FutureWarning about GPU env vars
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="ray")
            ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
        return True
    except Exception:
        return False


def run_ray_scans(symbols: List[str], scan_config, regime_tags: Optional[dict] = None, price_cache: Optional[dict] = None):
    """
    Run scans via Ray if enabled; return list of Series rows or None if Ray disabled/unavailable.
    
    PHASE 1 OPTIMIZATION: Accepts price_cache to pass to workers.
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

    # PHASE 1: Put price_cache in Ray object store for efficient sharing
    if price_cache:
        price_cache_ref = ray.put(price_cache)
    else:
        price_cache_ref = None

    @ray.remote
    def scan_symbol_remote(symbol: str, cache_ref):
        try:
            # Get cache from object store
            cache = ray.get(cache_ref) if cache_ref else None
            return _scan_symbol(
                symbol=symbol,
                lookback_days=cfg_dict["lookback_days"],
                trade_style=cfg_dict["trade_style"],
                price_cache=cache,  # PHASE 1: Pass pre-fetched data
            )
        except Exception:
            return None

    futures = [scan_symbol_remote.remote(sym, price_cache_ref) for sym in symbols]
    try:
        results = ray.get(futures)
    except Exception:
        return None
    rows = [r for r in results if r is not None]
    return rows
