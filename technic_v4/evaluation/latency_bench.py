from __future__ import annotations

import time
from typing import Dict, List

import pandas as pd

from technic_v4.scanner_core import ScanConfig, run_scan
from technic_v4.universe_loader import load_universe
from technic_v4.config.settings import get_settings
from technic_v4.infra.logging import get_logger

logger = get_logger()


def _apply_overrides(override_settings: Dict[str, bool]):
    """
    Context manager-like helper to temporarily override boolean settings.
    Returns a restore function that reverts overrides.
    """
    settings = get_settings()
    original = {}
    for k, v in override_settings.items():
        if hasattr(settings, k):
            original[k] = getattr(settings, k)
            setattr(settings, k, v)
    def restore():
        for k, v in original.items():
            setattr(settings, k, v)
    return restore


def time_scan(universe: List[str], description: str, override_settings: Dict[str, bool]) -> None:
    """
    Run a scan for the given universe with certain Settings flags overridden,
    print timing information and basic stats (num results and signal counts).
    """
    restore = _apply_overrides(override_settings)
    settings = get_settings()
    try:
        cfg = ScanConfig(
            max_symbols=len(universe),
        )
        # Scanner reads its own universe list; we can set a custom list via config if supported.
        # For now rely on max_symbols and the default loader ordering.
        start = time.perf_counter()
        df, status = run_scan(cfg)
        elapsed = time.perf_counter() - start
        strong = (df["Signal"] == "Strong Long").sum() if not df.empty and "Signal" in df.columns else 0
        longs = (df["Signal"] == "Long").sum() if not df.empty and "Signal" in df.columns else 0
        logger.info(
            "[LATENCY] %s | mode=%s | elapsed=%.2fs | results=%d | strong=%d | long=%d | status=%s",
            description,
            override_settings,
            elapsed,
            len(df),
            strong,
            longs,
            status,
        )
    finally:
        restore()


def _pick_universe(n: int) -> List[str]:
    uni = load_universe()
    return [u.symbol for u in uni[:n]]


def main():
    sizes = [50, 250, 1000]
    modes = {
        "baseline": {"use_ml_alpha": False, "use_tft_features": False, "use_meta_alpha": False, "use_deep_alpha": False},
        "full": {},
    }

    for size in sizes:
        universe = _pick_universe(size)
        for mode, overrides in modes.items():
            time_scan(universe, f"{mode}-universe-{size}", overrides)


if __name__ == "__main__":
    main()
