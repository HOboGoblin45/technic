from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

from technic_v4.config.settings import get_settings
from technic_v4.scanner_core import run_scan, ScanConfig, OUTPUT_DIR


SHADOW_LOG_PATH = OUTPUT_DIR / "shadow_signals.csv"


def run_shadow_scan(config: ScanConfig) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Run baseline (no ML/TFT/deep/meta) and new engine scans for the same config.
    Returns (df_baseline, df_new, status_new).
    """
    settings = get_settings()
    # Snapshot flags
    saved = {
        "use_ml_alpha": settings.use_ml_alpha,
        "use_tft_features": settings.use_tft_features,
        "use_deep_alpha": settings.use_deep_alpha,
        "use_meta_alpha": settings.use_meta_alpha,
        "use_explainability": settings.use_explainability,
        "enable_shadow_mode": settings.enable_shadow_mode,
    }
    try:
        # Baseline flags off
        settings.use_ml_alpha = False
        settings.use_tft_features = False
        settings.use_deep_alpha = False
        settings.use_meta_alpha = False
        settings.use_explainability = False
        settings.enable_shadow_mode = False
        df_baseline, _ = run_scan(config=config)

        # New engine with original flags (but shadow disabled to avoid recursion)
        for k, v in saved.items():
            setattr(settings, k, v)
        settings.enable_shadow_mode = False
        df_new, status_new = run_scan(config=config)

        # Tag modes
        if df_baseline is not None:
            df_baseline["engine_mode"] = "baseline"
        if df_new is not None:
            df_new["engine_mode"] = "new"
        return df_baseline, df_new, status_new
    finally:
        # Restore flags
        for k, v in saved.items():
            setattr(settings, k, v)


def append_shadow_log(as_of_date: str, df_baseline: pd.DataFrame, df_new: pd.DataFrame) -> None:
    """
    Append side-by-side baseline/new signals to CSV.
    """
    rows = []
    for df in (df_baseline, df_new):
        if df is None or df.empty:
            continue
        subset = df.copy()
        subset["as_of_date"] = as_of_date
        keep_cols = ["as_of_date", "engine_mode", "Symbol", "Signal", "TechRating", "AlphaScore"]
        for col in keep_cols:
            if col not in subset.columns:
                subset[col] = None
        rows.append(subset[keep_cols])
    if not rows:
        return
    out = pd.concat(rows, ignore_index=True)
    SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SHADOW_LOG_PATH, mode="a", header=not SHADOW_LOG_PATH.exists(), index=False)


__all__ = ["run_shadow_scan", "append_shadow_log", "SHADOW_LOG_PATH"]
