from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from technic_v4.config.settings import get_settings
from technic_v4.infra.logging import get_logger

logger = get_logger()


def tft_predict_horizons(symbol: str, history_df: pd.DataFrame) -> Dict[str, float]:
    """
    Returns horizon forecasts like:
    {"tft_horizon_5d": ..., "tft_horizon_10d": ...}

    If TFT is disabled or no model artifact is present, returns {}.
    Currently a stub; replace with real TFT/ONNX inference when available.
    """
    settings = get_settings()
    if not getattr(settings, "use_tft_features", False):
        return {}

    # TODO: wire to real TFT model (PyTorch Lightning or ONNX) when available.
    # Placeholder: check for an artifact path to decide whether to emit anything.
    model_path = Path("models/tft_price_forecast.ckpt")
    if not model_path.exists():
        logger.info("[tft] model not found at %s; skipping TFT features", model_path)
        return {}

    # Stub prediction: return empty until real model is loaded.
    logger.info("[tft] placeholder inference for %s; model loading not implemented", symbol)
    return {}


__all__ = ["tft_predict_horizons"]
