"""
Optional ML alpha inference.

Loads a default alpha model (if present) and scores cross-sectional features.
Intended to be a drop-in enhancer for the factor-based alpha blend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from technic_v4 import model_registry
from technic_v4.engine.alpha_models import LGBMAlphaModel, BaseAlphaModel
from technic_v4.engine import inference_engine
from technic_v4.engine.alpha_models.meta_alpha import MetaAlphaModel


DEFAULT_MODEL_PATH = Path("models/alpha/lgbm_v1.pkl")
DEFAULT_ONNX_PATH = Path("models/alpha/lgbm_v1.onnx")
DEFAULT_META_MODEL_PATH = Path("models/alpha/meta_alpha.pkl")


def load_default_alpha_model() -> Optional[BaseAlphaModel]:
    """
    Load the default alpha model if the artifact exists.
    Returns None if not found or failed to load.
    """
    reg_entry = None
    try:
        reg_entry = model_registry.get_latest_model("alpha_lgbm_v1")
    except Exception:
        reg_entry = None
    model_path = Path(reg_entry["path_pickle"]) if reg_entry and reg_entry.get("path_pickle") else DEFAULT_MODEL_PATH
    if not model_path.exists() and DEFAULT_MODEL_PATH.exists():
        model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        return None
    try:
        return LGBMAlphaModel.load(str(model_path))
    except Exception:
        return None


def score_alpha(df_features: pd.DataFrame) -> Optional[pd.Series]:
    """
    Score alpha given a feature DataFrame.
    Returns a Series aligned to df_features.index, or None if model unavailable.
    """
    reg_entry = None
    try:
        reg_entry = model_registry.get_latest_model("alpha_lgbm_v1")
    except Exception:
        reg_entry = None
    use_onnx = str(os.getenv("TECHNIC_USE_ONNX_ALPHA", "false")).lower() in {"1", "true", "yes"}
    candidate_onnx = Path(reg_entry["path_onnx"]) if reg_entry and reg_entry.get("path_onnx") else DEFAULT_ONNX_PATH
    if not candidate_onnx.exists() and DEFAULT_ONNX_PATH.exists():
        candidate_onnx = DEFAULT_ONNX_PATH
    if use_onnx and candidate_onnx.exists():
        sess = inference_engine.load_onnx_session(str(candidate_onnx))
        if sess is not None:
            try:
                return inference_engine.onnx_predict(sess, df_features)
            except Exception:
                pass
    model = load_default_alpha_model()
    if model is None:
        return None
    try:
        return model.predict(df_features)
    except Exception:
        return None


def load_meta_alpha_model() -> Optional[BaseAlphaModel]:
    """
    Load meta alpha model if present.
    """
    path = DEFAULT_META_MODEL_PATH
    if not path.exists():
        return None
    try:
        return MetaAlphaModel.load(str(path))
    except Exception:
        return None


def score_meta_alpha(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Score meta alpha using available blended features.
    """
    model = load_meta_alpha_model()
    if model is None:
        return None
    cols = []
    for col in ["factor_alpha", "ml_alpha", "AlphaScore"]:
        if col in df.columns:
            cols.append(col)
    cols.extend([c for c in df.columns if c.startswith("tft_forecast_h")])
    cols.extend([c for c in df.columns if c.startswith("regime_trend_") or c.startswith("regime_vol_")])
    if not cols:
        return None
    try:
        X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return model.predict(X)
    except Exception:
        return None
