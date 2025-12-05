"""
Optional ML alpha inference.

Loads a default alpha model (if present) and scores cross-sectional features.
Intended to be a drop-in enhancer for the factor-based alpha blend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from technic_v4.engine.alpha_models import LGBMAlphaModel, BaseAlphaModel
from technic_v4.engine import inference_engine


DEFAULT_MODEL_PATH = Path("models/alpha/lgbm_v1.pkl")
DEFAULT_ONNX_PATH = Path("models/alpha/lgbm_v1.onnx")


def load_default_alpha_model() -> Optional[BaseAlphaModel]:
    """
    Load the default alpha model if the artifact exists.
    Returns None if not found or failed to load.
    """
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
    use_onnx = str(os.getenv("TECHNIC_USE_ONNX_ALPHA", "false")).lower() in {"1", "true", "yes"}
    if use_onnx and DEFAULT_ONNX_PATH.exists():
        sess = inference_engine.load_onnx_session(str(DEFAULT_ONNX_PATH))
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
