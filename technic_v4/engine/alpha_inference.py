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


DEFAULT_MODEL_PATH = Path("models/alpha/lgbm_v1.pkl")


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
    model = load_default_alpha_model()
    if model is None:
        return None
    try:
        return model.predict(df_features)
    except Exception:
        return None
