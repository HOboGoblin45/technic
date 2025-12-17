from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from technic_v4.config.settings import get_settings

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "meta" / "winprob_10d.pkl"


_META_MODEL = None


def _load_model():
    global _META_MODEL
    if _META_MODEL is not None:
        return _META_MODEL

    if not MODEL_PATH.exists():
        logger.warning(
            "[META] win_prob_10d model not found at %s; skipping meta inference.",
            MODEL_PATH,
        )
        _META_MODEL = None
        return None

    import joblib

    _META_MODEL = joblib.load(MODEL_PATH)
    logger.info("[META] Loaded win_prob_10d model from %s", MODEL_PATH)
    return _META_MODEL


def _select_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and order the features for meta win_prob_10d inference.
    Must match the feature engineering used in train_meta_model.py.
    We use a "best effort" feature subset, dropping missing columns.
    """
    candidate_cols: List[str] = [
        "InstitutionalCoreScore",
        "TechRating",
        "alpha_blend",
        "AlphaScorePct",
        "QualityScore",
        "fundamental_quality_score",
        "ExplosivenessScore",
        "DollarVolume",
        "ATR14_pct",
        "RiskScore",
        "risk_score",
        # Any regime flags used at training time:
        "regime_trend_TRENDING_UP",
        "regime_trend_TRENDING_DOWN",
        "regime_trend_SIDEWAYS",
        "regime_vol_HIGH_VOL",
        "regime_vol_LOW_VOL",
    ]
    cols = [c for c in candidate_cols if c in df.columns]
    if not cols:
        raise ValueError(
            "[META] No meta features found in df; available columns: "
            f"{list(df.columns)[:20]}..."
        )
    feats = df[cols].copy()
    # Ensure numeric
    feats = feats.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return feats


def score_win_prob_10d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'win_prob_10d' column to df using the trained meta-model.

    If the model is missing, logs a warning and leaves df unchanged.
    """
    model = _load_model()
    if model is None:
        logger.warning("[META] No win_prob_10d model; skipping meta inference.")
        df["win_prob_10d"] = np.nan
        return df

    # If the saved object is a bundle dict, unwrap model/features if present
    model_obj = model
    features = None
    if isinstance(model, dict):
        model_obj = model.get("model", model)
        features = model.get("features")

    feats = _select_meta_features(df)
    if features:
        # Reorder/align to training feature order when available
        cols = [c for c in features if c in feats.columns]
        if cols:
            feats = feats[cols]

    # Support both scikit-learn style predict_proba and regressor-style predict
    if hasattr(model_obj, "predict_proba"):
        probs = model_obj.predict_proba(feats.values)[:, 1]
    else:
        preds = model_obj.predict(feats.values)
        # Clip to [0, 1] as a probability
        probs = np.clip(preds, 0.0, 1.0)

    df = df.copy()
    df["win_prob_10d"] = probs
    logger.info("[META] Scored win_prob_10d for %d rows.", df.shape[0])
    return df
