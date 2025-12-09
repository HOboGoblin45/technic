"""
Optional ML alpha inference.

Loads default alpha models (LGBM/XGB), optional deep model, and meta alpha.
Intended as a drop-in enhancer for the factor-based alpha blend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

from technic_v4 import model_registry
from technic_v4.config.settings import get_settings
from technic_v4.engine import inference_engine
from technic_v4.engine.alpha_models import (
    BaseAlphaModel,
    LGBMAlphaModel,
    XGBAlphaModel,
    EnsembleAlphaModel,
)
from technic_v4.engine.alpha_models.meta_alpha import MetaAlphaModel
from technic_v4.infra.logging import get_logger

logger = get_logger()

# Optional torch typing for pylance
if TYPE_CHECKING:  # pragma: no cover
    import torch as torch_mod
    import torch.nn as nn_mod
else:
    torch_mod = None
    nn_mod = None

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False
    torch = None
    nn = None


DEFAULT_MODEL_PATH = Path("models/alpha/lgbm_v1.pkl")
DEFAULT_ONNX_PATH = Path("models/alpha/lgbm_v1.onnx")
DEFAULT_META_MODEL_PATH = Path("models/alpha/meta_alpha.pkl")
DEFAULT_DEEP_MODEL_PATH = Path("models/alpha/deep_alpha.pt")

# Local XGB bundles (non-registry) for alpha_xgb_v1 (5d) and alpha_xgb_v1_10d (10d)
_XGB_BUNDLE_5D: Optional[dict] = None
_XGB_BUNDLE_10D: Optional[dict] = None
_XGB_MODEL_PATH_5D = os.getenv("TECHNIC_ALPHA_MODEL_PATH", "models/alpha/xgb_v1.pkl")
_XGB_MODEL_PATH_10D = os.getenv(
    "TECHNIC_ALPHA_MODEL_PATH_10D", "models/alpha/xgb_v1_10d.pkl"
)


def _load_xgb_bundle(path: str, cache: dict | None) -> tuple[Optional[dict], dict | None]:
    """Internal helper to load an XGB joblib bundle with simple caching."""
    if cache is not None:
        return cache, cache
    if not os.path.exists(path):
        raise FileNotFoundError(f"Alpha model bundle not found at {path}")
    logger.info("[ALPHA] loading XGB bundle from %s", path)
    bundle = joblib.load(path)
    return bundle, bundle


def load_xgb_bundle_5d() -> dict:
    """Lazy-load 5d XGB alpha model bundle."""
    global _XGB_BUNDLE_5D
    bundle, _XGB_BUNDLE_5D = _load_xgb_bundle(_XGB_MODEL_PATH_5D, _XGB_BUNDLE_5D)
    return bundle  # type: ignore[return-value]


def load_xgb_bundle_10d() -> dict:
    """Lazy-load 10d XGB alpha model bundle (if present)."""
    global _XGB_BUNDLE_10D
    bundle, _XGB_BUNDLE_10D = _load_xgb_bundle(_XGB_MODEL_PATH_10D, _XGB_BUNDLE_10D)
    return bundle  # type: ignore[return-value]


# -------------------------------
# Core model loading helpers (LGBM / registry)
# -------------------------------


def _load_model_by_entry(entry: dict) -> Optional[BaseAlphaModel]:
    if not entry:
        return None
    path = entry.get("path_pickle")
    if not path:
        return None
    model_name = entry.get("model_name", "")
    try:
        if model_name == "alpha_xgb_v1":
            logger.info("[alpha] loading XGB model from %s", path)
            return XGBAlphaModel.load(path)
        if model_name == "alpha_ensemble_v1":
            logger.info("[alpha] loading ensemble model from %s", path)
            return EnsembleAlphaModel.load(path)
        logger.info("[alpha] loading LGBM model from %s", path)
        return LGBMAlphaModel.load(path)
    except Exception:
        logger.warning("[alpha] failed to load model at %s", path, exc_info=True)
        return None


def load_default_alpha_model() -> Optional[BaseAlphaModel]:
    reg_entry = None
    try:
        reg_entry = model_registry.load_model("alpha_lgbm_v1")
    except Exception:
        reg_entry = None
    settings = get_settings()
    env_model_name = settings.alpha_model_name or os.getenv("TECHNIC_ALPHA_MODEL_NAME")
    if env_model_name:
        try:
            reg_entry = model_registry.load_model(env_model_name)
        except Exception:
            pass
    if reg_entry:
        loaded = _load_model_by_entry(reg_entry)
        if loaded:
            return loaded
    model_path = (
        Path(reg_entry["path_pickle"])
        if reg_entry and reg_entry.get("path_pickle")
        else DEFAULT_MODEL_PATH
    )
    if not model_path.exists() and DEFAULT_MODEL_PATH.exists():
        model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        return None
    try:
        logger.info("[alpha] loading default LGBM model from %s", model_path)
        return LGBMAlphaModel.load(str(model_path))
    except Exception:
        logger.warning(
            "[alpha] failed to load default model %s", model_path, exc_info=True
        )
        return None


def _score_with_bundle(
    df_features: pd.DataFrame, bundle: dict, label: str = "5d"
) -> Optional[pd.Series]:
    """Helper to score df_features with a given XGB bundle."""
    model = bundle.get("model")
    feature_cols = bundle.get("features") or []

    if model is None or not feature_cols:
        logger.warning("[ALPHA] XGB bundle missing model or feature list for %s", label)
        return None

    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        logger.warning(
            "[ALPHA] missing feature columns for %s XGB model: %s", label, missing
        )
        return None

    X = (
        df_features[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    try:
        preds = model.predict(X)
        logger.info("[ALPHA] %s XGB prediction succeeded, first few=%s", label, preds[:5])
        return pd.Series(preds, index=df_features.index)
    except Exception:
        logger.warning("[ALPHA] %s XGB prediction failed", label, exc_info=True)
        return None

def score_alpha(df_features: pd.DataFrame) -> Optional[pd.Series]:
    """
    Score ML alpha (5-day horizon) using the local XGB bundle (models/alpha/xgb_v1.pkl).
    This assumes df_features already contains the feature columns referenced by the bundle.
    """
    if df_features is None or df_features.empty:
        return None

    logger.info(
        "[ALPHA] score_alpha called with shape=%s columns=%s",
        df_features.shape,
        list(df_features.columns),
    )

    try:
        bundle = load_xgb_bundle_5d()
    except FileNotFoundError:
        logger.warning(
            "[ALPHA] XGB 5d bundle not found at %s; returning None", _XGB_MODEL_PATH_5D
        )
        return None
    except Exception:
        logger.warning("[ALPHA] failed to load 5d XGB bundle", exc_info=True)
        return None

    return _score_with_bundle(df_features, bundle, label="5d")


def score_alpha_10d(df_features: pd.DataFrame) -> Optional[pd.Series]:
    """
    Score ML alpha (10-day horizon) using a second XGB bundle (models/alpha/xgb_v1_10d.pkl).
    Returns None if the model is missing or prediction fails.
    """
    if df_features is None or df_features.empty:
        return None

    logger.info(
        "[ALPHA] score_alpha_10d called with shape=%s columns=%s",
        df_features.shape,
        list(df_features.columns),
    )

    try:
        bundle = load_xgb_bundle_10d()
    except FileNotFoundError:
        logger.info(
            "[ALPHA] XGB 10d bundle not found at %s; skipping 10d alpha",
            _XGB_MODEL_PATH_10D,
        )
        return None
    except Exception:
        logger.warning("[ALPHA] failed to load 10d XGB bundle", exc_info=True)
        return None

    return _score_with_bundle(df_features, bundle, label="10d")

# -------------------------------
# Meta alpha
# -------------------------------


def load_meta_alpha_model() -> Optional[BaseAlphaModel]:
    path = DEFAULT_META_MODEL_PATH
    if not path.exists():
        return None
    try:
        logger.info("[alpha] loading meta model from %s", path)
        return MetaAlphaModel.load(str(path))
    except Exception:
        logger.warning(
            "[alpha] failed to load meta model %s", path, exc_info=True
        )
        return None


def score_meta_alpha(df: pd.DataFrame) -> Optional[pd.Series]:
    model = load_meta_alpha_model()
    if model is None:
        return None
    cols: list[str] = []
    for col in ["factor_alpha", "ml_alpha", "AlphaScore"]:
        if col in df.columns:
            cols.append(col)
    cols.extend([c for c in df.columns if c.startswith("tft_forecast_h")])
    cols.extend(
        [
            c
            for c in df.columns
            if c.startswith("regime_trend_") or c.startswith("regime_vol_")
        ]
    )
    if not cols:
        return None
    try:
        X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return model.predict(X)
    except Exception:
        return None


# -------------------------------
# Deep alpha (sequential) placeholder
# -------------------------------
if HAVE_TORCH:

    class _TinyLSTM(nn.Module):  # pragma: no cover
        def __init__(self, input_dim: int, hidden_dim: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            out = self.head(h[-1])
            return out.squeeze(-1)

else:
    _TinyLSTM = None  # type: ignore


def load_deep_alpha_model(path: Path = DEFAULT_DEEP_MODEL_PATH):
    if not HAVE_TORCH or _TinyLSTM is None:
        return None
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu")
        input_dim = (
            state.get("meta", {}).get("input_dim", 4)
            if isinstance(state, dict)
            else 4
        )
        model = _TinyLSTM(input_dim=input_dim)
        model.load_state_dict(state.get("state_dict", state))
        model.eval()
        return model
    except Exception:
        return None


def _build_seq_from_history(
    history_df: pd.DataFrame, seq_len: int = 60
) -> Optional[np.ndarray]:
    if history_df is None or history_df.empty:
        return None
    df = history_df.tail(seq_len + 1).copy()
    if df.empty or len(df) < seq_len:
        return None
    df["ret"] = df["Close"].pct_change()
    df["vol"] = df["Volume"].pct_change().fillna(0)
    df["rsi_proxy"] = df["Close"].rolling(14).apply(
        lambda x: (
            x.diff().clip(lower=0).mean()
            / (x.diff().abs().mean() + 1e-6)
        )
    )
    feats = (
        df[["ret", "vol", "rsi_proxy"]]
        .fillna(0)
        .tail(seq_len)
        .values.astype("float32")
    )
    return feats


def score_deep_alpha_single(history_df: pd.DataFrame) -> Optional[float]:
    """
    Score deep alpha for a single symbol using recent history.
    """
    if not HAVE_TORCH:
        return None
    model = load_deep_alpha_model()
    if model is None:
        return None
    seq = _build_seq_from_history(history_df)
    if seq is None:
        return None
    with torch.no_grad():
        tens = torch.tensor(seq).unsqueeze(0)  # (1, seq_len, feat_dim)
        out = model(tens).cpu().numpy().ravel()
        return float(out[0]) if len(out) > 0 else None
    return None
