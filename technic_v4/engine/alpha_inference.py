"""
Optional ML alpha inference.

Loads default alpha models (LGBM/XGB), optional deep model, and meta alpha.
Intended as a drop-in enhancer for the factor-based alpha blend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from technic_v4 import model_registry
from technic_v4.engine import inference_engine
from technic_v4.engine.alpha_models import BaseAlphaModel, LGBMAlphaModel, XGBAlphaModel
from technic_v4.engine.alpha_models.meta_alpha import MetaAlphaModel

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False
    torch = None
    nn = None


DEFAULT_MODEL_PATH = Path("models/alpha/lgbm_v1.pkl")
DEFAULT_ONNX_PATH = Path("models/alpha/lgbm_v1.onnx")
DEFAULT_META_MODEL_PATH = Path("models/alpha/meta_alpha.pkl")
DEFAULT_DEEP_MODEL_PATH = Path("models/alpha/deep_alpha.pt")


# -------------------------------
# Core model loading helpers
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
            return XGBAlphaModel.load(path)
        return LGBMAlphaModel.load(path)
    except Exception:
        return None


def load_default_alpha_model() -> Optional[BaseAlphaModel]:
    reg_entry = None
    try:
        reg_entry = model_registry.get_active_model("alpha_lgbm_v1") or model_registry.get_latest_model("alpha_lgbm_v1")
    except Exception:
        reg_entry = None
    env_model_name = os.getenv("TECHNIC_ALPHA_MODEL_NAME")
    if env_model_name:
        try:
            reg_entry = model_registry.get_active_model(env_model_name) or model_registry.get_latest_model(env_model_name)
        except Exception:
            pass
    if reg_entry:
        loaded = _load_model_by_entry(reg_entry)
        if loaded:
            return loaded
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
    reg_entry = None
    try:
        preferred = os.getenv("TECHNIC_ALPHA_MODEL_NAME")
        if preferred:
            reg_entry = model_registry.get_active_model(preferred) or model_registry.get_latest_model(preferred)
        if reg_entry is None:
            reg_entry = model_registry.get_active_model("alpha_lgbm_v1") or model_registry.get_latest_model("alpha_lgbm_v1")
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


# -------------------------------
# Meta alpha
# -------------------------------

def load_meta_alpha_model() -> Optional[BaseAlphaModel]:
    path = DEFAULT_META_MODEL_PATH
    if not path.exists():
        return None
    try:
        return MetaAlphaModel.load(str(path))
    except Exception:
        return None


def score_meta_alpha(df: pd.DataFrame) -> Optional[pd.Series]:
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
        input_dim = state.get("meta", {}).get("input_dim", 4) if isinstance(state, dict) else 4
        model = _TinyLSTM(input_dim=input_dim)
        model.load_state_dict(state.get("state_dict", state))
        model.eval()
        return model
    except Exception:
        return None


def _build_seq_from_history(history_df: pd.DataFrame, seq_len: int = 60) -> Optional[np.ndarray]:
    if history_df is None or history_df.empty:
        return None
    df = history_df.tail(seq_len + 1).copy()
    if df.empty or len(df) < seq_len:
        return None
    df["ret"] = df["Close"].pct_change()
    df["vol"] = df["Volume"].pct_change().fillna(0)
    df["rsi_proxy"] = df["Close"].rolling(14).apply(lambda x: (x.diff().clip(lower=0).mean() / (x.diff().abs().mean() + 1e-6)))
    feats = df[["ret", "vol", "rsi_proxy"]].fillna(0).tail(seq_len).values.astype("float32")
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
