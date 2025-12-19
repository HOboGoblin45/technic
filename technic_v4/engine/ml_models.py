"""
Model wrappers for GPU-first alpha prediction with explainability hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import shap
import torch
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelResult:
    model: Any
    shap_explainer: Optional[Any]
    feature_importance: Optional[np.ndarray]


def train_lgbm_regressor(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    enable_shap: bool = True,
) -> ModelResult:
    """
    Train a LightGBM regressor on GPU (if available). Returns model and SHAP explainer.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    base_params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": -1,
        "num_leaves": 127,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "regression",
        "device": "gpu",
        "verbosity": -1,
    }
    if params:
        base_params.update(params)

    model = lgb.LGBMRegressor(**base_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        verbose=False,
    )

    explainer = None
    if enable_shap:
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = None

    fi = getattr(model, "feature_importances_", None)
    return ModelResult(model=model, shap_explainer=explainer, feature_importance=fi)


def explain_predictions(
    model_result: ModelResult, X_sample: np.ndarray, max_samples: int = 200
) -> Optional[np.ndarray]:
    """
    Compute SHAP values for a sample batch.
    """
    if model_result.shap_explainer is None:
        return None
    subset = X_sample[:max_samples]
    try:
        shap_vals = model_result.shap_explainer.shap_values(subset)
        return shap_vals
    except Exception:
        return None
