from __future__ import annotations

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

import pandas as pd
try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

from technic_v4.engine.alpha_models.base import BaseAlphaModel


class XGBAlphaModel(BaseAlphaModel):
    """
    XGBoost-based cross-sectional alpha model.
    """

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        tree_method: str = "hist",
        **kwargs,
    ) -> None:
        if xgb is None:
            raise ImportError("xgboost is required for XGBAlphaModel")
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "tree_method": tree_method,
            "objective": "reg:squarederror",
        }
        params.update(kwargs)
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)

    def save(self, path: str) -> None:
        if joblib is None:
            raise ImportError("joblib is required to save the model")
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "XGBAlphaModel":
        if joblib is None:
            raise ImportError("joblib is required to load the model")
        inst = cls()
        inst.model = joblib.load(path)
        return inst
