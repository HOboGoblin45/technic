from __future__ import annotations

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

import pandas as pd
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

from technic_v4.engine.alpha_models.base import BaseAlphaModel


class LGBMAlphaModel(BaseAlphaModel):
    """
    LightGBM-based cross-sectional alpha model.
    """

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        num_leaves: int = 127,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        device: str = "gpu",
        **kwargs,
    ) -> None:
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "regression",
            "device": device,
            "verbosity": -1,
        }
        params.update(kwargs)
        if lgb is None:
            raise ImportError("lightgbm is required for LGBMAlphaModel")
        self.model = lgb.LGBMRegressor(**params)

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
    def load(cls, path: str) -> "LGBMAlphaModel":
        inst = cls()
        if joblib is None:
            raise ImportError("joblib is required to load the model")
        inst.model = joblib.load(path)
        return inst
