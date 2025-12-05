from __future__ import annotations

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

import pandas as pd
try:
    from sklearn.linear_model import Ridge
except Exception:  # pragma: no cover
    Ridge = None

from technic_v4.engine.alpha_models.base import BaseAlphaModel


class MetaAlphaModel(BaseAlphaModel):
    """
    Meta-model to blend multiple signal sources into a final alpha score.
    Default regressor: Ridge.
    """

    def __init__(self, **kwargs) -> None:
        alpha = kwargs.pop("alpha", 1.0)
        if Ridge is None:
            raise ImportError("scikit-learn is required for MetaAlphaModel")
        self.model = Ridge(alpha=alpha)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)

    def save(self, path: str) -> None:
        if joblib is None:
            raise ImportError("joblib is required to save the meta alpha model")
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "MetaAlphaModel":
        if joblib is None:
            raise ImportError("joblib is required to load the meta alpha model")
        inst = cls()
        inst.model = joblib.load(path)
        return inst
