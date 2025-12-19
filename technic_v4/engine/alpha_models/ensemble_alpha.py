from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from technic_v4.engine.alpha_models.base import BaseAlphaModel


class EnsembleAlphaModel(BaseAlphaModel):
    """
    Simple weighted-mean ensemble for cross-sectional alpha models.

    Holds a list of BaseAlphaModel instances and combines their predictions
    using configurable weights (defaults to equal weights).
    """

    def __init__(self, models: Optional[List[BaseAlphaModel]] = None, weights: Optional[Iterable[float]] = None):
        self.models: List[BaseAlphaModel] = models or []
        self.weights = np.array(list(weights), dtype=float) if weights is not None else None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not self.models:
            raise ValueError("EnsembleAlphaModel requires at least one underlying model.")
        for m in self.models:
            m.fit(X, y)
        return None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.models:
            return pd.Series(index=X.index, dtype=float)
        preds = []
        for m in self.models:
            preds.append(m.predict(X))
        if not preds:
            return pd.Series(index=X.index, dtype=float)

        preds_mat = np.vstack([p.values for p in preds])
        if self.weights is None or len(self.weights) != preds_mat.shape[0]:
            w = np.ones(preds_mat.shape[0], dtype=float)
        else:
            w = self.weights
        weighted = np.average(preds_mat, axis=0, weights=w)
        return pd.Series(weighted, index=X.index)

    def save(self, path: str) -> None:
        payload = {
            "weights": self.weights,
            "models": self.models,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "EnsembleAlphaModel":
        payload = joblib.load(path)
        mdl = cls(models=payload.get("models") or [])
        weights = payload.get("weights")
        mdl.weights = np.array(weights, dtype=float) if weights is not None else None
        return mdl
