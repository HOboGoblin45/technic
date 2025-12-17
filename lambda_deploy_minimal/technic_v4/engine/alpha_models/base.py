from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseAlphaModel(ABC):
    """
    Abstract base interface for cross-sectional alpha models.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseAlphaModel":
        ...
