"""Meta-signal composer using PCA + ridge regression."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def build_meta_signal(signal_matrix: np.ndarray, target: np.ndarray, n_components: int = 5) -> tuple[np.ndarray, PCA, Ridge]:
    """Construct a meta-signal from individual signal outputs."""
    pca = PCA(n_components=min(n_components, signal_matrix.shape[1]))
    comps = pca.fit_transform(signal_matrix)
    model = Ridge(alpha=1.0)
    model.fit(comps, target)
    meta_signal = model.predict(comps)
    return meta_signal, pca, model


__all__ = ["build_meta_signal"]
