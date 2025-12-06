"""Strategy clustering engine for grouping similar behaviors."""

from __future__ import annotations

try:
    from sklearn.cluster import KMeans  # type: ignore

    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    HAVE_SKLEARN = False


def cluster_strategies(feature_matrix, n_clusters: int = 4):
    """Group users with similar signal-following behaviors."""
    if not HAVE_SKLEARN:
        raise ImportError("scikit-learn required for strategy clustering.")
    model = KMeans(n_clusters=n_clusters, n_init="auto")
    labels = model.fit_predict(feature_matrix)
    return labels


__all__ = ["cluster_strategies"]
