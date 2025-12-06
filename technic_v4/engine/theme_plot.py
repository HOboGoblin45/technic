"""Portfolio theme visualizer using PCA + KMeans."""

from __future__ import annotations

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt  # type: ignore

    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False


def plot_themes(embeddings, tickers, n_clusters: int = 5, show: bool = True):
    """Cluster and plot portfolio assets by sector/theme."""
    if not HAVE_DEPS:
        raise ImportError("sklearn/matplotlib required for theme plotting.")

    reduced = PCA(n_components=2).fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(reduced)
    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=kmeans.labels_)
    for i, t in enumerate(tickers):
        ax.annotate(t, (reduced[i, 0], reduced[i, 1]))
    ax.set_title("Thematic Portfolio Clusters")
    if show:
        plt.show()
    return fig, scatter


__all__ = ["plot_themes"]
