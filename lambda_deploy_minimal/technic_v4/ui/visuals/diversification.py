"""Portfolio diversification heatmap visualization."""

from __future__ import annotations

try:
    import seaborn as sns  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    HAVE_PLOTS = True
except ImportError:  # pragma: no cover
    HAVE_PLOTS = False


def plot_diversification(allocation_dict):
    """Generate a sector/asset heatmap of current portfolio allocation."""
    if not HAVE_PLOTS:
        raise ImportError("seaborn/matplotlib required for diversification heatmap.")
    sns.heatmap(
        [list(allocation_dict.values())],
        annot=True,
        xticklabels=allocation_dict.keys(),
        yticklabels=["Allocation"],
    )
    plt.title("Portfolio Diversification Heatmap")
    plt.show()


__all__ = ["plot_diversification"]
