"""Competitive return heatmap / bar comparison."""

from __future__ import annotations

try:
    import matplotlib.pyplot as plt  # type: ignore

    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False


def plot_peer_comparison(technic_return, competitor_returns, show: bool = True):
    """Compare Technic alpha performance vs peers."""
    if not HAVE_MPL:
        raise ImportError("matplotlib is required for peer comparison plotting.")
    names = list(competitor_returns.keys()) + ["Technic"]
    values = list(competitor_returns.values()) + [technic_return]
    fig, ax = plt.subplots()
    ax.bar(names, values)
    ax.set_title("Technic vs Competitor Alpha Returns")
    if show:
        plt.show()
    return fig, ax


__all__ = ["plot_peer_comparison"]
