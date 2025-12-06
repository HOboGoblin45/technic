"""Competitor signal analysis comparator with correlation heatmap."""

from __future__ import annotations

import pandas as pd

try:
    import seaborn as sns  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    HAVE_PLOTS = True
except ImportError:  # pragma: no cover
    HAVE_PLOTS = False


def compare_signals(technic_signals, external_signals, show: bool = True):
    """Compare Technic vs external signals via correlation heatmap."""
    if not HAVE_PLOTS:
        raise ImportError("seaborn/matplotlib are required for signal comparison plotting.")
    df = pd.DataFrame({"Technic": technic_signals, "External": external_signals})
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, ax=ax)
    ax.set_title("Signal Correlation Matrix")
    if show:
        plt.show()
    return fig, ax


__all__ = ["compare_signals"]
