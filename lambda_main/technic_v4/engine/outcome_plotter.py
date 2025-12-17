"""Multi-scenario outcome plotter for optimistic/expected/pessimistic paths."""

from __future__ import annotations

try:
    import matplotlib.pyplot as plt  # type: ignore

    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False


def plot_scenarios(low, mid, high, show: bool = True):
    """Plot pessimistic, expected, and optimistic scenarios on one chart."""
    if not HAVE_MPL:
        raise ImportError("matplotlib is required to plot scenarios.")
    fig, ax = plt.subplots()
    ax.plot(low, label="Pessimistic")
    ax.plot(mid, label="Expected")
    ax.plot(high, label="Optimistic")
    ax.legend()
    ax.set_title("Trade Outcome Scenarios")
    if show:
        plt.show()
    return fig, ax


__all__ = ["plot_scenarios"]
