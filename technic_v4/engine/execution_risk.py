"""Execution risk plotting utilities."""

from __future__ import annotations

try:
    import matplotlib.pyplot as plt  # type: ignore

    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False


def plot_execution_risk(volumes, slippage, show: bool = True):
    """Scatter plot of slippage vs trade volume for calibration."""
    if not HAVE_MPL:
        raise ImportError("matplotlib is not installed; cannot plot execution risk.")

    fig, ax = plt.subplots()
    ax.scatter(volumes, slippage)
    ax.set_xlabel("Volume")
    ax.set_ylabel("Slippage")
    ax.set_title("Execution Risk by Trade Volume")
    ax.grid(True)
    if show:
        plt.show()
    return fig


__all__ = ["plot_execution_risk"]
