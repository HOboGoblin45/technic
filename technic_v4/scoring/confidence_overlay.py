"""Model confidence score overlay visualization."""

from __future__ import annotations

try:
    import matplotlib.pyplot as plt  # type: ignore

    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False


def confidence_overlay_plot(probabilities, labels, show: bool = True):
    """Visualize prediction confidence alongside labels."""
    if not HAVE_MPL:
        raise ImportError("matplotlib is required for confidence overlay plotting.")
    fig, ax = plt.subplots()
    scatter = ax.scatter(range(len(probabilities)), probabilities, c=labels, cmap="coolwarm")
    ax.set_title("Prediction Confidence Overlay")
    if show:
        plt.show()
    return fig, scatter


__all__ = ["confidence_overlay_plot"]
