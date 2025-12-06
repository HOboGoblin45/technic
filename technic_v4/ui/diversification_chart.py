"""Portfolio diversification radar chart renderer."""

from __future__ import annotations

try:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False


def render_radar_chart(factor_data):
    """Visualize factor exposure across sectors/momentum/volatility."""
    if not HAVE_DEPS:
        raise ImportError("matplotlib and numpy required for radar chart.")
    labels = list(factor_data.keys())
    values = list(factor_data.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.show()
    return fig, ax


__all__ = ["render_radar_chart"]
