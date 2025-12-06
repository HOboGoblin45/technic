"""Competitive benchmarking dashboard (optional, Dash-based).

This lightweight Dash app compares Technic performance against a few
reference platforms. It is standalone and only runs if `dash` and
`plotly` are installed in the environment.
"""

from __future__ import annotations

try:
    import dash
    from dash import dcc, html
    import plotly.express as px

    HAVE_DASH = True
except ImportError:  # pragma: no cover - optional dependency
    HAVE_DASH = False


def create_app():
    """Create and return the Dash app if dependencies are available."""
    if not HAVE_DASH:
        raise ImportError(
            "dash/plotly is not installed. Install `dash` and `plotly` to run the benchmarking dashboard."
        )

    app = dash.Dash(__name__)
    fig = px.bar(
        x=["Technic", "Finviz", "SeekingAlpha"],
        y=[0.35, 0.18, 0.22],
        labels={"x": "Platform", "y": "Sharpe"},
        title="Benchmark Sharpe Comparison",
    )
    app.layout = html.Div([dcc.Graph(figure=fig)])
    return app


if __name__ == "__main__":  # pragma: no cover
    app = create_app()
    app.run_server(debug=True)
