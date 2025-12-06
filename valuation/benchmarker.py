"""Institutional valuation benchmark."""

from __future__ import annotations

import json
from pathlib import Path

TOOLS = ["TradeIdeas", "Kavout", "Koyfin", "Ziggma", "Technic"]
METRICS = ["model_granularity", "data_latency", "nl_loop", "customizability", "alpha_refresh"]


def benchmark():
    matrix = {tool: {m: "tbd" for m in METRICS} for tool in TOOLS}
    Path("valuation").mkdir(exist_ok=True)
    Path("valuation/feature_matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    Path("valuation/internal_estimate.txt").write_text("Internal estimate tbd", encoding="utf-8")


if __name__ == "__main__":
    benchmark()
