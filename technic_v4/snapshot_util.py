"""Diagnostic snapshot tool."""

from __future__ import annotations

import os
import json
from datetime import datetime


def save_diagnostic_snapshot(model_state, signals, latency, diff):
    folder = f"./snapshots/{datetime.now().strftime('%Y%m%d')}/"
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "snapshot.json"), "w") as f:
        json.dump(
            {
                "model": str(model_state),
                "signals": signals,
                "latency": latency,
                "return_diff": diff,
            },
            f,
        )


__all__ = ["save_diagnostic_snapshot"]
