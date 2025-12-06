"""Operator onboarding CLI stub."""

from __future__ import annotations

import json
from pathlib import Path


def onboard():
    config = {
        "aws_configured": False,
        "agent_presets": "tbd",
        "cron_scheduled": False,
        "institutional_hash": "tbd",
    }
    Path("ops").mkdir(exist_ok=True)
    Path("ops/onboarding.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    onboard()
