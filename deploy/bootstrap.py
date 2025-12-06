"""Deployment bootstrapper readiness report."""

from __future__ import annotations

import os
import json
from pathlib import Path

CRITICAL_VARS = ["POLYGON_API_KEY", "TECHNIC_API_BASE"]


def check_readiness() -> dict:
    missing = [v for v in CRITICAL_VARS if not os.getenv(v)]
    report = {
        "missing_env": missing,
        "has_config": Path("config/mode_flags.yaml").exists(),
        "has_requirements": Path("requirements.txt").exists(),
    }
    return report


def main():
    report = check_readiness()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
