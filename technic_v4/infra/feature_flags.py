"""Institutional feature toggles."""

from __future__ import annotations

import os
import yaml
from pathlib import Path

DEFAULT_FLAGS = {
    "IS_ENTERPRISE_MODE": False,
    "DISABLE_PUBLIC_API": False,
    "ENFORCE_QUANT_REVIEW": False,
}


def load_flags(path: Path = Path("config/feature_flags.yaml")) -> dict:
    flags = dict(DEFAULT_FLAGS)
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            flags.update(loaded)
        except Exception:
            pass
    # ENV overrides
    for k in DEFAULT_FLAGS:
        if os.getenv(k) is not None:
            flags[k] = str(os.getenv(k)).lower() in {"1", "true", "yes"}
    return flags


__all__ = ["load_flags", "DEFAULT_FLAGS"]
