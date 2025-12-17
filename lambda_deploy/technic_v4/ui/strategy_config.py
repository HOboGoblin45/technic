"""User strategy control panel configuration defaults."""

from __future__ import annotations

user_config = {
    "risk_tolerance": "moderate",
    "use_options_model": True,
    "alpha_modules": ["momentum", "flow", "nlp"],
}

__all__ = ["user_config"]
