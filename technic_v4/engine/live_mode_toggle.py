"""User-configurable trade mode toggle (simulate/paper/live)."""

from __future__ import annotations

from typing import Literal


class TradeMode:
    """Manage current trading mode with safe validation."""

    allowed_modes = ("simulate", "paper", "live")

    def __init__(self, mode: Literal["simulate", "paper", "live"] = "simulate"):
        if mode not in self.allowed_modes:
            raise ValueError(f"Invalid mode '{mode}'. Allowed: {self.allowed_modes}")
        self.mode = mode

    def set_mode(self, new_mode: str) -> None:
        """Set mode if valid; otherwise raise ValueError."""
        if new_mode not in self.allowed_modes:
            raise ValueError(f"Invalid mode '{new_mode}'. Allowed: {self.allowed_modes}")
        self.mode = new_mode

    def get_mode(self) -> str:
        """Return current mode."""
        return self.mode


__all__ = ["TradeMode"]
