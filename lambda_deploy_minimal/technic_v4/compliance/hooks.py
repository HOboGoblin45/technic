"""Governance compliance hooks for audit/oversight readiness."""

from __future__ import annotations


def compliance_hook(action_type: str, payload) -> None:
    """Insert compliance logging hooks."""
    print(f"[COMPLIANCE] {action_type}: {payload}")


__all__ = ["compliance_hook"]
