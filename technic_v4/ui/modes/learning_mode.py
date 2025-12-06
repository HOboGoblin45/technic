"""Learning mode dashboard helpers for beginner-friendly explanations."""

from __future__ import annotations


def generate_learning_tip(signal) -> str:
    """Explain model reasoning in simpler terms per trade."""
    return (
        f"This trade was recommended because {signal['top_feature']} "
        "increased historically before rallies."
    )


__all__ = ["generate_learning_tip"]
