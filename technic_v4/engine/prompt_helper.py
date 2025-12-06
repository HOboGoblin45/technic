"""Context-aware LLM prompt generator."""

from __future__ import annotations


def generate_prompt(symbol: str, sentiment_score: float, macro_state: str) -> str:
    """Create a contextual prompt for LLM responses."""
    context = f"{symbol} is in a {macro_state} regime with sentiment score {sentiment_score:.2f}."
    return f"Given {context}, what is the expected move over 5 days?"


__all__ = ["generate_prompt"]
