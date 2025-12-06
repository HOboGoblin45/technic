"""Daily trade summary generator using a text-generation model (GPT-2)."""

from __future__ import annotations

try:
    from transformers import pipeline

    _generator = pipeline("text-generation", model="gpt2")
    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    HAVE_TRANSFORMERS = False
    _generator = None


def generate_daily_summary(logs: str, max_length: int = 150) -> str:
    """Generate a short summary of trades from raw log text."""
    if not HAVE_TRANSFORMERS or _generator is None:
        return logs
    try:
        summary = _generator(
            f"Summarize the following trades: {logs}", max_length=max_length
        )[0]["generated_text"]
        return summary
    except Exception:
        return logs


__all__ = ["generate_daily_summary"]
