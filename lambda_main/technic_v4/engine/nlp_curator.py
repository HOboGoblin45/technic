"""LLM-aware NLP curation engine to normalize user prompts/descriptions."""

from __future__ import annotations

try:
    from transformers import pipeline

    _curate = pipeline("text2text-generation")
    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    HAVE_TRANSFORMERS = False
    _curate = None


def normalize_prompt(user_text: str) -> str:
    """Standardize a user-provided prompt for consistency."""
    if not HAVE_TRANSFORMERS or _curate is None:
        return user_text
    try:
        return _curate(f"Standardize this query: {user_text}", max_length=80)[0]["generated_text"]
    except Exception:
        return user_text


__all__ = ["normalize_prompt"]
