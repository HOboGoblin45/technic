"""NLP-based explanation simplifier to reduce jargon in LLM outputs."""

from __future__ import annotations

try:
    from transformers import pipeline

    _rewrite = pipeline("text2text-generation", model="pszemraj/pegasus-xsum-distill-base")
    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    HAVE_TRANSFORMERS = False
    _rewrite = None


def simplify_explanation(text: str) -> str:
    """Rewrite an explanation into simpler English if the model is available."""
    if not HAVE_TRANSFORMERS or _rewrite is None:
        return text
    try:
        out = _rewrite(text, max_length=50, do_sample=False)
        return out[0]["generated_text"]
    except Exception:
        return text


__all__ = ["simplify_explanation"]
