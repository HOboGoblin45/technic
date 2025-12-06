"""NLP confidence qualifier for LLM explanations."""

from __future__ import annotations

try:
    from transformers import pipeline

    _explainer = pipeline("text-classification", return_all_scores=True)
    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    HAVE_TRANSFORMERS = False
    _explainer = None


def annotate_confidence(text: str) -> str:
    """Append a confidence score (softmax) to a piece of text."""
    if not HAVE_TRANSFORMERS or _explainer is None:
        return text
    try:
        scores = _explainer(text)[0]
        confidence = max(s["score"] for s in scores)
        return f"{text} (Confidence: {confidence:.2f})"
    except Exception:
        return text


__all__ = ["annotate_confidence"]
