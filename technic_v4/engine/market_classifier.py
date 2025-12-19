"""AI market state classifier (Trending, Choppy, Mean-Reverting)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def classify_market(X, y):
    """Fit a supervised classifier for current market regime labels."""
    model = RandomForestClassifier().fit(X, y)
    return model


__all__ = ["classify_market"]
