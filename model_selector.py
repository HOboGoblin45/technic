from __future__ import annotations

"""
Adaptive model selector: evaluate multiple models on a validation set and pick the one with lowest MSE.
"""

from joblib import load
from sklearn.metrics import mean_squared_error


def select_best_model(models: dict, X_val, y_val):
    """
    Evaluate each model (name -> path) on (X_val, y_val) and return the best model name and all scores.
    """
    scores = {}
    for name, path in models.items():
        model = load(path)
        preds = model.predict(X_val)
        scores[name] = mean_squared_error(y_val, preds)
    best = sorted(scores.items(), key=lambda x: x[1])
    return best[0][0], scores
