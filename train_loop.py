from __future__ import annotations

"""
Rolling walk-forward training loop.
Retrains a model every `retrain_every` samples and saves timestamped checkpoints.
"""

import os
from datetime import datetime

from joblib import dump

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def rolling_train(X, y, model_fn, retrain_every: int = 5):
    """
    Rolling walk-forward training: fit a fresh model on each block of size retrain_every.
    Saves each checkpoint to CHECKPOINT_DIR/model_<timestamp>.joblib.
    """
    for start in range(0, len(X), retrain_every):
        X_train, y_train = X[start:start + retrain_every], y[start:start + retrain_every]
        model = model_fn()
        model.fit(X_train, y_train)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dump(model, f"{CHECKPOINT_DIR}/model_{timestamp}.joblib")
