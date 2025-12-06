"""Continuous adversarial probe tests."""

from __future__ import annotations

import numpy as np


def run_probe(model, base_data: np.ndarray) -> float:
    """Inject edge-case synthetic data and log confidence change."""
    synthetic = base_data.copy()
    # Sudden illiquidity / conflicting momentum
    synthetic *= np.random.choice([0.9, 1.1], size=synthetic.shape)
    pred_base = model.predict(base_data)
    pred_stress = model.predict(synthetic)
    # Return relative confidence change
    return float(np.mean(pred_stress) - np.mean(pred_base))


__all__ = ["run_probe"]
