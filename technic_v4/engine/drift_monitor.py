"""Simple model drift detection based on MSE ratio."""

from __future__ import annotations

from sklearn.metrics import mean_squared_error


def detect_drift(
    baseline_mse: float, y_true, y_pred, threshold: float = 0.5
) -> bool:
    """Return True if current MSE exceeds baseline by more than `threshold` fraction.

    Args:
        baseline_mse: Reference MSE from validation or training.
        y_true: Ground-truth targets.
        y_pred: Model predictions aligned with y_true.
        threshold: Fractional tolerance; 0.5 means 50% higher MSE triggers drift.

    """
    if baseline_mse <= 0:
        raise ValueError("baseline_mse must be positive.")
    current_mse = mean_squared_error(y_true, y_pred)
    drift_score = current_mse / baseline_mse
    return drift_score > (1 + threshold)


__all__ = ["detect_drift"]
