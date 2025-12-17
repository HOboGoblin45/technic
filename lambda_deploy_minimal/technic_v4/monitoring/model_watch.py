"""Model watchdog monitor."""

from __future__ import annotations

import numpy as np


def monitor_model_errors(error_log):
    if len(error_log) < 10:
        return False
    if error_log[-1] > np.mean(error_log[-10:]) + 2 * np.std(error_log[-10:]):
        return True  # trigger alert
    return False


__all__ = ["monitor_model_errors"]
