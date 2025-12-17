"""Alpha rollout guard comparing live vs cloned outputs."""

from __future__ import annotations

import numpy as np


def check_alpha_guard(live_alpha: np.ndarray, clone_alpha: np.ndarray, threshold: float = 0.8) -> bool:
    """Return False to halt live use if correlation below threshold."""
    if live_alpha.shape != clone_alpha.shape or live_alpha.size == 0:
        return False
    corr = np.corrcoef(live_alpha, clone_alpha)[0, 1]
    return corr >= threshold


__all__ = ["check_alpha_guard"]
