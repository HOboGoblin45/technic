"""Compliance policy annotator."""

from __future__ import annotations

import functools
from typing import Callable


def _tag(tag_name: str):
    def decorator(fn: Callable):
        setattr(fn, "_policy_tag", tag_name)
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator


GDPR_sensitive = _tag("GDPR_sensitive")
FINRA_required = _tag("FINRA_required")
CCPA_user_data = _tag("CCPA_user_data")


__all__ = ["GDPR_sensitive", "FINRA_required", "CCPA_user_data"]
