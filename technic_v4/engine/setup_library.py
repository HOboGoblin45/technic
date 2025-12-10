"""
Lightweight setup library: derive simple setup tags/buckets for scan rows.

Currently buckets by InstitutionalCoreScore (ICS) quantiles and PlayStyle.
Intended to feed explanations / grouping without heavy clustering.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from technic_v4.engine import meta_experience


def _ics_bucket_label(score: float, meta: Optional[meta_experience.MetaExperience]) -> str:
    if meta is None or meta.edges is None or score is None or np.isnan(score):
        return ""
    try:
        b = int(np.digitize([score], meta.edges[1:-1], right=True)[0])
    except Exception:
        return ""
    # Labels 0..n-1 -> Q1..Qn
    return f"ICS_Q{b + 1}"


def classify_setup(row: pd.Series, meta: Optional[meta_experience.MetaExperience]) -> str:
    """
    Return a compact setup tag like "Stable | ICS_Q4".
    """
    playstyle = str(row.get("PlayStyle") or "").strip()
    ics = row.get("InstitutionalCoreScore")
    try:
        ics = float(ics) if ics is not None else np.nan
    except Exception:
        ics = np.nan

    bucket = _ics_bucket_label(ics, meta)
    parts = []
    if playstyle:
        parts.append(playstyle)
    if bucket:
        parts.append(bucket)
    return " | ".join(parts)


__all__ = ["classify_setup"]
