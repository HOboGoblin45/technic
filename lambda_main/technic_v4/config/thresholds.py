from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = REPO_ROOT / "config" / "score_thresholds.json"


@dataclass
class HorizonThresholds:
    ics_core_min: float
    ics_satellite_min: float
    win_prob_min: float
    quality_min: float


@dataclass
class ScoreThresholds:
    """Container for 5d / 10d thresholds."""

    fwd_ret_5d: HorizonThresholds
    fwd_ret_10d: HorizonThresholds


def _load_raw_json() -> dict:
    if not THRESHOLDS_PATH.exists():
        # Safe default thresholds if config is missing
        return {
            "defaults": {
                "ics_core_min": 75.0,
                "ics_satellite_min": 65.0,
                "win_prob_5d_min": 0.55,
                "win_prob_10d_min": 0.58,
                "quality_min": 0.0,
            }
        }
    with THRESHOLDS_PATH.open("r") as f:
        return json.load(f)


def _horizon_from_config(config: dict, horizon: str, win_prob_key: str) -> HorizonThresholds:
    defaults = config.get("defaults", {}) or {}
    horiz = config.get(horizon, {}) or {}

    def _get(name: str, fallback_key: Optional[str] = None, default_val: float = 0.0) -> float:
        if name in horiz:
            return float(horiz[name])
        if fallback_key and fallback_key in defaults:
            return float(defaults[fallback_key])
        if name in defaults:
            return float(defaults[name])
        return default_val

    ics_core_min = _get("ics_core_min", default_val=75.0)
    ics_satellite_min = _get("ics_satellite_min", default_val=65.0)
    win_prob_min = _get(win_prob_key, default_val=0.55)
    quality_min = _get("quality_min", default_val=0.0)

    return HorizonThresholds(
        ics_core_min=ics_core_min,
        ics_satellite_min=ics_satellite_min,
        win_prob_min=win_prob_min,
        quality_min=quality_min,
    )


def load_score_thresholds() -> ScoreThresholds:
    """
    Load 5d and 10d thresholds from config/score_thresholds.json with
    sensible defaults if the file is missing or incomplete.

    The idea:
      - fwd_ret_5d: uses 'win_prob_5d_min'
      - fwd_ret_10d: uses 'win_prob_10d_min'
    """
    cfg = _load_raw_json()

    thr_5d = _horizon_from_config(cfg, "fwd_ret_5d", "win_prob_5d_min")
    thr_10d = _horizon_from_config(cfg, "fwd_ret_10d", "win_prob_10d_min")

    return ScoreThresholds(
        fwd_ret_5d=thr_5d,
        fwd_ret_10d=thr_10d,
    )
