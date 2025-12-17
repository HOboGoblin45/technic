from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
RISK_PROFILES_PATH = REPO_ROOT / "config" / "risk_profiles.json"


@dataclass
class RiskProfile:
    name: str
    label: str
    min_ics: float
    min_tech_rating: float
    max_atr_pct: float
    min_price: float
    min_dollar_volume: float
    allow_runners: bool
    min_quality_score: float
    min_win_prob_10d: float


def _load_raw_profiles() -> Dict[str, dict]:
    if not RISK_PROFILES_PATH.exists():
        # Safe, very conservative defaults if the file is missing.
        return {
            "conservative": {
                "label": "Conservative (fallback)",
                "min_ics": 80.0,
                "min_tech_rating": 22.0,
                "max_atr_pct": 0.025,
                "min_price": 15.0,
                "min_dollar_volume": 10_000_000,
                "allow_runners": False,
                "min_quality_score": 0.0,
                "min_win_prob_10d": 0.60,
            },
            "defaults": {"profile": "conservative"},
        }
    with RISK_PROFILES_PATH.open("r") as f:
        return json.load(f)


def _build_profile(name: str, data: dict) -> RiskProfile:
    return RiskProfile(
        name=name,
        label=data.get("label", name),
        min_ics=float(data.get("min_ics", 0.0)),
        min_tech_rating=float(data.get("min_tech_rating", 0.0)),
        max_atr_pct=float(data.get("max_atr_pct", 1.0)),
        min_price=float(data.get("min_price", 0.0)),
        min_dollar_volume=float(data.get("min_dollar_volume", 0.0)),
        allow_runners=bool(data.get("allow_runners", True)),
        min_quality_score=float(data.get("min_quality_score", float("-inf"))),
        min_win_prob_10d=float(data.get("min_win_prob_10d", 0.0)),
    )


def load_risk_profiles() -> Dict[str, RiskProfile]:
    raw = _load_raw_profiles()
    profiles: Dict[str, RiskProfile] = {}
    for name, data in raw.items():
        if name == "defaults":
            continue
        profiles[name] = _build_profile(name, data or {})
    return profiles


def get_default_profile_name() -> str:
    raw = _load_raw_profiles()
    defaults = raw.get("defaults", {}) or {}
    return str(defaults.get("profile", "balanced"))


def get_risk_profile(name: str | None) -> RiskProfile:
    profiles = load_risk_profiles()
    if not name:
        name = get_default_profile_name()
    name = name.lower()
    if name not in profiles:
        # Fallback to default if unknown profile is requested
        name = get_default_profile_name()
    return profiles[name]
