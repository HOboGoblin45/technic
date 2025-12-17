from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal


TradeStyle = Literal["swing", "position"]


@dataclass(frozen=True)
class StrategyProfile:
    """
    Encapsulates common scan/trade presets that map to ScanConfig + RiskSettings + alpha flags.
    """

    name: str
    description: str
    trade_style: TradeStyle
    risk_pct: float
    target_rr: float
    min_tech_rating: float
    use_ml_alpha: bool
    use_meta_alpha: bool
    use_tft_features: bool
    use_portfolio_optimizer: bool


def _profiles() -> Dict[str, StrategyProfile]:
    return {
        "conservative_swing": StrategyProfile(
            name="Conservative Swing",
            description="Lower risk, swing-style scans with ML alpha and TFT features, portfolio aware.",
            trade_style="swing",
            risk_pct=0.5,
            target_rr=2.0,
            min_tech_rating=15.0,
            use_ml_alpha=True,
            use_meta_alpha=False,
            use_tft_features=True,
            use_portfolio_optimizer=True,
        ),
        "balanced_swing": StrategyProfile(
            name="Balanced Swing",
            description="Default swing risk, blends ML alpha, enables TFT features and portfolio optimizer.",
            trade_style="swing",
            risk_pct=1.0,
            target_rr=2.0,
            min_tech_rating=12.0,
            use_ml_alpha=True,
            use_meta_alpha=True,
            use_tft_features=True,
            use_portfolio_optimizer=True,
        ),
        "aggressive_swing": StrategyProfile(
            name="Aggressive Swing",
            description="Higher risk swing with ML+meta alpha; lighter portfolio constraints.",
            trade_style="swing",
            risk_pct=1.5,
            target_rr=2.5,
            min_tech_rating=10.0,
            use_ml_alpha=True,
            use_meta_alpha=True,
            use_tft_features=True,
            use_portfolio_optimizer=False,
        ),
        "conservative_position": StrategyProfile(
            name="Conservative Position",
            description="Position-style, lower risk, higher TechRating floor with ML alpha.",
            trade_style="position",
            risk_pct=0.6,
            target_rr=2.0,
            min_tech_rating=16.0,
            use_ml_alpha=True,
            use_meta_alpha=False,
            use_tft_features=True,
            use_portfolio_optimizer=True,
        ),
        "aggressive_position": StrategyProfile(
            name="Aggressive Position",
            description="Position-style with higher risk_pct and meta alpha enabled.",
            trade_style="position",
            risk_pct=1.2,
            target_rr=2.5,
            min_tech_rating=12.0,
            use_ml_alpha=True,
            use_meta_alpha=True,
            use_tft_features=True,
            use_portfolio_optimizer=True,
        ),
    }


def get_strategy_profile(name: str) -> StrategyProfile:
    profs = _profiles()
    if name not in profs:
        raise KeyError(f"Unknown strategy profile: {name}")
    return profs[name]


def list_strategy_profiles() -> Dict[str, StrategyProfile]:
    return _profiles()
