from __future__ import annotations

"""
Lightweight alerting hooks for Technic.
- Detects interesting events (high-conviction signals, regime shifts, metric degradation).
- Outputs to stdout/log file; provider/webhook/email stubs are placeholders.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd


AlertLevel = Literal["info", "warning", "critical"]


@dataclass
class Alert:
    level: AlertLevel
    category: str
    message: str
    payload: Optional[dict]
    timestamp: datetime


def detect_alerts_from_scan(
    results_df: pd.DataFrame,
    regime: Optional[dict],
    scoreboard_summary: Optional[dict] = None,
    last_regime_path: str = "data_cache/last_regime.json",
) -> List[Alert]:
    """
    Produce alerts based on scan results, regime, and scoreboard metrics.
    Rules are deliberately simple placeholders; make configurable later.
    """
    alerts: List[Alert] = []
    now = datetime.utcnow()

    # High-conviction signals
    if results_df is not None and not results_df.empty:
        strong = results_df[(results_df.get("TechRating", 0) >= 18) & (results_df.get("AlphaScore", 0) >= 15)]
        for _, row in strong.head(5).iterrows():
            msg = f"High-conviction signal: {row.get('Symbol')} TechRating={row.get('TechRating'):.1f} AlphaScore={row.get('AlphaScore'):.1f}"
            alerts.append(Alert(level="info", category="signal", message=msg, payload=row.to_dict(), timestamp=now))

    # Regime change detection (persist last regime)
    prev_regime = {}
    lr_path = Path(last_regime_path)
    if lr_path.exists():
        try:
            prev_regime = json.loads(lr_path.read_text(encoding="utf-8"))
        except Exception:
            prev_regime = {}
    if regime:
        try:
            lr_path.parent.mkdir(parents=True, exist_ok=True)
            lr_path.write_text(json.dumps(regime), encoding="utf-8")
        except Exception:
            pass
        if prev_regime and (prev_regime.get("vol") != regime.get("vol") or prev_regime.get("trend") != regime.get("trend")):
            alerts.append(
                Alert(
                    level="warning",
                    category="regime",
                    message=f"Regime change: trend {prev_regime.get('trend')} -> {regime.get('trend')} , vol {prev_regime.get('vol')} -> {regime.get('vol')}",
                    payload={"previous": prev_regime, "current": regime},
                    timestamp=now,
                )
            )

    # Scoreboard degradation
    if scoreboard_summary:
        ic_val = scoreboard_summary.get("ic") or scoreboard_summary.get("ic_30d")
        if ic_val is not None and pd.notna(ic_val) and ic_val < 0:
            alerts.append(
                Alert(
                    level="warning",
                    category="scoreboard",
                    message=f"Negative IC detected: {ic_val:.3f}",
                    payload=scoreboard_summary,
                    timestamp=now,
                )
            )

    return alerts


def log_alerts_to_file(alerts: List[Alert], path: str = "logs/alerts.log") -> None:
    if not alerts:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for a in alerts:
            payload = {
                "level": a.level,
                "category": a.category,
                "message": a.message,
                "payload": a.payload,
                "timestamp": a.timestamp.isoformat(),
            }
            f.write(json.dumps(payload) + "\n")


def print_alerts(alerts: List[Alert]) -> None:
    for a in alerts:
        print(f"[ALERT][{a.level.upper()}][{a.category}] {a.timestamp.isoformat()} - {a.message}")


def send_alerts_via_webhook(alerts: List[Alert], url: str) -> None:
    """
    Placeholder for webhook integration. No network I/O implemented here.
    """
    # TODO: implement webhook dispatch with requests.post
    return
