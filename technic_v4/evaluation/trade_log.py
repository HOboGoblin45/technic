from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd

TradeDirection = Literal["long", "short"]


@dataclass
class TradeRecord:
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: TradeDirection
    size: float  # position size (shares or dollar allocation)
    risk_per_share: float
    realized_R: Optional[float] = None
    strategy_profile_name: Optional[str] = None
    signal_type: Optional[str] = None
    notes: Optional[str] = None


def append_trades(records: List[TradeRecord], path: str = "data_cache/trade_log.jsonl") -> None:
    """
    Append trade records as JSON lines to the given path.
    """
    if not records:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for rec in records:
            payload = asdict(rec)
            # datetime serialization
            payload["entry_date"] = payload["entry_date"].isoformat() if payload.get("entry_date") else None
            payload["exit_date"] = payload["exit_date"].isoformat() if payload.get("exit_date") else None
            f.write(json.dumps(payload) + "\n")


def load_trades(path: str = "data_cache/trade_log.jsonl") -> List[TradeRecord]:
    """
    Load trade records from JSONL.
    """
    p = Path(path)
    if not p.exists():
        return []
    out: List[TradeRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            rec = TradeRecord(
                symbol=obj.get("symbol", ""),
                entry_date=pd.to_datetime(obj.get("entry_date")),
                exit_date=pd.to_datetime(obj.get("exit_date")) if obj.get("exit_date") else None,
                entry_price=float(obj.get("entry_price", 0) or 0),
                exit_price=float(obj.get("exit_price")) if obj.get("exit_price") is not None else None,
                direction=obj.get("direction") or "long",
                size=float(obj.get("size", 0) or 0),
                risk_per_share=float(obj.get("risk_per_share", 0) or 0),
                realized_R=float(obj.get("realized_R")) if obj.get("realized_R") is not None else None,
                strategy_profile_name=obj.get("strategy_profile_name"),
                signal_type=obj.get("signal_type"),
                notes=obj.get("notes"),
            )
            out.append(rec)
        except Exception:
            continue
    return out


def compute_trade_pnl(records: List[TradeRecord]) -> pd.DataFrame:
    """
    Compute per-trade PnL metrics.
    """
    if not records:
        return pd.DataFrame()
    rows = []
    for rec in records:
        exit_px = rec.exit_price
        if exit_px is None:
            continue
        pnl_per_share = exit_px - rec.entry_price
        if rec.direction == "short":
            pnl_per_share *= -1
        r_mult = None
        if rec.risk_per_share not in (0, None):
            r_mult = pnl_per_share / rec.risk_per_share
        pnl_dollar = pnl_per_share * rec.size
        rows.append(
            {
                "Symbol": rec.symbol,
                "EntryDate": rec.entry_date,
                "ExitDate": rec.exit_date,
                "Direction": rec.direction,
                "EntryPrice": rec.entry_price,
                "ExitPrice": exit_px,
                "Size": rec.size,
                "RiskPerShare": rec.risk_per_share,
                "R_multiple": r_mult,
                "PnL_per_share": pnl_per_share,
                "PnL_dollar": pnl_dollar,
                "StrategyProfile": rec.strategy_profile_name,
                "SignalType": rec.signal_type,
                "Notes": rec.notes,
            }
        )
    return pd.DataFrame(rows)


def summarize_trades(df_trades: pd.DataFrame) -> dict:
    """
    Summarize PnL by symbol, strategy, and signal type.
    """
    if df_trades is None or df_trades.empty:
        return {}
    total_pnl = df_trades["PnL_dollar"].sum()
    r_mult = df_trades["R_multiple"].dropna()
    wins = df_trades[df_trades["PnL_dollar"] > 0]
    win_rate = len(wins) / len(df_trades) if len(df_trades) > 0 else None
    by_symbol = df_trades.groupby("Symbol")["PnL_dollar"].sum().sort_values(ascending=False).to_dict()
    by_strategy = df_trades.groupby("StrategyProfile")["PnL_dollar"].sum().sort_values(ascending=False).to_dict()
    by_signal = df_trades.groupby("SignalType")["PnL_dollar"].sum().sort_values(ascending=False).to_dict()
    return {
        "total_pnl": float(total_pnl),
        "avg_R": float(r_mult.mean()) if not r_mult.empty else None,
        "win_rate": float(win_rate) if win_rate is not None else None,
        "by_symbol": by_symbol,
        "by_strategy": by_strategy,
        "by_signal": by_signal,
    }
