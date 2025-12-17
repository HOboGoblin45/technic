from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd

from technic_v4 import data_engine
from technic_v4.engine.trade_management_engine import suggest_trade_updates


@dataclass
class SimTrade:
    symbol: str
    entry_date: date
    entry_price: float
    stop_price: float
    target_price: float
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None  # "hit_target", "hit_stop", "time_exit"


def _forward_prices(symbol: str, start: date, days: int) -> pd.DataFrame:
    df = data_engine.get_price_history(symbol, days=days + 5, freq="daily")
    if df is None or df.empty:
        return pd.DataFrame()
    # ensure Date column
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.set_index("Date")
    else:
        df = df.copy()
        df.index = pd.to_datetime(df.index).date
    df = df.sort_index()
    return df[df.index >= start]


def _simulate_trade(trade: SimTrade, adaptive: bool = False, max_days: int = 20) -> SimTrade:
    prices = _forward_prices(trade.symbol, trade.entry_date, max_days)
    if prices.empty:
        trade.exit_date = trade.entry_date
        trade.exit_price = trade.entry_price
        trade.outcome = "no_data"
        return trade

    cur_stop = trade.stop_price
    cur_target = trade.target_price
    entry_seen = False

    for i, (dt_idx, row) in enumerate(prices.iterrows()):
        if dt_idx < trade.entry_date:
            continue
        if not entry_seen and dt_idx >= trade.entry_date:
            entry_seen = True
        if not entry_seen:
            continue

        close = float(row.get("Close", row.get("close", row.get("Adj Close", trade.entry_price))))
        days_since_entry = (dt_idx - trade.entry_date).days

        if adaptive:
            try:
                updates = suggest_trade_updates(
                    symbol=trade.symbol,
                    entry=trade.entry_price,
                    stop=cur_stop,
                    target=cur_target,
                    history_df=prices.loc[:dt_idx],
                    days_since_entry=days_since_entry,
                )
                if updates:
                    cur_stop = float(updates.get("new_stop", cur_stop))
                    cur_target = float(updates.get("new_target", cur_target))
            except Exception:
                pass

        if close >= cur_target:
            trade.exit_date = dt_idx
            trade.exit_price = cur_target
            trade.outcome = "hit_target"
            return trade
        if close <= cur_stop:
            trade.exit_date = dt_idx
            trade.exit_price = cur_stop
            trade.outcome = "hit_stop"
            return trade
        if i >= max_days - 1:
            trade.exit_date = dt_idx
            trade.exit_price = close
            trade.outcome = "time_exit"
            return trade

    # If loop ends without exit
    last_dt = prices.index[-1]
    trade.exit_date = last_dt
    trade.exit_price = float(prices.iloc[-1].get("Close", trade.entry_price))
    trade.outcome = "time_exit"
    return trade


def simulate_trades_static(trades: List[SimTrade], max_days: int = 20) -> List[SimTrade]:
    return [_simulate_trade(t, adaptive=False, max_days=max_days) for t in trades]


def simulate_trades_adaptive(trades: List[SimTrade], max_days: int = 20) -> List[SimTrade]:
    return [_simulate_trade(t, adaptive=True, max_days=max_days) for t in trades]


def _summary(trades: List[SimTrade]) -> dict:
    if not trades:
        return {"count": 0}
    pnl = []
    wins = 0
    for t in trades:
        if t.exit_price is None or t.entry_price == 0:
            continue
        ret = (t.exit_price / t.entry_price) - 1
        pnl.append(ret)
        if ret > 0:
            wins += 1
    return {
        "count": len(trades),
        "avg_ret": float(pd.Series(pnl).mean()) if pnl else 0.0,
        "win_rate": wins / len(trades) if trades else 0.0,
    }


def _sample_trades_from_scan(max_trades: int = 5) -> List[SimTrade]:
    # Lazy import to avoid heavy deps at module load
    from technic_v4.scanner_core import ScanConfig, run_scan

    cfg = ScanConfig(max_symbols=100)
    df, _ = run_scan(cfg)
    if df is None or df.empty:
        return []
    df = df[df["Signal"].isin(["Strong Long", "Long"])].head(max_trades)
    trades: List[SimTrade] = []
    today = date.today()
    for _, row in df.iterrows():
        entry = float(row.get("Entry") or row.get("Close") or 0)
        stop = float(row.get("Stop") or entry * 0.97)
        target = float(row.get("Target") or entry * 1.05)
        if entry <= 0:
            continue
        trades.append(
            SimTrade(
                symbol=str(row.get("Symbol")),
                entry_date=today,
                entry_price=entry,
                stop_price=stop,
                target_price=target,
            )
        )
    return trades


def main():
    trades = _sample_trades_from_scan(max_trades=5)
    if not trades:
        print("No trades to simulate")
        return
    static_res = simulate_trades_static([t for t in trades], max_days=15)
    adaptive_res = simulate_trades_adaptive([t for t in trades], max_days=15)

    print("Static summary:", _summary(static_res))
    print("Adaptive summary:", _summary(adaptive_res))


if __name__ == "__main__":
    # Lazy imports to avoid circulars
    from technic_v4.scanner_core import ScanConfig, run_scan

    main()
