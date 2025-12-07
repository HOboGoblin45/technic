from datetime import date
from typing import List

from technic_v4.scanner_core import run_scan, ScanConfig
from technic_v4.evaluation.paper_portfolio import (
    SimTrade,
    simulate_trades_static,
    simulate_trades_adaptive,
)


def build_trades_from_scan(trade_date: date, max_trades: int = 5) -> List[SimTrade]:
    cfg = ScanConfig(
        max_symbols=max_trades,
        trade_style="Short-term swing",
        min_tech_rating=10.0,
    )
    df, status = run_scan(cfg)
    print(status)

    df = df.sort_values("TechRating", ascending=False).head(max_trades)
    trades: List[SimTrade] = []

    for _, r in df.iterrows():
        symbol = str(r["Symbol"])
        entry_price = float(r["Entry"])
        stop_price = float(r["Stop"])
        target_price = float(r["Target"])
        trades.append(
            SimTrade(
                symbol=symbol,
                entry_date=trade_date,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
            )
        )

    return trades


def summarize(label: str, trades: List[SimTrade]) -> None:
    realized_rr = []
    wins = 0
    for t in trades:
        if t.exit_price is None or t.stop_price is None or t.entry_price is None:
            continue
        # Reward/Risk ratio based on initial risk
        rr = (t.exit_price - t.entry_price) / (t.entry_price - t.stop_price)
        realized_rr.append(rr)
        if t.target_price is not None and t.exit_price >= t.target_price:
            wins += 1
    avg_rr = sum(realized_rr) / len(realized_rr) if realized_rr else 0.0
    win_rate = wins / len(trades) if trades else 0.0
    print(f"[{label}] avg RR={avg_rr:.2f}, win_rate={win_rate:.1%}")


def main():
    today = date.today()
    trades = build_trades_from_scan(today, max_trades=5)
    print(f"Simulating {len(trades)} trades from {today}")

    static_res = simulate_trades_static(trades)
    adaptive_res = simulate_trades_adaptive(trades)

    summarize("STATIC", static_res)
    summarize("ADAPTIVE", adaptive_res)


if __name__ == "__main__":
    main()
