import pandas as pd
import pytest

# Stub LLM dependency so technic_app imports cleanly in tests
import types, sys
sys.modules["generate_copilot_answer"] = types.SimpleNamespace(generate_copilot_answer=lambda *a, **k: "ok")
from technic_v4.ui import technic_app


def test_backtester_runs(monkeypatch):
    # Fake price history
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=120, freq="D"),
            "Close": [100 + i * 0.5 for i in range(120)],
        }
    ).set_index("Date")

    def fake_price_history(symbol: str, days: int, use_intraday: bool = False):
        return df.tail(days)

    monkeypatch.setattr(technic_app, "ui_price_history", fake_price_history)
    equity_df, metrics = technic_app.run_ma_backtest("AAPL", days=90, ma_fast=10, ma_slow=30, initial_capital=10000)
    assert equity_df is not None
    assert "Equity" in equity_df.columns
    assert metrics["total_return"] is not None
