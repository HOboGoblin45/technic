import pandas as pd
import numpy as np
import pytest

from datetime import date

from technic_v4.evaluation import eval_harness


def test_backtest_top_n_smoke(monkeypatch):
    # Provide deterministic price history and scores
    def fake_price_history(symbol, days, freq="daily"):
        idx = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
        base = 100 + np.arange(days)
        df = pd.DataFrame({"Open": base, "High": base + 1, "Low": base - 1, "Close": base, "Volume": 1_000_000}, index=idx)
        df.index.name = "Date"
        return df

    def fake_compute_scores(df, trade_style=None, fundamentals=None):
        out = pd.DataFrame(index=df.index)
        out["TechRating"] = np.linspace(10, 50, len(df))
        return out

    monkeypatch.setattr("technic_v4.data_engine.get_price_history", fake_price_history)
    monkeypatch.setattr("technic_v4.engine.scoring.compute_scores", fake_compute_scores)

    universe = ["AAPL", "MSFT"]
    start = date.today().replace(year=date.today().year - 1)
    end = start

    df = eval_harness.backtest_top_n(universe, start, end, top_n=1, forward_horizon_days=2, lookback_days=10)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"symbol", "as_of", "fwd_ret"}.issubset(df.columns)
