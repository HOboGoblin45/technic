import pandas as pd

import technic_v4.scanner_core as sc
from technic_v4.universe_loader import UniverseRow
from technic_v4.data_layer.fundamentals import FundamentalsSnapshot
import technic_v4.engine.feature_engine as fe


def test_run_scan_smoke(monkeypatch):
    # Universe with two symbols
    monkeypatch.setattr(sc, "load_universe", lambda: [UniverseRow("AAA", "Tech", "SW", "SW"), UniverseRow("BBB", "Tech", "SW", "SW")])
    # Provide minimal feature set
    monkeypatch.setattr(
        fe,
        "build_features",
        lambda df, fundamentals=None: pd.Series(
            {
                "sma_20": 10.0,
                "sma_50": 9.5,
                "sma_200": 9.0,
                "sma_20_above_50": 1.0,
                "rsi_14": 60.0,
                "macd_hist": 0.1,
                "pct_from_high20": -1.0,
                "ret_5d": 0.02,
                "atr_pct_14": 0.01,
                "vol_spike_ratio": 1.0,
            }
        ),
    )

    def fake_history(symbol: str, days: int, use_intraday: bool = False, end_date=None):
        idx = pd.date_range("2024-01-01", periods=60, freq="D")
        return pd.DataFrame(
            {
                "Open": [10.0] * 60,
                "High": [10.5] * 60,
                "Low": [9.5] * 60,
                "Close": [10.0 + i * 0.01 for i in range(60)],
                "Volume": [1_000_000] * 60,
            },
            index=idx,
        )

    monkeypatch.setattr(sc, "get_stock_history_df", fake_history)
    monkeypatch.setattr(sc, "get_fundamentals", lambda sym: FundamentalsSnapshot({}))

    df, status = sc.run_scan()
    assert not df.empty
    assert isinstance(status, str)
