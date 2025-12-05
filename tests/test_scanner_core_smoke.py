import pandas as pd

import technic_v4.scanner_core as sc
from technic_v4.universe_loader import UniverseRow
from technic_v4.data_layer.fundamentals import FundamentalsSnapshot
import technic_v4.engine.feature_engine as fe


def test_run_scan_smoke(monkeypatch):
    # Universe with two symbols
    monkeypatch.setattr(sc, "load_universe", lambda: [UniverseRow("AAA", "Tech", "SW", "SW"), UniverseRow("BBB", "Tech", "SW", "SW")])
    # Avoid feature overlap during join: return empty feature set
    monkeypatch.setattr(fe, "build_features", lambda df, fundamentals=None: pd.DataFrame(index=df.index))

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
