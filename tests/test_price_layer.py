import pytest

from technic_v4.data_layer import price_layer


def test_cache_roundtrip(monkeypatch):
    # Use a small dummy dataframe
    import pandas as pd

    df = pd.DataFrame({"Close": [1.0, 1.1]})
    price_layer._store_in_cache("AAPL", 10, False, df)
    out = price_layer._get_from_cache("AAPL", 10, False)
    assert out is not None
    assert list(out["Close"]) == [1.0, 1.1]


def test_realtime_last_empty():
    assert price_layer.get_realtime_last("MSFT") is None


def test_price_layer_graceful(monkeypatch):
    # Monkeypatch polygon calls to avoid network
    def fake_hist(symbol, days):
        import pandas as pd

        return pd.DataFrame({"Close": [100, 101, 102]})

    monkeypatch.setattr(price_layer, "_polygon_history", fake_hist)
    out = price_layer.get_stock_history_df("AAPL", days=5, use_intraday=False)
    assert not out.empty
    assert "Close" in out.columns
