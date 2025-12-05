import pandas as pd

from technic_v4.engine import feature_engine


def test_build_features_basic():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(1, 31),
            "High": [x + 1 for x in range(1, 31)],
            "Low": range(1, 31),
            "Close": [x + 0.5 for x in range(1, 31)],
            "Volume": [1_000_000] * 30,
        },
        index=idx,
    )
    feats = feature_engine.build_features(df)
    assert not feats.empty
    for col in ["MomentumScore", "ATR_pct", "RSI14"]:
        assert col in feats.columns
    latest = feature_engine.get_latest_features(df)
    assert isinstance(latest, pd.Series)
    assert not latest.empty
