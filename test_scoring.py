import pandas as pd

from technic_v4.data_layer.polygon_client import get_stock_history_df
from technic_v4.engine.scoring import compute_scores

if __name__ == "__main__":
    df = get_stock_history_df("AAPL", days=150)
    scored = compute_scores(df)

    print(
        scored[[
            "Close",
            "RSI14",
            "MA10",
            "MA20",
            "MA50",
            "MACD",
            "MACD_signal",
            "MACD_hist",
            "BB_width",
            "ADX14",
            "TrendScore",
            "MomentumScore",
            "ExplosivenessScore",
            "BreakoutScore",
            "VolumeScore",
            "VolatilityScore",
            "OscillatorScore",
            "TrendQualityScore",
            "RiskScore",
            "TechRating_raw",
            "TechRating",
            "Signal",
            "TradeType",
            "EntryHint",
            "StopHint",
            "Target1",
            "Target2",
            "RR_T2",
        ]].tail(15)
    )
