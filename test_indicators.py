import pandas as pd
from technic_v4.data_layer.polygon_client import get_stock_history_df
from technic_v4.indicators import calculate_indicators

# Fetch sample data
df = get_stock_history_df("AAPL", days=100)
df = calculate_indicators(df)  # Make sure this line is present

# Check the output
print(df[[
    "Close", "RSI14", "ATR14_pct", "RVOL20", "PctFromHigh20", "TrendStrength50",
    "MA10", "MA20", "MA50", "SlopeMA20", 
    "MACD", "MACD_signal", "BB_upper", "BB_lower", "BB_width", "BB_pctB", 
    "ADX14"
]].tail(15))

