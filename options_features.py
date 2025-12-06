from __future__ import annotations

"""
Options flow sentiment feature using Polygon.io API.
Computes call/put volume ratio over a period.
"""

import pandas as pd

try:
    from polygon import RESTClient
except Exception:  # pragma: no cover
    RESTClient = None


def get_options_flow(api_key: str, ticker: str, start: str, end: str):
    """
    Fetch options aggregate data and compute call/put volume ratio.
    Returns a DataFrame with timestamp and call_put_ratio.
    """
    if RESTClient is None:
        raise ImportError("polygon RESTClient not available")
    client = RESTClient(api_key)
    activity = client.get_aggs(ticker, multiplier=1, timespan="day", from_=start, to=end)
    df = pd.DataFrame(activity)
    if "call_volume" in df and "put_volume" in df:
        df["call_put_ratio"] = df["call_volume"] / df["put_volume"].replace(0, pd.NA)
    else:
        df["call_put_ratio"] = pd.NA
    return df[["timestamp", "call_put_ratio"]]
