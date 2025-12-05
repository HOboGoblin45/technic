from __future__ import annotations

"""
Factor/style exposure and attribution.

Uses simple ETF proxies for factors and linear regression to estimate betas and contributions.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm  # type: ignore
    HAVE_SM = True
except Exception:  # pragma: no cover
    HAVE_SM = False

from technic_v4.data_layer.price_layer import get_stock_history_df


FACTOR_PROXIES = {
    "MKT": "SPY",   # market
    "VAL": "VTV",   # value
    "MOM": "MTUM",  # momentum
    "SIZE": "IWM",  # size
}


def load_factor_series(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> pd.DataFrame:
    """
    Load daily returns for factor proxies between start_date and end_date.
    Returns DataFrame indexed by date with columns matching FACTOR_PROXIES keys.
    """
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    data: Dict[str, pd.Series] = {}
    for fac, ticker in FACTOR_PROXIES.items():
        try:
            df = get_stock_history_df(symbol=ticker, start_date=start_ts, end_date=end_ts, use_intraday=False)
            if df is None or df.empty:
                continue
            close = df["Close"]
            rets = close.pct_change().dropna()
            data[fac] = rets
        except Exception:
            continue
    if not data:
        return pd.DataFrame()
    fac_df = pd.DataFrame(data)
    fac_df = fac_df.loc[(fac_df.index >= start_ts) & (fac_df.index <= end_ts)]
    return fac_df


def compute_portfolio_returns(trade_history_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.Series:
    """
    Compute a simple daily portfolio return series given trade history and price data.
    - trade_history_df should contain columns ['date', 'Symbol', 'Weight'] (weights sum to 1 per day).
    - price_data should contain 'Date', 'Symbol', 'Close'.
    """
    if trade_history_df is None or trade_history_df.empty or price_data is None or price_data.empty:
        return pd.Series(dtype=float)
    # Pivot price data into wide close matrix
    price_data = price_data.copy()
    price_data["Date"] = pd.to_datetime(price_data["Date"])
    close_wide = price_data.pivot_table(index="Date", columns="Symbol", values="Close").sort_index()
    weights_by_day = {}
    for dt, group in trade_history_df.groupby("date"):
        weights = group.set_index("Symbol")["Weight"]
        weights_by_day[pd.to_datetime(dt)] = weights
    returns = close_wide.pct_change().dropna()
    port_returns = []
    idx = []
    for dt, row in returns.iterrows():
        # find latest weights at or before dt
        applicable = [d for d in weights_by_day.keys() if d <= dt]
        if not applicable:
            continue
        latest_day = max(applicable)
        w = weights_by_day[latest_day].reindex(row.index).fillna(0.0)
        port_ret = float((w * row).sum())
        port_returns.append(port_ret)
        idx.append(dt)
    return pd.Series(port_returns, index=idx)


def estimate_factor_exposure(portfolio_returns: pd.Series, factor_df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate factor betas via linear regression.
    """
    if portfolio_returns is None or portfolio_returns.empty or factor_df is None or factor_df.empty:
        return {}
    aligned = portfolio_returns.to_frame("ret").join(factor_df, how="inner").dropna()
    if aligned.empty:
        return {}
    y = aligned["ret"].values
    X = aligned[factor_df.columns].values
    X = np.column_stack([np.ones(len(X)), X])  # add intercept
    if HAVE_SM:
        model = sm.OLS(y, X).fit()
        params = model.params
        r2 = float(model.rsquared)
    else:
        # numpy fallback
        params, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ params
        ss_tot = ((y - y.mean()) ** 2).sum()
        ss_res = ((y - y_hat) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    res = {
        "alpha": float(params[0]),
        "r2": float(r2),
    }
    for i, fac in enumerate(factor_df.columns, start=1):
        res[f"beta_{fac}"] = float(params[i])
    return res


def factor_attribution_report(portfolio_returns: pd.Series, factor_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute factor attribution: betas, contributions, residual volatility.
    """
    if portfolio_returns is None or portfolio_returns.empty or factor_df is None or factor_df.empty:
        return {}
    est = estimate_factor_exposure(portfolio_returns, factor_df)
    if not est:
        return {}
    contrib = {}
    for fac in factor_df.columns:
        beta = est.get(f"beta_{fac}")
        if beta is None:
            continue
        contrib[fac] = beta * factor_df[fac].mean()
    resid = float(portfolio_returns.std())
    return {
        "alpha": est.get("alpha"),
        "betas": {k: v for k, v in est.items() if k.startswith("beta_")},
        "factor_contrib": contrib,
        "r2": est.get("r2"),
        "residual_vol": resid,
    }

