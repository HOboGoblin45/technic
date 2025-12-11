from __future__ import annotations

"""
Lightweight sanity check for the options pipeline.

Runs:
  - fetch chain snapshot via OptionChainService
  - apply filter_liquid_contracts
  - compute sweetness/quality and OptionTrade summary

Prints a small table for a few liquid tickers.
"""

from typing import List

import pandas as pd

from technic_v4.data_layer.options_data import OptionChainService
from technic_v4.engine.options_selector import filter_liquid_contracts, select_option_candidates


def summarize_symbol(symbol: str) -> pd.DataFrame:
    svc = OptionChainService()
    snap, _meta = svc.fetch_chain_snapshot(symbol, contract_type="call")
    if not snap:
        return pd.DataFrame(
            [{"Symbol": symbol, "Summary": "No chain snapshot returned.", "OptionQualityScore": pd.NA}]
        )

    spot = None
    try:
        ua = snap[0].get("underlying_asset") or {}
        spot = ua.get("price") or ua.get("last_price")
    except Exception:
        spot = None

    picks = select_option_candidates(
        snap,
        direction="call",
        trade_style="swing",
        underlying_price=spot,
        tech_rating=25,
        risk_score=0.6,
    )
    if not picks:
        return pd.DataFrame(
            [{"Symbol": symbol, "Summary": "No option candidates after filters.", "OptionQualityScore": pd.NA}]
        )

    rows: List[dict] = []
    for p in picks[:5]:
        rows.append(
            {
                "Symbol": symbol,
                "OptionSymbol": p.get("ticker") or p.get("option_symbol"),
                "Expiration": p.get("expiration"),
                "Side": p.get("contract_type"),
                "Strike": p.get("strike"),
                "Sweetness": p.get("option_sweetness_score"),
                "OptionQualityScore": p.get("option_sweetness_score"),
                "Summary": p.get("reason") or "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    symbols = ["AAPL", "MSFT", "SPY"]
    frames = [summarize_symbol(sym) for sym in symbols]
    out = pd.concat(frames, ignore_index=True)
    print(out[["Symbol", "OptionSymbol", "Expiration", "Side", "Strike", "Sweetness", "OptionQualityScore", "Summary"]])


if __name__ == "__main__":
    main()
