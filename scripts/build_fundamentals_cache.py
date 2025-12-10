import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from technic_v4.universe_loader import load_universe
from requests import Session

FMP_API_KEY = os.environ.get("FMP_API_KEY", "YOUR_FMP_API_KEY")
STABLE_BASE = "https://financialmodelingprep.com/stable"
REQUEST_DELAY = float(os.environ.get("FUNDAMENTALS_REQUEST_DELAY", "0.25"))
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0

DATA_DIR = Path("data_cache")
FUND_DIR = DATA_DIR / "fundamentals"
FUND_DIR.mkdir(exist_ok=True, parents=True)


# --- HTTP HELPER --------------------------------------------------------------

def fmp_get(resource: str, params: Dict = None, session: Optional[Session] = None) -> List[dict]:
    """
    Generic helper for FMP stable API.

    resource: e.g. "income-statement", "balance-sheet-statement"
    """
    if FMP_API_KEY in (None, "", "YOUR_FMP_API_KEY"):
        raise RuntimeError("FMP_API_KEY not set in environment.")

    url = f"{STABLE_BASE}/{resource.lstrip('/')}"
    q = dict(params or {})
    q["apikey"] = FMP_API_KEY
    client = session or requests

    attempt = 0
    while True:
        try:
            r = client.get(url, params=q, timeout=20)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                if isinstance(data, dict):
                    return [data]
                return []
            return data
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 429 and attempt < MAX_RETRIES:
                delay = BACKOFF_FACTOR ** attempt
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except Exception:
                        pass
                print(f"[fundamentals] 429 rate limit on {resource}; sleeping {delay:.2f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
                attempt += 1
                continue
            raise


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b is None or float(b) == 0.0:
            return None
        return float(a) / float(b)
    except Exception:
        return None


def growth(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    try:
        if curr is None or prev is None or float(prev) == 0.0:
            return None
        return float(curr) / float(prev) - 1.0
    except Exception:
        return None


# --- CORE FETCH ---------------------------------------------------------------

def fetch_fundamentals(symbol: str, session: Optional[Session] = None) -> Dict:
    sym = symbol.upper()

    # Pull last 2 ANNUAL statements (Starter supports annual statements). :contentReference[oaicite:3]{index=3}
    income = fmp_get(
        "income-statement",
        {"symbol": sym, "period": "annual", "limit": 2},
        session=session,
    )
    balance = fmp_get(
        "balance-sheet-statement",
        {"symbol": sym, "period": "annual", "limit": 2},
        session=session,
    )
    cash = fmp_get(
        "cash-flow-statement",
        {"symbol": sym, "period": "annual", "limit": 2},
        session=session,
    )
    profile = fmp_get("profile", {"symbol": sym}, session=session)

    inc_cur = income[0] if income else {}
    inc_prev = income[1] if len(income) > 1 else {}
    bal_cur = balance[0] if balance else {}
    bal_prev = balance[1] if len(balance) > 1 else {}
    cf_cur = cash[0] if cash else {}
    prof = profile[0] if profile else {}

    # Income statement items
    revenue = inc_cur.get("revenue")
    revenue_prev = inc_prev.get("revenue")
    gross_profit = inc_cur.get("grossProfit")
    operating_income = inc_cur.get("operatingIncome") or inc_cur.get("ebit")
    ebitda = inc_cur.get("ebitda")
    net_income = inc_cur.get("netIncome")
    interest_expense = inc_cur.get("interestExpense")

    # Balance sheet items
    total_assets = bal_cur.get("totalAssets")
    total_equity = bal_cur.get("totalStockholdersEquity") or bal_cur.get("totalEquity")
    total_debt = (
        bal_cur.get("totalDebt")
        or (
            (bal_cur.get("shortTermDebt") or 0)
            + (bal_cur.get("longTermDebt") or 0)
            if ("shortTermDebt" in bal_cur or "longTermDebt" in bal_cur)
            else None
        )
    )

    # Cash flow items
    operating_cash_flow = cf_cur.get("operatingCashFlow")
    capex = cf_cur.get("capitalExpenditure") or cf_cur.get("capitalExpenditures")
    free_cash_flow = cf_cur.get("freeCashFlow")
    if free_cash_flow is None and operating_cash_flow is not None and capex is not None:
        # Capex often negative; free CF ≈ opCF - capex
        try:
            free_cash_flow = float(operating_cash_flow) - float(capex)
        except Exception:
            pass

    # Profile: market cap, EV, dividend yield, etc. :contentReference[oaicite:4]{index=4}
    market_cap = prof.get("mktCap") or prof.get("marketCap")
    enterprise_value = prof.get("enterpriseValue")
    dividend_yield = prof.get("lastDivYield") or prof.get("dividendYield")

    # Fallback approximate yield: annualized lastDiv / price
    if dividend_yield is None and prof.get("lastDiv") and prof.get("price"):
        try:
            dividend_yield = (float(prof["lastDiv"]) * 4.0) / float(prof["price"])
        except Exception:
            pass

    # Margins
    gross_margin = safe_div(gross_profit, revenue)
    operating_margin = safe_div(operating_income, revenue)
    net_margin = safe_div(net_income, revenue)

    # Returns
    return_on_equity = safe_div(net_income, total_equity)
    return_on_assets = safe_div(net_income, total_assets)

    # Growth metrics
    revenue_growth = growth(revenue, revenue_prev)
    sales_growth = revenue_growth  # alias
    eps_cur = inc_cur.get("eps") or inc_cur.get("epsDiluted")
    eps_prev = inc_prev.get("eps") or inc_prev.get("epsDiluted")
    eps_growth = growth(eps_cur, eps_prev)
    earnings_growth = growth(net_income, inc_prev.get("netIncome"))

    fundamentals = {
        "symbol": sym,
        "market_cap": market_cap,
        "free_cash_flow": free_cash_flow,
        "operating_cash_flow": operating_cash_flow,
        "enterprise_value": enterprise_value,
        "ebitda": ebitda,
        "operating_income": operating_income,
        "net_income": net_income,
        "revenue": revenue,
        "total_debt": total_debt,
        "total_equity": total_equity,
        "total_assets": total_assets,
        "return_on_equity": return_on_equity,
        "return_on_assets": return_on_assets,
        "gross_profit": gross_profit,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "revenue_growth": revenue_growth,
        "sales_growth": sales_growth,
        "eps_growth": eps_growth,
        "earnings_growth": earnings_growth,
        "dividend_yield": dividend_yield,
        "interest_expense": interest_expense,
    }

    return fundamentals


def write_fundamentals(symbols: List[str]) -> None:
    unique_syms = []
    seen = set()
    for s in symbols:
        u = s.upper()
        if u and u not in seen:
            seen.add(u)
            unique_syms.append(u)

    with Session() as session:
        for sym in unique_syms:
            try:
                data = fetch_fundamentals(sym, session=session)
            except Exception as exc:
                print(f"[fundamentals] WARN: failed for {sym}: {exc}")
                continue

            out_path = FUND_DIR / f"{sym}.json"
            with out_path.open("w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            print(f"[fundamentals] Wrote {out_path}")
            if REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)


if __name__ == "__main__":
    universe = load_universe()
    symbols = [row.symbol for row in universe]
    write_fundamentals(symbols)
