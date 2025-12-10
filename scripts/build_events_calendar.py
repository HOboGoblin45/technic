import csv
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import requests

# --- CONFIG -------------------------------------------------------------------

FMP_API_KEY = os.environ.get("FMP_API_KEY", "YOUR_FMP_API_KEY")
STABLE_BASE = "https://financialmodelingprep.com/stable"

DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True, parents=True)

EVENTS_CSV = DATA_DIR / "events_calendar.csv"

# Range of earnings we care about: past 1y + next 1y
EARNINGS_LOOKBACK_DAYS = 365
EARNINGS_LOOKAHEAD_DAYS = 365


# --- HTTP HELPER --------------------------------------------------------------

def fmp_get(resource: str, params: Dict = None) -> List[dict]:
    """
    Generic helper for FMP stable API.

    resource: e.g. "earnings-calendar", "dividends", "income-statement"
    """
    if FMP_API_KEY in (None, "", "YOUR_FMP_API_KEY"):
        raise RuntimeError("FMP_API_KEY not set in environment.")

    url = f"{STABLE_BASE}/{resource.lstrip('/')}"
    q = dict(params or {})
    q["apikey"] = FMP_API_KEY
    r = requests.get(url, params=q, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        # Some endpoints wrap in an object; normalize to list
        if isinstance(data, dict):
            return [data]
        return []
    return data


# --- EARNINGS -----------------------------------------------------------------

def build_earnings_map(symbols: List[str]) -> Dict[str, dict]:
    """
    Use ONE bulk call to /stable/earnings-calendar and then
    split into last/next per symbol.

    Returns:
      { "AAPL": {
          "last_date": "YYYY-MM-DD" or "",
          "next_date": "YYYY-MM-DD" or "",
          "surprise_positive": bool
        }, ... }
    """
    today = date.today()
    from_date = (today - timedelta(days=EARNINGS_LOOKBACK_DAYS)).isoformat()
    to_date = (today + timedelta(days=EARNINGS_LOOKAHEAD_DAYS)).isoformat()

    print(f"[events] Fetching earnings calendar {from_date} -> {to_date}")
    rows = fmp_get("earnings-calendar", {"from": from_date, "to": to_date})

    sym_set = {s.upper() for s in symbols}
    buckets: Dict[str, dict] = defaultdict(lambda: {"past": [], "future": []})

    for row in rows:
        sym = (row.get("symbol") or "").upper()
        if sym not in sym_set:
            continue

        # FMP uses "date" for announcement date; sometimes also dateReported/reportDate
        d_str = (
            row.get("date")
            or row.get("dateReported")
            or row.get("reportDate")
        )
        if not d_str:
            continue

        try:
            d = datetime.fromisoformat(str(d_str)[:10]).date()
        except Exception:
            continue

        if d < today:
            buckets[sym]["past"].append((d, row))
        else:
            buckets[sym]["future"].append((d, row))

    # Build per-symbol summary
    result: Dict[str, dict] = {}
    for sym in sym_set:
        past = buckets[sym]["past"]
        future = buckets[sym]["future"]

        last_date_iso = ""
        next_date_iso = ""
        surprise_positive = False

        if past:
            last_date, last_row = max(past, key=lambda x: x[0])
            last_date_iso = last_date.isoformat()

            # Try EPS surprise (actual vs estimate).
            # Field names vary: eps / epsEstimated, actualEps / estimatedEps, etc.
            eps_candidates = [
                last_row.get("eps"),
                last_row.get("actualEps"),
                last_row.get("epsActual"),
            ]
            est_candidates = [
                last_row.get("epsEstimated"),
                last_row.get("estimatedEps"),
                last_row.get("epsEstimate"),
            ]

            eps = next((x for x in eps_candidates if x not in (None, "")), None)
            est = next((x for x in est_candidates if x not in (None, "")), None)

            try:
                if eps is not None and est is not None:
                    surprise_positive = float(eps) > float(est)
            except Exception:
                surprise_positive = False

        if future:
            next_date, _ = min(future, key=lambda x: x[0])
            next_date_iso = next_date.isoformat()

        result[sym] = {
            "last_date": last_date_iso,
            "next_date": next_date_iso,
            "surprise_positive": surprise_positive,
        }

    return result


# --- DIVIDENDS ----------------------------------------------------------------

def get_dividend_ex_date(symbol: str) -> str:
    """
    Use /stable/dividends?symbol=SYM
    Return next ex-date if available, else last ex-date, else "".
    """
    today = date.today()
    try:
        rows = fmp_get("dividends", {"symbol": symbol})
    except Exception as exc:
        print(f"[events] WARN: dividends failed for {symbol}: {exc}")
        return ""

    future_dates: List[date] = []
    past_dates: List[date] = []

    for row in rows:
        # Docs say "date" is the ex-dividend date for the Dividends Company API. :contentReference[oaicite:0]{index=0}
        d_str = row.get("date")
        if not d_str:
            continue
        try:
            d = datetime.fromisoformat(str(d_str)[:10]).date()
        except Exception:
            continue

        if d >= today:
            future_dates.append(d)
        else:
            past_dates.append(d)

    if future_dates:
        return min(future_dates).isoformat()
    if past_dates:
        return max(past_dates).isoformat()
    return ""


# --- DRIVER -------------------------------------------------------------------

def build_events_csv(symbols: List[str]) -> None:
    """
    Build data_cache/events_calendar.csv for the given symbol universe.
    """
    symbols_u = [s.upper() for s in symbols]
    earnings_map = build_earnings_map(symbols_u)

    fieldnames = [
        "symbol",
        "next_earnings_date",
        "last_earnings_date",
        "earnings_surprise_flag",
        "dividend_ex_date",
    ]

    with EVENTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sym in symbols_u:
            earn = earnings_map.get(sym, {})
            last_e = earn.get("last_date", "")
            next_e = earn.get("next_date", "")
            surprise = bool(earn.get("surprise_positive", False))
            div_ex = get_dividend_ex_date(sym)

            writer.writerow(
                {
                    "symbol": sym,
                    "next_earnings_date": next_e,
                    "last_earnings_date": last_e,
                    "earnings_surprise_flag": surprise,
                    "dividend_ex_date": div_ex,
                }
            )

    print(f"[events] Wrote {EVENTS_CSV}")


if __name__ == "__main__":
    # TODO: Replace this with your real Technic universe loader
    test_symbols = ["AAPL", "MSFT", "ODP", "VMEO", "VRNT"]
    build_events_csv(test_symbols)
