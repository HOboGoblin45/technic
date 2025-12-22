"""
Lightweight FastAPI surface to expose the existing technic engine for the new Flutter UI.

Endpoints are intentionally minimal and defensive so the app can start even if optional
data sources (e.g., Polygon) or keys are missing. Flesh out payloads as you hook in the
actual UI consumers.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make sure technic_v4 is importable when running from repo root
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

try:
    from technic_v4 import scanner_core
    from technic_v4.engine import trade_planner
except Exception:  # pragma: no cover - keep API alive even if imports fail
    scanner_core = None  # type: ignore
    trade_planner = None  # type: ignore

# Optional options stack
try:
    from technic_v4.data_layer.options_data import OptionChainService
    from technic_v4.engine.options_selector import select_option_candidates
except Exception:  # pragma: no cover
    OptionChainService = None  # type: ignore
    select_option_candidates = None  # type: ignore

# Optional UI helpers (for cached picklists)
try:
    from technic_v4.ui.technic_app import get_universe_stats  # type: ignore
except Exception:  # pragma: no cover
    get_universe_stats = None  # type: ignore

# Optional price/fundamentals for symbol detail
try:
    from technic_v4.data_layer.price_layer import get_stock_history_df, get_realtime_last  # type: ignore
    from technic_v4.data_layer.fundamentals import get_fundamentals  # type: ignore
except Exception:  # pragma: no cover
    get_stock_history_df = None  # type: ignore
    get_realtime_last = None  # type: ignore
    get_fundamentals = None  # type: ignore

app = FastAPI(title="Technic API", version="0.1.0")

# Add CORS middleware to allow Flutter web app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:*",
        "http://127.0.0.1:*",
        "https://technic-m5vn.onrender.com",
        "https://*.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Pydantic response models
# -------------------------------------------------------------------


class Mover(BaseModel):
    ticker: str
    delta: Optional[float]
    note: Optional[str]
    isPositive: bool


class IdeaItem(BaseModel):
    title: Optional[str]
    ticker: Optional[str]
    meta: Optional[str]
    plan: Optional[str]


class ScanResultItem(BaseModel):
    ticker: Optional[str]
    signal: Optional[str]
    rrr: Optional[str]
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    techRating: Optional[float]
    riskScore: Optional[float]
    sector: Optional[str]
    industry: Optional[str]


class ScanResponse(BaseModel):
    results: List[ScanResultItem]
    movers: List[Mover]
    ideas: List[IdeaItem]
    log: Optional[str] = None


class OptionCandidate(BaseModel):
    ticker: Optional[str]
    contract_type: Optional[str]
    strike: Optional[float]
    expiration: Optional[str]
    dte: Optional[int]
    delta: Optional[float]
    iv: Optional[float]
    open_interest: Optional[int]
    volume: Optional[int]
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    last: Optional[float]
    spread_pct: Optional[float]
    breakeven: Optional[float]
    underlying: Optional[float]
    moneyness: Optional[float]
    score: Optional[float]
    reason: Optional[str]


class OptionsResponse(BaseModel):
    ticker: str
    direction: str
    trade_style: str
    candidates: List[OptionCandidate]


class SymbolHistoryPoint(BaseModel):
    date: Any = Field(..., description="ISO date or datetime")
    Open: Optional[float] = None
    High: Optional[float] = None
    Low: Optional[float] = None
    Close: Optional[float] = None
    Volume: Optional[float] = None


class SymbolResponse(BaseModel):
    ticker: str
    last: Optional[float]
    history: List[SymbolHistoryPoint]
    fundamentals: Dict[str, Any]
    options_available: bool


class UniverseStats(BaseModel):
    sectors: List[str]
    subindustries: List[str]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# -------------------------------------------------------------------
# V1 API Routes (Mobile App Compatible)
# -------------------------------------------------------------------

class ScanRequestV1(BaseModel):
    """Request body for V1 scan endpoint"""
    max_symbols: int = Field(default=50, ge=1, le=6000)
    trade_style: str = Field(default="Short-term swing")
    min_tech_rating: float = Field(default=0.0, ge=0.0, le=100.0)
    sectors: Optional[List[str]] = None
    lookback_days: Optional[int] = Field(default=90, ge=1, le=365)
    options_mode: Optional[str] = None


@app.post("/v1/scan", response_model=ScanResponse)
def run_scan_v1(body: ScanRequestV1) -> Dict[str, Any]:
    """
    V1 API: Run equity scanner with POST body (mobile app compatible).
    
    Accepts JSON body with scan parameters and returns structured results.
    """
    if scanner_core is None:
        raise HTTPException(500, "scanner_core unavailable in this session")

    try:
        cfg = None
        if hasattr(scanner_core, "ScanConfig"):
            cfg = scanner_core.ScanConfig(
                max_symbols=body.max_symbols,
                min_tech_rating=body.min_tech_rating,
                trade_style=body.trade_style,
                lookback_days=body.lookback_days or 90,
            )
        df, log = scanner_core.run_scan(config=cfg)
    except Exception as exc:
        raise HTTPException(500, f"scan failed: {exc}") from exc

    # Apply sector filter if provided
    if body.sectors and len(body.sectors) > 0:
        if "Sector" in df.columns:
            df = df[df["Sector"].isin(body.sectors)]

    # Structured payload for Flutter
    def _movers(df: pd.DataFrame) -> List[Dict[str, Any]]:
        if not {"Symbol", "RewardRisk", "Signal"} <= set(df.columns):
            return []
        movers_df = (
            df[["Symbol", "RewardRisk", "Signal"]]
            .assign(
                ticker=lambda x: x["Symbol"],
                delta=lambda x: x["RewardRisk"].round(2),
                note=lambda x: x["Signal"],
                isPositive=lambda x: x["RewardRisk"].fillna(0) >= 0,
            )[["ticker", "delta", "note", "isPositive"]]
            .head(6)
        )
        return movers_df.to_dict(orient="records")

    def _ideas(df: pd.DataFrame) -> List[Dict[str, Any]]:
        cols = set(df.columns)
        if "Symbol" not in cols:
            return []
        ideas_df = (
            df.sort_values("RewardRisk", ascending=False)
            .head(5)
            .assign(
                title=lambda x: x["Signal"] if "Signal" in cols else "Idea",
                ticker=lambda x: x["Symbol"],
                meta=lambda x: (
                    "R:R "
                    + x.get("RewardRisk", pd.Series()).fillna(0).round(2).astype(str)
                    + (" • TechRating " + x.get("TechRating", pd.Series()).fillna(0).round(1).astype(str)
                       if "TechRating" in cols else "")
                ),
                plan=lambda x: (
                    "Entry " + x.get("EntryPrice", pd.Series()).fillna("").astype(str)
                    + " • Stop " + x.get("StopPrice", pd.Series()).fillna("").astype(str)
                    + " • Target " + x.get("TargetPrice", pd.Series()).fillna("").astype(str)
                ),
            )[["title", "ticker", "meta", "plan"]]
        )
        return ideas_df.to_dict(orient="records")

    def _results(df: pd.DataFrame) -> List[Dict[str, Any]]:
        cols = set(df.columns)
        out = []
        for _, row in df.iterrows():
            out.append(
                {
                    "ticker": row.get("Symbol"),
                    "signal": row.get("Signal"),
                    "rrr": f"R:R {row.get('RewardRisk'):.2f}" if pd.notna(row.get("RewardRisk")) else None,
                    "entry": row.get("EntryPrice"),
                    "stop": row.get("StopPrice"),
                    "target": row.get("TargetPrice"),
                    "techRating": row.get("TechRating") if "TechRating" in cols else None,
                    "riskScore": row.get("RiskScore") if "RiskScore" in cols else None,
                    "sector": row.get("Sector"),
                    "industry": row.get("Industry"),
                }
            )
        return out

    df_sorted = df.sort_values("RewardRisk", ascending=False)

    return {
        "results": _results(df_sorted),
        "movers": _movers(df_sorted),
        "ideas": _ideas(df_sorted),
        "log": log,
        "universe_size": len(df),
        "symbols_scanned": len(df),
    }


@app.get("/v1/symbol/{ticker}", response_model=SymbolResponse)
def symbol_detail_v1(ticker: str, days: int = Query(90, ge=1, le=365)) -> Dict[str, Any]:
    """
    V1 API: Get detailed symbol information (mobile app compatible).
    """
    return symbol_detail(ticker, days, intraday=False)


@app.post("/v1/copilot")
def copilot_v1(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    V1 API: Copilot endpoint (mobile app compatible).
    """
    return copilot(body)


@app.get("/v1/universe_stats", response_model=UniverseStats)
def universe_stats_v1() -> Dict[str, Any]:
    """
    V1 API: Get universe statistics (mobile app compatible).
    """
    return universe_stats()


# -------------------------------------------------------------------
# Original API Routes (Backward Compatible)
# -------------------------------------------------------------------

@app.get("/scan", response_model=ScanResponse)
def run_scan(
    max_symbols: int = Query(50, ge=1, le=500),
    min_tech_rating: float = Query(0.0, ge=0.0, le=100.0),
    sort_by: str = Query("RewardRisk"),
    ascending: bool = Query(False),
    include_log: bool = Query(True),
) -> Dict[str, Any]:
    """
    Run the equity scanner and return summary data.

    Query params:
      - max_symbols: limit how many symbols to scan (default 50)
      - min_tech_rating: filter to minimum TechRating (default 0.0)
      - sort_by: column to sort results (default RewardRisk)
      - ascending: sort order (default False)
      - include_log: include scan log text (default True)
    """
    if scanner_core is None:
        raise HTTPException(500, "scanner_core unavailable in this session")

    try:
        cfg = None
        if hasattr(scanner_core, "ScanConfig"):
            cfg = scanner_core.ScanConfig(max_symbols=max_symbols, min_tech_rating=min_tech_rating)
        df, log = scanner_core.run_scan(config=cfg)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"scan failed: {exc}") from exc

    records = df.to_dict(orient="records")

    # Structured payload for Flutter
    def _movers(df: pd.DataFrame) -> List[Dict[str, Any]]:
        if not {"Symbol", "RewardRisk", "Signal"} <= set(df.columns):
            return []
        movers_df = (
            df[["Symbol", "RewardRisk", "Signal"]]
            .assign(
                ticker=lambda x: x["Symbol"],
                delta=lambda x: x["RewardRisk"].round(2),
                note=lambda x: x["Signal"],
                isPositive=lambda x: x["RewardRisk"].fillna(0) >= 0,
            )[["ticker", "delta", "note", "isPositive"]]
            .head(6)
        )
        return movers_df.to_dict(orient="records")

    def _ideas(df: pd.DataFrame) -> List[Dict[str, Any]]:
        cols = set(df.columns)
        if "Symbol" not in cols:
            return []
        ideas_df = (
            df.sort_values("RewardRisk", ascending=False)
            .head(5)
            .assign(
                title=lambda x: x["Signal"] if "Signal" in cols else "Idea",
                ticker=lambda x: x["Symbol"],
                meta=lambda x: (
                    "R:R "
                    + x.get("RewardRisk", pd.Series()).fillna(0).round(2).astype(str)
                    + (" • TechRating " + x.get("TechRating", pd.Series()).fillna(0).round(1).astype(str)
                       if "TechRating" in cols else "")
                ),
                plan=lambda x: (
                    "Entry " + x.get("EntryPrice", pd.Series()).fillna("").astype(str)
                    + " • Stop " + x.get("StopPrice", pd.Series()).fillna("").astype(str)
                    + " • Target " + x.get("TargetPrice", pd.Series()).fillna("").astype(str)
                ),
            )[["title", "ticker", "meta", "plan"]]
        )
        return ideas_df.to_dict(orient="records")

    def _results(df: pd.DataFrame) -> List[Dict[str, Any]]:
        cols = set(df.columns)
        out = []
        for _, row in df.iterrows():
            out.append(
                {
                    "ticker": row.get("Symbol"),
                    "signal": row.get("Signal"),
                    "rrr": f"R:R {row.get('RewardRisk'):.2f}" if pd.notna(row.get("RewardRisk")) else None,
                    "entry": row.get("EntryPrice"),
                    "stop": row.get("StopPrice"),
                    "target": row.get("TargetPrice"),
                    "techRating": row.get("TechRating") if "TechRating" in cols else None,
                    "riskScore": row.get("RiskScore") if "RiskScore" in cols else None,
                    "sector": row.get("Sector"),
                    "industry": row.get("Industry"),
                }
            )
        return out

    # optional sort
    if sort_by in df.columns:
        df_sorted = df.sort_values(sort_by, ascending=ascending)
    else:
        df_sorted = df

    payload = {
        "results": _results(df_sorted),
        "movers": _movers(df_sorted),
        "ideas": _ideas(df_sorted),
    }
    if include_log:
        payload["log"] = log
    return payload


@app.get("/options/{ticker}", response_model=OptionsResponse)
def options_candidates(
    ticker: str,
    direction: str = "call",
    trade_style: str = "Swing",
    tech_rating: Optional[float] = None,
    risk_score: Optional[float] = None,
    price_target: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Return ranked option candidates for a ticker.
    """
    if OptionChainService is None or select_option_candidates is None:
        raise HTTPException(500, "Options module unavailable")

    svc = OptionChainService()
    try:
        chain, meta = svc.fetch_chain_snapshot(symbol=ticker, contract_type=direction)
        underlying_px = None
        # Derive underlying price from snapshot payload if present
        for item in chain:
            underlying_obj = item.get("underlying_asset") or {}
            underlying_px = (
                underlying_obj.get("last")
                or underlying_obj.get("price")
                or underlying_obj.get("last_price")
            )
            if underlying_px is not None:
                break
        picks: List[Dict[str, Any]] = select_option_candidates(
            chain=chain,
            direction=direction,
            trade_style=trade_style,
            underlying_price=underlying_px,
            tech_rating=tech_rating,
            risk_score=risk_score,
            price_target=price_target,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"Options lookup failed: {exc}") from exc

    return {
        "ticker": ticker.upper(),
        "direction": direction,
        "trade_style": trade_style,
        "candidates": picks,
    }


@app.post("/copilot")
def copilot(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Proxy to the Quant Copilot LLM helper.
    """
    from technic_v4.ui.generate_copilot_answer import generate_copilot_answer

    question = (body or {}).get("question")
    if not question:
        raise HTTPException(400, "question is required")
    try:
        answer = generate_copilot_answer(question)
    except Exception as exc:  # pragma: no cover
        # Keep the endpoint alive even if LLM is unreachable
        return {"answer": f"Copilot unavailable: {exc}"}
    return {"answer": answer}

# Alias path for clients that expect /api/copilot
@app.post("/api/copilot")
def copilot_alias(body: Dict[str, Any]) -> Dict[str, Any]:
    return copilot(body)


@app.get("/universe_stats", response_model=UniverseStats)
def universe_stats() -> Dict[str, Any]:
    """
    Return cached universe sectors/subindustries for picklists.
    """
    if get_universe_stats is None:
        raise HTTPException(500, "Universe stats unavailable")
    try:
        stats = get_universe_stats()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"universe stats failed: {exc}") from exc

    return {
        "sectors": [name for name, _ in stats.get("sectors", [])],
        "subindustries": [name for name, _ in stats.get("subindustries", [])],
    }


@app.get("/symbol/{ticker}", response_model=SymbolResponse)
def symbol_detail(ticker: str, days: int = 90, intraday: bool = False) -> Dict[str, Any]:
    """
    Return a basic symbol snapshot: latest price, recent history, and fundamentals if available.
    """
    if get_stock_history_df is None:
        raise HTTPException(500, "price layer unavailable")

    symbol = ticker.upper()
    try:
        df = get_stock_history_df(symbol, days=days, use_intraday=intraday)
        history = df.reset_index().rename(columns={"Date": "date"}).to_dict(orient="records")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"history fetch failed: {exc}") from exc

    last_price = None
    if get_realtime_last:
        last = get_realtime_last(symbol)
        if last is not None:
            last_price = float(last)

    fundamentals: Dict[str, Any] = {}
    if get_fundamentals:
        try:
            fundamentals = get_fundamentals(symbol) or {}
        except Exception:
            fundamentals = {}

    # Flag to show options badge if an options chain likely exists
    options_available = OptionChainService is not None and select_option_candidates is not None

    return {
        "ticker": symbol,
        "last": last_price,
        "history": history,
        "fundamentals": fundamentals,
        "options_available": options_available,
    }


# Convenience launcher: uvicorn api:app --reload
def _main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)


if __name__ == "__main__":  # pragma: no cover
    _main()
