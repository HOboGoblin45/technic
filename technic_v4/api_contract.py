"""
Minimal API contract and optional FastAPI app for Flutter integration.

- Defines Pydantic models for scan responses.
- Provides a create_app() helper that spins up a FastAPI app
  if FastAPI is installed. (Safe to import even when FastAPI
  is not present.)
"""
from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from technic_v4.scanner_core import ScanConfig, run_scan
try:
    from technic_v4.data_layer.options_data import OptionChainService
    from technic_v4.engine.options_selector import select_option_candidates
except Exception:
    OptionChainService = None  # type: ignore
    select_option_candidates = None  # type: ignore

try:
    from technic_v4.data_layer.price_layer import get_stock_history_df
except Exception:
    get_stock_history_df = None

try:
    from technic_v4.data_layer.fundamentals import get_fundamentals
except Exception:
    get_fundamentals = None

try:
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:  # FastAPI not required for core functionality
    FastAPI = None  # type: ignore
    Depends = None  # type: ignore
    HTTPException = None  # type: ignore
    JSONResponse = None  # type: ignore


# -------------------------
# Pydantic models
# -------------------------

class ScanItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    symbol: str = Field(..., alias="Symbol")
    tech_rating: float = Field(..., alias="TechRating")
    signal: str = Field(..., alias="Signal")
    mu_total: Optional[float] = Field(None, alias="MuTotal")
    mu_hat: Optional[float] = Field(None, alias="MuHat")
    mu_ml: Optional[float] = Field(None, alias="MuMl")
    alpha_score: Optional[float] = Field(None, alias="AlphaScore")
    risk_score: Optional[float] = Field(None, alias="risk_score")
    regime_trend: Optional[str] = Field(None, alias="RegimeTrend")
    regime_vol: Optional[str] = Field(None, alias="RegimeVol")
    option_picks: Optional[list] = Field(None, alias="OptionPicks")
    entry: Optional[float] = Field(None, alias="EntryPrice")
    stop: Optional[float] = Field(None, alias="StopPrice")
    target: Optional[float] = Field(None, alias="TargetPrice")
    reward_risk: Optional[float] = Field(None, alias="RewardRisk")
    position_size: Optional[float] = Field(None, alias="PositionSize")
    sector: Optional[str] = Field(None, alias="Sector")
    industry: Optional[str] = Field(None, alias="Industry")
    subindustry: Optional[str] = Field(None, alias="SubIndustry")


class ScanResponse(BaseModel):
    total: int
    count: int
    items: List[ScanItem]


class ScanRequest(BaseModel):
    max_symbols: Optional[int] = Field(None, ge=1, le=2000)
    lookback_days: Optional[int] = Field(None, ge=30, le=1500)
    min_tech_rating: Optional[float] = None
    allow_shorts: Optional[bool] = False
    trade_style: Optional[str] = None
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)


class HealthResponse(BaseModel):
    status: str = "ok"


class OptionItem(BaseModel):
    ticker: str
    contract_type: str
    strike: float
    expiration: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    delta: Optional[float] = None
    iv: Optional[float] = None
    oi: Optional[float] = None
    volume: Optional[float] = None
    spread_pct: Optional[float] = None
    score: Optional[float] = None


class OptionsResponse(BaseModel):
    symbol: str
    direction: str
    total: int
    count: int
    items: List[OptionItem]


class OptionsRequest(BaseModel):
    symbol: str
    direction: str = Field("call", pattern="^(call|put)$")
    trade_style: str = "Short-term swing"
    limit: int = Field(3, ge=1, le=10)


class FundamentalsSnapshot(BaseModel):
    sector: Optional[str] = None
    industry: Optional[str] = None
    subindustry: Optional[str] = None
    pe: Optional[float] = None
    peg: Optional[float] = None
    market_cap: Optional[float] = None


class SymbolDetailResponse(BaseModel):
    symbol: str
    has_history: bool
    has_fundamentals: bool
    fundamentals: FundamentalsSnapshot = FundamentalsSnapshot()
    error: Optional[str] = None


# -------------------------
# FastAPI factory (optional)
# -------------------------

def _require_fastapi():
    if FastAPI is None:
        raise ImportError("FastAPI not installed. Please `pip install fastapi uvicorn`.")


def _auth_dep():
    api_key = os.getenv("TECHNIC_API_KEY")
    if not api_key:
        return None  # no auth enforced

    def _check(x_api_key: Optional[str] = None):
        if x_api_key != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return _check


def create_app() -> "FastAPI":
    _require_fastapi()
    app = FastAPI(title="Technic API", version="1.0.0")
    auth_dep = _auth_dep()

    # Structured error handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.status_code, "message": exc.detail}},
        )

    @app.get("/health", response_model=HealthResponse, dependencies=[] if auth_dep is None else [Depends(auth_dep)])
    def health():
        return HealthResponse()

    @app.post("/scan", response_model=ScanResponse, dependencies=[] if auth_dep is None else [Depends(auth_dep)])
    def scan(req: ScanRequest):
        cfg = ScanConfig(
            max_symbols=req.max_symbols,
            lookback_days=req.lookback_days or 150,
            min_tech_rating=req.min_tech_rating or 0.0,
            allow_shorts=bool(req.allow_shorts),
            trade_style=req.trade_style or "Short-term swing",
        )
        try:
            df, status = run_scan(cfg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Scan failed: {exc}")
        if df is None or df.empty:
            return ScanResponse(total=0, count=0, items=[])

        df = df.reset_index(drop=True)
        total = len(df)
        start = max(req.offset, 0)
        end = max(req.offset, 0) + max(req.limit, 0)
        sliced = df.iloc[start:end] if req.limit is not None else df
        items = sliced.to_dict(orient="records")
        # Normalize field names to match Pydantic model aliases
        items_norm = []
        for rec in items:
            rec["tech_rating"] = rec.get("TechRating")
            rec["signal"] = rec.get("Signal")
            rec["mu_total"] = rec.get("MuTotal")
            rec["mu_hat"] = rec.get("MuHat")
            rec["mu_ml"] = rec.get("MuMl")
            rec["alpha_score"] = rec.get("AlphaScore")
            rec["risk_score"] = rec.get("risk_score") or rec.get("RiskScore")
            rec["regime_trend"] = rec.get("RegimeTrend")
            rec["regime_vol"] = rec.get("RegimeVol")
            rec["option_picks"] = rec.get("OptionPicks")
            rec["entry"] = rec.get("EntryPrice")
            rec["stop"] = rec.get("StopPrice")
            rec["target"] = rec.get("TargetPrice")
            rec["reward_risk"] = rec.get("RewardRisk")
            rec["position_size"] = rec.get("PositionSize")
            rec["sector"] = rec.get("Sector")
            rec["industry"] = rec.get("Industry")
            rec["subindustry"] = rec.get("SubIndustry")
            rec["symbol"] = rec.get("Symbol")
            items_norm.append(rec)

        return ScanResponse(total=total, count=len(items_norm), items=items_norm)

    @app.post("/options", response_model=OptionsResponse, dependencies=[] if auth_dep is None else [Depends(auth_dep)])
    def options(req: OptionsRequest):
        if OptionChainService is None or select_option_candidates is None:
            raise HTTPException(status_code=503, detail="Options module unavailable")

        svc = OptionChainService(api_key=os.getenv("POLYGON_API_KEY"))
        try:
            chain, meta = svc.fetch_chain_snapshot(symbol=req.symbol, contract_type=req.direction)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Options fetch failed: {exc}")

        # Attempt to get underlying price from chain metadata
        underlying_px = None
        try:
            if chain:
                underlying_obj = chain[0].get("underlying_asset") or {}
                underlying_px = underlying_obj.get("last") or underlying_obj.get("price")
        except Exception:
            pass

        picks = select_option_candidates(
            chain=chain,
            direction=req.direction,
            trade_style=req.trade_style,
            underlying_price=underlying_px,
            tech_rating=None,
            risk_score=None,
            price_target=None,
            signal=None,
        )

        items = []
        for p in picks[: max(1, req.limit)]:
            items.append(
                OptionItem(
                    ticker=p.get("ticker"),
                    contract_type=p.get("contract_type"),
                    strike=p.get("strike"),
                    expiration=p.get("expiration"),
                    bid=p.get("bid"),
                    ask=p.get("ask"),
                    mid=p.get("mid"),
                    delta=p.get("delta"),
                    iv=p.get("iv"),
                    oi=p.get("open_interest"),
                    volume=p.get("volume"),
                    spread_pct=p.get("spread_pct"),
                    score=p.get("score"),
                )
            )

        return OptionsResponse(
            symbol=req.symbol.upper(),
            direction=req.direction,
            total=len(chain),
            count=len(items),
            items=items,
        )

    @app.get("/symbol/{symbol}", response_model=SymbolDetailResponse, dependencies=[] if auth_dep is None else [Depends(auth_dep)])
    def symbol_detail(symbol: str):
        symbol_u = symbol.upper()
        has_history = False
        if get_stock_history_df is not None:
            try:
                hist = get_stock_history_df(symbol_u, days=60)
                has_history = bool(hist is not None and not hist.empty)
            except Exception:
                has_history = False

        fundamentals = FundamentalsSnapshot()
        has_fundamentals = False
        if get_fundamentals is not None:
            try:
                snap = get_fundamentals(symbol_u)
                if snap and getattr(snap, "raw", None):
                    fundamentals = FundamentalsSnapshot(
                        sector=snap.get("sector"),
                        industry=snap.get("industry"),
                        subindustry=snap.get("subindustry"),
                        pe=snap.get("pe") or snap.get("PE"),
                        peg=snap.get("peg"),
                        market_cap=snap.get("market_cap") or snap.get("marketcap"),
                    )
                    has_fundamentals = True
            except Exception:
                has_fundamentals = False

        return SymbolDetailResponse(
            symbol=symbol_u,
            has_history=has_history,
            has_fundamentals=has_fundamentals,
            fundamentals=fundamentals,
            error=None,
        )

    return app


# Convenience for running via `python -m technic_v4.api_contract`
if __name__ == "__main__":  # pragma: no cover
    _require_fastapi()
    import uvicorn  # type: ignore

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
