"""
API contract and optional FastAPI app for Flutter/mobile integration (v5).

Includes richer scan outputs: alpha, portfolio weights, explanations, options strategies,
scoreboard metrics, and regime tags. Responses are versioned and error responses are structured.
"""
from __future__ import annotations

import os
from typing import List, Optional, Dict, Any, Literal

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

try:
    from technic_v4.evaluation import scoreboard as eval_scoreboard
except Exception:
    eval_scoreboard = None  # type: ignore

# -------------------------
# Pydantic models
# -------------------------

class OptionStrategyDTO(BaseModel):
    symbol: str
    strategy_type: str
    legs: list
    expected_value: Optional[float] = None
    expected_return_pct: Optional[float] = None
    risk_score: Optional[float] = None
    notes: Optional[str] = None


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
    portfolio_weight: Optional[float] = Field(None, alias="Weight")
    regime_trend: Optional[str] = Field(None, alias="RegimeTrend")
    regime_vol: Optional[str] = Field(None, alias="RegimeVol")
    regime_state_id: Optional[int] = Field(None, alias="RegimeStateId")
    option_picks: Optional[list] = Field(None, alias="OptionPicks")
    option_strategies: Optional[List[OptionStrategyDTO]] = Field(None, alias="OptionStrategies")
    explanation: Optional[str] = Field(None, alias="Explanation")
    entry: Optional[float] = Field(None, alias="EntryPrice")
    stop: Optional[float] = Field(None, alias="StopPrice")
    target: Optional[float] = Field(None, alias="TargetPrice")
    reward_risk: Optional[float] = Field(None, alias="RewardRisk")
    position_size: Optional[float] = Field(None, alias="PositionSize")
    sector: Optional[str] = Field(None, alias="Sector")
    industry: Optional[str] = Field(None, alias="Industry")
    subindustry: Optional[str] = Field(None, alias="SubIndustry")
    tft_forecast_h1: Optional[float] = Field(None, alias="tft_forecast_h1")
    tft_forecast_h3: Optional[float] = Field(None, alias="tft_forecast_h3")
    tft_forecast_h5: Optional[float] = Field(None, alias="tft_forecast_h5")


class ScoreboardSummaryDTO(BaseModel):
    ic_30d: Optional[float] = None
    ic_90d: Optional[float] = None
    precision10_30d: Optional[float] = None
    hit_rate_30d: Optional[float] = None
    avgR_30d: Optional[float] = None
    sharpe_90d: Optional[float] = None


class ScanResponse(BaseModel):
    api_version: str = "v1.0"
    total: int
    count: int
    items: List[ScanItem]
    scoreboard_summary: Optional[ScoreboardSummaryDTO] = None
    regime: Optional[dict] = None
    metadata: Optional[dict] = None


class LiteScanResultItem(BaseModel):
    symbol: str
    price: Optional[float] = None
    tech_rating: float
    alpha_score: Optional[float] = None
    signal: str
    sector: Optional[str] = None
    short_explanation: Optional[str] = None
    profile_name: Optional[str] = None


class LiteScanResponse(BaseModel):
    api_version: str = "v1.0"
    items: List[LiteScanResultItem]
    regime: Optional[dict] = None
    metadata: Optional[dict] = None


class ScanRequest(BaseModel):
    max_symbols: Optional[int] = Field(None, ge=1, le=2000)
    lookback_days: Optional[int] = Field(None, ge=30, le=1500)
    min_tech_rating: Optional[float] = None
    allow_shorts: Optional[bool] = False
    trade_style: Optional[str] = None
    strategy_profile_name: Optional[str] = None
    profile: Optional[str] = Field(
        default=None,
        description="Risk profile to use for the scan: conservative | balanced | aggressive. If omitted, the default profile is used.",
    )
    risk_pct: Optional[float] = None
    target_rr: Optional[float] = None
    use_ml_alpha: Optional[bool] = None
    use_meta_alpha: Optional[bool] = None
    use_tft_features: Optional[bool] = None
    use_portfolio_optimizer: Optional[bool] = None
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)
    options_mode: Optional[Literal["stock_only", "stock_plus_options"]] = Field(
        default=None,
        description=(
            "User preference for options suggestions. "
            "'stock_only' suppresses options ideas in the response; "
            "'stock_plus_options' returns both stock and options plans. "
            "If omitted, defaults to 'stock_plus_options'."
        ),
    )


class HealthResponse(BaseModel):
    status: str = "ok"


class ErrorDTO(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


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
            content={"api_version": "v1.0", "error": {"code": str(exc.status_code), "message": exc.detail}},
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
            risk_pct=req.risk_pct or 1.0,
            target_rr=req.target_rr or 2.0,
            strategy_profile_name=req.strategy_profile_name,
            profile=req.profile,
        )
        # Feature flags
        if req.use_ml_alpha is not None:
            os.environ["TECHNIC_USE_ML_ALPHA"] = "1" if req.use_ml_alpha else "0"
        if req.use_meta_alpha is not None:
            os.environ["TECHNIC_USE_META_ALPHA"] = "1" if req.use_meta_alpha else "0"
        if req.use_tft_features is not None:
            os.environ["TECHNIC_USE_TFT_FEATURES"] = "1" if req.use_tft_features else "0"
        if req.use_portfolio_optimizer is not None:
            os.environ["USE_PORTFOLIO_OPTIMIZER"] = "1" if req.use_portfolio_optimizer else "0"
        try:
            df, status = run_scan(cfg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Scan failed: {exc}")
        if df is None or df.empty:
            return ScanResponse(total=0, count=0, items=[], regime=None, metadata={"status": status})

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
            rec["portfolio_weight"] = rec.get("Weight")
            rec["regime_trend"] = rec.get("RegimeTrend")
            rec["regime_vol"] = rec.get("RegimeVol")
            rec["option_picks"] = rec.get("OptionPicks")
            rec["option_strategies"] = rec.get("OptionStrategies")
            rec["explanation"] = rec.get("Explanation")
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

        sb_summary = None
        if eval_scoreboard is not None:
            try:
                sb_raw = eval_scoreboard.compute_history_metrics(n=10)
                # Basic mapping to ScoreboardSummaryDTO; same values used for both 30/90 placeholders for now
                sb_summary = ScoreboardSummaryDTO(
                    ic_30d=sb_raw.get("ic"),
                    ic_90d=sb_raw.get("ic"),
                    precision10_30d=sb_raw.get("precision_at_n"),
                    hit_rate_30d=sb_raw.get("hit_rate"),
                    avgR_30d=sb_raw.get("avg_R"),
                    sharpe_90d=None,
                )
            except Exception:
                sb_summary = None
        metadata = {
            "profile_name": req.profile or req.strategy_profile_name or cfg.strategy_profile_name,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "universe_size": total,
            "status": status,
        }
        return ScanResponse(
            api_version="v1.0",
            total=total,
            count=len(items_norm),
            items=items_norm,
            scoreboard_summary=sb_summary,
            regime=None,
            metadata=metadata,
        )

    @app.post("/lite-scan", response_model=LiteScanResponse, dependencies=[] if auth_dep is None else [Depends(auth_dep)])
    def lite_scan(req: ScanRequest):
        """
        Lightweight scan endpoint for mobile clients; returns only essentials.
        """
        cfg = ScanConfig(
            max_symbols=req.max_symbols,
            lookback_days=req.lookback_days or 150,
            min_tech_rating=req.min_tech_rating or 0.0,
            allow_shorts=bool(req.allow_shorts),
            trade_style=req.trade_style or "Short-term swing",
            risk_pct=req.risk_pct or 1.0,
            target_rr=req.target_rr or 2.0,
            strategy_profile_name=req.strategy_profile_name,
            profile=req.profile,
        )
        if req.use_ml_alpha is not None:
            os.environ["TECHNIC_USE_ML_ALPHA"] = "1" if req.use_ml_alpha else "0"
        if req.use_meta_alpha is not None:
            os.environ["TECHNIC_USE_META_ALPHA"] = "1" if req.use_meta_alpha else "0"
        if req.use_tft_features is not None:
            os.environ["TECHNIC_USE_TFT_FEATURES"] = "1" if req.use_tft_features else "0"
        try:
            df, status = run_scan(cfg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Scan failed: {exc}")
        if df is None or df.empty:
            return LiteScanResponse(api_version="v1.0", items=[], regime=None, metadata={"status": status})

        df = df.reset_index(drop=True)
        top = df.head(min(20, len(df)))
        items: List[LiteScanResultItem] = []
        for _, rec in top.iterrows():
            sym = rec.get("Symbol")
            price = rec.get("Last") or rec.get("Close") or None
            expl = rec.get("Explanation") or ""
            short_expl = (expl[:140] + "...") if expl and len(expl) > 140 else (expl or None)
            items.append(
                LiteScanResultItem(
                    symbol=sym,
                    price=float(price) if price is not None else None,
                    tech_rating=float(rec.get("TechRating", 0) or 0),
                    alpha_score=float(rec.get("AlphaScore")) if rec.get("AlphaScore") is not None else None,
                    signal=str(rec.get("Signal", "")),
                    sector=rec.get("Sector"),
                    short_explanation=short_expl,
                    profile_name=req.strategy_profile_name or cfg.strategy_profile_name,
                )
            )
        metadata = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "status": status,
            "count": len(items),
            "profile_name": req.strategy_profile_name or cfg.strategy_profile_name,
        }
        return LiteScanResponse(api_version="v1.0", items=items, regime=None, metadata=metadata)

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
