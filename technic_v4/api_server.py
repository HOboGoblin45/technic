from __future__ import annotations

import os
import json
import time
import hashlib
from typing import Any, List, Optional, Literal

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from technic_v4.config.settings import get_settings
from technic_v4.config.product import PRODUCT
from technic_v4.config.pricing import PLANS
from technic_v4.scanner_core import ScanConfig, run_scan, OUTPUT_DIR as SCAN_OUTPUT_DIR
from technic_v4.ui.generate_copilot_answer import generate_copilot_answer

# Initialize logging
from technic_v4.infra.logging import get_logger
logger = get_logger()

# Initialize Redis cache
try:
    from technic_v4.cache.redis_cache import redis_cache
    REDIS_AVAILABLE = redis_cache.available
    logger.info(f"[API] Redis cache available: {REDIS_AVAILABLE}")
except Exception as e:
    redis_cache = None
    REDIS_AVAILABLE = False
    logger.warning(f"[API] Redis cache not available: {e}")

# Initialize AWS Lambda client
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
LAMBDA_FUNCTION_NAME = os.getenv('LAMBDA_FUNCTION_NAME', 'technic-scanner')
USE_LAMBDA = os.getenv('USE_LAMBDA', 'false').lower() == 'true'

lambda_client = None
logger.info(f"[API INIT] USE_LAMBDA env var: {os.getenv('USE_LAMBDA', 'not set')}")
logger.info(f"[API INIT] USE_LAMBDA parsed: {USE_LAMBDA}")
logger.info(f"[API INIT] AWS_REGION: {AWS_REGION}")
logger.info(f"[API INIT] LAMBDA_FUNCTION_NAME: {LAMBDA_FUNCTION_NAME}")

if USE_LAMBDA:
    try:
        import boto3
        logger.info(f"[API INIT] boto3 imported successfully")
        lambda_client = boto3.client('lambda', region_name=AWS_REGION)
        logger.info(f"[API INIT] Lambda client initialized: {LAMBDA_FUNCTION_NAME} in {AWS_REGION}")
        
        # Test Lambda connectivity
        try:
            response = lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
            logger.info(f"[API INIT] Lambda function verified: {response['Configuration']['FunctionName']}")
        except Exception as verify_error:
            logger.error(f"[API INIT] Lambda function verification failed: {verify_error}")
            
    except Exception as e:
        logger.error(f"[API INIT] Lambda client initialization failed: {e}", exc_info=True)
        lambda_client = None
else:
    logger.info(f"[API INIT] Lambda disabled (USE_LAMBDA={USE_LAMBDA})")

app = FastAPI(
    title="Technic API",
    version="1.0.0",
    description="Technic scan API: equity/options idea generation (with Lambda & Redis)",
)


# -----------------------------
# Auth
# -----------------------------
def get_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    """
    Simple header-based API key check. If TECHNIC_API_KEY is unset, allow all (dev mode).
    """
    expected = os.getenv("TECHNIC_API_KEY")
    if expected is None:
        return ""
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


# -----------------------------
# Schemas
# -----------------------------
class ScanRequest(BaseModel):
    universe: Optional[List[str]] = None  # currently not wired; default loader is used
    max_symbols: int = 25
    trade_style: str = "Short-term swing"
    min_tech_rating: float = 0.0
    options_mode: Optional[str] = "stock_plus_options"
    sectors: Optional[List[str]] = None
    lookback_days: Optional[int] = None


class CopilotRequest(BaseModel):
    question: str
    symbol: Optional[str] = None
    options_mode: Optional[Literal["stock_only", "stock_plus_options"]] = None


class ScanResultRow(BaseModel):
    symbol: str
    signal: str
    techRating: float
    alphaScore: Optional[float] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    rationale: Optional[str] = None
    optionTrade: Optional[dict] = None
    merit_score: Optional[float] = None
    merit_band: Optional[str] = None
    merit_flags: Optional[str] = None
    merit_summary: Optional[str] = None


class ScanResponse(BaseModel):
    status: str
    disclaimer: str
    results: List[ScanResultRow]
    universe_size: Optional[int] = None  # Total symbols in selected universe
    symbols_scanned: Optional[int] = None  # Actual symbols scanned


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, Any]:
    settings = get_settings()
    return {
        "api_version": app.version,
        "use_ml_alpha": settings.use_ml_alpha,
        "use_tft_features": settings.use_tft_features,
        "lambda_enabled": USE_LAMBDA and lambda_client is not None,
        "redis_enabled": REDIS_AVAILABLE,
    }


def _get_cache_key(req: ScanRequest) -> str:
    """Generate cache key from scan request"""
    config_dict = {
        "max_symbols": req.max_symbols,
        "trade_style": req.trade_style,
        "min_tech_rating": req.min_tech_rating,
        "options_mode": req.options_mode,
        "sectors": sorted(req.sectors) if req.sectors else None,
        "lookback_days": req.lookback_days,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    hash_key = hashlib.md5(config_str.encode()).hexdigest()
    return f"scan_v1:{hash_key}"


def _check_cache(cache_key: str) -> Optional[dict]:
    """Check if scan result is cached"""
    if not REDIS_AVAILABLE or not redis_cache:
        return None
    
    try:
        cached = redis_cache.get(cache_key)
        if cached:
            logger.info(f"[API] Cache HIT: {cache_key}")
            return json.loads(cached) if isinstance(cached, str) else cached
    except Exception as e:
        logger.warning(f"[API] Cache read error: {e}")
    
    return None


def _set_cache(cache_key: str, data: dict, ttl: int = 300) -> None:
    """Cache scan result"""
    if not REDIS_AVAILABLE or not redis_cache:
        return
    
    try:
        redis_cache.set(cache_key, json.dumps(data), ttl=ttl)
        logger.info(f"[API] Cache SET: {cache_key} (TTL: {ttl}s)")
    except Exception as e:
        logger.warning(f"[API] Cache write error: {e}")


def _invoke_lambda(req: ScanRequest) -> Optional[dict]:
    """Invoke AWS Lambda for heavy computation"""
    if not lambda_client:
        return None
    
    try:
        logger.info(f"[API] Invoking Lambda: {LAMBDA_FUNCTION_NAME}")
        start_time = time.time()
        
        # Prepare payload
        payload = {
            "max_symbols": req.max_symbols,
            "trade_style": req.trade_style,
            "min_tech_rating": req.min_tech_rating,
            "options_mode": req.options_mode or "stock_plus_options",
            "sectors": req.sectors,
            "lookback_days": req.lookback_days,
        }
        
        # Invoke Lambda
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        # Parse response
        response_payload = json.loads(response['Payload'].read())
        
        if response_payload.get('statusCode') != 200:
            error_body = json.loads(response_payload.get('body', '{}'))
            logger.error(f"[API] Lambda error: {error_body.get('error', 'Unknown error')}")
            return None
        
        # Parse successful response
        body = json.loads(response_payload['body'])
        elapsed = time.time() - start_time
        logger.info(f"[API] Lambda completed in {elapsed:.2f}s")
        
        return body
        
    except Exception as e:
        logger.error(f"[API] Lambda invocation failed: {e}", exc_info=True)
        return None


@app.get("/meta")
def meta():
    return {
        "name": PRODUCT.name,
        "tagline": PRODUCT.tagline,
        "short_description": PRODUCT.short_description,
        "website_url": PRODUCT.website_url,
        "docs_url": PRODUCT.docs_url,
        "plans": [
            {
                "id": p.id,
                "name": p.name,
                "price_usd_per_month": p.price_usd_per_month,
            }
            for p in PLANS
        ],
    }


@app.get("/v1/plans")
def list_plans():
    return [
        {
            "id": p.id,
            "name": p.name,
            "price_usd_per_month": p.price_usd_per_month,
            "description": p.description,
            "features": p.features,
        }
        for p in PLANS
    ]


def _format_scan_results(df: pd.DataFrame) -> List[ScanResultRow]:
    if df is None or df.empty:
        return []

    def _float_or_none(val):
        try:
            if val == val:
                return float(val)
        except Exception:
            return None
        return None

    def _maybe_option(row: pd.Series):
        if "OptionPicks" in row and row["OptionPicks"]:
            return row["OptionPicks"][0]
        if "OptionTrade" in row and row["OptionTrade"]:
            return row["OptionTrade"][0] if isinstance(row["OptionTrade"], list) else row["OptionTrade"]
        return None

    rows: List[ScanResultRow] = []
    for _, r in df.iterrows():
        rows.append(
            ScanResultRow(
                symbol=str(r.get("Symbol", "")),
                signal=str(r.get("Signal", "")),
                techRating=_float_or_none(r.get("TechRating")) or 0.0,
                alphaScore=_float_or_none(r.get("AlphaScore")),
                entry=_float_or_none(r.get("Entry")),
                stop=_float_or_none(r.get("Stop")),
                target=_float_or_none(r.get("Target")),
                rationale=str(r.get("Rationale") or r.get("Explanation") or ""),
                optionTrade=_maybe_option(r),
                merit_score=_float_or_none(r.get("MeritScore")),
                merit_band=str(r.get("MeritBand") or ""),
                merit_flags=str(r.get("MeritFlags") or ""),
                merit_summary=str(r.get("MeritSummary") or ""),
            )
        )
    return rows


def _load_latest_scan_row(symbol: str) -> Optional[pd.Series]:
    """
    Best-effort: load the latest technic_scan_results.csv and return the first row
    matching the given symbol (case-insensitive). Returns None on any failure.
    """
    try:
        path = SCAN_OUTPUT_DIR / "technic_scan_results.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "Symbol" not in df.columns:
            return None
        sym = symbol.upper().strip()
        mask = df["Symbol"].astype(str).str.upper().str.strip() == sym
        sub = df.loc[mask]
        if sub.empty:
            return None
        return sub.iloc[0]
    except Exception:
        return None


@app.post("/v1/scan", response_model=ScanResponse)
def scan_endpoint(req: ScanRequest, api_key: str = Depends(get_api_key)) -> ScanResponse:
    """
    Run a scan and return a stable, versioned schema.
    
    Flow:
    1. Check Redis cache
    2. If cached -> return immediately
    3. If not cached:
       a. For large scans (>50 symbols) -> try Lambda
       b. If Lambda fails or small scan -> use Render
    4. Cache the result
    """
    from technic_v4.universe_loader import load_universe
    
    start_time = time.time()
    options_mode = req.options_mode or "stock_plus_options"
    
    # Generate cache key
    cache_key = _get_cache_key(req)
    
    # Check cache first
    cached_result = _check_cache(cache_key)
    if cached_result:
        logger.info(f"[API] Returning cached result (took {time.time() - start_time:.2f}s)")
        return ScanResponse(**cached_result)
    
    # Calculate universe size
    universe = load_universe()
    universe_size = len(universe)
    
    if req.sectors:
        sector_set = {s.lower().strip() for s in req.sectors}
        filtered_universe = [
            row for row in universe
            if row.sector and row.sector.lower().strip() in sector_set
        ]
        universe_size = len(filtered_universe)
    
    # Decide execution strategy
    use_lambda = USE_LAMBDA and lambda_client and req.max_symbols > 50
    
    # Debug logging for Lambda decision
    logger.info(f"[LAMBDA DECISION] USE_LAMBDA={USE_LAMBDA}, lambda_client={'initialized' if lambda_client else 'None'}, max_symbols={req.max_symbols}, threshold=50")
    logger.info(f"[LAMBDA DECISION] Will use Lambda: {use_lambda}")
    
    if use_lambda:
        # Try Lambda for large scans
        logger.info(f"[API] Using Lambda for scan ({req.max_symbols} symbols)")
        lambda_result = _invoke_lambda(req)
        
        if lambda_result and lambda_result.get('results'):
            # Lambda succeeded
            elapsed = time.time() - start_time
            logger.info(f"[API] Lambda scan completed in {elapsed:.2f}s")
            
            # Format response
            response_data = {
                "status": f"Scan completed via Lambda in {elapsed:.1f}s",
                "disclaimer": " ".join(PRODUCT.disclaimers),
                "results": lambda_result['results'],
                "universe_size": universe_size,
                "symbols_scanned": len(lambda_result['results']),
            }
            
            # Cache the result
            _set_cache(cache_key, response_data, ttl=300)
            
            return ScanResponse(**response_data)
        else:
            # Lambda failed, fall back to Render
            logger.warning("[API] Lambda failed, falling back to Render")
    
    # Run scan locally on Render
    logger.info(f"[API] Using Render for scan ({req.max_symbols} symbols)")
    
    cfg = ScanConfig(
        max_symbols=req.max_symbols,
        trade_style=req.trade_style,
        min_tech_rating=req.min_tech_rating,
        options_mode=options_mode,
        sectors=req.sectors,
        lookback_days=req.lookback_days,
    )
    
    # Run the scan (returns 3 values: df, status_text, performance_metrics)
    result = run_scan(cfg)
    if len(result) == 3:
        df, status_text, performance_metrics = result
    else:
        # Backward compatibility
        df, status_text = result
        performance_metrics = {}
    
    # Track how many symbols were actually scanned
    symbols_scanned = len(df) if df is not None and not df.empty else 0
    
    elapsed = time.time() - start_time
    logger.info(f"[API] Render scan completed in {elapsed:.2f}s")

    disclaimer = " ".join(PRODUCT.disclaimers)
    
    response_data = {
        "status": f"{status_text} (Render: {elapsed:.1f}s)",
        "disclaimer": disclaimer,
        "results": _format_scan_results(df),
        "universe_size": universe_size,
        "symbols_scanned": symbols_scanned,
    }
    
    # Cache the result
    _set_cache(cache_key, response_data, ttl=300)

    return ScanResponse(**response_data)


@app.post("/v1/copilot")
async def copilot(request: CopilotRequest, api_key: str = Depends(get_api_key)):
    """
    Generate a Copilot answer for a question, optionally scoped to a symbol from
    the latest scan output.

    Request:
      {
        "question": "Explain this setup in plain language",
        "symbol": "ODP"  // optional
      }
    Response:
      {
        "answer": "…human-readable explanation…"
      }
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")

    row = None
    options_mode = request.options_mode or "stock_plus_options"
    if request.symbol:
        row = _load_latest_scan_row(request.symbol)

    try:
        answer = generate_copilot_answer(
            question=request.question,
            row=row,
            options_mode=options_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Copilot error: {exc}") from exc

    return {"answer": answer}


class PricePoint(BaseModel):
    date: str
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int


class Fundamentals(BaseModel):
    pe: Optional[float] = None
    eps: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    market_cap: Optional[float] = None


class EventInfo(BaseModel):
    next_earnings: Optional[str] = None
    days_to_earnings: Optional[int] = None
    next_dividend: Optional[str] = None
    dividend_amount: Optional[float] = None


class SymbolDetailResponse(BaseModel):
    symbol: str
    last_price: Optional[float] = None
    change_pct: Optional[float] = None
    history: List[PricePoint]
    fundamentals: Optional[Fundamentals] = None
    events: Optional[EventInfo] = None
    
    # MERIT & Scores
    merit_score: Optional[float] = None
    merit_band: Optional[str] = None
    merit_flags: Optional[str] = None
    merit_summary: Optional[str] = None
    tech_rating: Optional[float] = None
    win_prob_10d: Optional[float] = None
    quality_score: Optional[float] = None
    ics: Optional[float] = None
    ics_tier: Optional[str] = None
    alpha_score: Optional[float] = None
    risk_score: Optional[str] = None
    
    # Factor breakdown
    momentum_score: Optional[float] = None
    value_score: Optional[float] = None
    quality_factor: Optional[float] = None
    growth_score: Optional[float] = None
    
    # Options
    options_available: bool = False


@app.get("/v1/symbol/{ticker}", response_model=SymbolDetailResponse)
def symbol_detail(ticker: str, days: int = 90, api_key: str = Depends(get_api_key)) -> SymbolDetailResponse:
    """
    Get detailed information for a specific symbol including:
    - Price history (candlestick data)
    - MERIT Score and all quantitative metrics
    - Fundamentals
    - Events (earnings, dividends)
    - Factor breakdown
    
    This endpoint pulls data from the latest scan results if available,
    otherwise returns basic price history.
    """
    from technic_v4 import data_engine
    from technic_v4.data_layer.events import get_event_info
    
    ticker = ticker.upper().strip()
    
    # Get price history
    try:
        history_df = data_engine.get_price_history(ticker, days, freq="daily")
        if history_df is None or history_df.empty:
            raise HTTPException(status_code=404, detail=f"No price data found for {ticker}")
        
        # Convert to list of PricePoint
        history_df = history_df.reset_index()
        if 'Date' in history_df.columns:
            history_df['date'] = history_df['Date'].astype(str)
        elif 'index' in history_df.columns:
            history_df['date'] = history_df['index'].astype(str)
        
        history_list = []
        for _, row in history_df.iterrows():
            history_list.append(PricePoint(
                date=str(row.get('date', row.get('Date', ''))),
                Open=float(row.get('Open', 0)),
                High=float(row.get('High', 0)),
                Low=float(row.get('Low', 0)),
                Close=float(row.get('Close', 0)),
                Volume=int(row.get('Volume', 0)),
            ))
        
        last_price = float(history_df['Close'].iloc[-1]) if not history_df.empty else None
        
        # Calculate change percentage
        change_pct = None
        if len(history_df) >= 2:
            prev_close = float(history_df['Close'].iloc[-2])
            if prev_close > 0:
                change_pct = ((last_price - prev_close) / prev_close) * 100
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching price data: {str(e)}")
    
    # Try to get data from latest scan results
    scan_row = _load_latest_scan_row(ticker)
    
    # Extract metrics from scan row if available
    merit_score = None
    merit_band = None
    merit_flags = None
    merit_summary = None
    tech_rating = None
    win_prob_10d = None
    quality_score = None
    ics = None
    ics_tier = None
    alpha_score = None
    risk_score = None
    momentum_score = None
    value_score = None
    quality_factor = None
    growth_score = None
    
    if scan_row is not None:
        def _safe_float(val):
            try:
                if val == val:  # Check for NaN
                    return float(val)
            except:
                pass
            return None
        
        merit_score = _safe_float(scan_row.get("MeritScore"))
        merit_band = str(scan_row.get("MeritBand", "")) if scan_row.get("MeritBand") else None
        merit_flags = str(scan_row.get("MeritFlags", "")) if scan_row.get("MeritFlags") else None
        merit_summary = str(scan_row.get("MeritSummary", "")) if scan_row.get("MeritSummary") else None
        tech_rating = _safe_float(scan_row.get("TechRating"))
        win_prob_10d = _safe_float(scan_row.get("win_prob_10d"))
        quality_score = _safe_float(scan_row.get("QualityScore"))
        ics = _safe_float(scan_row.get("InstitutionalCoreScore"))
        ics_tier = str(scan_row.get("ICS_Tier", "")) if scan_row.get("ICS_Tier") else None
        alpha_score = _safe_float(scan_row.get("AlphaScore"))
        risk_score = str(scan_row.get("RiskScore", "")) if scan_row.get("RiskScore") else None
        
        # Factor scores (if available)
        momentum_score = _safe_float(scan_row.get("MomentumScore"))
        value_score = _safe_float(scan_row.get("ValueScore"))
        quality_factor = _safe_float(scan_row.get("QualityFactor"))
        growth_score = _safe_float(scan_row.get("GrowthScore"))
    
    # Get fundamentals
    fundamentals = None
    try:
        fund_data = data_engine.get_fundamentals(ticker)
        if fund_data:
            fundamentals = Fundamentals(
                pe=fund_data.get("pe"),
                eps=fund_data.get("eps"),
                roe=fund_data.get("roe"),
                debt_to_equity=fund_data.get("debt_to_equity"),
                market_cap=fund_data.get("market_cap"),
            )
    except:
        pass
    
    # Get events
    events = None
    try:
        event_data = get_event_info(ticker)
        if event_data:
            events = EventInfo(
                next_earnings=event_data.get("next_earnings"),
                days_to_earnings=event_data.get("days_to_earnings"),
                next_dividend=event_data.get("next_dividend"),
                dividend_amount=event_data.get("dividend_amount"),
            )
    except:
        pass
    
    return SymbolDetailResponse(
        symbol=ticker,
        last_price=last_price,
        change_pct=change_pct,
        history=history_list,
        fundamentals=fundamentals,
        events=events,
        merit_score=merit_score,
        merit_band=merit_band,
        merit_flags=merit_flags,
        merit_summary=merit_summary,
        tech_rating=tech_rating,
        win_prob_10d=win_prob_10d,
        quality_score=quality_score,
        ics=ics,
        ics_tier=ics_tier,
        alpha_score=alpha_score,
        risk_score=risk_score,
        momentum_score=momentum_score,
        value_score=value_score,
        quality_factor=quality_factor,
        growth_score=growth_score,
        options_available=False,  # TODO: Check if options data exists
    )


# -----------------------------
# Dev entrypoint
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "technic_v4.api_server:app",
        host="0.0.0.0",
        port=int(os.getenv("TECHNIC_API_PORT", "8502")),
        reload=True,
    )