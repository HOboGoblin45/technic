from __future__ import annotations

import os
from typing import Any, List, Optional, Literal

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from technic_v4.config.settings import get_settings
from technic_v4.config.product import PRODUCT
from technic_v4.config.pricing import PLANS
from technic_v4.scanner_core import ScanConfig, run_scan, OUTPUT_DIR as SCAN_OUTPUT_DIR
from technic_v4.ui.generate_copilot_answer import generate_copilot_answer

app = FastAPI(
    title="Technic API",
    version="1.0.0",
    description="Technic scan API: equity/options idea generation",
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


class ScanResponse(BaseModel):
    status: str
    disclaimer: str
    results: List[ScanResultRow]


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
    }


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
    """
    options_mode = req.options_mode or "stock_plus_options"
    cfg = ScanConfig(
        max_symbols=req.max_symbols,
        trade_style=req.trade_style,
        min_tech_rating=req.min_tech_rating,
        options_mode=options_mode,
        sectors=req.sectors,
        lookback_days=req.lookback_days,
    )
    df, status_text = run_scan(cfg)

    disclaimer = " ".join(PRODUCT.disclaimers)

    return ScanResponse(status=status_text, disclaimer=disclaimer, results=_format_scan_results(df))


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
