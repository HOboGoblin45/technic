from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from technic_v4.config.settings import get_settings
from technic_v4.scanner_core import run_scan, ScanConfig
from technic_v4.infra.logging import get_logger

logger = get_logger()
app = FastAPI(title="Technic API", version="1.0.0")


class ScanRequest(BaseModel):
    universe: Optional[List[str]] = None  # currently not wired; default loader used
    max_symbols: int = 25
    trade_style: str = "Short-term swing"
    min_tech_rating: float = 0.0
    lookback_days: int = 120
    allow_shorts: bool = False
    only_tradeable: bool = True
    sectors: Optional[List[str]] = None
    subindustries: Optional[List[str]] = None
    industry_contains: Optional[str] = None


class ScanResultRow(BaseModel):
    symbol: str
    signal: str
    techRating: float
    alphaScore: Optional[float] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    rationale: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None


class ScanResponse(BaseModel):
    status: str
    results: List[ScanResultRow]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": "1.0.0"}


def _format_scan_results(df: pd.DataFrame) -> List[ScanResultRow]:
    if df is None or df.empty:
        return []

    def _val(row, col):
        return row[col] if col in row and pd.notna(row[col]) else None

    rows: List[ScanResultRow] = []
    for _, row in df.iterrows():
        rows.append(
            ScanResultRow(
                symbol=str(_val(row, "Symbol") or ""),
                signal=str(_val(row, "Signal") or ""),
                techRating=float(_val(row, "TechRating") or 0.0),
                alphaScore=_val(row, "AlphaScore"),
                entry=_val(row, "Entry"),
                stop=_val(row, "Stop"),
                target=_val(row, "Target"),
                rationale=str(
                    _val(row, "Rationale")
                    or _val(row, "Explanation")
                    or ""
                ),
                sector=_val(row, "Sector"),
                industry=_val(row, "Industry"),
            )
        )
    return rows


@app.post("/v1/scan", response_model=ScanResponse)
def scan_v1(req: ScanRequest) -> ScanResponse:
    """
    Versioned scan endpoint with a stable schema.
    """
    cfg = ScanConfig(
        max_symbols=req.max_symbols,
        lookback_days=req.lookback_days,
        min_tech_rating=req.min_tech_rating,
        trade_style=req.trade_style,
        allow_shorts=req.allow_shorts,
        only_tradeable=req.only_tradeable,
        sectors=req.sectors,
        subindustries=req.subindustries,
        industry_contains=req.industry_contains,
    )
    logger.info("[API] /v1/scan request: %s", cfg)
    df, status = run_scan(config=cfg)
    return ScanResponse(status=status, results=_format_scan_results(df))


# Legacy endpoint kept for backward compatibility
@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest) -> ScanResponse:
    cfg = ScanConfig(
        max_symbols=req.max_symbols,
        lookback_days=req.lookback_days,
        min_tech_rating=req.min_tech_rating,
        trade_style=req.trade_style,
        allow_shorts=req.allow_shorts,
        only_tradeable=req.only_tradeable,
        sectors=req.sectors,
        subindustries=req.subindustries,
        industry_contains=req.industry_contains,
    )
    logger.info("[API] /scan request: %s", cfg)
    df, status = run_scan(config=cfg)
    return ScanResponse(status=status, results=_format_scan_results(df))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("technic_v4.api_server:app", host="0.0.0.0", port=8000, reload=True)
