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
    max_symbols: int = 200
    lookback_days: int = 120
    min_tech_rating: float = 0.0
    trade_style: str = "Short-term swing"
    allow_shorts: bool = False
    only_tradeable: bool = True
    sectors: Optional[List[str]] = None
    subindustries: Optional[List[str]] = None
    industry_contains: Optional[str] = None


class ScanResponse(BaseModel):
    status: str
    count: int
    results: List[dict]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": "1.0.0"}


@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest) -> ScanResponse:
    """
    Run the Technic scan and return results as JSON.
    """
    settings = get_settings()
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
    logger.info("[API] scan request: %s", cfg)
    df, status = run_scan(config=cfg)
    if df is None or df.empty:
        return ScanResponse(status=status, count=0, results=[])
    return ScanResponse(status=status, count=len(df), results=df.to_dict(orient="records"))
