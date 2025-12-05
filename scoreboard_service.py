"""
Lightweight scoreboard API for Technic.

Run:
    uvicorn scoreboard_service:app --reload --port 8000

Env:
    SCOREBOARD_API_TOKEN   (optional) bearer token for auth
    SCOREBOARD_STORAGE_DIR (optional) directory for JSON storage, default ./data_cache/scoreboards
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse

API_TOKEN = os.getenv("SCOREBOARD_API_TOKEN") or None
STORAGE_DIR = Path(os.getenv("SCOREBOARD_STORAGE_DIR") or "./data_cache/scoreboards")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Technic Scoreboard API", version="1.0.0")


def _auth(header_val: str | None) -> None:
    if API_TOKEN is None:
        return
    token = (header_val or "").removeprefix("Bearer ").strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _path(user_id: str) -> Path:
    safe = "".join(ch for ch in user_id if ch.isalnum() or ch in ("-", "_"))
    return STORAGE_DIR / f"{safe or 'default'}.json"


def _load(user_id: str) -> List[Dict[str, Any]]:
    p = _path(user_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(user_id: str, payload: List[Dict[str, Any]]) -> None:
    p = _path(user_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/scoreboard/{user_id}")
def get_scoreboard(user_id: str, authorization: str | None = Header(default=None)):
    _auth(authorization)
    return {"scoreboard": _load(user_id)}


@app.post("/scoreboard/{user_id}")
def set_scoreboard(
    user_id: str,
    payload: Dict[str, Any],
    authorization: str | None = Header(default=None),
):
    _auth(authorization)
    entries = payload.get("scoreboard")
    if not isinstance(entries, list):
        raise HTTPException(status_code=400, detail="scoreboard must be a list")
    _save(user_id, entries)
    return JSONResponse({"status": "saved", "count": len(entries)})
