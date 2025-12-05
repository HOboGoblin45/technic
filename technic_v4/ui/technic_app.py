from __future__ import annotations

import os
import json
import threading
import sys
import urllib.parse
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from collections import Counter

import pandas as pd
import streamlit as st
import numpy as np
import datetime as dt
import requests

try:
    import altair as alt

    HAVE_ALTAIR = True
except ImportError:
    HAVE_ALTAIR = False

# Quant Copilot (LLM) – now in a separate module
from generate_copilot_answer import generate_copilot_answer

try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

TECHNIC_API_PORT = int(os.getenv("TECHNIC_API_PORT", "8502"))  # default JSON API port
_api_server_started = False
_api_server_port_in_use: int | None = None
_api_status_path = Path(__file__).resolve().parent / "api_server_status.log"


def _log_api(msg: str) -> None:
    print(msg, flush=True)
    try:
        _api_status_path.parent.mkdir(parents=True, exist_ok=True)
        with _api_status_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _start_api_server():
    """
    Spin up a tiny HTTP server that exposes JSON endpoints:
    - /api/scanner
    - /api/movers
    - /api/ideas
    - /api/scoreboard
    - /api/copilot?prompt=...
    """
    global _api_server_started, _api_server_port_in_use
    if _api_server_started:
        return

    class _ApiHandler(BaseHTTPRequestHandler):
        def _json(self, payload: Any, status: int = 200):
            body = json.dumps(payload, default=str)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            path = parsed.path.rstrip("/") or "/"

            try:
                if path == "/api/scanner":
                    params = self._with_api_defaults(params)
                    cfg = _scan_config_from_params(params)
                    scan_df, _scan_log = run_scan(config=cfg)
                    self._json(_format_scan_results(scan_df))
                elif path == "/api/movers":
                    params = self._with_api_defaults(params)
                    cfg = _scan_config_from_params(params)
                    scan_df, _scan_log = run_scan(config=cfg)
                    self._json(_format_movers(scan_df))
                elif path == "/api/ideas":
                    params = self._with_api_defaults(params)
                    cfg = _scan_config_from_params(params)
                    scan_df, _scan_log = run_scan(config=cfg)
                    self._json(_format_ideas(scan_df))
                elif path == "/api/scoreboard":
                    self._json(_format_scoreboard_payload())
                elif path == "/api/copilot":
                    prompt_param = (params.get("prompt") or [""])[0]
                    self._json(_copilot_payload(prompt_param))
                elif path == "/api/copilot" and self.command == "GET":
                    prompt_param = (params.get("prompt") or [""])[0]
                    self._json(_copilot_payload(prompt_param))
                elif path == "/api/universe_stats":
                    self._json(_universe_stats())
                elif path == "/api/symbol":
                    symbol_param = (params.get("symbol") or params.get("ticker") or [""])[0]
                    lookback = int((params.get("lookback") or params.get("days") or ["90"])[0])
                    trade_style = (params.get("trade_style") or ["Short-term swing"])[0]
                    self._json(_single_symbol_scan(symbol_param, lookback, trade_style))
                else:
                    self._json({"error": f"unknown path '{path}'"}, status=404)
            except Exception as exc:  # pragma: no cover - defensive
                self._json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

        def do_POST(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = {}
            if length > 0:
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except Exception:
                    body = {}
            try:
                if path == "/api/copilot":
                    prompt = body.get("prompt") or body.get("question") or ""
                    self._json(_copilot_payload(prompt))
                elif path == "/api/symbol":
                    symbol_param = body.get("symbol") or body.get("ticker") or ""
                    lookback = int(body.get("lookback") or body.get("days") or 90)
                    trade_style = body.get("trade_style") or "Short-term swing"
                    self._json(_single_symbol_scan(symbol_param, lookback, trade_style))
                else:
                    self._json({"error": f"unknown path '{path}'"}, status=404)
            except Exception as exc:  # pragma: no cover
                self._json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

        def log_message(self, format, *args):  # pragma: no cover - silence server logs
            return

        @staticmethod
        def _with_api_defaults(params: dict[str, list[str]]) -> dict[str, list[str]]:
            """
            Apply forgiving defaults for API calls so the Flutter client
            gets data even when no query params are provided.
            """
            merged = {k: v[:] for k, v in params.items()}
            def _ensure(key: str, value: str):
                if key not in merged or not merged.get(key):
                    merged[key] = [value]

            # Allow full-universe scans by default
            _ensure("max_symbols", "6000")
            _ensure("lookback_days", "90")
            _ensure("min_tech_rating", "0")
            _ensure("allow_shorts", "true")
            _ensure("only_tradeable", "false")
            _ensure("trade_style", "Short-term swing")
            return merged

    def _bind_server(port: int):
        ThreadingHTTPServer.allow_reuse_address = True
        server = ThreadingHTTPServer(("", port), _ApiHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    # Trace entry
    _log_api("starting api server")

    # Try the requested port, fall back to next if busy
    for candidate_port in (TECHNIC_API_PORT, TECHNIC_API_PORT + 1, TECHNIC_API_PORT + 2):
        try:
            srv = _bind_server(candidate_port)
            _api_server_started = True
            _api_server_port_in_use = candidate_port
            msg = f"[technic API] serving JSON on http://localhost:{candidate_port}"
            _log_api(msg)
            break
        except Exception as exc:
            _api_server_started = False
            _api_server_port_in_use = None
            err = f"[technic API] failed to start on port {candidate_port}: {exc}"
            _log_api(err)
            continue


_start_api_server()

# -------------------------------------------------------------------
# Path & project-level imports
# -------------------------------------------------------------------

SPARKLINE_LOOKBACK_DAYS = 60
SPARKLINE_MAX_ROWS = 30  # tighter cap to keep scans snappy

# Price chart ranges for Symbol Detail
# Each entry: (label, {"days": int, "intraday": bool})
CHART_TIMEFRAMES = [
    ("1D", {"days": 3, "intraday": True}),
    ("1W", {"days": 10, "intraday": False}),
    ("1M", {"days": 30, "intraday": False}),
    ("3M", {"days": 90, "intraday": False}),
    ("6M", {"days": 180, "intraday": False}),
    ("1Y", {"days": 365, "intraday": False}),
    ("3Y", {"days": 1095, "intraday": False}),
]
# Quick chips to keep the header tidy
CHART_QUICK_LABELS = ["1D", "1W", "1M", "3M"]
CHART_DEFAULT = "3M"

BRAND_PRIMARY = "#b6ff3b"
BRAND_SECONDARY = "#5eead4"
BRAND_TEXT = "#e5e7eb"
BRAND_MUTED = "#9ca3af"
BRAND_DARK = "#0f172a"

# Screener presets (applied to controls when preset changes)
PRESET_CONFIG = {
    "Swing Breakouts": {
        "lookback_days": 120,
        "min_tech_rating": 12.0,
        "ui_style": "Swing",
        "risk_pct": 1.0,
        "rr_multiple": 2.0,
        "allow_shorts": False,
        "only_tradeable": True,
        "require_breakout_tag": True,
        "require_momentum_tag": True,
        "sf_breakouts": True,
        "sf_smooth_trends": True,
        "sf_exclude_choppy": True,
        "max_symbols": 300,
    },
    "Long-term Quality Leaders": {
        "lookback_days": 365,
        "min_tech_rating": 16.0,
        "ui_style": "Long-term",
        "risk_pct": 0.8,
        "rr_multiple": 2.0,
        "allow_shorts": False,
        "only_tradeable": True,
        "sf_smooth_trends": True,
        "sf_low_risk": True,
        "sf_high_conviction": True,
        "require_momentum_tag": False,
        "max_symbols": 400,
    },
    "Low-risk Trend Followers": {
        "lookback_days": 180,
        "min_tech_rating": 12.0,
        "ui_style": "Medium-term",
        "risk_pct": 0.8,
        "rr_multiple": 2.0,
        "allow_shorts": False,
        "only_tradeable": True,
        "sf_smooth_trends": True,
        "sf_low_risk": True,
        "sf_exclude_choppy": True,
    },
    "High-conviction Momentum": {
        "lookback_days": 150,
        "min_tech_rating": 18.0,
        "ui_style": "Swing",
        "risk_pct": 1.0,
        "rr_multiple": 2.5,
        "allow_shorts": False,
        "only_tradeable": True,
        "require_breakout_tag": True,
        "require_momentum_tag": True,
        "sf_breakouts": True,
        "sf_high_conviction": True,
    },
}

# News
NEWS_LIMIT = 8

# Live refresh
LIVE_REFRESH_SECONDS = 30

# Options cache (chain snapshot)
OPTION_CACHE_TTL = 8 * 60  # seconds

# Cloud sync
SCOREBOARD_SYNC_URL = os.getenv("SCOREBOARD_SYNC_URL")
SCOREBOARD_SYNC_TOKEN = os.getenv("SCOREBOARD_SYNC_TOKEN")
SCOREBOARD_USER_ID = os.getenv("SCOREBOARD_USER_ID", "default")
# Supabase (preferred cloud sync)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_USER_ID = os.getenv("SUPABASE_USER_ID")  # uuid matching auth.uid() when RLS is on

# API key (Polygon / Massive) — env-first with gentle fallback to config.py
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    try:
        from technic_v4.config import POLYGON_API_KEY as _cfg_key

        POLYGON_API_KEY = _cfg_key
    except Exception:
        POLYGON_API_KEY = None

def inject_premium_theme(mode: str = "dark") -> None:
    """
    Global app-level CSS to give Technic a polished, 'funded startup' look.
    - Dark, low-glare background (focus + trust)
    - Calm cyan/green accents (competence / profit)
    - Clear hierarchy for cards, tables and tabs
    """
    light = mode == "light"
    bg = "#f8fafc" if light else "#02040d"
    bg_alt = "#e2e8f0" if light else "#030711"
    surface = "#ffffff" if light else "rgba(10,16,28,0.98)"
    surface_soft = "#f8fafc" if light else "rgba(12,18,32,0.92)"
    border = "rgba(226,232,240,0.9)" if light else "rgba(148,163,184,0.28)"
    ink = "#0f172a" if light else "#e5e7eb"
    table_head = "#e2e8f0" if light else "rgba(10,16,28,0.98)"
    head_text = "#0f172a" if light else "#e5e7eb"
    hover_row = "rgba(182,255,59,0.12)" if light else "rgba(182,255,59,0.06)"
    sidebar_bg = "#f1f5f9" if light else "rgba(10,16,28,0.96)"
    st.markdown(
        """
        <style>
        :root {
            --technic-bg: """ + bg + """;
            --technic-bg-alt: """ + bg_alt + """;
            --technic-surface: """ + surface + """;
            --technic-surface-soft: """ + surface_soft + """;
            --technic-border-subtle: """ + border + """;
            --technic-brand-primary: #b6ff3b;     /* neon growth green */
            --technic-brand-primary-soft: rgba(182,255,59,0.16);
            --technic-brand-ink: """ + ink + """;
            --technic-accent: var(--technic-brand-primary);
            --technic-accent-soft: var(--technic-brand-primary-soft);
            --technic-positive: #9ef01a;         /* profit green */
            --technic-negative: #ff6b81;         /* loss red */
            --technic-warning:  #facc15;         /* soft amber */
        }

        /* App background + typography */
        .stApp {
            background: """ + ("linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%)" if light else "radial-gradient(circle at 20% 10%, rgba(182,255,59,0.08), transparent 32%), radial-gradient(circle at 80% 0%, rgba(59,130,246,0.06), transparent 36%), linear-gradient(180deg, #02040d 0%, #01030a 100%)") + """;
            color: var(--technic-brand-ink);
            font-family: "Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Sticky header container */
        .technic-header {
            position: sticky;
            top: 0;
            z-index: 900;
            background: linear-gradient(to bottom, rgba(8,12,22,0.96), rgba(8,12,22,0.9));
            border-bottom: 1px solid rgba(182,255,59,0.35);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }

        .technic-header-inner {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0.65rem 1.25rem 0.55rem;
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }

        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-symbol {
            width: 36px;
            height: 36px;
            border-radius: 12px 12px 6px 6px;
            background: linear-gradient(135deg, rgba(182,255,59,0.9), rgba(16,185,129,0.6));
            box-shadow: 0 10px 30px rgba(182,255,59,0.25);
            position: relative;
            transform: rotate(-6deg);
            overflow: hidden;
        }
        .brand-symbol::after {
            content: "";
            position: absolute;
            inset: 6px 10px 6px 14px;
            border-radius: 999px;
            border: 2px solid rgba(255,255,255,0.75);
            transform: skewX(-8deg);
        }
        .brand-symbol::before {
            content: "";
            position: absolute;
            width: 10px;
            height: 16px;
            background: rgba(255,255,255,0.18);
            top: 4px;
            right: 6px;
            border-radius: 6px;
            transform: rotate(-6deg);
        }
        .brand-wordmark h1 {
            margin: 0;
            font-size: 1.35rem;
            letter-spacing: 0.08em;
            font-weight: 800;
            color: #f8fafc;
            text-transform: uppercase;
        }
        .brand-wordmark .tagline {
            font-size: 0.85rem;
            color: #cbd5e1;
        }

        /* Side bar – keep it clean and legible */
        section[data-testid="stSidebar"] {
            background: """ + sidebar_bg + """;
            border-right: 1px solid rgba(182,255,59,0.22);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
        }

        /* Main container width & spacing */
        .main .block-container {
            padding-top: 0.9rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }

        /* Primary buttons (Run Scan, Ask Copilot, etc.) */
        .stButton>button {
            border-radius: 999px;
            border: 1px solid rgba(182,255,59,0.45);
            background: linear-gradient(135deg, #b6ff3b, #5eead4);
            color: #08101c;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            box-shadow: 0 18px 40px rgba(0,0,0,0.55);
        }

        .stButton>button:hover {
            border-color: rgba(182,255,59,0.8);
            background: linear-gradient(135deg, #d9ff6c, #8df8df);
            color: #04060f;
        }

        .stButton>button:focus-visible {
            outline: 2px solid var(--technic-brand-primary);
            outline-offset: 2px;
        }

        /* Metric chips, badges, etc. */
        .metric-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.18rem 0.65rem;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.5);
            font-size: 0.75rem;
            color: #e5e7eb;
            background: radial-gradient(circle at top left, rgba(56,189,248,0.18), rgba(15,23,42,1));
        }

        .metric-chip--good {
            border-color: rgba(34,197,94,0.7);
            background: radial-gradient(circle at top left, rgba(34,197,94,0.18), rgba(15,23,42,1));
        }

        .metric-chip--bad {
            border-color: rgba(248,113,113,0.7);
            background: radial-gradient(circle at top left, rgba(248,113,113,0.18), rgba(15,23,42,1));
        }

        /* Card container */
        .technic-card {
            background: linear-gradient(135deg, var(--technic-surface), var(--technic-surface-soft));
            border-radius: 18px;
            border: 1px solid var(--technic-border-subtle);
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 26px 60px rgba(0,0,0,0.65);
        }

        .technic-card h3,
        .technic-card h4 {
            margin-top: 0;
            margin-bottom: 0.4rem;
            font-weight: 600;
        }

        /* DataFrames / tables */
        .stDataFrame, .stTable {
            border-radius: 14px !important;
            overflow: hidden !important;
            border: 1px solid rgba(148,163,184,0.25);
            background-color: """ + ("#ffffff" if light else "rgba(10,16,28,0.96)") + """;
        }

        .stDataFrame table {
            font-size: 0.80rem;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
            gap: 0.25rem;
            border-bottom: 1px solid rgba(55,65,81,0.9);
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.4rem 0.9rem;
            border-radius: 999px 999px 0 0;
            background: transparent;
            color: #9ca3af;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: radial-gradient(circle at top, var(--technic-accent-soft), rgba(15,23,42,0.98));
            color: #e5e7eb;
            font-weight: 600;
        }

        /* Progress bar label */
        .technic-progress-label {
            text-align: center;
            font-size: 0.9rem;
            color: #e5e7eb;
            margin-top: 0.25rem;
        }

        /* Responsive tweaks */
        @media (max-width: 900px) {
            .technic-header-inner {
                align-items: flex-start;
                padding: 0.45rem 0.85rem 0.35rem;
            }
            .technictitle h1 {
                font-size: 1.05rem;
                letter-spacing: 0.18em;
            }
            .technic-subtitle {
                font-size: 0.78rem;
                text-align: left;
            }
            .main .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }
            .technic-card {
                padding: 0.9rem 0.8rem;
                border-radius: 16px;
            }
            .stDataFrame table {
                font-size: 0.78rem;
            }
        }

        /* Sidebar polish */
        section[data-testid="stSidebar"] .block-container {
            padding-top: 0.5rem;
        }
        .sidebar-card {
            background: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(2,6,23,0.9));
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 18px;
            padding: 0.85rem 0.9rem;
            margin-bottom: 0.9rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 12px 28px rgba(0,0,0,0.35);
        }
        .sidebar-card h2, .sidebar-card h3, .sidebar-card h4 {
            margin: 0 0 0.35rem 0;
            letter-spacing: 0.05em;
            font-size: 0.9rem;
            color: #e5e7eb;
        }
        section[data-testid="stSidebar"] label {
            color: #cbd5e1 !important;
            font-weight: 600;
        }

        /* Tabs styled + centered */
        .technic-tabs-container [role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.45rem;
            padding: 0.45rem;
            margin: 0.25rem 0 0.75rem 0;
            border-radius: 14px;
            border: 1px solid rgba(148,163,184,0.35);
            background: linear-gradient(135deg, rgba(10,16,28,0.9), rgba(10,16,28,0.8));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 10px 30px rgba(0,0,0,0.35);
        }
        .technic-tabs-container [role="radiogroup"] > label,
        .technic-tabs-container [role="radio"] {
            padding: 0.45rem 1rem;
            border-radius: 12px;
            border: 1px solid transparent;
            background: rgba(148,163,184,0.09);
            color: #cbd5e1;
            font-weight: 700;
            letter-spacing: 0.03em;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            cursor: pointer;
        }
        .technic-tabs-container [role="radio"][aria-checked="true"],
        .technic-tabs-container [role="radiogroup"] > label[data-checked="true"] {
            border-color: rgba(56,189,248,0.6);
            background: radial-gradient(circle at top left, rgba(182,255,59,0.35), rgba(10,16,28,0.95));
            color: #f8fafc;
            box-shadow: 0 8px 24px rgba(182,255,59,0.2);
        }

        /* Trade idea cards */
        .idea-card {
            background: linear-gradient(150deg, rgba(182,255,59,0.12), rgba(94,234,212,0.10));
            border: 1px solid rgba(182,255,59,0.28);
            border-radius: 16px;
            padding: 0.75rem 0.85rem;
            box-shadow: 0 14px 36px rgba(0,0,0,0.45);
            height: 100%;
        }
        .idea-card__top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            margin-bottom: 0.3rem;
            color: #e5e7eb;
        }
        .idea-chip {
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: rgba(56,189,248,0.25);
            color: #e0f2fe;
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .idea-signal {
            opacity: 0.9;
            font-weight: 600;
        }
        .idea-symbol {
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            margin-bottom: 0.4rem;
        }
        .idea-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.25rem;
            font-size: 0.9rem;
        }
        .idea-metrics span {
            display: block;
            color: #94a3b8;
            font-size: 0.75rem;
        }
        .idea-footer {
            margin-top: 0.45rem;
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: #cbd5e1;
        }

        /* Results table tweaks */
        .stDataFrame thead tr th {
            position: sticky !important;
            top: 0;
            z-index: 2;
            background: """ + table_head + """ !important;
            font-weight: 700 !important;
            color: """ + head_text + """ !important;
            border-bottom: 1px solid rgba(148,163,184,0.25) !important;
        }
        .stDataFrame tbody tr:nth-child(odd) {
            background: rgba(148,163,184,0.02);
        }
        .stDataFrame tbody tr:hover {
            background: """ + hover_row + """;
        }
        .stDataFrame td, .stDataFrame th {
            padding: 8px 10px !important;
        }

        /* Ticker */
        .ticker {
            overflow: hidden;
            border-radius: 14px;
            border: 1px solid rgba(148,163,184,0.25);
            background: """ + ("linear-gradient(90deg, #f8fafc, #e2e8f0)" if light else "linear-gradient(90deg, rgba(10,16,28,0.9), rgba(12,18,32,0.9))") + """;
            box-shadow: 0 12px 30px rgba(0,0,0,0.12);
        }
        .ticker__track {
            display: inline-flex;
            gap: 24px;
            white-space: nowrap;
            animation: ticker 44s linear infinite;
            padding: 4px 10px;
            min-width: 100%;
        }
        @keyframes ticker {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        .ticker__item {
            display: inline-flex;
            gap: 8px;
            align-items: center;
            font-weight: 700;
            font-size: 0.95rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# Add project root (parent of "technic_v4") to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local persistence for scoreboard
SCOREBOARD_PATH = ROOT / "data_cache" / "scoreboard.json"

# Try to import the unified price layer; if unavailable, sparklines are disabled.
try:
    from technic_v4.data_layer.price_layer import (
        get_stock_history_df as ui_price_history,
        get_realtime_last,
        start_realtime_stream,
        get_multi_timeframes,
        get_stream_status,
    )
except Exception:
    ui_price_history = None
    get_realtime_last = None
    start_realtime_stream = None
    get_multi_timeframes = None
    get_stream_status = None

try:
    from technic_v4.data_layer.fundamentals import get_fundamentals
except Exception:
    get_fundamentals = None

try:
    from technic_v4.data_layer.relative_strength import rs_change_percentile
except Exception:
    rs_change_percentile = None

try:
    from technic_v4.data_layer.options_data import OptionChainService
except Exception:
    OptionChainService = None  # type: ignore

try:
    from technic_v4.engine.options_selector import select_option_candidates
except Exception:
    select_option_candidates = None  # type: ignore

try:
    from technic_v4.engine.scoring import compute_scores
    from technic_v4.engine.trade_planner import plan_trades, RiskSettings
except Exception:
    compute_scores = None  # type: ignore
    plan_trades = None  # type: ignore
    RiskSettings = None  # type: ignore

FUNDAMENTAL_FIELDS = ["PE", "PEG", "Piotroski", "AltmanZ", "EPS_Growth"]

from technic_v4.scanner_core import run_scan, ScanConfig
from technic_v4.universe_loader import load_universe

# Optional: yfinance fundamentals fallback
try:
    import yfinance as yf
except ImportError:
    yf = None


def _float_or_none(val: Any) -> float | None:
    try:
        f = float(val)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _clean_num(val: Any) -> float | None:
    """
    Return None for NaN/None; otherwise float(value).
    Keeps JSON compliant (no NaN).
    """
    try:
        f = float(val)
    except Exception:
        return None
    if f != f:  # NaN check
        return None
    return f


def _remote_headers() -> dict[str, str]:
    if SCOREBOARD_SYNC_TOKEN:
        return {"Authorization": f"Bearer {SCOREBOARD_SYNC_TOKEN}"}
    return {}


def supabase_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def supabase_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_KEY or "",
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def supabase_fetch_scoreboard() -> list[dict[str, Any]]:
    if not supabase_enabled() or not SUPABASE_USER_ID:
        return []
    try:
        url = SUPABASE_URL.rstrip("/") + "/rest/v1/scoreboards"
        params = {
            "select": "entries",
            "user_id": f"eq.{SUPABASE_USER_ID}",
        }
        resp = requests.get(url, headers=supabase_headers(), params=params, timeout=8)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if not data:
            return []
        first = data[0] or {}
        entries = first.get("entries")
        return entries if isinstance(entries, list) else []
    except Exception:
        return []


def supabase_save_scoreboard(entries: list[dict[str, Any]]) -> None:
    if not supabase_enabled() or not SUPABASE_USER_ID:
        return
    try:
        url = SUPABASE_URL.rstrip("/") + "/rest/v1/scoreboards"
        payload = [
            {
                "user_id": SUPABASE_USER_ID,
                "entries": entries,
            }
        ]
        requests.post(url, headers=supabase_headers(), json=payload, timeout=8)
    except Exception:
        return


def load_scoreboard_from_disk() -> list[dict[str, Any]]:
    # Prefer Supabase if configured
    if supabase_enabled():
        sb = supabase_fetch_scoreboard()
        if sb:
            return sb

    # Prefer remote if configured
    if SCOREBOARD_SYNC_URL:
        try:
            resp = requests.get(
                f"{SCOREBOARD_SYNC_URL.rstrip('/')}/{SCOREBOARD_USER_ID}",
                headers=_remote_headers(),
                timeout=6,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and isinstance(data.get("scoreboard"), list):
                    return data["scoreboard"]
        except Exception:
            pass

    if not SCOREBOARD_PATH.exists():
        return []
    try:
        with open(SCOREBOARD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        return []
    return []


def persist_scoreboard_to_disk(entries: list[dict[str, Any]]) -> None:
    try:
        SCOREBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SCOREBOARD_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
    except Exception:
        pass


def sync_scoreboard_to_remote(entries: list[dict[str, Any]]) -> None:
    """
    Optional best-effort sync to a remote endpoint if SCOREBOARD_SYNC_URL is set.
    Sends JSON via POST; failure is silent.
    """
    # Supabase preferred if available
    if supabase_enabled():
        supabase_save_scoreboard(entries)
        return

    url = SCOREBOARD_SYNC_URL
    if not url:
        return

    def _sync():
        try:
            requests.post(
                f"{url.rstrip('/')}/{SCOREBOARD_USER_ID}",
                json={"scoreboard": entries},
                headers=_remote_headers(),
                timeout=5,
            )
        except Exception:
            pass

    threading.Thread(target=_sync, daemon=True).start()


def get_latest_close(symbol: str, days: int = 5) -> float | None:
    """
    Fetch the latest close for a symbol using the unified price layer.
    Returns None gracefully if data is unavailable.
    """
    if get_realtime_last:
        live = get_realtime_last(symbol)
        if live is not None:
            return live

    if ui_price_history is None:
        return None
    try:
        hist = ui_price_history(symbol=symbol, days=days, use_intraday=True)
    except Exception:
        return None

    if hist is None or hist.empty:
        return None

    last_row = hist.iloc[-1]
    close = last_row.get("Close")
    try:
        if pd.notna(close):
            return float(close)
    except Exception:
        return None
    return None


def ensure_scoreboard_state() -> None:
    if "scoreboard_entries" not in st.session_state:
        st.session_state["scoreboard_entries"] = []

    if not st.session_state.get("scoreboard_loaded", False):
        st.session_state["scoreboard_entries"] = load_scoreboard_from_disk()
        st.session_state["scoreboard_loaded"] = True


def add_row_to_scoreboard(row: pd.Series) -> None:
    ensure_scoreboard_state()
    now_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    entry = _float_or_none(row.get("EntryPrice"))
    stop = _float_or_none(row.get("StopPrice"))
    target = _float_or_none(row.get("TargetPrice"))
    rr = _float_or_none(row.get("RewardRisk"))
    size = _float_or_none(row.get("PositionSize"))

    st.session_state["scoreboard_entries"].append(
        {
            "symbol": str(row.get("Symbol", "")).upper(),
            "signal": str(row.get("Signal", "")),
            "trade_type": str(row.get("TradeType", row.get("Signal", ""))),
            "entry": entry,
            "stop": stop,
            "target": target,
            "rr": rr,
            "position_size": size,
            "added_at": now_ts,
            "note": "From scan",
        }
    )
    persist_scoreboard_to_disk(st.session_state["scoreboard_entries"])
    sync_scoreboard_to_remote(st.session_state["scoreboard_entries"])


def build_scoreboard_df(entries: list[dict[str, Any]]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for item in entries:
        sym = str(item.get("symbol", "")).upper()
        entry = _float_or_none(item.get("entry"))
        stop = _float_or_none(item.get("stop"))
        target = _float_or_none(item.get("target"))

        last_price = get_latest_close(sym) if sym else None

        status = "Active"
        outcome = None
        if last_price is None:
            status = "Price unavailable"
        elif target is not None and last_price >= target:
            status = "Target hit"
            outcome = "win"
        elif stop is not None and last_price <= stop:
            status = "Stopped out"
            outcome = "loss"

        pnl_pct = None
        if last_price is not None and entry not in (None, 0):
            pnl_pct = ((last_price - entry) / entry) * 100.0

        rows.append(
            {
                "Symbol": sym,
                "TradeType": item.get("trade_type") or "-",
                "Added": item.get("added_at", ""),
                "Entry": entry,
                "Stop": stop,
                "Target": target,
                "RR": item.get("rr"),
                "Last": last_price,
                "PnL%": pnl_pct,
                "Status": status,
                "Outcome": outcome or "",
            }
        )

    df = pd.DataFrame(rows)
    return df


def summarize_scoreboard(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "total": 0,
            "open": 0,
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "success_rate": None,
        }

    wins = int((df["Status"] == "Target hit").sum())
    losses = int((df["Status"] == "Stopped out").sum())
    closed = wins + losses
    total = len(df)
    open_trades = total - closed
    success_rate = (wins / closed * 100.0) if closed > 0 else None

    return {
        "total": total,
        "open": open_trades,
        "closed": closed,
        "wins": wins,
        "losses": losses,
        "success_rate": success_rate,
    }


def _first_param(params: dict[str, list[str]], key: str, default: Any = None) -> Any:
    values = params.get(key) or []
    if not values:
        return default
    return values[0]


def _parse_bool_param(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_param(value: str | None, default: int) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def _parse_float_param(value: str | None, default: float) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


def _parse_list_param(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",")]
    parts = [p for p in parts if p]
    return parts or None


def _scan_config_from_params(params: dict[str, list[str]]) -> ScanConfig:
    base_cfg = ScanConfig()
    ui_style = _first_param(params, "ui_style")
    trade_style = _first_param(params, "trade_style", base_cfg.trade_style)
    if ui_style:
        ui_style_l = ui_style.lower()
        if ui_style_l.startswith("swing"):
            trade_style = "Short-term swing"
        elif ui_style_l.startswith("medium"):
            trade_style = "Medium-term swing"
        elif ui_style_l.startswith("long"):
            trade_style = "Position / longer-term"

    sectors = _parse_list_param(_first_param(params, "sectors"))
    subindustries = _parse_list_param(_first_param(params, "subindustries"))

    return ScanConfig(
        max_symbols=_parse_int_param(_first_param(params, "max_symbols"), base_cfg.max_symbols),
        lookback_days=_parse_int_param(_first_param(params, "lookback_days"), base_cfg.lookback_days),
        min_tech_rating=_parse_float_param(_first_param(params, "min_tech_rating"), base_cfg.min_tech_rating),
        account_size=_parse_float_param(_first_param(params, "account_size"), base_cfg.account_size),
        risk_pct=_parse_float_param(_first_param(params, "risk_pct"), base_cfg.risk_pct),
        target_rr=_parse_float_param(_first_param(params, "target_rr"), base_cfg.target_rr),
        trade_style=trade_style or base_cfg.trade_style,
        allow_shorts=_parse_bool_param(_first_param(params, "allow_shorts"), base_cfg.allow_shorts),
        only_tradeable=_parse_bool_param(_first_param(params, "only_tradeable"), base_cfg.only_tradeable),
        sectors=sectors,
        subindustries=subindustries,
        industry_contains=_first_param(params, "industry") or None,
    )


def _format_scan_results(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []

    cols = set(df.columns)
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rr_val = _float_or_none(row.get("RewardRisk"))
        rrr_txt = f"R:R {rr_val:.2f}" if rr_val is not None else None
        out.append(
            {
                "ticker": row.get("Symbol"),
                "signal": row.get("Signal"),
                "rrr": rrr_txt,
                "entry": _clean_num(row.get("EntryPrice")),
                "stop": _clean_num(row.get("StopPrice")),
                "target": _clean_num(row.get("TargetPrice")),
                "note": row.get("TradeType") or row.get("Signal") or "",
                "techRating": _clean_num(row.get("TechRating")) if "TechRating" in cols else None,
                "riskScore": _clean_num(row.get("RiskScore")) if "RiskScore" in cols else None,
                "sector": row.get("Sector"),
                "industry": row.get("Industry"),
            }
        )
    return out


def _format_movers(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    if not {"Symbol", "RewardRisk", "Signal"} <= set(df.columns):
        return []
    movers: list[dict[str, Any]] = []
    # Prefer names with real reward/risk and non-Avoid signals
    filtered = df.copy()
    filtered = filtered[filtered["Signal"].isin(["Strong Long", "Long", "Strong Short", "Short"]) | df["Signal"].notna()]
    filtered = filtered[pd.notna(filtered["RewardRisk"]) | pd.notna(filtered.get("TechRating", None))]
    for _, r in filtered.head(6).iterrows():
        delta_val = _clean_num(r.get("RewardRisk"))
        if delta_val is None:
            delta_val = _clean_num(r.get("TechRating"))
        movers.append(
            {
                "ticker": r.get("Symbol"),
                "delta": delta_val,
                "note": r.get("Signal"),
                "isPositive": False if delta_val is None else delta_val >= 0,
            }
        )
    return movers


def _format_ideas(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty or "Symbol" not in df.columns:
        return []

    def _stringify(val: Any) -> str:
        if val is None:
            return ""
        try:
            if pd.isna(val):
                return ""
        except Exception:
            pass
        return str(val)

    working = df
    if {"RewardRisk", "EntryPrice", "StopPrice", "TargetPrice"} <= set(df.columns):
        working = working[
            working["RewardRisk"].notna()
            & working["EntryPrice"].notna()
            & working["StopPrice"].notna()
            & working["TargetPrice"].notna()
        ]
    if "RewardRisk" in working.columns:
        working = working.sort_values("RewardRisk", ascending=False)

    ideas_df = working.head(5)
    ideas: list[dict[str, Any]] = []
    for _, row in ideas_df.iterrows():
        rr_val = _float_or_none(row.get("RewardRisk"))
        rr_txt = f"R:R {rr_val:.2f}" if rr_val is not None else "R:R n/a"
        tech_val = _float_or_none(row.get("TechRating"))
        meta_parts = [rr_txt]
        if tech_val is not None:
            meta_parts.append(f"TechRating {tech_val:.1f}")

        option_pick = None
        if (
            OptionChainService is not None
            and select_option_candidates is not None
            and POLYGON_API_KEY
        ):
            direction = "put" if str(row.get("Signal", "")).lower().find("short") >= 0 else "call"
            try:
                svc = OptionChainService()
                chain, meta = svc.fetch_chain_snapshot(symbol=str(row["Symbol"]), contract_type=direction)
                picks = select_option_candidates(
                    chain=chain,
                    direction=direction,
                    trade_style="Short-term swing",
                    underlying_price=None,
                    tech_rating=_float_or_none(row.get("TechRating")),
                    risk_score=_float_or_none(row.get("RiskScore")),
                    price_target=_float_or_none(row.get("TargetPrice")),
                    signal=row.get("Signal"),
                )
                if picks:
                    top = picks[0]
                    option_pick = {
                        "ticker": top.get("ticker"),
                        "contract_type": top.get("contract_type"),
                        "strike": top.get("strike"),
                        "expiration": top.get("expiration"),
                        "dte": top.get("dte"),
                        "delta": top.get("delta"),
                        "iv": top.get("iv"),
                        "bid": top.get("bid"),
                        "ask": top.get("ask"),
                        "mid": top.get("mid"),
                    }
            except Exception:
                option_pick = None

        ideas.append(
            {
                "title": row.get("Signal") or "Idea",
                "ticker": row.get("Symbol"),
                "meta": " | ".join(meta_parts),
                "plan": (
                    f"Entry {_stringify(_clean_num(row.get('EntryPrice')))} | "
                    f"Stop {_stringify(_clean_num(row.get('StopPrice')))} | "
                    f"Target {_stringify(_clean_num(row.get('TargetPrice')))}"
                ),
                "sparkline": [],
                "option": option_pick,
            }
        )
    return ideas


def _format_scoreboard_payload() -> list[dict[str, Any]]:
    ensure_scoreboard_state()
    sb_entries = st.session_state.get("scoreboard_entries", [])
    sb_df = build_scoreboard_df(sb_entries)
    stats = summarize_scoreboard(sb_df)

    avg_pnl = None
    if not sb_df.empty and "PnL%" in sb_df.columns:
        try:
            avg_pnl = float(sb_df["PnL%"].dropna().mean())
        except Exception:
            avg_pnl = None

    def _pnl_text(val: float | None) -> str:
        if val is None or pd.isna(val):
            return "+0.0%"
        sign = "+" if val > 0 else ""
        return f"{sign}{val:.1f}%"

    win_rate_txt = "n/a"
    if stats.get("success_rate") is not None:
        win_rate_txt = f"{stats['success_rate']:.0f}% win"

    slices: list[dict[str, Any]] = [
        {
            "label": "Scoreboard",
            "pnl": _pnl_text(avg_pnl),
            "winRate": win_rate_txt,
            "horizon": f"{stats['open']} open / {stats['total']} total",
            "accent": BRAND_PRIMARY,
        }
    ]

    if stats.get("closed"):
        slices.append(
            {
                "label": "Closed",
                "pnl": f"W {stats['wins']} / L {stats['losses']}",
                "winRate": win_rate_txt,
                "horizon": "Completed trades",
                "accent": BRAND_SECONDARY,
            }
        )

    return slices


def _universe_stats() -> dict[str, Any]:
    try:
        universe = load_universe()
    except Exception as exc:
        return {"error": f"universe load failed: {exc}"}
    total = len(universe)
    counter: Counter[str] = Counter()
    sub_counter: Counter[str] = Counter()
    for u in universe:
        if u.sector:
            counter[u.sector] += 1
        if u.subindustry:
            sub_counter[u.subindustry] += 1
    sectors = [{"name": name, "count": count} for name, count in counter.most_common()]
    subindustries = [{"name": name, "count": count} for name, count in sub_counter.most_common()]
    return {"total": total, "sectors": sectors, "subindustries": subindustries}


def _copilot_payload(prompt: str) -> dict[str, Any]:
    prompt_clean = (prompt or "").strip()
    if not prompt_clean:
        return {"role": "assistant", "body": "prompt is required"}
    try:
        # Try preferred signatures gracefully
        try:
            answer = generate_copilot_answer(prompt_clean, row=None)  # type: ignore[arg-type]
        except TypeError:
            answer = generate_copilot_answer(prompt_clean)
    except Exception as exc:
        return {"role": "assistant", "body": f"Copilot error: {exc}"}
    return {"role": "assistant", "body": answer}


# ------------------------------------------------------------------
# Lightweight JSON API shim (used by Flutter client)
# ------------------------------------------------------------------
def _single_symbol_scan(symbol: str, lookback: int, trade_style: str) -> dict[str, Any]:
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"error": "symbol is required"}
    if ui_price_history is None or compute_scores is None or plan_trades is None or RiskSettings is None:
        return {"error": "price layer or scoring unavailable"}
    try:
        use_intraday = "short-term" in trade_style.lower() or "swing" in trade_style.lower()
        hist = ui_price_history(sym, days=lookback, use_intraday=use_intraday)
        if hist is None or hist.empty:
            return {"error": f"no history for {sym}"}
        scored = compute_scores(hist, trade_style=trade_style)
        latest = scored.iloc[-1:].copy()
        latest["Symbol"] = sym
        risk = RiskSettings(account_size=10_000.0, risk_pct=1.0, target_rr=2.0, trade_style=trade_style)
        planned = plan_trades(latest, risk)
        return {"results": _format_scan_results(planned)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

_api_params = st.experimental_get_query_params()
_api_target = (_api_params.get("api") or [None])[0]
if _api_target:
    api_kind = str(_api_target).lower()

    if api_kind in {"scanner", "movers", "ideas"}:
        cfg = _scan_config_from_params(_api_params)
        progress_messages: list[str] = []

        def _progress_cb(symbol: str, idx: int, total: int):
            # idx is 1-based inside run_scan
            pct = int((idx / max(total, 1)) * 100)
            progress_messages.append(f"{symbol.upper()} ({idx}/{total}) {pct}%")

        try:
            scan_df, _scan_log = run_scan(config=cfg, progress_cb=_progress_cb)
        except Exception as exc:
            st.json({"error": f"scan failed: {exc}"})
            st.stop()

        if scan_df is None or scan_df.empty:
            st.json([])
            st.stop()

        if api_kind == "scanner":
            st.json(
                {
                    "results": _format_scan_results(scan_df),
                    "progress": progress_messages[-1] if progress_messages else None,
                }
            )
        elif api_kind == "movers":
            st.json(_format_movers(scan_df))
        else:
            st.json(_format_ideas(scan_df))
        st.stop()

    elif api_kind == "scoreboard":
        st.json(_format_scoreboard_payload())
        st.stop()

    elif api_kind == "copilot":
        prompt_param = (_api_params.get("prompt") or [""])[0]
        st.json(_copilot_payload(prompt_param))
        st.stop()

    else:
        st.json({"error": f"unknown api '{api_kind}'"})
        st.stop()

# -------------------------------------------------------------------
# Table styling helpers
# -------------------------------------------------------------------

SECTOR_COLORS = {
    "Energy": "#ff8a65",
    "Materials": "#ffb74d",
    "Industrials": "#64b5f6",
    "Consumer Discretionary": "#ba68c8",
    "Consumer Staples": "#4db6ac",
    "Health Care": "#e57373",
    "Financials": "#81c784",
    "Information Technology": "#4fc3f7",
    "Communication Services": "#9575cd",
    "Utilities": "#90a4ae",
    "Real Estate": "#a1887f",
    "Unknown": "#e0e0e0",
}


def style_results_table(df: pd.DataFrame, compact: bool = False) -> "pd.io.formats.style.Styler":
    """
    Build a styled DataFrame for Streamlit.

    - Heatmap for TechRating
    - Color-coded Sector column
    - Basic bold header styling
    """
    if df is None or df.empty:
        return df.style  # type: ignore[return-value]

    styler = df.style

    # TechRating gradient
    if "TechRating" in df.columns:
        max_val = float(df["TechRating"].max())
        min_val = float(df["TechRating"].min())
        rng = max_val - min_val if max_val != min_val else 1.0

        def _techrating_color(val: float) -> str:
            if pd.isna(val):
                return ""
            rel = (val - min_val) / rng if rng != 0 else 0.5
            if rel >= 0.7:
                return "background-color: rgba(0, 200, 83, 0.25);"  # green
            elif rel >= 0.4:
                return "background-color: rgba(255, 235, 59, 0.25);"  # yellow
            else:
                return "background-color: rgba(244, 67, 54, 0.18);"  # red

        styler = styler.applymap(_techrating_color, subset=["TechRating"])

    # Core subscores: subtle blue heatmap
    score_cols = [
        c
        for c in (
            "TrendScore",
            "MomentumScore",
            "ExplosivenessScore",
            "BreakoutScore",
            "VolumeScore",
            "VolatilityScore",
            "OscillatorScore",
            "TrendQualityScore",
        )
        if c in df.columns
    ]
    if score_cols:
        styler = styler.background_gradient(
            subset=score_cols,
            cmap="Greens",
        )

    # MatchMode "badge" styling (Strict vs Relaxed)
    if "MatchMode" in df.columns:
        def style_match_mode(val: Any) -> str:
            text = str(val or "").lower()
            if "strict" in text:
                # strict = high confidence
                return (
                    "background-color: rgba(46, 204, 113, 0.18); "
                    "color: #b2fab4; font-weight: 600; border-radius: 999px; "
                    "padding: 0.05rem 0.4rem; text-align: center;"
                )
            elif "relaxed" in text:
                # relaxed = looser idea
                return (
                    "background-color: rgba(241, 196, 15, 0.14); "
                    "color: #ffe082; font-weight: 500; border-radius: 999px; "
                    "padding: 0.05rem 0.4rem; text-align: center;"
                )
            else:
                return (
                    "background-color: rgba(96, 125, 139, 0.25); "
                    "color: #cfd8dc; border-radius: 999px; "
                    "padding: 0.05rem 0.4rem; text-align: center;"
                )

        styler = styler.applymap(style_match_mode, subset=["MatchMode"])

    # If you later add a SetupTags column, make it look like compact pills
    if "SetupTags" in df.columns:
        styler = styler.set_properties(
            subset=["SetupTags"],
            **{
                "font-size": "0.75rem",
                "color": "#eceff1",
                "white-space": "pre-wrap",
            },
        )

    # Sector coloring
    if "Sector" in df.columns:

        def _sector_color(sector: str) -> str:
            color = SECTOR_COLORS.get(str(sector), SECTOR_COLORS["Unknown"])
            return f"background-color: {color}; color: black;"

        styler = styler.applymap(_sector_color, subset=["Sector"])

    # Tooltips for composite / RS context
    try:
        bench = (getattr(df, "attrs", {}) or {}).get("rs_benchmark", "SPY")
        tooltip_df = pd.DataFrame("", index=df.index, columns=df.columns)
        for idx, row in df.iterrows():
            comp = _float_or_none(row.get("CompositeScore"))
            tech = _float_or_none(row.get("TechRating"))
            rs = _float_or_none(row.get("RS_pct"))
            fscore = _float_or_none(row.get("Piotroski"))
            if comp is not None:
                tooltip_df.at[idx, "CompositeScore"] = (
                    f"Composite = {comp:.1f} (Tech {'' if tech is None else f'{tech:.1f}'}, "
                    f"RS {'' if rs is None else f'{rs:.0f}%'} vs {bench}, "
                    f"F {'' if fscore is None else f'{fscore:.1f}'}/9)"
                )
            if tech is not None:
                tooltip_df.at[idx, "TechRating"] = "TechRating: multi-factor technical strength (higher is better)."
            if rs is not None:
                tooltip_df.at[idx, "RS_pct"] = f"Relative strength vs {bench}: {rs:.0f}th percentile."
        styler = styler.set_tooltips(tooltip_df)
    except Exception:
        pass

    header_bg = "#0f172a" if not compact else "#111827"
    font_size = "0.85rem" if not compact else "0.80rem"
    padding = "6px 8px" if not compact else "4px 6px"

    styler = styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("background-color", header_bg),
                    ("color", "#e5e7eb"),
                ],
            },
        ]
    )

    # Keep index simple
    styler = styler.set_properties(**{"font-size": font_size, "padding": padding})

    return styler


def render_results_heatmap(df: pd.DataFrame, color_col: str = "TechRating") -> None:
    """Render a simple sector/symbol heatmap for quick visual scan."""
    if not HAVE_ALTAIR or df is None or df.empty:
        st.write("Heatmap unavailable (missing Altair or data).")
        return
    data = df.copy()
    data["Sector"] = data["Sector"].fillna("Unknown").astype(str)
    data = data.sort_values(color_col, ascending=False).head(40)  # keep it light
    data["Symbol"] = data["Symbol"].astype(str)
    chart = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X("Symbol:N", sort=None, axis=alt.Axis(labelAngle=0, labelLimit=80)),
            y=alt.Y("Sector:N", sort=None),
            color=alt.Color(
                f"{color_col}:Q",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=color_col),
            ),
            tooltip=["Symbol", "Sector", f"{color_col}", "TechRating", "CompositeScore"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def render_multi_chart_grid(symbols: list[str], days: int = 90, cols: int = 3) -> None:
    """Show small multiple line charts for a handful of symbols."""
    if not HAVE_ALTAIR or ui_price_history is None:
        st.write("Chart grid unavailable (missing Altair or price layer).")
        return
    cards = []
    for sym in symbols:
        try:
            hist = ui_price_history(sym, days=days, use_intraday=False)
        except Exception:
            hist = None
        if hist is None or hist.empty:
            continue
        df = hist.reset_index()
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        card = (
            alt.Chart(df)
            .mark_line(color=BRAND_PRIMARY, strokeWidth=2)
            .encode(
                x=alt.X("Date:T", axis=None),
                y=alt.Y("Close:Q", axis=None),
                tooltip=["Date:T", alt.Tooltip("Close:Q", format=".2f")],
            )
            .properties(title=sym, height=120)
        )
        cards.append(card)
    if not cards:
        st.write("No charts to show.")
        return
    # Arrange into a grid
    rows = []
    for i in range(0, len(cards), cols):
        row = alt.hconcat(*cards[i : i + cols])
        rows.append(row)
    grid = alt.vconcat(*rows)
    st.altair_chart(grid, use_container_width=True)


def run_simple_backtest(
    df: pd.DataFrame,
    hold_days: int = 20,
    slippage_pct: float = 0.0,
    commission_r: float = 0.0,
    risk_per_trade: float | None = None,
) -> dict:
    """
    Lightweight backtest:
    - Enter at EntryPrice on next bar open (approx: use EntryPrice itself)
    - Exit if TargetPrice or StopPrice hit (use end-of-day check), else exit after `hold_days` bars
    - Computes win rate, avg R multiple, and a simple equity curve (assuming 1R risk per trade)
    - Slippage/commission are applied against R (risk) units to keep it simple
    """
    results = []
    equity = [1.0]  # start at 1R
    pnl_dollars: list[float] = []
    for _, row in df.iterrows():
        sym = row.get("Symbol")
        entry = _float_or_none(row.get("EntryPrice"))
        stop = _float_or_none(row.get("StopPrice"))
        target = _float_or_none(row.get("TargetPrice"))
        if entry is None or stop is None or target is None:
            continue
        try:
            hist = ui_price_history(sym, days=hold_days + 5, use_intraday=False)
        except Exception:
            hist = None
        if hist is None or hist.empty or "Close" not in hist.columns:
            continue
        closes = hist["Close"].reset_index(drop=True)
        hit = None
        pnl_r = -1.0  # assume loss if no target
        for i in range(min(hold_days, len(closes))):
            px = closes.iloc[i]
            if target is not None and px >= target:
                hit = "target"
                pnl_r = (target - entry) / (entry - stop) if (entry - stop) != 0 else 1.0
                break
            if stop is not None and px <= stop:
                hit = "stop"
                pnl_r = -1.0
                break
        if hit is None:
            # exit at last observed price
            px = closes.iloc[min(hold_days - 1, len(closes) - 1)]
            pnl_r = (px - entry) / (entry - stop) if (entry - stop) != 0 else 0.0
        # apply simple slippage/commission in R units
        pnl_net = pnl_r - (abs(pnl_r) * slippage_pct) - commission_r
        results.append(pnl_net)
        equity.append(equity[-1] + pnl_net)
        if risk_per_trade is not None:
            pnl_dollars.append(pnl_net * risk_per_trade)
    if not results:
        return {"trades": 0}
    wins = [r for r in results if r > 0]
    out = {
        "trades": len(results),
        "win_rate": len(wins) / len(results) * 100.0,
        "avg_r": sum(results) / len(results),
        "equity": equity,
    }
    if pnl_dollars:
        out["pnl_sum"] = sum(pnl_dollars)
        out["pnl_avg"] = sum(pnl_dollars) / len(pnl_dollars)
    return out

def build_short_summary(row: pd.Series) -> str:
    """
    Create a compact 1-sentence summary for the Symbol Detail header.
    Pulls the most important scores to give a quick human-readable snapshot.
    """

    tr = row.get("TechRating", None)
    signal = row.get("Signal", "Neutral")
    trend = row.get("TrendScore", None)
    momentum = row.get("MomentumScore", None)
    vol = row.get("VolatilityScore", None)

    parts = []

    # Signal
    if isinstance(signal, str):
        parts.append(f"{signal}")

    # TechRating
    if tr is not None and not pd.isna(tr):
        parts.append(f"TechRating {tr:.1f}")

    # Trend
    if trend is not None and not pd.isna(trend):
        if trend >= 3:
            parts.append("strong trend")
        elif trend >= 1:
            parts.append("mild uptrend")
        elif trend <= -2:
            parts.append("weak/negative trend")

    # Momentum
    if momentum is not None and not pd.isna(momentum):
        if momentum >= 2:
            parts.append("momentum strong")
        elif momentum <= -1:
            parts.append("momentum weakening")

    # Volatility
    if vol is not None and not pd.isna(vol):
        if vol >= 2:
            parts.append("high volatility")
        elif vol <= -1:
            parts.append("low volatility")

    # If no components detected
    if not parts:
        return "No major technical attributes detected."

    return ", ".join(parts).capitalize() + "."


def _get_live_price(symbol: str, df_hint: pd.DataFrame | None = None) -> float | None:
    """Best-effort current price: realtime tick if available, else 'Last' from results."""
    if get_realtime_last:
        try:
            px = get_realtime_last(symbol)
            if px is not None:
                return float(px)
        except Exception:
            pass
    if df_hint is not None and not df_hint.empty and "Symbol" in df_hint.columns:
        rows = df_hint[df_hint["Symbol"].astype(str) == str(symbol)]
        if not rows.empty:
            candidate = rows.iloc[0].get("Last") or rows.iloc[0].get("Close")
            try:
                return float(candidate)
            except Exception:
                return None
    return None

def _current_trade_style() -> str:
    cfg = st.session_state.get("scan_config")
    if cfg is not None and hasattr(cfg, "trade_style"):
        val = getattr(cfg, "trade_style")
        if val:
            return str(val)
    return "Short-term swing"

def describe_signal(signal: str | None) -> str:
    if not signal:
        return "No technical signal assigned."
    signal = signal.strip()

    if signal == "Strong Long":
        return "High-quality long setup based on trend, momentum, and volatility."
    if signal == "Long":
        return "Reasonable long setup; quality and conviction are moderate."
    if signal == "Avoid":
        return "No clear edge based on the current technical configuration."
    if signal == "Strong Short":
        return "High-conviction short setup; trend and momentum favor the downside."
    if signal == "Short":
        return "Short setup exists but conviction is moderate."
    return f"Technical signal: {signal}"

def add_setup_tags_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'SetupTags' text column to a scored DataFrame, using build_setup_tags().
    The tags are joined with ' · ' for compact display in the main table.
    """
    if df is None or df.empty:
        return df

    tags_list: list[str] = []
    for _, r in df.iterrows():
        tags = build_setup_tags(r)
        # Join tags with a middle dot separator, e.g. "Breakout · Momentum · Low risk"
        tags_list.append(" · ".join(tags))

    df = df.copy()
    df["SetupTags"] = tags_list
    return df

def build_narrative_from_row(row: pd.Series) -> str:
    parts: list[str] = []

    symbol = row.get("Symbol", "Unknown")

    tech = float(row.get("TechRating", np.nan))
    risk = float(row.get("RiskScore", np.nan)) if "RiskScore" in row else np.nan
    signal = str(row.get("Signal", ""))

    trend = float(row.get("TrendScore", 0.0))
    momentum = float(row.get("MomentumScore", 0.0))
    vol = float(row.get("VolatilityScore", 0.0))
    breakout = float(row.get("BreakoutScore", 0.0))
    explosiveness = float(row.get("ExplosivenessScore", 0.0))
    tq = float(row.get("TrendQualityScore", 0.0))

    parts.append(f"{symbol} currently has a TechRating of {tech:.1f}.")

    if not pd.isna(risk):
        parts.append(
            f"RiskScore is {risk:.2f} (0 = higher volatility / wider stops, 1 = lower volatility / tighter risk)."
        )

    if signal:
        parts.append(describe_signal(signal))

    if trend >= 2:
        parts.append("Trend strength is firmly bullish with a clear directional bias.")
    elif trend >= 1:
        parts.append("Trend is constructive and tilts bullish.")
    elif trend <= -2:
        parts.append("Trend is firmly bearish with sustained downside pressure.")
    elif trend <= -1:
        parts.append("Trend tilts bearish with downside bias.")

    if momentum >= 2:
        parts.append("Momentum is strong, showing aggressive follow-through in the current direction.")
    elif momentum >= 1:
        parts.append("Momentum is supportive but not extreme.")
    elif momentum <= -1:
        parts.append("Momentum is fading or favoring the opposite direction.")

    if tq >= 2:
        parts.append("Trend quality is high, with clean swings and relatively low noise.")
    elif tq >= 1:
        parts.append("Trend quality is acceptable with mostly orderly pullbacks.")
    elif tq <= -1:
        parts.append("Trend quality is poor, with choppy or whipsaw-like price action.")

    if breakout >= 2 or explosiveness >= 2:
        parts.append("Recent price action shows breakout-style behavior with strong extension.")
    elif breakout >= 1 or explosiveness >= 1:
        parts.append("Breakout characteristics are present but not extreme.")
    elif breakout <= -1 or explosiveness <= -1:
        parts.append("Price has struggled to follow through on breakouts.")

    if vol >= 2:
        parts.append("Volatility is elevated, which can increase both opportunity and risk.")
    elif vol >= 1:
        parts.append("Volatility is somewhat elevated but still within a tradable range.")
    elif vol <= -1:
        parts.append("Volatility is relatively muted.")

    if "TradeType" in row and isinstance(row["TradeType"], str) and row["TradeType"]:
        parts.append(f"Trade type is classified as: {row['TradeType']} setup.")

    unique_parts: list[str] = []
    for p in parts:
        if p not in unique_parts:
            unique_parts.append(p)

    return " ".join(unique_parts)


def build_conditions(row: pd.Series) -> dict[str, str]:
    def to_float(name: str, default: float = 0.0) -> float:
        val = row.get(name, default)
        try:
            return float(val)
        except Exception:
            return default

    conditions: dict[str, str] = {}

    trend = to_float("TrendScore")
    momentum = to_float("MomentumScore")
    explosiveness = to_float("ExplosivenessScore")
    breakout = to_float("BreakoutScore")
    volume = to_float("VolumeScore")
    vol = to_float("VolatilityScore")
    tq = to_float("TrendQualityScore")

    if trend >= 2:
        conditions["Trend"] = "Trend strength is firmly bullish."
    elif trend >= 1:
        conditions["Trend"] = "Trend is constructive and tilts bullish."
    elif trend <= -2:
        conditions["Trend"] = "Trend strength is firmly bearish."
    elif trend <= -1:
        conditions["Trend"] = "Trend tilts bearish."
    else:
        conditions["Trend"] = "Trend is neutral or range-bound."

    if tq >= 2:
        conditions["Trend Quality"] = "Trend quality is high and swings are relatively clean."
    elif tq >= 1:
        conditions["Trend Quality"] = "Trend quality is acceptable with moderate noise."
    elif tq <= -1:
        conditions["Trend Quality"] = "Price action is choppy with frequent whipsaws."
    else:
        conditions["Trend Quality"] = "Trend quality is neutral."

    if momentum >= 2:
        conditions["Momentum"] = "Momentum is strong with aggressive follow-through."
    elif momentum >= 1:
        conditions["Momentum"] = "Momentum is supportive but not extreme."
    elif momentum <= -1:
        conditions["Momentum"] = "Momentum tilts bearish / fading."
    else:
        conditions["Momentum"] = "Momentum is mixed or flat."

    if explosiveness >= 2:
        conditions["Breakout / Explosiveness"] = "Recent moves have been explosive with strong extension."
    elif explosiveness >= 1:
        conditions["Breakout / Explosiveness"] = "Moves show reasonable follow-through after breakouts."
    elif explosiveness <= -1:
        conditions["Breakout / Explosiveness"] = "Breakouts tend to fade or lack follow-through."
    else:
        conditions["Breakout / Explosiveness"] = "Explosiveness is neutral."

    if breakout >= 2:
        conditions["Breakout setup"] = "Breakout conditions are strongly present."
    elif breakout >= 1:
        conditions["Breakout setup"] = "Breakout conditions are moderately present."
    elif breakout <= -1:
        conditions["Breakout setup"] = "Breakout conditions are weak or failing."
    else:
        conditions["Breakout setup"] = "Breakout characteristics are neutral."

    if volume >= 2:
        conditions["Volume"] = "Volume is well above normal, supporting conviction."
    elif volume >= 1:
        conditions["Volume"] = "Volume is slightly above normal."
    elif volume <= -1:
        conditions["Volume"] = "Volume is below normal and may limit conviction."
    else:
        conditions["Volume"] = "Volume is roughly in line with normal."

    if vol >= 2:
        conditions["Volatility"] = "Volatility is elevated; expect larger swings."
    elif vol >= 1:
        conditions["Volatility"] = "Volatility is somewhat elevated."
    elif vol <= -1:
        conditions["Volatility"] = "Volatility is relatively low."
    else:
        conditions["Volatility"] = "Volatility is neutral."

    return conditions


def _perf_over(df: pd.DataFrame, periods: int = 10) -> float | None:
    """
    Simple return over last `periods` rows.
    """
    if df is None or df.empty or len(df) < periods:
        return None
    start = df["Close"].iloc[-periods]
    end = df["Close"].iloc[-1]
    if start in (None, 0) or pd.isna(start) or pd.isna(end):
        return None
    return (end - start) / start * 100.0

def build_setup_tags(row: pd.Series) -> list[str]:
    """
    Infer a small set of 'setup tags' from the scoring model for quick visual labeling.
    Examples: Breakout, Smooth trend, Momentum, Low risk, High conviction, Choppy, etc.
    """
    def get(name: str, default: float = 0.0) -> float:
        val = row.get(name, default)
        try:
            return float(val)
        except Exception:
            return float(default)

    trend = get("TrendScore")
    tq = get("TrendQualityScore")
    mom = get("MomentumScore")
    expl = get("ExplosivenessScore")
    brk = get("BreakoutScore")
    vol = get("VolatilityScore")
    risk = get("RiskScore")
    tech = get("TechRating")
    signal = str(row.get("Signal", "") or "").strip()

    tags: list[str] = []

    # Directional bias / conviction
    if signal in ("Strong Long", "Long"):
        tags.append("Bullish bias")
    elif signal:
        tags.append(f"{signal} bias")

    if tech >= 22:
        tags.append("High conviction")
    elif tech >= 18:
        tags.append("Quality setup")

    # Trend / structure
    if trend >= 2.0 and tq >= 2.0:
        tags.append("Smooth trend")
    elif trend >= 1.0 and tq >= 1.0:
        tags.append("Uptrend")
    elif tq < 0.0:
        tags.append("Choppy")

    # Breakout / momentum profile
    if brk >= 2.0 and expl >= 1.0:
        tags.append("Breakout")
    elif brk >= 1.0:
        tags.append("Breakout watch")

    if mom >= 2.0:
        tags.append("Momentum")
    elif mom <= -1.0:
        tags.append("Momentum fading")

    # Volatility / risk character
    if risk >= 0.80:
        tags.append("Low risk")
    elif risk >= 0.65:
        tags.append("Moderate risk")
    elif risk <= 0.45:
        tags.append("High risk")

    if vol >= 2.0:
        tags.append("High volatility")
    elif vol <= -1.0:
        tags.append("Quiet volatility")

    # De-duplicate while preserving order
    seen = set()
    deduped: list[str] = []
    for t in tags:
        if t not in seen:
            deduped.append(t)
            seen.add(t)

    # Don’t overwhelm: keep the first few most relevant tags
    return deduped[:4]


def _setup_tag_style(tag: str) -> str:
    """
    Return an inline CSS style for a given tag.
    We use simple heuristics based on the tag text.
    """
    base = (
        "display:inline-block; padding:2px 8px; margin:0 4px 4px 0; "
        "border-radius:9999px; font-size:0.75rem; font-weight:500;"
    )

    text = tag.lower()

    if any(k in text for k in ["breakout", "momentum"]):
        # Bright blue
        return base + "background-color:rgba(25,118,210,0.85); color:white;"
    if any(k in text for k in ["trend", "bullish", "quality", "high conviction", "low risk"]):
        # Greenish
        return base + "background-color:rgba(56,142,60,0.85); color:white;"
    if any(k in text for k in ["high risk", "choppy", "fading", "high volatility"]):
        # Orange/red warning
        return base + "background-color:rgba(245,124,0,0.9); color:white;"
    if any(k in text for k in ["quiet", "moderate", "neutral"]):
        # Neutral gray
        return base + "background-color:rgba(120,144,156,0.8); color:white;"

    # Fallback pill
    return base + "background-color:rgba(96,125,139,0.8); color:white;"


def render_setup_tags(tags: list[str]) -> str:
    """
    Build an HTML string to render setup tags as pill-style badges.
    """
    spans = [
        f'<span style="{_setup_tag_style(tag)}">{tag}</span>'
        for tag in tags
    ]
    return "<div style='margin-top:4px; margin-bottom:4px;'>" + "".join(spans) + "</div>"

def apply_smart_filters(
    df: pd.DataFrame,
    *,
    only_breakouts: bool = False,
    only_smooth_trends: bool = False,
    exclude_choppy: bool = False,
    high_conviction_only: bool = False,
    low_risk_only: bool = False,
    top_by_sector: bool = False,
) -> pd.DataFrame:
    """
    Apply high-level Smart Filters on top of an already-scored DataFrame.
    All filters are optional and can be combined.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Only breakout-style setups
    if only_breakouts:
        if "BreakoutScore" in out.columns:
            mask = out["BreakoutScore"] >= 2
        elif "TradeType" in out.columns:
            mask = out["TradeType"].astype(str).str.contains("Breakout", case=False)
        else:
            mask = pd.Series(False, index=out.index)
        out = out[mask]

    # Smooth trend: strong trend, decent quality, not crazy volatility
    if only_smooth_trends:
        trend = out.get("TrendScore", 0)
        trendq = out.get("TrendQualityScore", 0)
        vol = out.get("VolatilityScore", 0)
        mask = (trend >= 2) & (trendq >= 1) & (vol >= 0)
        out = out[mask]

    # Exclude choppy / noisy names
    if exclude_choppy:
        trendq = out.get("TrendQualityScore", 0)
        vol = out.get("VolatilityScore", 0)
        mask = (trendq > -1) & (vol > -1)
        out = out[mask]

    # High-conviction only
    if high_conviction_only:
        if "Signal" in out.columns:
            mask = out["Signal"].isin(["Strong Long"])
        elif "TechRating" in out.columns:
            mask = out["TechRating"] >= 20.0
        else:
            mask = pd.Series(False, index=out.index)
        out = out[mask]

    # Low-risk only (higher RiskScore = lower risk)
    if low_risk_only and "RiskScore" in out.columns:
        mask = out["RiskScore"] >= 0.65
        out = out[mask]

    # Top N per sector by TechRating (here: top 3)
    if top_by_sector and "Sector" in out.columns and "TechRating" in out.columns:
        out = out.sort_values("TechRating", ascending=False)
        out = (
            out.groupby("Sector", group_keys=False)
            .head(3)
            .reset_index(drop=True)
        )

    return out


@st.cache_data(show_spinner=False)
def fetch_symbol_news(symbol: str, limit: int = NEWS_LIMIT) -> list[dict[str, str]]:
    """
    Lightweight news fetcher using Polygon / Massive reference news.
    Returns a list of dicts with title, source, time, and url.
    """
    if not symbol or not POLYGON_API_KEY:
        return []

    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": symbol.upper(),
        "limit": int(limit),
        "apiKey": POLYGON_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception:
        return []

    results = data.get("results") or []
    news: list[dict[str, str]] = []
    for item in results:
        published_raw = item.get("published_utc") or ""
        try:
            published_dt = dt.datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            published_txt = published_dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            published_txt = published_raw or ""

        publisher = item.get("publisher")
        if isinstance(publisher, dict):
            pub_name = publisher.get("name", "") or publisher.get("title", "") or ""
        else:
            pub_name = str(publisher or "")

        news.append(
            {
                "title": item.get("title", "").strip(),
                "source": (item.get("source") or "").strip() or pub_name.strip() or "News",
                "published": published_txt,
                "url": item.get("article_url", "").strip(),
            }
        )
    return news


@st.cache_data(show_spinner=False)
def fetch_bulk_news(symbols: list[str], per_symbol: int = 3, max_items: int = 25) -> list[dict[str, str]]:
    """
    Pull news for multiple symbols and merge into a single list sorted by time.
    """
    seen_urls = set()
    items: list[dict[str, str]] = []

    for sym in symbols:
        sym = sym.upper().strip()
        if not sym:
            continue
        sym_news = fetch_symbol_news(sym, limit=per_symbol)
        for item in sym_news:
            url = item.get("url", "")
            if url and url in seen_urls:
                continue
            seen_urls.add(url)
            enriched = dict(item)
            enriched["symbol"] = sym
            items.append(enriched)

    # Sort by published desc when possible
    def _parse_dt(val: str) -> float:
        try:
            return dt.datetime.strptime(val, "%Y-%m-%d %H:%M UTC").timestamp()
        except Exception:
            return 0.0

    items.sort(key=lambda x: _parse_dt(x.get("published", "")), reverse=True)
    return items[:max_items]


@st.cache_data(show_spinner=False, ttl=OPTION_CACHE_TTL)
def fetch_option_recos(
    symbol: str,
    direction: str,
    trade_style: str,
    underlying: float,
    tech_rating: float | None,
    risk_score: float | None,
    price_target: float | None,
    signal: str | None,
) -> dict[str, Any]:
    """
    Cached wrapper that fetches the option chain snapshot and returns scored picks.
    """
    direction = (direction or "call").lower()
    if OptionChainService is None or select_option_candidates is None:
        return {"picks": [], "meta": {"cached": False}, "chain_count": 0}

    svc = OptionChainService(api_key=POLYGON_API_KEY)
    chain, meta = svc.fetch_chain_snapshot(symbol=symbol, contract_type=direction)
    picks = select_option_candidates(
        chain=chain,
        direction=direction,
        trade_style=trade_style,
        underlying_price=underlying,
        tech_rating=tech_rating,
        risk_score=risk_score,
        price_target=price_target,
        signal=signal,
    )

    return {"picks": picks, "meta": meta, "chain_count": len(chain)}


def build_ticker_html(df: pd.DataFrame, max_items: int = 8) -> str:
    items = []
    for _, r in df.head(max_items).iterrows():
        sym = str(r.get("Symbol", ""))
        last = _float_or_none(r.get("Last"))
        if last is None or pd.isna(last):
            last = _float_or_none(r.get("Close"))
        if last is None or pd.isna(last):
            last = _float_or_none(r.get("EntryPrice"))

        entry = _float_or_none(r.get("EntryPrice"))
        if entry is not None and pd.isna(entry):
            entry = None

        change = None
        if last is not None and entry not in (None, 0):
            if not pd.isna(last) and not pd.isna(entry):
                change = (last - entry) / entry * 100.0

        signal = str(r.get("Signal", "") or "")
        color = "#22c55e" if "Long" in signal else "#fb7185" if "Short" in signal else BRAND_TEXT
        change_txt = f"{change:+.2f}%" if change is not None else ""
        price_txt = f"{last:.2f}" if last is not None and not pd.isna(last) else "--"
        items.append(f"<span class='ticker__item' style='color:{color};'>{sym} {price_txt} {change_txt}</span>")
    # duplicate items for seamless scroll only when multiple symbols
    track_items = items + items if len(items) > 1 else items
    track = " ".join(track_items)
    return f"<div class='ticker'><div class='ticker__track'>{track}</div></div>"


def run_ma_backtest(
    symbol: str,
    days: int,
    ma_fast: int,
    ma_slow: int,
    initial_capital: float = 10_000.0,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """
    Simple moving-average crossover backtest (long-only).
    - Enter when fast MA crosses above slow MA.
    - Exit to cash when fast crosses below slow.
    Returns (equity_df, metrics).
    """
    if ui_price_history is None:
        return None, {"error": "Price layer unavailable"}

    try:
        hist = ui_price_history(symbol=symbol, days=days, use_intraday=False)
    except Exception as exc:
        return None, {"error": f"Price fetch failed: {exc}"}

    if hist is None or hist.empty or "Close" not in hist.columns:
        return None, {"error": "No price data"}

    df = hist.copy()
    df = df.reset_index()
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    df = df.sort_values("Date")
    df["Return"] = df["Close"].pct_change().fillna(0.0)
    df["MA_fast"] = df["Close"].rolling(ma_fast).mean()
    df["MA_slow"] = df["Close"].rolling(ma_slow).mean()

    df["Signal"] = (df["MA_fast"] > df["MA_slow"]).astype(int)
    df["Position"] = df["Signal"].shift(1).fillna(0)  # trade on next bar

    df["StratRet"] = df["Return"] * df["Position"]
    df["Equity"] = initial_capital * (1 + df["StratRet"]).cumprod()
    df["BuyHold"] = initial_capital * (df["Close"] / df["Close"].iloc[0])

    # Metrics
    total_return = (df["Equity"].iloc[-1] / initial_capital) - 1
    buyhold_return = (df["BuyHold"].iloc[-1] / initial_capital) - 1

    years = max(len(df) / 252.0, 1e-6)
    cagr = (df["Equity"].iloc[-1] / initial_capital) ** (1 / years) - 1

    rolling_max = df["Equity"].cummax()
    drawdowns = (df["Equity"] / rolling_max) - 1
    max_dd = drawdowns.min()

    sharpe = None
    if df["StratRet"].std() > 0:
        sharpe = df["StratRet"].mean() / df["StratRet"].std() * np.sqrt(252)

    # Trade stats (entries/exits)
    trades: list[float] = []
    positions = df["Position"].values
    closes = df["Close"].values
    entry_price = None
    for i in range(1, len(df)):
        if positions[i - 1] == 0 and positions[i] == 1:
            entry_price = closes[i]
        elif positions[i - 1] == 1 and positions[i] == 0 and entry_price:
            trades.append((closes[i] / entry_price) - 1)
            entry_price = None
    # If still in trade at end
    if entry_price:
        trades.append((closes[-1] / entry_price) - 1)

    win_rate = (np.mean([t > 0 for t in trades]) * 100) if trades else None
    avg_trade = (np.mean(trades) * 100) if trades else None

    metrics = {
        "total_return": total_return,
        "buyhold_return": buyhold_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_trade": avg_trade,
    }

    return df[["Date", "Equity", "BuyHold", "MA_fast", "MA_slow"]], metrics


def attach_sparklines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a 'Sparkline' column: list of recent Close prices per symbol.
    Uses the price layer, with a hard cap to avoid hammering Polygon.
    """
    if df is None or df.empty:
        return df

    if ui_price_history is None or "Symbol" not in df.columns:
        return df

    df = df.copy()
    # Only fetch for top rows to avoid slow scans
    df_top = df.head(SPARKLINE_MAX_ROWS)
    spark_values: list[list[float] | None] = []

    for i, (_, row) in enumerate(df_top.iterrows()):
        sym = str(row.get("Symbol", "")).upper()

        if i >= SPARKLINE_MAX_ROWS or not sym:
            spark_values.append(None)
            continue

        try:
            hist = ui_price_history(
                symbol=sym,
                days=SPARKLINE_LOOKBACK_DAYS,
                use_intraday=False,
            )
            if hist is not None and not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].tail(SPARKLINE_LOOKBACK_DAYS).tolist()
                spark_values.append(closes)
            else:
                spark_values.append(None)
        except Exception:
            spark_values.append(None)

    # Pad the rest with None (no sparkline for lower-ranked rows)
    while len(spark_values) < len(df):
        spark_values.append(None)

    df["Sparkline"] = spark_values
    return df


def attach_fundamentals_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich results with cached fundamentals (PE, PEG, Piotroski, Altman Z, EPS growth).
    Limits to a subset to keep scans responsive.
    """
    if df is None or df.empty or get_fundamentals is None or "Symbol" not in df.columns:
        return df

    df = df.copy()
    pe_list, peg_list, fscore_list, z_list, epsg_list = [], [], [], [], []
    max_rows = min(len(df), 200)
    for idx, row in enumerate(df.itertuples(index=False)):
        sym = str(getattr(row, "Symbol", "")).upper()
        if idx >= max_rows or not sym:
            pe_list.append(None); peg_list.append(None); fscore_list.append(None); z_list.append(None); epsg_list.append(None)
            continue
        snap = get_fundamentals(sym) if sym else None
        if snap is None or not getattr(snap, "raw", None):
            # Try yfinance as a quick fallback
            if yf is not None and sym:
                try:
                    info = yf.Ticker(sym).info
                    pe_list.append(_float_or_none(info.get("trailingPE")))
                    peg_list.append(_float_or_none(info.get("pegRatio")))
                    fscore_list.append(None)
                    z_list.append(None)
                    epsg_list.append(None)
                    continue
                except Exception:
                    pass
            pe_list.append(None); peg_list.append(None); fscore_list.append(None); z_list.append(None); epsg_list.append(None)
            continue
        pe_list.append(_float_or_none(snap.get("pe") or snap.get("PE")))
        peg_list.append(_float_or_none(snap.get("peg")))
        fscore_list.append(_float_or_none(snap.get("piotroski") or snap.get("piotroski_fscore")))
        z_list.append(_float_or_none(snap.get("altman_z") or snap.get("altman_zscore")))
        epsg_list.append(_float_or_none(snap.get("eps_growth") or snap.get("eps_5y_cagr")))
    df["PE"] = pe_list
    df["PEG"] = peg_list
    df["Piotroski"] = fscore_list
    df["AltmanZ"] = z_list
    df["EPS_Growth"] = epsg_list
    return df


def attach_relative_strength(df: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """
    Append RS percentile vs benchmark (last ~40 bars vs history).
    """
    if df is None or df.empty or "Symbol" not in df.columns or rs_change_percentile is None:
        return df
    df = df.copy()
    rs_list: list[float | None] = []
    for _, row in df.iterrows():
        sym = str(row.get("Symbol", "")).upper()
        try:
            rs_val = rs_change_percentile(sym, benchmark=benchmark, days=200, lookback=40)
        except Exception:
            rs_val = None
        rs_list.append(rs_val)
    df["RS_pct"] = rs_list
    df.attrs["rs_benchmark"] = benchmark
    return df


def apply_fundamental_filters(
    df: pd.DataFrame,
    min_piotroski: float | None,
    max_pe: float | None,
    max_peg: float | None,
    min_eps_g: float | None,
    min_rs_pct: float | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if min_piotroski is not None and "Piotroski" in out.columns:
        out = out[(out["Piotroski"].fillna(-999) >= min_piotroski)]
    if max_pe is not None and "PE" in out.columns:
        out = out[(out["PE"].fillna(9999) <= max_pe)]
    if max_peg is not None and "PEG" in out.columns:
        out = out[(out["PEG"].fillna(9999) <= max_peg)]
    if min_eps_g is not None and "EPS_Growth" in out.columns:
        out = out[(out["EPS_Growth"].fillna(-999) >= min_eps_g)]
    if min_rs_pct is not None and "RS_pct" in out.columns:
        out = out[(out["RS_pct"].fillna(-999) >= min_rs_pct)]
    return out


def apply_custom_formula(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    """
    Apply a user-entered boolean formula with a limited, safer parser.
    Supported:
      - Columns: alphanumerics + underscore (e.g., PE, TechRating, RiskScore)
      - Operators: < <= > >= == != & |
      - Parentheses
    Example: "(PE < 40) & (TechRating > 15) & (RiskScore > 0.7)"
    """
    if df is None or df.empty:
        return df
    expr = (formula or "").strip()
    if not expr:
        return df

    import re

    token_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[<>=!]=|[<>&|()]|==|!=")
    tokens = token_pattern.findall(expr)
    if not tokens:
        return df

    safe_tokens = []
    for tok in tokens:
        if tok in {"&", "|", "(", ")", "<", ">", "<=", ">=", "==", "!="}:
            safe_tokens.append(tok)
        else:
            # treat as column name
            col = tok
            if col not in df.columns:
                safe_tokens.append("False")
            else:
                safe_tokens.append(f"@df['{col}']")

    safe_expr = " ".join(safe_tokens)
    try:
        mask = eval(safe_expr, {"df": df})
        if isinstance(mask, pd.Series) and mask.dtype == bool:
            return df[mask]
    except Exception:
        return df
    return df


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def attach_technical_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach basic technical flags:
    - above_sma50: Close > 50-day SMA
    - above_sma200: Close > 200-day SMA
    - weekly_uptrend: last weekly close > previous weekly close
    - golden_cross: SMA50 > SMA200
    """
    if df is None or df.empty or ui_price_history is None or "Symbol" not in df.columns:
        return df
    df = df.copy()
    above50, above200, weekly_up, golden = [], [], [], []
    for _, row in df.iterrows():
        sym = str(row.get("Symbol", "")).upper()
        try:
            hist = ui_price_history(symbol=sym, days=260, use_intraday=False)
            if hist is None or hist.empty or "Close" not in hist.columns:
                above50.append(None); above200.append(None); weekly_up.append(None); golden.append(None); continue
            close = hist["Close"]
            sma50 = _sma(close, 50)
            sma200 = _sma(close, 200)
            latest = close.iloc[-1]
            above50.append(bool(latest > sma50.iloc[-1])) if not pd.isna(sma50.iloc[-1]) else above50.append(None)
            above200.append(bool(latest > sma200.iloc[-1])) if not pd.isna(sma200.iloc[-1]) else above200.append(None)
            if not pd.isna(sma50.iloc[-1]) and not pd.isna(sma200.iloc[-1]):
                golden.append(bool(sma50.iloc[-1] > sma200.iloc[-1]))
            else:
                golden.append(None)

            # weekly uptrend from multi-timeframe
            if get_multi_timeframes is not None:
                mtf = get_multi_timeframes(sym, days=400)
                wk = mtf.get("weekly")
                if wk is not None and len(wk) >= 2 and "Close" in wk.columns:
                    weekly_up.append(bool(wk["Close"].iloc[-1] > wk["Close"].iloc[-2]))
                else:
                    weekly_up.append(None)
            else:
                weekly_up.append(None)
        except Exception:
            above50.append(None); above200.append(None); weekly_up.append(None); golden.append(None)
    df["Above_SMA50"] = above50
    df["Above_SMA200"] = above200
    df["Weekly_Uptrend"] = weekly_up
    df["Golden_Cross"] = golden
    return df


def attach_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach ATR% (14-day ATR / close * 100) and simple volatility flags.
    """
    if df is None or df.empty or ui_price_history is None or "Symbol" not in df.columns:
        return df
    df = df.copy()
    atr_list: list[float | None] = []
    for _, row in df.iterrows():
        sym = str(row.get("Symbol", "")).upper()
        try:
            hist = ui_price_history(symbol=sym, days=60, use_intraday=False)
            if hist is None or hist.empty:
                atr_list.append(None)
                continue
            h, l, c = hist["High"], hist["Low"], hist["Close"]
            tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=14, min_periods=14).mean()
            atr_pct = (atr / c) * 100.0
            val = atr_pct.iloc[-1] if len(atr_pct.dropna()) else None
            atr_list.append(float(val) if val is not None and not pd.isna(val) else None)
        except Exception:
            atr_list.append(None)
    df["ATR_pct"] = atr_list
    return df


def compute_composite_score(row: pd.Series) -> float | None:
    """
    Lightweight composite: TechRating (scaled 0-40) + RS (0-30) + Piotroski (0-30).
    Missing values are treated as neutral.
    """
    tech = _float_or_none(row.get("TechRating"))
    rs = _float_or_none(row.get("RS_pct"))
    fscore = _float_or_none(row.get("Piotroski"))

    parts = []
    if tech is not None:
        parts.append(min(max(tech, 0.0), 40.0))
    if rs is not None:
        parts.append(min(max(rs / 100.0 * 30.0, 0.0), 30.0))
    if fscore is not None:
        parts.append(min(max(fscore / 9.0 * 30.0, 0.0), 30.0))

    if not parts:
        return None
    return sum(parts)


def attach_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df["CompositeScore"] = df.apply(compute_composite_score, axis=1)
    return df


def compute_market_regime() -> str:
    """
    Market regime heuristic:
    - Tries VIX (if yfinance is available)
        Low: VIX < 15, Medium: 15-25, High: >25
    - Falls back to SPY ATR% (14d):
        Low: <1.5, Medium: 1.5-2.5, High: >2.5
    """
    if yf is not None:
        try:
            vix = yf.Ticker("^VIX").history(period="1mo")
            if not vix.empty:
                val = float(vix["Close"].iloc[-1])
                if val < 15:
                    return "low vol"
                elif val < 25:
                    return "medium vol"
                else:
                    return "high vol"
        except Exception:
            pass
    try:
        spy = ui_price_history("SPY", days=60, use_intraday=False)
        if spy is None or spy.empty:
            return "unknown"
        h, l, c = spy["High"], spy["Low"], spy["Close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean()
        atr_pct = (atr / c) * 100.0
        val = atr_pct.iloc[-1]
        if val < 1.5:
            return "low vol"
        elif val < 2.5:
            return "medium vol"
        else:
            return "high vol"
    except Exception:
        return "unknown"


def apply_technical_filters(
    df: pd.DataFrame,
    require_weekly_up: bool = False,
    require_above_50: bool = False,
    require_above_200: bool = False,
    require_golden_cross: bool = False,
    min_atr_pct: float | None = None,
    max_atr_pct: float | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if require_weekly_up and "Weekly_Uptrend" in out.columns:
        out = out[out["Weekly_Uptrend"] == True]
    if require_above_50 and "Above_SMA50" in out.columns:
        out = out[out["Above_SMA50"] == True]
    if require_above_200 and "Above_SMA200" in out.columns:
        out = out[out["Above_SMA200"] == True]
    if require_golden_cross and "Golden_Cross" in out.columns:
        out = out[out["Golden_Cross"] == True]
    if min_atr_pct is not None and "ATR_pct" in out.columns:
        out = out[out["ATR_pct"].fillna(-999) >= min_atr_pct]
    if max_atr_pct is not None and "ATR_pct" in out.columns:
        out = out[out["ATR_pct"].fillna(9999) <= max_atr_pct]
    return out

def render_trade_idea_card(row: pd.Series) -> str:
    """
    Compact HTML card for the "Personal Quant trade ideas" strip.
    Keeps key numbers front-and-center for quick scanning.
    """
    sym = str(row.get("Symbol", "-"))
    signal = str(row.get("Signal", "") or "").strip() or "Idea"

    direction = "Long"
    if "avoid" in signal.lower():
        direction = "Avoid"
    elif "short" in signal:
        direction = "Short"

    signal_display = signal if signal.lower() != direction.lower() else ""

    def _safe_float(val, default: float = float("nan")) -> float:
        try:
            return float(val)
        except Exception:
            return default

    entry = _safe_float(row.get("EntryPrice"))
    stop = _safe_float(row.get("StopPrice"))
    target = _safe_float(row.get("TargetPrice"))
    rr = _safe_float(row.get("RewardRisk"))
    size = _safe_float(row.get("PositionSize"))
    tech = _safe_float(row.get("TechRating"))

    rr_txt = f"{rr:.1f}:1" if not pd.isna(rr) else "-"
    size_txt = f"{int(size):,}" if not pd.isna(size) else "-"
    tech_txt = f"{tech:.1f}" if not pd.isna(tech) else "-"

    def fmt_price(val: float) -> str:
        return f"{val:.2f}" if not pd.isna(val) else "-"

    return f"""
    <div class="idea-card">
      <div class="idea-card__top">
        <div class="idea-chip">{direction}</div>
        <div class="idea-signal">{signal_display}</div>
      </div>
      <div class="idea-symbol">{sym}</div>
      <div class="idea-metrics">
        <div><span>Entry</span><strong>{fmt_price(entry)}</strong></div>
        <div><span>Stop</span><strong>{fmt_price(stop)}</strong></div>
        <div><span>Target</span><strong>{fmt_price(target)}</strong></div>
      </div>
      <div class="idea-footer">
        <div>R:R <strong>{rr_txt}</strong></div>
        <div>Size <strong>{size_txt}</strong></div>
        <div>TechRating <strong>{tech_txt}</strong></div>
      </div>
    </div>
    """

def make_score_bar_chart(scores: dict[str, float]) -> "matplotlib.figure.Figure | None":
    """
    Dashboard-style horizontal bar chart for the score breakdown.

    - Bars sorted by absolute score (most important factors at the top)
    - Fixed x-range (-3 to +3) across symbols
    - Light x-grid and zero line
    - Numeric value label on each bar
    """
    if not HAVE_MPL:
        return None

    import matplotlib.pyplot as plt  # type: ignore[import]

    if not scores:
        return None

    # Sort factors by absolute strength (strongest at the top)
    sorted_items = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    labels = [k for k, _ in sorted_items]
    values = [float(v) for _, v in sorted_items]

    n = len(values)

    # Height scales with number of factors so it doesn't feel cramped
    fig_height = 0.4 * n + 1.5
    fig, ax = plt.subplots(figsize=(5.5, fig_height), dpi=120)

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, values)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    # Consistent visual range
    ax.set_xlim(-3.0, 3.0)
    ax.set_xlabel("Score (clipped ±3)")

    # Zero line + light grid for quick reading
    ax.axvline(0, linewidth=1)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.3)

    # Cleaner frame
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Add numeric labels on each bar
    for bar, val in zip(bars, values):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        # Position label just outside the bar, left/right depending on sign
        if val >= 0:
            ha = "left"
            x_text = x + 0.1
        else:
            ha = "right"
            x_text = x - 0.1
        ax.text(
            x_text,
            y,
            f"{val:.1f}",
            va="center",
            ha=ha,
            fontsize=8,
        )

    fig.tight_layout()
    return fig

def render_option_play(row: pd.Series, trade_style: str | None) -> None:
    """
    Pull + display Polygon option picks for the current symbol.
    """
    symbol = str(row.get("Symbol", "")).upper()
    st.markdown('<div class="technic-card">', unsafe_allow_html=True)
    st.markdown("##### Options play (Polygon)")

    if not symbol:
        st.write("No symbol selected.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if OptionChainService is None or select_option_candidates is None:
        st.info("Options module unavailable in this session (imports failed).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not POLYGON_API_KEY:
        st.info("Set POLYGON_API_KEY to enable options data.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    direction = "call"
    signal_txt = str(row.get("Signal", "") or "").lower()
    if "short" in signal_txt:
        direction = "put"

    trade_style_label = trade_style or "Short-term swing"
    # Prefer live tick if available for accurate moneyness/delta targeting
    underlying_px = _get_live_price(symbol) or (
        _float_or_none(row.get("Last"))
        or _float_or_none(row.get("Close"))
        or _float_or_none(row.get("EntryPrice"))
    )
    if underlying_px is None:
        st.write("No underlying price available to score options.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    tech_rating = _float_or_none(row.get("TechRating"))
    risk_score = _float_or_none(row.get("RiskScore"))
    price_target = _float_or_none(row.get("TargetPrice"))

    with st.spinner("Fetching chain snapshot and scoring contracts..."):
        try:
            opt_payload = fetch_option_recos(
                symbol=symbol,
                direction=direction,
                trade_style=trade_style_label,
                underlying=float(underlying_px),
                tech_rating=tech_rating,
                risk_score=risk_score,
                price_target=price_target,
                signal=signal_txt,
            )
        except Exception as exc:
            st.error(f"Options lookup failed: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

    picks = opt_payload.get("picks") or []
    meta = opt_payload.get("meta") or {}
    chain_count = opt_payload.get("chain_count", len(picks))
    cache_flag = meta.get("cached")
    st.caption(
        f"Chain scanned: {chain_count} contracts"
        + ("" if cache_flag is None else f" | {'Cached' if cache_flag else 'Fresh'}")
    )

    if not picks:
        st.write("No option candidates met the liquidity and spread filters.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    top_n = min(3, len(picks))
    st.caption(f"Top {top_n} {direction.upper()} candidates for {trade_style_label}.")

    for pick in picks[:top_n]:
        header = (
            f"{symbol} ${pick.get('strike', '')} {pick.get('contract_type', '').upper()} "
            f"exp {pick.get('expiration') or 'N/A'}"
        )
        dte = pick.get("dte")
        delta = pick.get("delta")
        iv = pick.get("iv")
        oi = pick.get("open_interest")
        vol = pick.get("volume")
        spread_pct = pick.get("spread_pct")
        breakeven = pick.get("breakeven")
        mny = pick.get("moneyness")
        score = pick.get("score")
        bid = pick.get("bid")
        ask = pick.get("ask")
        mid = pick.get("mid") or pick.get("last")

        delta_txt = f"{delta:.2f}" if delta is not None else "N/A"
        iv_txt = f"{iv:.2f}" if iv is not None else "N/A"
        dte_txt = f"{int(dte)}d" if dte is not None else "N/A"
        spread_txt = f"{spread_pct*100:.1f}%" if spread_pct is not None else "N/A"
        breakeven_txt = f"${breakeven:.2f}" if breakeven is not None else "N/A"
        mny_txt = f"{mny*100:.1f}%" if mny is not None else "N/A"
        score_txt = f"{score:.1f}" if score is not None else "N/A"
        bid_ask_txt = (
            f"${bid:.2f} x ${ask:.2f}" if bid is not None and ask is not None else "N/A"
        )
        mid_txt = f"${mid:.2f}" if mid is not None else "N/A"

        st.markdown(
            f"""
            <div class="technic-card" style="margin-top:0.25rem;padding:0.85rem 1rem;border:1px solid rgba(148,163,184,0.35);">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="font-weight:700;color:{'#16a34a' if direction=='call' else '#ef4444'};">{header}</div>
                <div class="metric-chip metric-chip--good">Score {score_txt}</div>
              </div>
              <div style="display:flex;flex-wrap:wrap;gap:0.75rem;font-size:0.9rem;margin-top:0.25rem;">
                <div>Delta <strong>{delta_txt}</strong></div>
                <div>IV <strong>{iv_txt}</strong></div>
                <div>DTE <strong>{dte_txt}</strong></div>
                <div>Spread <strong>{spread_txt}</strong></div>
                <div>OI <strong>{oi if oi is not None else 'N/A'}</strong></div>
                <div>Vol <strong>{vol if vol is not None else 'N/A'}</strong></div>
                <div>Breakeven <strong>{breakeven_txt}</strong></div>
                <div>Moneyness <strong>{mny_txt}</strong></div>
              </div>
              <div style="display:flex;flex-wrap:wrap;gap:0.75rem;font-size:0.9rem;margin-top:0.25rem;">
                <div>Bid/Ask <strong>{bid_ask_txt}</strong></div>
                <div>Mid/Last <strong>{mid_txt}</strong></div>
              </div>
              <div style="font-size:0.9rem;margin-top:0.4rem;color:#cbd5e1;">{pick.get('reason','')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    top_pick = picks[0]
    if st.button(
        "Ask Copilot about this option",
        key=f"option_copilot_{symbol}",
        use_container_width=True,
    ):
        prompt = (
            f"Explain why {top_pick.get('ticker')} "
            f"({top_pick.get('contract_type','').upper()} {top_pick.get('strike')} "
            f"exp {top_pick.get('expiration')}) fits a {trade_style_label} "
            f"{'bullish' if direction == 'call' else 'bearish'} setup on {symbol}. "
            f"Underlying price about {underlying_px:.2f}, delta {delta_txt}, "
            f"DTE {dte_txt}, spread {spread_txt}, breakeven {breakeven_txt}. "
            "Keep it short and focus on liquidity, delta/time alignment, and cost."
        )
        try:
            answer = generate_copilot_answer(prompt, row)
            st.write(answer)
        except Exception as exc:
            st.error(f"Copilot error: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_symbol_detail_panel(row: pd.Series) -> None:
    """
    Symbol Detail tab: show a short natural-language summary on the left
    and a compact set of key metrics on the right using st.metric().
    """

    if row is None or row.empty:
        st.write("No symbol selected.")
        return

    # Pull core fields
    symbol = str(row.get("Symbol", "—"))
    signal = row.get("Signal", None)
    tech = row.get("TechRating", None)
    trend = row.get("TrendScore", None)
    vol = row.get("VolatilityScore", None)
    risk = row.get("RiskScore", None)
    trade_style = None
    cfg = st.session_state.get("scan_config")
    if cfg is not None and hasattr(cfg, "trade_style"):
        trade_style = getattr(cfg, "trade_style")

    # NEW: build narrative + conditions once, and reuse below
    narration = build_narrative_from_row(row)
    conditions = build_conditions(row)

    # ---- Layout: two columns ----------------------------------------------
    col1, col2 = st.columns([2, 1])

    # Left: symbol + text summary
    with col1:
        st.subheader("Symbol")
        st.markdown(f"**{symbol}**")

        parts: list[str] = []

        if isinstance(signal, str) and signal:
            parts.append(signal)

        if isinstance(tech, (int, float)) and not pd.isna(tech):
            parts.append(f"TechRating {tech:.1f}")

        if isinstance(trend, (int, float)) and not pd.isna(trend):
            if trend >= 0.7:
                parts.append("strong trend")
            elif trend >= 0.4:
                parts.append("mild uptrend")
            elif trend <= -0.7:
                parts.append("strong downtrend")
            elif trend <= -0.4:
                parts.append("mild downtrend")

        if isinstance(vol, (int, float)) and not pd.isna(vol):
            if vol >= 0.7:
                parts.append("high volatility")
            elif vol <= 0.3:
                parts.append("low volatility")

        if isinstance(risk, (int, float)) and not pd.isna(risk):
            if risk >= 0.7:
                parts.append("low risk")
            elif risk <= 0.3:
                parts.append("high risk")

        summary = ", ".join(parts) if parts else "No detailed scores available."
        st.write(summary)

    # Right: numeric metrics
    with col2:
        st.subheader("Key metrics")

        last = row.get("Last", None)
        if isinstance(last, (int, float)) and not pd.isna(last):
            st.metric("Last price", f"{last:.2f}")
        elif last is not None:
            st.metric("Last price", str(last))

        if isinstance(tech, (int, float)) and not pd.isna(tech):
            st.metric("TechRating", f"{tech:.1f}")

        if isinstance(risk, (int, float)) and not pd.isna(risk):
            st.metric("Risk score (0 = high risk, 1 = low risk)", f"{risk:.2f}")

        # ✅ fixed: no ambiguous NA boolean
        if signal is not None and not pd.isna(signal) and str(signal).strip() != "":
            st.metric("Signal", str(signal))

    # Price chart with trade levels
    st.markdown('<div class="technic-card">', unsafe_allow_html=True)
    st.markdown("##### Price & trade map")
    if ui_price_history is None:
        st.write("Price layer unavailable in this session.")
    else:
        tf_labels = [lbl for lbl, _ in CHART_TIMEFRAMES]
        tf_map = {lbl: cfg for lbl, cfg in CHART_TIMEFRAMES}
        default_tf_idx = tf_labels.index(CHART_DEFAULT) if CHART_DEFAULT in tf_labels else 0
        # Start streaming for this symbol if possible
        if start_realtime_stream and POLYGON_API_KEY:
            try:
                start_realtime_stream({symbol}, POLYGON_API_KEY)
            except Exception:
                pass

        # Quick range chips (focused set to reduce clutter)
        quick_labels = [lbl for lbl in CHART_QUICK_LABELS if lbl in tf_labels]
        if quick_labels:
            quick_cols = st.columns(len(quick_labels))
            for col, lbl in zip(quick_cols, quick_labels):
                with col:
                    if st.button(lbl, key=f"quick_tf_{symbol}_{lbl}", use_container_width=True):
                        st.session_state[f"chart_tf_{symbol}"] = lbl

        col_tf, col_auto = st.columns([2, 1])
        with col_tf:
            tf_choice = st.radio(
                "Range",
                tf_labels,
                index=default_tf_idx,
                horizontal=True,
                key=f"chart_tf_{symbol}",
            )
        with col_auto:
            auto_refresh_chart = st.checkbox(
                "Auto-refresh chart",
                value=False,
                key=f"chart_autorefresh_{symbol}",
            )
            chart_refresh_interval = st.slider(
                "Every (sec)",
                min_value=10,
                max_value=120,
                value=30,
                step=5,
                key=f"chart_refresh_interval_{symbol}",
            )
            if auto_refresh_chart and hasattr(st, "autorefresh"):
                st.autorefresh(
                    interval=chart_refresh_interval * 1000,
                    key=f"chart_autorefresh_token_{symbol}",
                )
            elif auto_refresh_chart:
                try:
                    st.rerun()
                except Exception:
                    pass

        tf_cfg = tf_map.get(tf_choice, {"days": 90, "intraday": False})
        days = int(tf_cfg.get("days", 90))
        use_intraday = bool(tf_cfg.get("intraday", False))

        try:
            hist = ui_price_history(symbol=symbol, days=days, use_intraday=use_intraday)
        except Exception:
            hist = None

        if hist is None or hist.empty:
            st.write("No price history available.")
        else:
            chart_df = hist.reset_index()
            if "Date" not in chart_df.columns:
                chart_df = chart_df.rename(columns={chart_df.columns[0]: "Date"})

            chart_df = chart_df.dropna(subset=["Close"])

            # Append live tick if available
            if get_realtime_last:
                live_px = get_realtime_last(symbol)
                if live_px is not None:
                    live_row = pd.DataFrame([{"Date": pd.Timestamp.utcnow(), "Close": live_px}])
                    chart_df = pd.concat([chart_df, live_row], ignore_index=True)

            entry_val = _float_or_none(row.get("EntryPrice"))
            stop_val = _float_or_none(row.get("StopPrice"))
            target_val = _float_or_none(row.get("TargetPrice"))

        if HAVE_ALTAIR:
            volume_layer = None
            # Build candlestick-like chart if OHLC available; otherwise line
            axis_x = alt.Axis(
                format="%b %d",
                labelColor=BRAND_TEXT,
                title=None,
                grid=False,
                labelFontSize=11,
                tickColor="rgba(148,163,184,0.35)",
            )
            axis_y = alt.Axis(
                labelColor=BRAND_TEXT,
                grid=True,
                gridColor="rgba(148,163,184,0.18)",
                title="",
                labelFontSize=11,
                tickColor="rgba(148,163,184,0.35)",
            )
            show_ma = st.checkbox("Show 50/200-day MAs", value=False, key=f"ma_toggle_{symbol}")

            if {"Open", "High", "Low", "Close"}.issubset(chart_df.columns):
                base = alt.Chart(chart_df).encode(
                    x=alt.X("Date:T", axis=axis_x)
                )
                rule = base.mark_rule(strokeWidth=1, color="#94a3b8").encode(
                    y="Low:Q",
                    y2="High:Q",
                )
                bar = base.mark_bar(size=5).encode(
                    y="Open:Q",
                    y2="Close:Q",
                    color=alt.condition(
                        "datum.Close >= datum.Open",
                        alt.value("#22c55e"),
                        alt.value("#fb7185"),
                    ),
                )
                price_layer = alt.layer(rule, bar)
            else:
                price_layer = (
                    alt.Chart(chart_df)
                    .mark_line(color=BRAND_PRIMARY, strokeWidth=2)
                    .encode(
                        x=alt.X("Date:T", axis=axis_x),
                        y=alt.Y("Close:Q", axis=axis_y),
                    )
                )

            # Optional moving averages for extra context
            if show_ma:
                try:
                    chart_df = chart_df.sort_values("Date")
                    chart_df["SMA50"] = chart_df["Close"].rolling(window=50, min_periods=10).mean()
                    chart_df["SMA200"] = chart_df["Close"].rolling(window=200, min_periods=20).mean()
                    ma_layers = []
                    for col, color in (("SMA50", "#60a5fa"), ("SMA200", "#a78bfa")):
                        if chart_df[col].notna().any():
                            ma_layers.append(
                                alt.Chart(chart_df)
                                .mark_line(strokeWidth=1.3, color=color, opacity=0.9)
                                .encode(x=alt.X("Date:T", axis=axis_x), y=alt.Y(f"{col}:Q", axis=axis_y))
                            )
                    if ma_layers:
                        price_layer = alt.layer(price_layer, *ma_layers)
                except Exception:
                    pass

            # Optional volume bars if volume present
            if "Volume" in chart_df.columns:
                vol_scale = alt.Scale(domain=[0, chart_df["Volume"].max() * 1.1])
                volume_layer = (
                    alt.Chart(chart_df)
                    .mark_bar(opacity=0.25, color="#9ca3af")
                    .encode(
                        x=alt.X("Date:T", axis=axis_x),
                        y=alt.Y("Volume:Q", scale=vol_scale, axis=alt.Axis(labelColor=BRAND_TEXT, title="Vol", labelFontSize=10)),
                    )
                    .properties(height=80)
                )

                levels = []
                if entry_val is not None:
                    levels.append(
                        {"label": "Entry", "price": entry_val, "color": BRAND_PRIMARY}
                    )
                if stop_val is not None:
                    levels.append(
                        {"label": "Stop", "price": stop_val, "color": "#fb7185"}
                    )
                if target_val is not None:
                    levels.append(
                        {"label": "Target", "price": target_val, "color": "#22c55e"}
                    )

                rule_layers = []
                if levels:
                    color_scale = alt.Scale(
                        domain=[lvl["label"] for lvl in levels],
                        range=[lvl["color"] for lvl in levels],
                    )
                    rule_layers.append(
                        alt.Chart(pd.DataFrame(levels))
                        .mark_rule(strokeWidth=1.5)
                        .encode(
                            y="price:Q",
                            color=alt.Color("label:N", scale=color_scale, legend=None),
                        )
                    )
                    rule_layers.append(
                        alt.Chart(pd.DataFrame(levels))
                        .mark_text(align="left", dx=6, dy=-4, fontSize=11)
                        .encode(
                            y="price:Q",
                            text="label:N",
                            color=alt.Color("label:N", scale=color_scale, legend=None),
                        )
                    )

                chart_main = alt.layer(price_layer, *rule_layers).properties(height=280)
                if volume_layer is not None:
                    combo = alt.vconcat(chart_main, volume_layer).resolve_scale(x="shared")
                    st.altair_chart(combo, use_container_width=True)
                else:
                    st.altair_chart(chart_main, use_container_width=True)
            else:
                st.line_chart(chart_df.set_index("Date")["Close"])
    st.markdown("</div>", unsafe_allow_html=True)

    # ----- Options play (Polygon) -----------------------------------------
    try:
        render_option_play(row, trade_style)
    except Exception as exc:
        st.warning(f"Options section unavailable: {exc}")

    # ----- Trader chart (single chart with line/candle toggle) ------------
    if ui_price_history is not None:
        symbol_local = str(row.get("Symbol", "")).upper()
        if symbol_local:
            st.markdown('<div class="technic-card">', unsafe_allow_html=True)
            st.markdown("##### Trader chart")

            tf_options = {
                "1D (5m)": {"days": 2, "intraday": True, "title": "1D (5m)"},
                "5D (5m)": {"days": 6, "intraday": True, "title": "5D (5m)"},
                "3M (1D)": {"days": 90, "intraday": False, "title": "3M (daily)"},
                "1Y (1D)": {"days": 365, "intraday": False, "title": "1Y (daily)"},
            }
            tf_choice = st.radio(
                "Range",
                list(tf_options.keys()),
                index=0,
                horizontal=True,
                key=f"trader_tf_{symbol_local}",
            )
            tf_cfg = tf_options[tf_choice]
            chart_type = st.radio(
                "Style",
                ["Candles", "Line"],
                index=0,
                horizontal=True,
                key=f"trader_style_{symbol_local}",
            )
            show_vol = st.checkbox("Show volume", value=True, key=f"trader_vol_{symbol_local}")

            try:
                hist = ui_price_history(
                    symbol=symbol_local,
                    days=int(tf_cfg["days"]),
                    use_intraday=bool(tf_cfg["intraday"]),
                )
            except Exception:
                hist = None

            if hist is None or hist.empty:
                st.write("No price history available.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                chart_df = hist.reset_index()
                if "Date" not in chart_df.columns:
                    chart_df = chart_df.rename(columns={chart_df.columns[0]: "Date"})
                chart_df = chart_df.dropna(subset=["Close"])

                if get_realtime_last:
                    live_px = get_realtime_last(symbol_local)
                    if live_px is not None:
                        live_row = pd.DataFrame([{"Date": pd.Timestamp.utcnow(), "Close": live_px}])
                        chart_df = pd.concat([chart_df, live_row], ignore_index=True)

                entry_val = _float_or_none(row.get("EntryPrice"))
                stop_val = _float_or_none(row.get("StopPrice"))
                target_val = _float_or_none(row.get("TargetPrice"))

                if HAVE_ALTAIR:
                    axis_x = alt.Axis(
                        format="%b %d",
                        labelColor=BRAND_TEXT,
                        title=None,
                        grid=False,
                        labelFontSize=11,
                        tickColor="rgba(148,163,184,0.35)",
                    )
                    axis_y = alt.Axis(
                        labelColor=BRAND_TEXT,
                        grid=True,
                        gridColor="rgba(148,163,184,0.18)",
                        title="",
                        labelFontSize=11,
                        tickColor="rgba(148,163,184,0.35)",
                    )

                    volume_layer = None
                    if show_vol and "Volume" in chart_df.columns:
                        vol_scale = alt.Scale(domain=[0, chart_df["Volume"].max() * 1.1])
                        volume_layer = (
                            alt.Chart(chart_df)
                            .mark_bar(opacity=0.3, color="#94a3b8")
                            .encode(
                                x=alt.X("Date:T", axis=axis_x),
                                y=alt.Y("Volume:Q", scale=vol_scale, axis=alt.Axis(labelColor=BRAND_TEXT, title="Vol", labelFontSize=10)),
                            )
                            .properties(height=70)
                        )

                    if chart_type == "Candles" and {"Open", "High", "Low", "Close"}.issubset(chart_df.columns):
                        base = alt.Chart(chart_df).encode(x=alt.X("Date:T", axis=axis_x))
                        rule = base.mark_rule(strokeWidth=1, color="#94a3b8").encode(
                            y="Low:Q",
                            y2="High:Q",
                        )
                        bar = base.mark_bar(size=5).encode(
                            y="Open:Q",
                            y2="Close:Q",
                            color=alt.condition(
                                "datum.Close >= datum.Open",
                                alt.value("#22c55e"),
                                alt.value("#fb7185"),
                            ),
                        )
                        price_layer = alt.layer(rule, bar)
                    else:
                        price_layer = (
                            alt.Chart(chart_df)
                            .mark_line(color=BRAND_PRIMARY, strokeWidth=2)
                            .encode(
                                x=alt.X("Date:T", axis=axis_x),
                                y=alt.Y("Close:Q", axis=axis_y),
                            )
                        )

                    levels = []
                    if entry_val is not None:
                        levels.append({"label": "Entry", "price": entry_val, "color": BRAND_PRIMARY})
                    if stop_val is not None:
                        levels.append({"label": "Stop", "price": stop_val, "color": "#fb7185"})
                    if target_val is not None:
                        levels.append({"label": "Target", "price": target_val, "color": "#22c55e"})

                    rule_layers = []
                    if levels:
                        color_scale = alt.Scale(
                            domain=[lvl["label"] for lvl in levels],
                            range=[lvl["color"] for lvl in levels],
                        )
                        rule_layers.append(
                            alt.Chart(pd.DataFrame(levels))
                            .mark_rule(strokeWidth=1.5)
                            .encode(
                                y="price:Q",
                                color=alt.Color("label:N", scale=color_scale, legend=None),
                            )
                        )
                        rule_layers.append(
                            alt.Chart(pd.DataFrame(levels))
                            .mark_text(align="left", dx=6, dy=-4, fontSize=11)
                            .encode(
                                y="price:Q",
                                text="label:N",
                                color=alt.Color("label:N", scale=color_scale, legend=None),
                            )
                        )

                    chart_main = alt.layer(price_layer, *rule_layers).properties(height=320, title=tf_cfg["title"])
                    if volume_layer is not None:
                        combo = alt.vconcat(chart_main, volume_layer).resolve_scale(x="shared")
                        st.altair_chart(combo, use_container_width=True)
                    else:
                        st.altair_chart(chart_main, use_container_width=True)
                else:
                    st.line_chart(chart_df.set_index("Date")["Close"], height=320, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # ----- Setup tags as pills ---------------------------------------------
    tags = build_setup_tags(row)
    if tags:
        st.markdown('<div class="technic-card">', unsafe_allow_html=True)
        st.markdown("##### Setup tags")
        st.markdown(render_setup_tags(tags), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ----- Fundamentals snapshot ------------------------------------------
    if get_fundamentals is not None:
        snap = get_fundamentals(symbol)
        if snap.raw:
            st.markdown('<div class="technic-card">', unsafe_allow_html=True)
            st.markdown("##### Fundamentals")
            cols = st.columns(3)
            with cols[0]:
                pe = snap.get("pe") or snap.get("PE")
                if pe is not None:
                    st.metric("P/E", f"{float(pe):.1f}")
                z = snap.get("altman_z") or snap.get("altman_zscore")
                if z is not None:
                    st.metric("Altman Z", f"{float(z):.2f}")
            with cols[1]:
                fscore = snap.get("piotroski") or snap.get("piotroski_fscore")
                if fscore is not None:
                    st.metric("Piotroski F", f"{float(fscore):.1f}")
                peg = snap.get("peg")
                if peg is not None:
                    st.metric("PEG", f"{float(peg):.2f}")
            with cols[2]:
                eps_g = snap.get("eps_growth") or snap.get("eps_5y_cagr")
                if eps_g is not None:
                    st.metric("EPS growth", f"{float(eps_g):.1f}%")
                margin = snap.get("profit_margin")
                if margin is not None:
                    st.metric("Profit margin", f"{float(margin):.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

    # ----- Multi-timeframe performance ------------------------------------
    if get_multi_timeframes is not None:
        try:
            mtf = get_multi_timeframes(symbol, days=400)
            weekly = mtf.get("weekly")
            monthly = mtf.get("monthly")
            perf_week = _perf_over(weekly, periods=min(4, len(weekly))) if weekly is not None else None
            perf_month = _perf_over(monthly, periods=min(3, len(monthly))) if monthly is not None else None
            st.markdown('<div class="technic-card">', unsafe_allow_html=True)
            st.markdown("##### Multi-timeframe pulse")
            cols = st.columns(2)
            with cols[0]:
                if perf_week is not None:
                    st.metric("4-week change", f"{perf_week:+.1f}%")
                else:
                    st.write("No weekly data")
            with cols[1]:
                if perf_month is not None:
                    st.metric("3-month change", f"{perf_month:+.1f}%")
                else:
                    st.write("No monthly data")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            pass

    # ----- Score breakdown (left) + conditions (right) ---------------------
    score_keys = [
        "TrendScore",
        "MomentumScore",
        "ExplosivenessScore",
        "BreakoutScore",
        "VolumeScore",
        "VolatilityScore",
        "OscillatorScore",
        "TrendQualityScore",
    ]
    score_dict: dict[str, float] = {}
    for k in score_keys:
        if k in row and not pd.isna(row[k]):
            val = float(row[k])
            score_dict[k] = max(-3.0, min(3.0, val))

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<div class="technic-card">', unsafe_allow_html=True)
        st.markdown("##### Score breakdown")
        if HAVE_ALTAIR and score_dict:
            chart_df = (
                pd.DataFrame({"Factor": list(score_dict.keys()), "Score": list(score_dict.values())})
                .sort_values("Score", key=abs, ascending=False)
            )
            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadius=6)
                .encode(
                    x=alt.X("Score:Q", scale=alt.Scale(domain=[-3, 3])),
                    y=alt.Y("Factor:N", sort=chart_df["Factor"].tolist()),
                    color=alt.condition(
                        "datum.Score >= 0",
                        alt.value(BRAND_PRIMARY),
                        alt.value("#fb7185"),
                    ),
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            fig = make_score_bar_chart(score_dict)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
            else:
                st.write("Score breakdown chart unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="technic-card">', unsafe_allow_html=True)
        st.markdown("##### Technical conditions snapshot")
        for label, text in conditions.items():
            st.write(f"• {label}: {text}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ----- Trade plan (if present) -----------------------------------------
    entry = row.get("EntryPrice", None)
    stop = row.get("StopPrice", None)
    target = row.get("TargetPrice", None)
    rr = row.get("RewardRisk", None)
    size = row.get("PositionSize", None)

    st.markdown('<div class="technic-card">', unsafe_allow_html=True)
    st.markdown("#### Personal Quant Plan")

    if (
        entry is None or stop is None or target is None or
        pd.isna(entry) or pd.isna(stop) or pd.isna(target)
    ):
        st.write(
            "Trade plan is unavailable for this symbol (missing ATR / volatility inputs). "
            "Try re-running the scan or using a different lookback window."
        )
    else:
        cols = st.columns(4)
        with cols[0]:
            st.caption("ENTRY")
            st.markdown(f"**${entry:,.2f}**")
        with cols[1]:
            st.caption("STOP")
            st.markdown(f"**${stop:,.2f}**")
        with cols[2]:
            st.caption("TARGET")
            st.markdown(f"**${target:,.2f}**")
        with cols[3]:
            st.caption("R:R")
            if rr is not None and not pd.isna(rr):
                st.markdown(f"**{rr:.2f}×**")
            else:
                st.markdown("—")

        if size is not None and not pd.isna(size):
            st.caption("POSITION SIZE (approx.)")
            st.markdown(f"**${size:,.0f}** notional")

        if st.button(
            "Add to scoreboard",
            key=f"add_scoreboard_{symbol}",
            use_container_width=True,
        ):
            add_row_to_scoreboard(row)
            st.success("Added to scoreboard for live tracking.")

    # Narrative block
    st.markdown("##### Narrative")
    st.write(narration)
    st.markdown("</div>", unsafe_allow_html=True)

@st.cache_data
def get_universe_stats():
    rows = load_universe()
    total = len(rows)
    sector_counts = Counter(r.sector or "Unknown" for r in rows)
    subindustry_counts = Counter(r.subindustry or "Other" for r in rows)

    sectors_sorted = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
    subindustries_sorted = sorted(
        subindustry_counts.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "total": total,
        "sectors": sectors_sorted,
        "subindustries": subindustries_sorted,
    }


@st.cache_data
def estimate_filtered_universe(
    sectors: tuple[str, ...] | None,
    subindustries: tuple[str, ...] | None,
    industry_contains: str | None,
) -> int:
    """
    Estimate how many symbols are in the current filtered universe
    (sector + subindustry + industry keyword), so we can set a smart
    max for the 'Max symbols to scan' slider.
    """
    rows = load_universe()
    industry_contains_l = (industry_contains or "").strip().lower() or None

    count = 0
    for r in rows:
        # Sector filter
        if sectors and (r.sector or "Unknown") not in sectors:
            continue

        # Subindustry filter
        if subindustries and (r.subindustry or "Other") not in subindustries:
            continue

        # Industry keyword filter
        if industry_contains_l:
            ind = (r.industry or "").lower()
            if industry_contains_l not in ind:
                continue

        count += 1

    return count


@st.cache_data
def get_sector_options():
    stats = get_universe_stats()
    return [name for name, _ in stats["sectors"]]


@st.cache_data
def get_subindustry_options():
    stats = get_universe_stats()
    return [name for name, _ in stats["subindustries"]]


st.set_page_config(
    page_title="Technic v4 - Multi-Factor Technical Scanner",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme toggle
theme_mode = st.sidebar.radio("Theme", ["Dark", "Light"], index=0, key="theme_mode_radio")
inject_premium_theme("light" if theme_mode == "Light" else "dark")

st.markdown(
    """
    <div class="technic-header">
      <div class="technic-header-inner">
        <div class="brand-lockup">
          <div class="brand-symbol" aria-hidden="true"></div>
          <div class="brand-wordmark">
            <h1>Technic</h1>
            <div class="tagline">Personal Quant</div>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

universe_stats = get_universe_stats()
regime = compute_market_regime()
regime_label = regime if regime != "unknown" else "n/a"
if st.session_state.get("scan_results") is None:
    st.markdown(
        f"""
        <div class="technic-card" style="margin-bottom:1rem;">
          <div style="display:flex;flex-wrap:wrap;gap:1rem;justify-content:space-between;align-items:center;">
            <div>
              <div style="font-size:0.9rem;color:#9ca3af;">Universe: {universe_stats['total']} symbols across {len(universe_stats['sectors'])} sectors and {len(universe_stats['subindustries'])} subindustries.</div>
              <h3 style="margin:4px 0 0 0;">Welcome to Technic</h3>
              <p style="margin:4px 0 0 0;color:#9ca3af;">Run a scan to see ranked setups, candlesticks with volume, live news, and Copilot explanations.</p>
            </div>
            <div style="display:flex;gap:0.6rem;flex-wrap:wrap;">
              <div class="metric-chip">Multi-factor signals</div>
              <div class="metric-chip">Ready-to-trade plans</div>
              <div class="metric-chip">News + Copilot</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.caption(
        f"Universe: {universe_stats['total']} symbols "
        f"across {len(universe_stats['sectors'])} sectors and "
        f"{len(universe_stats['subindustries'])} subindustries."
    )
    st.caption(f"Market regime (SPY ATR%): {regime_label}")

# Ticker strip (broad tape: scoreboard + top scan) with toggle
show_ticker = st.sidebar.checkbox("Show ticker strip", value=st.session_state.get("show_ticker", True), key="show_ticker")

ticker_sources: list[pd.DataFrame] = []
if show_ticker:
    if st.session_state.get("scan_results") is not None:
        ticker_sources.append(st.session_state["scan_results"].sort_values("TechRating", ascending=False).head(20))
    sb_df_for_tape = build_scoreboard_df(st.session_state.get("scoreboard_entries", []))
    if not sb_df_for_tape.empty:
        ticker_sources.append(sb_df_for_tape.head(20))

if show_ticker and ticker_sources:
    ticker_df = pd.concat(ticker_sources, ignore_index=True)
    ticker_df = ticker_df.drop_duplicates(subset=["Symbol"])
    ticker_html = build_ticker_html(ticker_df, max_items=30)
    st.markdown(ticker_html, unsafe_allow_html=True)
    # Stream status (best-effort)
    if get_stream_status is not None:
        try:
            st_status = get_stream_status()
            active = st_status.get("active")
            ages = st_status.get("last_tick_age", {})
            sel_sym = st.session_state.get("selected_symbol")
            age_txt = ""
            if sel_sym and ages:
                a = ages.get(str(sel_sym).upper())
                if a is not None:
                    age_txt = f" | {sel_sym} last tick {a:.1f}s ago"
            sym_txt = ""
            if st_status.get("symbols"):
                sym_txt = f" | subs: {', '.join(st_status['symbols'])}"
            st.caption(f"Stream: {'active' if active else 'idle'}{age_txt}{sym_txt}")
        except Exception:
            pass

# Live scan teaser (promising tickers while scanning)
if st.session_state.get("progress_symbols"):
    chips = " ".join(
        f"<span class='metric-chip metric-chip--good'>{sym}</span>"
        for sym in st.session_state["progress_symbols"][:10]
    )
    st.markdown(
        f"<div style='margin:6px 0 4px;'>Scanning: {chips}</div>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.header("🧭 Scan Settings")
    st.caption(f"Regime: **{regime_label.title()}** (auto-tuned defaults)")

    st.subheader("🎯 Preset")
    preset = st.selectbox(
        "Preset",
        [
            "Custom",
            "Swing Breakouts",
            "Long-term Quality Leaders",
            "Low-risk Trend Followers",
            "High-conviction Momentum",
        ],
        index=0,
        help="Presets give you suggested settings for different trading styles.",
    )

    # If preset changed and is not Custom, push defaults into widget state
    preset_cfg = PRESET_CONFIG.get(preset, {})
    if preset != st.session_state.get("last_preset") and preset != "Custom":
        def _set(label: str, val):
            st.session_state[label] = val

        _set("Lookback window (days)", preset_cfg.get("lookback_days", 180))
        _set("Minimum TechRating to show", preset_cfg.get("min_tech_rating", 10.0))
        _set("Allow short setups", preset_cfg.get("allow_shorts", False))
        _set("Only show tradeable signals", preset_cfg.get("only_tradeable", True))
        _set("Risk per trade (%)", preset_cfg.get("risk_pct", 1.0))
        _set("Target Reward:Risk", preset_cfg.get("rr_multiple", 2.0))
        _set("Trade style", preset_cfg.get("ui_style", "Swing"))
        _set("Max symbols to score", preset_cfg.get("max_symbols", 500))
        _set("Require Breakout setup", preset_cfg.get("require_breakout_tag", False))
        _set("Require Momentum setup", preset_cfg.get("require_momentum_tag", False))
        _set("Only potential breakouts", preset_cfg.get("sf_breakouts", False))
        _set("Only smooth trends", preset_cfg.get("sf_smooth_trends", False))
        _set("Low-risk only", preset_cfg.get("sf_low_risk", False))
        _set("Exclude choppy charts", preset_cfg.get("sf_exclude_choppy", False))
        _set("High-conviction only", preset_cfg.get("sf_high_conviction", False))
        _set("Top per sector", preset_cfg.get("sf_top_by_sector", False))
        st.session_state["last_preset"] = preset
    else:
        st.session_state["last_preset"] = preset

    if preset != "Custom":
        if preset == "Swing Breakouts":
            st.caption("Suggested: TechRating ~15, short-term swing, breakout + momentum tags.")
        elif preset == "Long-term Quality Leaders":
            st.caption("Suggested: Longer lookback, long-term style, higher TechRating floor, low volatility only.")
        elif preset == "Low-risk Trend Followers":
            st.caption("Suggested: Emphasize RiskScore >= 0.7, smooth trends, exclude choppy charts.")
        elif preset == "High-conviction Momentum":
            st.caption("Suggested: Strong Long only, high TechRating, strong momentum and explosiveness.")
        st.caption("You can adjust the sliders below to match or tweak the preset.")

    # Fast scan toggle: skip fundamentals/RS during scan for speed
    FAST_SCAN = st.checkbox(
        "Fast scan (skip fundamentals/RS during scan)",
        value=True,
    )

    with st.expander("⚙️ Scan Settings", expanded=True):
        lookback_days = st.slider(
            "Lookback window (days)",
            min_value=60,
            max_value=365,
            value=int(preset_cfg.get("lookback_days", 180)),
            step=30,
            help="Window used for computing scores and volatility metrics.",
        )

        min_tech_rating = st.slider(
            "Minimum TechRating to show",
            min_value=0.0,
            max_value=30.0,
            value=float(preset_cfg.get("min_tech_rating", 10.0)),
            step=0.5,
            help="Filter out the weakest names from the results.",
        )

    with st.expander("📐 Trade & Risk", expanded=True):
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            account_size = st.number_input(
                "Account size ($)",
                min_value=1_000.0,
                max_value=10_000_000.0,
                value=float(preset_cfg.get("account_size", 50_000.0)),
                step=1_000.0,
                help="Used to compute example position sizing.",
            )

            risk_pct = st.slider(
                "Risk per trade (%)",
                min_value=0.1,
                max_value=5.0,
                value=float(preset_cfg.get("risk_pct", 1.0)),
                step=0.1,
                help="Percent of account size at risk per trade (for sizing).",
            )

        with col_r2:
            rr_multiple = st.slider(
                "Target Reward:Risk",
                min_value=1.0,
                max_value=5.0,
                value=float(preset_cfg.get("rr_multiple", 2.0)),
                step=0.5,
                help="Used in example target calculations.",
            )

            ui_style = st.selectbox(
                "Trade style",
                options=["Swing", "Medium-term", "Long-term"],
                index=["Swing", "Medium-term", "Long-term"].index(
                    preset_cfg.get("ui_style", "Swing")
                ),
                help="Controls how the scoring engine weights different factors.",
            )

            style_map = {
                "Swing": "Short-term swing",
                "Medium-term": "Medium-term swing",
                "Long-term": "Position / longer-term",
            }
            trade_style_internal = style_map.get(ui_style, "Short-term swing")

        allow_shorts = st.checkbox(
            "Allow short setups",
            value=bool(preset_cfg.get("allow_shorts", False)),
            help="If enabled, the scanner may surface short candidates as well.",
        )

        only_tradeable = st.checkbox(
            "Only show tradeable signals",
            value=bool(preset_cfg.get("only_tradeable", True)),
            help="If enabled, hides 'Avoid' / low-conviction names.",
        )

    with st.expander("Universe Filters", expanded=False):
        all_sectors = get_sector_options()
        selected_sectors = st.multiselect(
            "Sectors to include (optional)",
            options=all_sectors,
            default=[],
            help="If empty, all sectors are included.",
        )

        all_subindustries = get_subindustry_options()
        selected_subindustries = st.multiselect(
            "Subindustries to include (optional)",
            options=all_subindustries,
            default=[],
            help="If empty, all subindustries are included.",
        )

        keyword_help = (
            "Filter by Industry via substring match. "
            "Example: 'semiconductor' to focus on chip names."
        )
        industry_keyword = st.text_input(
            "Industry keyword filter (optional)",
            value="",
            help=keyword_help,
        )

    # Dynamic max-symbols cap based on ALL filters
    filtered_count = estimate_filtered_universe(
        tuple(selected_sectors) if selected_sectors else None,
        tuple(selected_subindustries) if selected_subindustries else None,
        industry_keyword or None,
    )

    if filtered_count <= 0:
        # Fallback: if filters exclude everything, use the full universe size
        max_cap = universe_stats["total"]
    else:
        max_cap = filtered_count

    if max_cap < 1:
        max_cap = 1

    min_cap = 10 if max_cap >= 10 else 1
    step = 10 if max_cap >= 200 else 1

    max_symbols = st.slider(
        "Max symbols to score",
        min_value=int(min_cap),
        max_value=int(max_cap),
        value=min(int(preset_cfg.get("max_symbols", 500)), int(max_cap)),
        step=int(step),
        help=(
            "Upper limit of how many symbols are processed per scan. "
            "Drag to the maximum to scan the full filtered universe."
        ),
    )
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Smart Filters", expanded=False):

        require_breakout_tag = st.checkbox(
            "Require Breakout setup",
            value=bool(preset_cfg.get("require_breakout_tag", False)),
        )
        require_momentum_tag = st.checkbox(
            "Require Momentum setup",
            value=bool(preset_cfg.get("require_momentum_tag", False)),
        )
        only_high_conviction = st.checkbox(
            "Strong Long setups only",
            value=bool(preset_cfg.get("only_high_conviction", False)),
        )
        
        col_sf1, col_sf2 = st.columns(2)

        with col_sf1:
            sf_breakouts = st.checkbox(
                "Only potential breakouts",
                value=bool(preset_cfg.get("sf_breakouts", False)),
                help="Focus on names showing breakout-style behavior.",
            )
            sf_smooth_trends = st.checkbox(
                "Only smooth trends",
                value=bool(preset_cfg.get("sf_smooth_trends", False)),
                help="Trend + trend quality filters; hides messy charts.",
            )
            sf_low_risk = st.checkbox(
                "Low-risk only",
                value=bool(preset_cfg.get("sf_low_risk", False)),
                help="Require relatively high RiskScore (lower volatility / tighter risk).",
            )

        with col_sf2:
            sf_exclude_choppy = st.checkbox(
                "Exclude choppy charts",
                value=bool(preset_cfg.get("sf_exclude_choppy", False)),
                help="Hide low trend quality / high-volatility regimes.",
            )
        sf_high_conviction = st.checkbox(
            "High-conviction only",
            value=bool(preset_cfg.get("sf_high_conviction", False)),
            help="Only Strong Long / highest TechRatings.",
        )
        sf_top_by_sector = st.checkbox(
            "Top per sector",
            value=bool(preset_cfg.get("sf_top_by_sector", False)),
            help="Show the strongest names per sector (up to a few each).",
        )

        st.caption(
            "Tip: Start with a broad sector filter, then tighten Smart Filters to surface only the cleanest setups."
        )

    with st.expander("📊 Fundamental Filters (optional)", expanded=False):
        enable_fund_filters = st.checkbox("Enable fundamental filters", value=False)
        # Regime-aware defaults for fundamentals / RS (set once)
        if "fund_defaults_set" not in st.session_state:
            defaults = {
                "min_piotroski_default": 3.0,
                "max_pe_default": 60.0,
                "max_peg_default": 3.0,
                "min_eps_g_default": -10.0,
                "min_rs_pct_default": 0.0,
            }
            if regime == "high vol":
                defaults.update(
                    {"min_piotroski_default": 4.0, "max_pe_default": 45.0, "min_eps_g_default": 0.0, "min_rs_pct_default": 60.0}
                )
            elif regime == "medium vol":
                defaults.update({"min_piotroski_default": 3.5, "max_pe_default": 55.0, "min_rs_pct_default": 40.0})
            st.session_state.update(defaults)
            st.session_state["fund_defaults_set"] = True

        rs_bench = st.selectbox(
            "RS benchmark",
            options=["SPY", "QQQ", "IWM", "DIA"],
            index=0,
            help="Used for relative strength percentile.",
        )
        min_piotroski = st.slider(
            "Min Piotroski F",
            min_value=0.0,
            max_value=9.0,
            value=float(st.session_state.get("min_piotroski_default", 3.0)),
            step=0.5,
            disabled=not enable_fund_filters,
        )
        max_pe = st.slider(
            "Max P/E",
            min_value=1.0,
            max_value=150.0,
            value=float(st.session_state.get("max_pe_default", 60.0)),
            step=1.0,
            disabled=not enable_fund_filters,
        )
        max_peg = st.slider(
            "Max PEG",
            min_value=0.1,
            max_value=5.0,
            value=float(st.session_state.get("max_peg_default", 3.0)),
            step=0.1,
            disabled=not enable_fund_filters,
        )
        min_eps_g = st.slider(
            "Min EPS growth (%)",
            min_value=-50.0,
            max_value=80.0,
            value=float(st.session_state.get("min_eps_g_default", -10.0)),
            step=1.0,
            disabled=not enable_fund_filters,
        )
        min_rs_pct = st.slider(
            f"Min RS percentile vs {rs_bench}",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.get("min_rs_pct_default", 0.0)),
            step=5.0,
            disabled=not enable_fund_filters,
            help="Relative strength percentile over ~2 months.",
        )

    with st.expander("📈 Technical Filters (optional)", expanded=False):
        # Regime-aware defaults (only applied on first load)
        if "tech_defaults_set" not in st.session_state:
            if regime == "high vol":
                st.session_state["min_atr_pct_default"] = 1.0
                st.session_state["require_above_200_default"] = True
            elif regime == "medium vol":
                st.session_state["min_atr_pct_default"] = 0.5
                st.session_state["require_above_200_default"] = False
            else:
                st.session_state["min_atr_pct_default"] = 0.0
                st.session_state["require_above_200_default"] = False
            st.session_state["tech_defaults_set"] = True

        require_weekly_up = st.checkbox("Weekly uptrend", value=False)
        require_above_50 = st.checkbox("Above 50-day SMA", value=False)
        require_above_200 = st.checkbox("Above 200-day SMA", value=st.session_state.get("require_above_200_default", False))
        require_golden_cross = st.checkbox("Golden cross (SMA50 > SMA200)", value=False)
        min_atr_pct = st.slider(
            "Min ATR% (volatility)",
            min_value=0.0,
            max_value=25.0,
            value=st.session_state.get("min_atr_pct_default", 0.0),
            step=0.5,
        )
        max_atr_pct = st.slider(
            "Max ATR% (volatility)",
            min_value=0.0,
            max_value=25.0,
            value=25.0,
            step=0.5,
        )

    with st.expander("Advanced / Custom Formula", expanded=False):
        st.caption("Example: (PE < 40) & (TechRating > 15) & (RiskScore > 0.7)")
        custom_formula = st.text_input(
            "Boolean expression (uses column names)",
            value="",
            help="Use operators &, |, >, <, ==. Available columns include PE, PEG, Piotroski, TechRating, RiskScore, etc.",
        )

    with st.expander("🔔 Alerts (beta)", expanded=False):
        st.caption("Set a simple price alert for any symbol in your results.")
        alert_symbol = st.text_input("Symbol", value=st.session_state.get("selected_symbol", ""))
        alert_direction = st.radio("Direction", ["At or above", "At or below"], index=0, horizontal=True)
        alert_price = st.number_input("Trigger price", min_value=0.0, value=0.0, step=0.1)
        if st.button("Add alert", key="add_alert_button"):
            sym = alert_symbol.strip().upper()
            if sym and alert_price > 0:
                st.session_state["price_alerts"].append(
                    {
                        "symbol": sym,
                        "direction": "above" if alert_direction == "At or above" else "below",
                        "price": float(alert_price),
                        "triggered": False,
                    }
                )
                st.success(f"Alert added for {sym} at {alert_direction.lower()} {alert_price}.")
            else:
                st.warning("Enter a valid symbol and price.")
        # Manage alerts
        existing_alerts = st.session_state.get("price_alerts", [])
        if existing_alerts:
            st.caption("Active alerts")
            for idx, a in enumerate(existing_alerts):
                status = "triggered" if a.get("triggered") else "active"
                st.write(f"{idx+1}. {a['symbol']} {a['direction']} {a['price']:.2f} ({status})")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Clear triggered", key="clear_triggered_alerts"):
                    st.session_state["price_alerts"] = [a for a in existing_alerts if not a.get("triggered")]
                    st.success("Cleared triggered alerts.")
            with col_b:
                if st.button("Clear all alerts", key="clear_all_alerts"):
                    st.session_state["price_alerts"] = []
                    st.success("Cleared all alerts.")
            # Remove a single alert
            idx_to_remove = st.number_input(
                "Remove alert #", min_value=1, max_value=len(existing_alerts), value=1, step=1, key="remove_alert_idx"
            )
            if st.button("Remove selected alert", key="remove_alert_btn"):
                try:
                    pos = int(idx_to_remove) - 1
                    if 0 <= pos < len(existing_alerts):
                        existing_alerts.pop(pos)
                        st.session_state["price_alerts"] = existing_alerts
                        st.success("Removed alert.")
                except Exception:
                    st.warning("Enter a valid alert number.")
# ------------------------------------------------------------------
# Persistent scan state
# ------------------------------------------------------------------
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = None
    st.session_state["scan_status_msg"] = ""
    st.session_state["scan_config"] = None
    st.session_state["technic_last_updated"] = None
    st.session_state["selected_symbol"] = None

# Trigger flag for bottom “Re-run scan” button
if "trigger_scan" not in st.session_state:
    st.session_state["trigger_scan"] = False

# Quant Copilot state
if "copilot_last_q" not in st.session_state:
    st.session_state["copilot_last_q"] = ""
if "copilot_last_answer" not in st.session_state:
    st.session_state["copilot_last_answer"] = None
if "scoreboard_entries" not in st.session_state:
    st.session_state["scoreboard_entries"] = []
if "scoreboard_loaded" not in st.session_state:
    st.session_state["scoreboard_loaded"] = False
if "last_preset" not in st.session_state:
    st.session_state["last_preset"] = "Custom"
if "supabase_connected" not in st.session_state:
    st.session_state["supabase_connected"] = False
if "price_alerts" not in st.session_state:
    st.session_state["price_alerts"] = []  # list of dicts: {symbol, direction, price, triggered}

# ------------------------------------------------------------------
# Centered primary Run Scan button (top)
# ------------------------------------------------------------------
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    run_button_top = st.button(
        "Run Scan",
        type="primary",
        use_container_width=True,
        key="run_scan_top",
    )

# Combine top button + trigger from bottom
trigger_scan = st.session_state["trigger_scan"]
run_button = run_button_top or trigger_scan

# Reset trigger so it only fires once
st.session_state["trigger_scan"] = False

results_df: pd.DataFrame | None = st.session_state["scan_results"]
status_msg: str = st.session_state["scan_status_msg"]
config = st.session_state["scan_config"]

# ------------------------------------------------------------------
# Run the scan when requested
# ------------------------------------------------------------------
if run_button:
    # --- Live progress UI (centered) --------------------------------
    progress_container = st.container()
    with progress_container:
        col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
        with col_p2:
            progress_bar = st.progress(0.0)
            progress_placeholder = st.empty()

    def on_progress(symbol: str, idx: int, total: int) -> None:
        """Update the progress bar + ticker label while scanning."""
        if total <= 0:
            pct = 0.0
        else:
            pct = idx / float(total)

        # Track recently scanned symbols (dedup, max 15)
        if symbol:
            recent = st.session_state.get("progress_symbols", [])
            recent = [s for s in recent if s != symbol][:14]
            recent.insert(0, symbol)
            st.session_state["progress_symbols"] = recent

        progress_bar.progress(pct)
        progress_placeholder.markdown(
            f"""
            <div class="technic-progress-label">
              Scanning <code>{symbol}</code> ({idx} of {total})
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Build ScanConfig from current UI ---------------------------
    with st.spinner("Running Technic v4 scan..."):
        started = dt.datetime.now(dt.timezone.utc)

        config = ScanConfig(
            max_symbols=max_symbols,
            lookback_days=lookback_days,
            min_tech_rating=min_tech_rating,
            account_size=account_size,
            risk_pct=risk_pct,
            target_rr=rr_multiple,
            trade_style=trade_style_internal,
            allow_shorts=allow_shorts,
            only_tradeable=only_tradeable,
            sectors=selected_sectors or None,
            subindustries=selected_subindustries or None,
            industry_contains=industry_keyword or None,
        )

        df, status = run_scan(config, progress_cb=on_progress)

        finished = dt.datetime.now(dt.timezone.utc)
        st.session_state["technic_last_updated"] = finished
        st.session_state["scan_status_msg"] = status
        st.session_state["scan_config"] = config

    # Clear progress UI once scan is done
    progress_container.empty()
    st.session_state.pop("progress_symbols", None)

    # --- Post-processing: filters, Smart Filters, fallback ----------

    if df is None or df.empty:
        st.error(status or "No results returned. Check universe or data source.")
        results_df = None
        st.session_state["scan_results"] = None

    else:
        raw_results = df.copy()
        df_scored = df.copy()

        # Sort by TechRating
        if "TechRating" in df_scored.columns:
            df_scored = df_scored.sort_values("TechRating", ascending=False)

        # Sector filter
        if selected_sectors and "Sector" in df_scored.columns:
            df_scored = df_scored[df_scored["Sector"].isin(selected_sectors)]

        # Subindustry filter
        if selected_subindustries and "SubIndustry" in df_scored.columns:
            df_scored = df_scored[
                df_scored["SubIndustry"].isin(selected_subindustries)
            ]

        # Industry keyword filter
        kw = (industry_keyword or "").strip().lower()
        if kw and "Industry" in df_scored.columns:
            df_scored = df_scored[
                df_scored["Industry"].fillna("").str.lower().str.contains(kw)
            ]

        # Tag-based filters
        if require_breakout_tag:
            if "BreakoutScore" in df_scored.columns:
                df_scored = df_scored[df_scored["BreakoutScore"] >= 1]
            elif "TradeType" in df_scored.columns:
                df_scored = df_scored[df_scored["TradeType"].astype(str).str.contains("Breakout", case=False)]
        if require_momentum_tag:
            if "MomentumScore" in df_scored.columns:
                df_scored = df_scored[df_scored["MomentumScore"] >= 1]
            elif "TradeType" in df_scored.columns:
                df_scored = df_scored[df_scored["TradeType"].astype(str).str.contains("Momentum", case=False)]

        if only_high_conviction:
            if "Signal" in df_scored.columns:
                df_scored = df_scored[df_scored["Signal"].isin(["Strong Long", "Strong Short", "Long"])]
            elif "TechRating" in df_scored.columns:
                df_scored = df_scored[df_scored["TechRating"] >= 18.0]

        # Fundamentals enrichment + filters
        try:
            if not FAST_SCAN:
                df_scored = attach_fundamentals_cols(df_scored)
                df_scored = attach_relative_strength(df_scored, benchmark=rs_bench)
            df_scored = attach_technical_flags(df_scored)
            df_scored = attach_volatility(df_scored)
            if enable_fund_filters and not FAST_SCAN:
                df_scored = apply_fundamental_filters(
                    df_scored,
                    min_piotroski=min_piotroski,
                    max_pe=max_pe,
                    max_peg=max_peg,
                    min_eps_g=min_eps_g,
                    min_rs_pct=min_rs_pct,
                )
            df_scored = apply_technical_filters(
                df_scored,
                require_weekly_up=require_weekly_up,
                require_above_50=require_above_50,
                require_above_200=require_above_200,
                require_golden_cross=require_golden_cross,
                min_atr_pct=min_atr_pct,
                max_atr_pct=max_atr_pct,
            )
            if custom_formula.strip():
                df_scored = apply_custom_formula(df_scored, custom_formula)
        except Exception:
            pass

        # Smart Filters (high-level toggles)
        df_scored = apply_smart_filters(
            df_scored,
            only_breakouts=sf_breakouts,
            only_smooth_trends=sf_smooth_trends,
            exclude_choppy=sf_exclude_choppy,
            high_conviction_only=sf_high_conviction,
            low_risk_only=sf_low_risk,
            top_by_sector=sf_top_by_sector,
        )

        # Tradeable filter LAST
        if only_tradeable and "Signal" in df_scored.columns:
            allowed = {"Strong Long", "Long"}
            if allow_shorts:
                allowed |= {"Strong Short", "Short"}
            df_scored = df_scored[df_scored["Signal"].isin(allowed)]

        # ---- FALLBACK LOGIC: always show *something* when possible ---
        if df_scored is None or df_scored.empty:
            fallback = raw_results.copy()

            if "Signal" in fallback.columns:
                tradeable_mask = fallback["Signal"].isin(["Strong Long", "Long"])
                if tradeable_mask.any():
                    fallback = fallback[tradeable_mask]

            if "TechRating" in fallback.columns:
                fallback = fallback.sort_values("TechRating", ascending=False)
            try:
                if not FAST_SCAN:
                    fallback = attach_fundamentals_cols(fallback)
                    fallback = attach_relative_strength(fallback, benchmark=rs_bench)
                fallback = attach_technical_flags(fallback)
                fallback = attach_volatility(fallback)
                if enable_fund_filters and not FAST_SCAN:
                    fallback = apply_fundamental_filters(
                        fallback,
                        min_piotroski=min_piotroski,
                        max_pe=max_pe,
                        max_peg=max_peg,
                        min_eps_g=min_eps_g,
                        min_rs_pct=min_rs_pct,
                    )
                fallback = apply_technical_filters(
                    fallback,
                    require_weekly_up=require_weekly_up,
                    require_above_50=require_above_50,
                    require_above_200=require_above_200,
                    min_atr_pct=min_atr_pct,
                    max_atr_pct=max_atr_pct,
                )
                if custom_formula.strip():
                    fallback = apply_custom_formula(fallback, custom_formula)
            except Exception:
                pass

            try:
                fallback = attach_sparklines(fallback)
            except Exception:
                pass

            fallback = fallback.head(3)

            if fallback.empty:
                st.warning(
                    "No results matched your filters and no fallback recommendations were available."
                )
                results_df = None
                st.session_state["scan_results"] = None
            else:
                if "Signal" not in fallback.columns and "TechRating" in fallback.columns:
                    fallback["Signal"] = "Avoid"
                    fallback.loc[fallback["TechRating"] >= 22, "Signal"] = "Strong Long"
                    fallback.loc[
                        (fallback["TechRating"] >= 16) & (fallback["TechRating"] < 22),
                        "Signal",
                    ] = "Long"

                fallback = fallback.reset_index(drop=True)
                fallback["MatchMode"] = "Relaxed recommendation"
                n_reco = len(fallback)
                st.info(
                    f"No results matched your filters - showing top {n_reco} "
                    f"recommendation{'s' if n_reco != 1 else ''} instead."
                )

                results_df = fallback
                st.session_state["scan_results"] = results_df
        else:
            # Normal, strict-filter results
            try:
                df_scored = attach_sparklines(df_scored)
            except Exception:
                st.warning("Sparkline generation failed for some symbols.", icon="⚠️")

            results_df = df_scored.reset_index(drop=True)
            results_df["MatchMode"] = "Strict filters"

            st.session_state["scan_results"] = results_df

        # Persist status text
        status_msg = status
        st.session_state["scan_status_msg"] = status_msg

# ------------------------------------------------------------------
# RESULTS UI (tabs) – always uses session_state, not run_button
# ------------------------------------------------------------------
results_df: pd.DataFrame | None = st.session_state.get("scan_results")
status_msg: str = st.session_state.get("scan_status_msg", "")
config = st.session_state.get("scan_config")

if results_df is not None and not results_df.empty:
    # Final post-processing: SetupTags + ensure sparklines exist
    try:
        results_df = add_setup_tags_column(results_df)
    except Exception as e:
        st.warning(f"Setup tag generation failed: {e}")

    if "Sparkline" not in results_df.columns:
        try:
            results_df = attach_sparklines(results_df)
        except Exception as e:
            st.warning(f"Sparkline generation failed: {e}", icon="⚠️")

    # Cache back (for next rerun)
    st.session_state["scan_results"] = results_df

    # Ensure we have a selected symbol
    symbols_list = results_df["Symbol"].astype(str).tolist()
    if (
        "selected_symbol" not in st.session_state
        or st.session_state["selected_symbol"] not in symbols_list
    ):
        st.session_state["selected_symbol"] = symbols_list[0]

    # --- Main views / tabs (stateful) ------------------------------
    tab_names = [
        "Scan Results",
        "Symbol Detail",
        "Market Insight",
        "News Hub",
        "Backtester",
        "Scoreboard",
        "Quant Copilot",
    ]

    # Initialize active tab once
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = tab_names[0]

    # Radio-based tab selector that remembers the last choice
    tab_container = st.container()
    with tab_container:
        st.markdown('<div class="technic-tabs-container">', unsafe_allow_html=True)
        active_tab = st.radio(
            "View",
            tab_names,
            index=tab_names.index(st.session_state["active_tab"])
            if st.session_state["active_tab"] in tab_names
            else 0,
            horizontal=True,
            key="active_tab_radio",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state["active_tab"] = active_tab

    # Helper to fetch active row (used by Symbol Detail / Copilot)
    def _get_selected_row() -> pd.Series:
        sym = st.session_state.get("selected_symbol")
        if sym is None:
            return results_df.iloc[0]
        rows = results_df[results_df["Symbol"].astype(str) == str(sym)]
        if rows.empty:
            return results_df.iloc[0]
        return rows.iloc[0]

    # --- View 1: Scan Results ---------------------------------------
    if active_tab == "Scan Results":
        st.markdown('<div class="technic-card">', unsafe_allow_html=True)
        st.markdown("### Ranked setups")

        # Status line
        col_status_left, col_status_right = st.columns([3, 1])
        with col_status_left:
            if status_msg:
                st.caption(status_msg)
            last_ts = st.session_state.get("technic_last_updated")
            if last_ts is not None:
                st.caption(f"Last updated: {last_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # --- Personal Quant trade ideas (top of screen) -------------
        trade_cols = {
            "EntryPrice",
            "StopPrice",
            "TargetPrice",
            "RewardRisk",
            "PositionSize",
            "Signal",
            "Symbol",
        }
        if trade_cols.issubset(set(results_df.columns)):
            tradeable = results_df[
                results_df["Signal"].isin(
                    ["Strong Long", "Long", "Strong Short", "Short"]
                )
            ].copy()
            if tradeable.empty:
                tradeable = results_df.copy()

            top_ideas = tradeable.sort_values(
                "TechRating", ascending=False
            ).head(3)

            st.markdown("#### Personal Quant trade ideas")
            st.caption("Ready-to-trade snapshots tuned to your risk settings.")
            cols = st.columns(len(top_ideas)) if len(top_ideas) > 0 else []
            for col, (_, idea) in zip(cols, top_ideas.iterrows()):
                with col:
                    st.markdown(render_trade_idea_card(idea), unsafe_allow_html=True)
            if top_ideas.empty:
                st.write("No tradeable ideas yet - adjust filters or rerun the scan.")

        # Focus symbol selector + Jump-to-Copilot
        focus_col_left, focus_col_right = st.columns([3, 1])
        with focus_col_left:
            current_symbol = st.session_state.get("selected_symbol", symbols_list[0])
            focus_symbol = st.selectbox(
                "Focus symbol",
                options=symbols_list,
                index=symbols_list.index(str(current_symbol))
                if str(current_symbol) in symbols_list
                else 0,
                help="Pick which symbol to inspect in Symbol Detail / Quant Copilot.",
            )
            st.session_state["selected_symbol"] = focus_symbol

        with focus_col_right:
            if st.button("Explain in Quant Copilot", use_container_width=True):
                # Jump to Copilot tab for the currently selected symbol
                st.session_state.pop("active_tab_radio", None)
                st.session_state["active_tab"] = "Quant Copilot"
                try:
                    st.rerun()
                except Exception:
                    pass

        # Alerts status: show any pending alerts and check triggers
        alerts = st.session_state.get("price_alerts", [])
        if alerts:
            st.markdown("##### Alerts")
            hit_msgs = []
            for alert in alerts:
                if alert.get("triggered"):
                    continue
                sym = alert["symbol"]
                trig = alert["price"]
                direction = alert["direction"]
                live_px = _get_live_price(sym, df_hint=results_df)
                if live_px is None:
                    continue
                if (direction == "above" and live_px >= trig) or (direction == "below" and live_px <= trig):
                    alert["triggered"] = True
                    hit_msgs.append(f"{sym} hit {trig:.2f} ({live_px:.2f})")
            if hit_msgs:
                for msg in hit_msgs:
                    st.success(f"Alert: {msg}")
            pending = [a for a in alerts if not a.get("triggered")]
            if pending:
                st.caption(
                    "Active alerts: "
                    + ", ".join([f"{a['symbol']} {a['direction']} {a['price']:.2f}" for a in pending])
                )
            else:
                st.caption("No active alerts.")

        # Mini backtest (beta)
        with st.expander("Backtest (beta)", expanded=False):
            st.caption("Lightweight: uses Entry/Stop/Target vs daily closes; exits on target/stop or after N days.")
            bt_hold = st.slider("Max hold (trading days)", min_value=5, max_value=40, value=20, step=5, key="bt_hold_days")
            bt_slip = st.slider("Slippage (%)", min_value=0.0, max_value=2.0, value=0.0, step=0.1, key="bt_slip")
            bt_comm = st.number_input("Commission (R units)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="bt_comm")
            use_dollars = st.checkbox("Show $ PnL (risk per trade)", value=False, key="bt_dollars_toggle")
            risk_dollars = None
            if use_dollars:
                risk_dollars = st.number_input("Risk per trade ($)", min_value=10.0, max_value=100000.0, value=1000.0, step=50.0, key="bt_risk_dollars")
            if st.button("Run backtest on current results", key="run_bt"):
                bt_df = results_df.copy()
                bt = run_simple_backtest(bt_df, hold_days=bt_hold, slippage_pct=bt_slip / 100.0, commission_r=bt_comm, risk_per_trade=risk_dollars)
                if not bt or bt.get("trades", 0) == 0:
                    st.info("No trades could be tested (missing prices or plan levels).")
                else:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Trades", bt["trades"])
                    with c2:
                        st.metric("Win rate", f"{bt["win_rate"]:.1f}%")
                    with c3:
                        st.metric("Avg R", f"{bt["avg_r"]:.2f}")
                    if risk_dollars is not None and "pnl_sum" in bt:
                        st.caption(f"Total PnL:  | Avg/trade: ")
                    if HAVE_ALTAIR and "equity" in bt and len(bt["equity"]) > 1:
                        eq = pd.DataFrame({"Step": range(len(bt["equity"])), "Equity": bt["equity"]})
                        eq_chart = (
                            alt.Chart(eq)
                            .mark_line(color=BRAND_PRIMARY, strokeWidth=2)
                            .encode(x="Step:Q", y="Equity:Q")
                            .properties(height=120)
                        )
                        st.altair_chart(eq_chart, use_container_width=True)

        # Styled results table
        # Styled results table
        # Styled results table
        view_mode = st.radio(
            "Table view",
            ["Essential", "Full metrics"],
            index=0,
            horizontal=True,
            key="results_table_mode",
        )
        with st.expander("?? Visual views (optional)", expanded=False):
            st.caption("Use these only when you want a quick visual scan; default table stays clean.")
            heatmap_toggle = st.checkbox("Heatmap (sector x symbol)", value=False, key="heatmap_toggle")
            grid_toggle = st.checkbox("Multi-chart grid (top 6)", value=False, key="grid_toggle")
        card_mode = st.checkbox(
            "Compact cards (mobile-friendly)",
            value=True if st.session_state.get("active_tab") == "Scan Results" else False,
            help="Shows condensed cards instead of a wide grid.",
        )
        auto_cards = st.checkbox(
            "Auto card mode (mobile)",
            value=st.session_state.get("auto_cards", True),
            help="Automatically favor cards to avoid horizontal scrolling.",
        )
        force_cards = st.checkbox(
            "Force cards on small screens",
            value=st.session_state.get("force_cards", True),
            help="Always use cards instead of the grid.",
        )
        compact_table = st.checkbox(
            "Compact table rows",
            value=True,
            help="Tighter font/padding for smaller screens.",
        )
        st.session_state["force_cards"] = force_cards
        st.session_state["auto_cards"] = auto_cards
        sort_by = st.selectbox(
            "Sort by",
            options=["CompositeScore", "TechRating", "RS_pct", "RiskScore", "PE"],
            index=0,
            help="Choose a column to sort results.",
        )

        essential_order = [
            "Symbol",
            "Signal",
            "MatchMode",
        "TechRating",
        "CompositeScore",
        "RS_pct",
        "PE",
        "PEG",
        "Piotroski",
        "AltmanZ",
        "EPS_Growth",
            "RiskScore",
            "RewardRisk",
            "EntryPrice",
            "StopPrice",
            "TargetPrice",
            "PositionSize",
            "Sector",
            "Industry",
            "Sparkline",
        ]
        essential_cols = [c for c in essential_order if c in results_df.columns]
        remaining_cols = [c for c in results_df.columns if c not in essential_cols]

        table_df = (
            results_df[essential_cols + remaining_cols]
            if view_mode == "Essential"
            else results_df
        )

        if sort_by in table_df.columns:
            table_df = table_df.sort_values(sort_by, ascending=False, na_position="last")

        if heatmap_toggle:
            st.markdown("#### Heatmap")
            render_results_heatmap(table_df, color_col="CompositeScore" if "CompositeScore" in table_df.columns else "TechRating")

        if grid_toggle:
            st.markdown("#### Chart grid (spark overview)")
            top_syms = table_df["Symbol"].astype(str).head(6).tolist()
            render_multi_chart_grid(top_syms, days=90, cols=3)

        use_cards = force_cards or card_mode or auto_cards

        if use_cards:
            max_cards = 20
            opt_teaser_limit = 5
            trade_style_label = _current_trade_style()
            st.caption(f"Showing top {min(len(table_df), max_cards)} setups in card view.")
            for idx, (_, r) in enumerate(table_df.head(max_cards).iterrows()):
                sym = r.get("Symbol", "-")
                sig = r.get("Signal", "-")
                sector = r.get("Sector", "-")
                tech = r.get("TechRating", None)
                risk = r.get("RiskScore", None)
                comp = r.get("CompositeScore", None)
                rs = r.get("RS_pct", None)
                entry = r.get("EntryPrice", None)
                stop = r.get("StopPrice", None)
                target = r.get("TargetPrice", None)
                rr = r.get("RewardRisk", None)

                opt_teaser = ""
                show_opt_badge = False
                if (
                    idx < opt_teaser_limit
                    and POLYGON_API_KEY
                    and fetch_option_recos is not None
                ):
                    direction = "put" if isinstance(sig, str) and "Short" in sig else "call"
                    underlying_px = _get_live_price(sym) or _float_or_none(r.get("Last")) or _float_or_none(r.get("Close")) or _float_or_none(entry)
                    try:
                        if underlying_px is not None:
                            opt_payload = fetch_option_recos(
                                symbol=str(sym),
                                direction=direction,
                                trade_style=trade_style_label,
                                underlying=float(underlying_px),
                                tech_rating=_float_or_none(tech),
                                risk_score=_float_or_none(risk),
                                price_target=_float_or_none(target),
                                signal=str(sig),
                            )
                            picks = opt_payload.get("picks") or []
                            if picks:
                                p = picks[0]
                                show_opt_badge = True
                                delta_val = p.get("delta")
                                spread_val = p.get("spread_pct")
                                delta_txt = f"{delta_val:.2f}" if isinstance(delta_val, (int, float)) else "N/A"
                                spread_txt = (
                                    f"{spread_val*100:.1f}%"
                                    if isinstance(spread_val, (int, float))
                                    else "N/A"
                                )
                                opt_teaser = (
                                    f"<div style='margin-top:6px;padding:6px 8px;border-radius:10px;"
                                    "background:rgba(148,163,184,0.08);font-size:0.85rem;'>"
                                    f"<span style='font-weight:700;color:{'#16a34a' if direction=='call' else '#ef4444'};'>"
                                    f"{p.get('ticker','')}</span> "
                                    f"{p.get('contract_type','').upper()} "
                                    f"${p.get('strike','')} exp {p.get('expiration','')} "
                                    f"Δ {delta_txt} | spread {spread_txt}"
                                    "</div>"
                                )
                    except Exception:
                        opt_teaser = ""

                st.markdown(
                    f"""
                    <div class="technic-card" style="margin-bottom:0.6rem;padding:0.8rem 1rem;">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-weight:800;font-size:1.1rem;display:flex;align-items:center;gap:8px;">
                          <span>{sym}</span>{'<span style="font-size:0.75rem;padding:2px 8px;border-radius:999px;background:rgba(22,163,74,0.12);color:#22c55e;font-weight:700;">Options</span>' if show_opt_badge else ''}
                        </div>
                        <div style="font-weight:700;padding:4px 8px;border-radius:12px;background:rgba(158,240,26,0.15);color:#9ef01a;">{sig}</div>
                      </div>
                      <div style="display:flex;flex-wrap:wrap;gap:0.7rem;font-size:0.9rem;margin-top:0.35rem;">
                        <div>TechRating: <strong>{'' if pd.isna(tech) else f'{tech:.1f}'}</strong></div>
                        <div>RiskScore: <strong>{'' if pd.isna(risk) else f'{risk:.2f}'}</strong></div>
                        <div>Composite: <strong>{'' if pd.isna(comp) else f'{comp:.1f}'}</strong></div>
                        <div>RS pct: <strong>{'' if pd.isna(rs) else f'{rs:.0f}%'}</strong></div>
                        <div>R:R: <strong>{'' if pd.isna(rr) else f'{rr:.1f}:1'}</strong></div>
                        <div>Sector: <strong>{sector}</strong></div>
                      </div>
                      <div style="display:flex;flex-wrap:wrap;gap:1rem;font-size:0.9rem;margin-top:0.35rem;">
                        <div>Entry: <strong>{'' if pd.isna(entry) else f'{entry:.2f}'}</strong></div>
                        <div>Stop: <strong>{'' if pd.isna(stop) else f'{stop:.2f}'}</strong></div>
                        <div>Target: <strong>{'' if pd.isna(target) else f'{target:.2f}'}</strong></div>
                      </div>
                      {opt_teaser}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            styled = style_results_table(table_df, compact=compact_table)
            st.dataframe(
                styled,
                use_container_width=True,
                height=580,
            )

        # Bottom re-run button
        if st.button("Re-run scan with current settings", key="rerun_scan_bottom"):
            st.session_state["trigger_scan"] = True
            st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # --- View 2: Symbol Detail --------------------------------------
    elif active_tab == "Symbol Detail":
        row = _get_selected_row()
        render_symbol_detail_panel(row)

    # --- View 3: Market Insight ------------------------------------
    elif active_tab == "Market Insight":
        row = _get_selected_row()
        st.markdown('<div class="technic-card">', unsafe_allow_html=True)
        st.markdown("### Market insight")

        sector = str(row.get("Sector", "Unknown") or "Unknown")
        industry = str(row.get("Industry", "Unknown") or "Unknown")

        peers = results_df[results_df["Sector"] == sector] if "Sector" in results_df.columns else results_df
        peer_count = len(peers)

        def _pct_rank(series, value):
            if series is None or series.empty or pd.isna(value):
                return float("nan")
            try:
                series = series.dropna()
                if series.empty:
                    return float("nan")
                ranks = series.rank(pct=True, method="average")
                closest_idx = (series - value).abs().idxmin()
                return float(ranks.loc[closest_idx] * 100.0)
            except Exception:
                return float("nan")

        tech_val = float(row.get("TechRating", float("nan")))
        tech_rank = _pct_rank(peers["TechRating"], tech_val) if "TechRating" in peers.columns else float("nan")

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Sector", sector)
        with col_m2:
            st.metric("Industry", industry)
        with col_m3:
            st.metric("Peers in view", peer_count)

        tech_val_txt = f"{tech_val:.1f}" if pd.notna(tech_val) else "N/A"
        tech_rank_txt = f"{tech_rank:.0f}%" if pd.notna(tech_rank) else "N/A"

        col_m4, col_m5 = st.columns(2)
        with col_m4:
            st.metric("TechRating", tech_val_txt)
        with col_m5:
            st.metric(
                "TechRating percentile",
                tech_rank_txt,
                help="Percentile vs names in the same sector in this scan.",
            )

        st.markdown("#### How it stacks up")
        bullet = []
        if pd.notna(tech_rank):
            bullet.append(f"TechRating percentile within sector: {tech_rank_txt}.")
        if "Signal" in row:
            bullet.append(f"Signal: **{row['Signal']}**.")
        if "RewardRisk" in row and pd.notna(row["RewardRisk"]):
            bullet.append(f"Planned reward:risk of **{row['RewardRisk']:.1f}:1**.")
        if not bullet:
            bullet.append("No peer context available yet - run a broader scan.")
        for b in bullet:
            st.markdown(f"- {b}")

        # Optional Copilot narrative using current row + peer stats
        prompt = (
            "Compare this symbol to its sector peers in the current scan. "
            f"Sector: {sector}. Industry: {industry}. "
            f"TechRating: {tech_val_txt}. "
            f"Approx. percentile within sector: {tech_rank_txt}. "
            "Highlight whether its trend, momentum, and risk are stronger or weaker than the group. "
            "Keep it concise and plain English."
        )
        if st.button("Ask Copilot for sector context", use_container_width=True, key="copilot_sector_context"):
            with st.spinner("Consulting Copilot for market context..."):
                try:
                    insight = generate_copilot_answer(prompt, row)
                    st.write(insight)
                except Exception as e:
                    st.error(f"Copilot context error: {e}")

        st.markdown("#### Latest news for this symbol")
        symbol = str(row.get("Symbol", "")).upper()
        news_items = fetch_symbol_news(symbol)
        if not news_items:
            st.write("No recent headlines available right now.")
        else:
            for item in news_items:
                title = item.get("title") or "Untitled"
                source = item.get("source") or "News"
                published = item.get("published") or ""
                url = item.get("url") or ""
                if url:
                    st.markdown(
                        f"- [{title}]({url}) — {source} · {published}"
                    )
                else:
                    st.markdown(f"- {title} — {source} · {published}")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- View 4: News Hub -----------------------------------------
    elif active_tab == "News Hub":
        ensure_scoreboard_state()
        st.markdown('<div class="technic-card">', unsafe_allow_html=True)
        st.markdown("### News hub")
        st.caption("Fresh headlines for your top scan names and tracked trades.")

        # Build symbol list: top 5 by TechRating + tracked symbols
        symbols_for_news: list[str] = []
        if results_df is not None and "Symbol" in results_df.columns:
            top_syms = (
                results_df.sort_values("TechRating", ascending=False)["Symbol"]
                .astype(str)
                .head(5)
                .tolist()
            )
            symbols_for_news.extend(top_syms)

        sb_df = build_scoreboard_df(st.session_state.get("scoreboard_entries", []))
        if not sb_df.empty and "Symbol" in sb_df.columns:
            symbols_for_news.extend(sb_df["Symbol"].dropna().astype(str).tolist())

        symbols_for_news = [s for s in dict.fromkeys(symbols_for_news)]  # dedupe

        if not symbols_for_news:
            st.info("No symbols available yet. Run a scan or add to the scoreboard.")
        else:
            news_items = fetch_bulk_news(symbols_for_news, per_symbol=3, max_items=30)
            if not news_items:
                st.write("No recent headlines for these symbols.")
            else:
                for item in news_items:
                    title = item.get("title") or "Untitled"
                    source = item.get("source") or "News"
                    published = item.get("published") or ""
                    url = item.get("url") or ""
                    sym = item.get("symbol", "")
                    prefix = f"[{sym}] " if sym else ""
                    if url:
                        st.markdown(f"- {prefix}[{title}]({url}) — {source} · {published}")
                    else:
                        st.markdown(f"- {prefix}{title} — {source} · {published}")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- View 5: Scoreboard ----------------------------------------
    elif active_tab == "Backtester":
        st.markdown('<div class="technic-card">', unsafe_allow_html=True)
        st.markdown("### Strategy simulator (MA crossover)")
        st.caption(
            "Simple moving-average crossover backtest (long-only). "
            "Enter when fast MA crosses above slow; exit when it crosses below."
        )

        symbol_opts = results_df["Symbol"].astype(str).tolist() if results_df is not None else []
        if not symbol_opts:
            st.info("Run a scan first to pick a symbol.")
        else:
            col_bt1, col_bt2, col_bt3 = st.columns([2, 1, 1])
            with col_bt1:
                bt_symbol = st.selectbox(
                    "Symbol",
                    options=symbol_opts,
                    index=0,
                    key="bt_symbol",
                )
                bt_days = st.slider(
                    "Lookback (days)",
                    min_value=90,
                    max_value=1095,
                    value=365,
                    step=15,
                )
            with col_bt2:
                ma_fast = st.number_input(
                    "Fast MA window",
                    min_value=5,
                    max_value=100,
                    value=20,
                    step=1,
                )
                initial_capital = st.number_input(
                    "Initial capital ($)",
                    min_value=1000.0,
                    max_value=1_000_000.0,
                    value=10_000.0,
                    step=500.0,
                )
            with col_bt3:
                ma_slow = st.number_input(
                    "Slow MA window",
                    min_value=20,
                    max_value=250,
                    value=100,
                    step=5,
                )

            if ma_fast >= ma_slow:
                st.error("Fast MA must be smaller than Slow MA.")
            else:
                if st.button("Run backtest", use_container_width=True, key="run_backtest"):
                    with st.spinner("Simulating..."):
                        equity_df, metrics = run_ma_backtest(
                            bt_symbol,
                            bt_days,
                            ma_fast,
                            ma_slow,
                            initial_capital,
                        )
                    if equity_df is None:
                        st.error(metrics.get("error", "Backtest failed."))
                    else:
                        # Metrics
                        mcols = st.columns(5)

                        def fmt_pct(val):
                            return f"{val*100:.1f}%" if val is not None else "N/A"

                        mcols[0].metric("Total return", fmt_pct(metrics["total_return"]))
                        mcols[1].metric("Buy & Hold", fmt_pct(metrics["buyhold_return"]))
                        mcols[2].metric("CAGR", fmt_pct(metrics["cagr"]))
                        mcols[3].metric("Max drawdown", fmt_pct(metrics["max_dd"]))
                        sharpe_txt = f"{metrics['sharpe']:.2f}" if metrics["sharpe"] is not None else "N/A"
                        mcols[4].metric("Sharpe (daily 252)", sharpe_txt)

                        mcols2 = st.columns(3)
                        mcols2[0].metric("Trades", metrics["trades"])
                        mcols2[1].metric(
                            "Win rate",
                            fmt_pct((metrics["win_rate"] or 0) / 100)
                            if metrics["win_rate"] is not None
                            else "N/A",
                        )
                        mcols2[2].metric(
                            "Avg trade",
                            f"{metrics['avg_trade']:.1f}%"
                            if metrics["avg_trade"] is not None
                            else "N/A",
                        )

                        # Chart
                        if HAVE_ALTAIR:
                            chart_df = equity_df.rename(
                                columns={"Date": "Date", "Equity": "Strategy", "BuyHold": "Buy & Hold"}
                            )
                            chart = (
                                alt.Chart(chart_df)
                                .transform_fold(["Strategy", "Buy & Hold"], as_=["Series", "Value"])
                                .mark_line()
                                .encode(
                                    x="Date:T",
                                    y="Value:Q",
                                    color=alt.Color(
                                        "Series:N",
                                        scale=alt.Scale(
                                            domain=["Strategy", "Buy & Hold"],
                                            range=[BRAND_PRIMARY, "#9ca3af"],
                                        ),
                                    ),
                                )
                                .properties(height=360)
                            )
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.line_chart(
                                equity_df.set_index("Date")[["Equity", "BuyHold"]],
                                height=360,
                                use_container_width=True,
                            )

                        st.success("Backtest complete.")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- View 5: Scoreboard ----------------------------------------
    elif active_tab == "Scoreboard":
        ensure_scoreboard_state()
        sb_df = build_scoreboard_df(st.session_state["scoreboard_entries"])
        stats = summarize_scoreboard(sb_df)

        # Show sync status
        sync_status = "Local"
        if supabase_enabled():
            sync_status = "Supabase"
        elif SCOREBOARD_SYNC_URL:
            sync_status = "Remote URL"
        st.caption(f"Sync mode: {sync_status}")

        # Start realtime stream for tracked symbols (best-effort)
        if start_realtime_stream and POLYGON_API_KEY:
            symbols = set(sb_df["Symbol"].dropna().astype(str).tolist()) if not sb_df.empty else set()
            # also include currently selected focus symbol
            focus_sym = st.session_state.get("selected_symbol")
            if focus_sym:
                symbols.add(str(focus_sym))
            try:
                start_realtime_stream(symbols, POLYGON_API_KEY)
            except Exception:
                pass

        st.markdown('<div class="technic-card">', unsafe_allow_html=True)
        st.markdown("### Scoreboard")
        st.caption(
            "Tracks the setups you add from Symbol Detail. "
            "Status is based on live prices vs entry/stop/target. "
            "Use refresh to pull the latest ticks or turn on auto-refresh."
        )

        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            auto_refresh = st.checkbox(
                "Auto-refresh", value=False, key="scoreboard_auto_refresh"
            )
        with col_ctrl2:
            refresh_interval = st.slider(
                "Refresh interval (sec)",
                min_value=10,
                max_value=120,
                value=30,
                step=5,
                key="scoreboard_refresh_interval",
                help="How often to pull latest prices when auto-refresh is on.",
            )

        if auto_refresh:
            st_autorefresh = getattr(st, "autorefresh", None) or getattr(
                st, "experimental_rerun", None
            )
            if st_autorefresh and st_autorefresh != st.experimental_rerun:
                st.autorefresh(interval=refresh_interval * 1000, key="sb_autorefresh")
            else:
                # Fallback: manual rerun trigger based on time input
                st.experimental_rerun()

        refresh_now = st.button("Refresh prices now", key="scoreboard_refresh")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Tracked", stats["total"])
        with m2:
            st.metric("Open", stats["open"])
        with m3:
            st.metric("Closed", stats["closed"], help="Target hit + stopped out")
        with m4:
            sr = stats["success_rate"]
            sr_txt = f"{sr:.0f}%" if sr is not None else "N/A"
            st.metric("Win rate", sr_txt)

        ctrl1, ctrl2 = st.columns(2)
        with ctrl1:
            if st.button(
                "Reload from cloud/disk",
                use_container_width=True,
                help="Pulls from SCOREBOARD_SYNC_URL if set, otherwise from local file.",
            ):
                st.session_state["scoreboard_entries"] = load_scoreboard_from_disk()
                st.success("Scoreboard reloaded.")
        with ctrl2:
            if st.button("Clear scoreboard", use_container_width=True):
                st.session_state["scoreboard_entries"] = []
                persist_scoreboard_to_disk([])
                sync_scoreboard_to_remote([])
                st.success("Scoreboard cleared.")

        if sb_df.empty:
            st.info("No trades tracked yet. Add one from the Symbol Detail tab.")
        else:
            display_df = sb_df.copy()
            display_df["Entry"] = display_df["Entry"].map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "-"
            )
            display_df["Stop"] = display_df["Stop"].map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "-"
            )
            display_df["Target"] = display_df["Target"].map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "-"
            )
            display_df["Last"] = display_df["Last"].map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "-"
            )
            display_df["PnL%"] = display_df["PnL%"].map(
                lambda v: f"{v:+.2f}%" if pd.notna(v) else "-"
            )
            st.dataframe(display_df, use_container_width=True, height=480)

        # Curated headlines for tracked symbols (top 3)
        if not sb_df.empty:
            st.markdown("#### Headlines for your tracked symbols")
            unique_syms = [s for s in sb_df["Symbol"].dropna().unique().tolist() if s]
            for sym in unique_syms[:3]:
                st.markdown(f"**{sym}**")
                items = fetch_symbol_news(sym)
                if not items:
                    st.write("No recent headlines.")
                    continue
                for item in items[:3]:
                    title = item.get("title") or "Untitled"
                    source = item.get("source") or "News"
                    published = item.get("published") or ""
                    url = item.get("url") or ""
                    if url:
                        st.markdown(f"- [{title}]({url}) — {source} · {published}")
                    else:
                        st.markdown(f"- {title} — {source} · {published}")

        st.markdown("</div>", unsafe_allow_html=True)

        if refresh_now:
            st.experimental_rerun()

    # --- View 6: Quant Copilot --------------------------------------
    elif active_tab == "Quant Copilot":
        focus_symbol = st.session_state.get("selected_symbol")
        if focus_symbol is not None and "Symbol" in results_df.columns:
            try:
                row = results_df[
                    results_df["Symbol"].astype(str) == str(focus_symbol)
                ].iloc[0]
            except IndexError:
                row = results_df.iloc[0]
        else:
            row = results_df.iloc[0]

        st.markdown('<div class="technic-card">', unsafe_allow_html=True)
        st.markdown("### Quant Copilot (beta)")
        st.caption(
            "Ask for an explanation of this setup. "
            "Copilot combines Technic's scores with a large language model "
            "to give you a human-readable playbook."
        )

        user_q = st.text_area(
            "Question for Copilot:",
            placeholder=(
                "Examples:\n"
                "- Why is this symbol a Strong Long instead of Avoid?\n"
                "- How could someone size a position here given the RiskScore?\n"
                "- What are the main strengths and weaknesses of this setup?"
            ),
            key="copilot_question",
        )

        ask_clicked = st.button(
            "Ask Copilot",
            key="copilot_ask",
            use_container_width=True,
        )

        base_explanation = build_narrative_from_row(row)
        conditions = build_conditions(row)

        llm_answer = None
        default_q = (
            "Using only the scanner metrics and trade-plan fields "
            "(entry, stop, target, reward/risk, position size), "
            "outline an educational example trade plan for this symbol, "
            "including direction, entry zone, stop placement, target logic, "
            "and key risks."
        )
        effective_q = user_q.strip() or default_q

        if ask_clicked:
            with st.spinner("Thinking like a quant..."):
                try:
                    llm_answer = generate_copilot_answer(effective_q, row)
                    st.session_state["copilot_last_q"] = (
                        user_q.strip() or "[Auto] Example trade plan"
                    )
                    st.session_state["copilot_last_answer"] = llm_answer
                except Exception as e:
                    st.error(f"Copilot error: {e}")
                    llm_answer = None

        if not llm_answer:
            llm_answer = st.session_state.get("copilot_last_answer")
            user_q = st.session_state.get("copilot_last_q", "")

        if user_q.strip():
            st.markdown(f"**Your question:** {user_q.strip()}")

        st.markdown("#### Copilot view of this setup")

        if llm_answer:
            st.write(llm_answer)
            st.markdown("---")
            st.markdown(
                "_Below is the transparent explanation derived directly from "
                "the Technic model, so you can see exactly what the engine is seeing._"
            )

        st.write(base_explanation)

        st.markdown("##### Key technical drivers")
        for label, text in conditions.items():
            st.markdown(f"- **{label}:** {text}")

        st.markdown(
            "##### Positioning & risk\n"
            "Use the entry, stop, and target from the Personal Quant Plan. "
            "If you want to size more conservatively, lean on **RiskScore** and "
            "**volatility**: higher RiskScore and calmer volatility justify larger "
            "size; the opposite suggests scaling down."
        )

        st.markdown("</div>", unsafe_allow_html=True)

