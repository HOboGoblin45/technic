"""
Natural-language narrative generator for Copilot/summary.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd


def get_openai_client() -> Any:
    """
    Return an OpenAI client if OPENAI_API_KEY is set; otherwise None.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai

        openai.api_key = api_key
        return openai
    except Exception:
        return None


def build_scan_context_summary(results_df: pd.DataFrame, regime: dict, scoreboard_summary: dict | None) -> dict:
    """
    Build a compact context dict from scan results + regime + scoreboard.
    """
    context = {
        "regime": regime or {},
        "scoreboard": scoreboard_summary or {},
        "top_symbols": [],
    }
    if results_df is not None and not results_df.empty:
        cols = set(results_df.columns)
        for _, row in results_df.head(5).iterrows():
            context["top_symbols"].append(
                {
                    "Symbol": row.get("Symbol"),
                    "TechRating": row.get("TechRating"),
                    "AlphaScore": row.get("AlphaScore"),
                    "Sector": row.get("Sector"),
                    "Explanation": row.get("Explanation"),
                }
            )
    return context


def generate_narrative(context: dict, explanations: Dict[str, str] | None = None) -> str:
    """
    Generate a short narrative for the scan. Uses OpenAI if available; otherwise template fallback.
    """
    client = get_openai_client()
    regime = context.get("regime", {})
    sb = context.get("scoreboard", {})
    top_syms = context.get("top_symbols", [])

    if client is None:
        # Fallback template
        parts = []
        trend = regime.get("trend", "N/A")
        vol = regime.get("vol", "N/A")
        parts.append(f"Market regime: trend={trend}, vol={vol}.")
        if sb:
            parts.append(f"Scoreboard: IC={sb.get('ic')}, Precision@10={sb.get('precision_at_n')}, Hit={sb.get('hit_rate')}")
        if top_syms:
            parts.append("Top symbols:")
            for sym in top_syms:
                expl = explanations.get(sym["Symbol"]) if explanations else sym.get("Explanation")
                parts.append(f"- {sym.get('Symbol')}: TR={sym.get('TechRating')}, Alpha={sym.get('AlphaScore')} ({expl})")
        return " ".join(parts)

    # Use OpenAI chat completion
    sys_prompt = "You are a quantitative trading assistant. Explain signals clearly without making promises."
    user_prompt = (
        "Summarize today's scan for a trader.\n"
        f"Regime: {regime}\n"
        f"Scoreboard: {sb}\n"
        f"Top symbols (with explanations if any): {top_syms}\n"
    )
    try:
        resp = client.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return resp.choices[0].message["content"]
    except Exception:
        return "Could not generate narrative at this time."
