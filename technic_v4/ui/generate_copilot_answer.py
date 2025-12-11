from __future__ import annotations

from typing import Any
import os
import pandas as pd

from openai import OpenAI

# Optional local module for secrets; ignored if missing
try:  # noqa: SIM105
    import config_secrets  # type: ignore
except Exception:  # pragma: no cover - best effort import
    config_secrets = None


# --- Single shared LLM client for Copilot ------------------------------------
_llm_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    """
    Lazily create and cache a single OpenAI client for Quant Copilot.
    """
    global _llm_client
    if _llm_client is None:
        # Prefer environment variable; fallback to optional config_secrets module.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and config_secrets and getattr(config_secrets, "OPENAI_API_KEY", None):
            api_key = getattr(config_secrets, "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Set env var or config_secrets.OPENAI_API_KEY.")
        _llm_client = OpenAI(api_key=api_key.strip())
    return _llm_client


# --- Main Copilot function ----------------------------------------------------


def generate_copilot_answer(question: str, row: "pd.Series | dict | None" = None) -> str:
    """
    Turn the current symbol's metrics + the user's question into a detailed,
    plain-English answer using the OpenAI API.

    The answer is allowed to outline an example trade plan, but must always
    make it clear this is educational information, not personalized advice.
    It should reflect institutional signals such as ICS, win_prob_10d, QualityScore,
    and event context when available.
    """

    # Allow calls without a row (e.g., lightweight /api/copilot ping)
    if row is None:
        return (
            "Copilot is online but no symbol context was provided. "
            "Ask a question about a symbol from the scanner so I can tailor the response."
        )

    # Support both pandas Series and plain dict rows
    getter = row.get if hasattr(row, "get") else (lambda k, default=None: default)

    symbol = getter("Symbol", "Unknown")
    techrating = getter("TechRating", None)
    signal = getter("Signal", "Neutral")
    matchmode = getter("MatchMode", "Unknown")
    setuptags = getter("SetupTags", "")
    sector = getter("Sector", "Unknown")
    industry = getter("Industry", "Unknown")
    last = getter("Last", None)
    explanation = getter("Explanation", "")
    iv_rank = getter("iv_rank", getter("IVRank", None))
    regime_trend = getter("RegimeTrend", None)
    regime_vol = getter("RegimeVol", None)

    # Higher-level signals
    ics = getter("InstitutionalCoreScore", None)
    ics_tier = getter("ICS_Tier", getter("Tier", None))
    win_prob_10d = getter("win_prob_10d", None)
    quality = getter("QualityScore", getter("fundamental_quality_score", None))
    playstyle = getter("PlayStyle", None)
    is_ultra_risky = getter("IsUltraRisky", False)
    event_summary = getter("EventSummary", None)
    event_flags = getter("EventFlags", None)
    fundamental_snapshot = getter("FundamentalSnapshot", None)

    # Optional trade-plan fields if present on the row
    entry = getter("EntryPrice", None)
    stop = getter("StopPrice", None)
    target = getter("TargetPrice", None)
    rr = getter("RewardRisk", None)
    size = getter("PositionSize", None)

    # Format metrics cleanly for the prompt
    metrics_lines: list[str] = []
    if last is not None and not pd.isna(last):
        metrics_lines.append(f"- Last price: {last:.2f}")
    if techrating is not None and not pd.isna(techrating):
        metrics_lines.append(f"- TechRating: {techrating:.2f}")
    if ics is not None and not pd.isna(ics):
        metrics_lines.append(f"- InstitutionalCoreScore: {ics:.1f}")
    if ics_tier:
        metrics_lines.append(f"- ICS tier: {ics_tier}")
    if win_prob_10d is not None and not pd.isna(win_prob_10d):
        metrics_lines.append(f"- 10-day win probability (meta): {float(win_prob_10d):.2%}")
    if quality is not None and not pd.isna(quality):
        metrics_lines.append(f"- QualityScore (fundamentals): {float(quality):.1f}")
    if playstyle:
        metrics_lines.append(f"- PlayStyle: {playstyle}")
    if is_ultra_risky:
        metrics_lines.append("- Risk profile: ULTRA-RISKY / speculative")
    metrics_lines.append(f"- Signal: {signal}")
    metrics_lines.append(f"- Match mode: {matchmode}")
    metrics_lines.append(f"- Setup tags: {setuptags or 'None'}")
    metrics_lines.append(f"- Sector / Industry: {sector} / {industry}")

    if entry is not None and not pd.isna(entry):
        metrics_lines.append(f"- Planned entry price: {float(entry):.2f}")
    if stop is not None and not pd.isna(stop):
        metrics_lines.append(f"- Planned stop price: {float(stop):.2f}")
    if target is not None and not pd.isna(target):
        metrics_lines.append(f"- Planned target price: {float(target):.2f}")
    if rr is not None and not pd.isna(rr):
        metrics_lines.append(f"- Reward/Risk multiple: {float(rr):.2f}")
    if size is not None and not pd.isna(size):
        metrics_lines.append(f"- Position size (shares): {int(size)}")

    metrics_block = "\n".join(metrics_lines)

    system_msg = (
        "You are Quant Copilot inside a professional trading scanner app called Technic. "
        "Your job is to interpret the current symbol's technical context and, when asked, "
        "outline an example trade plan based ONLY on the provided scanner metrics.\n\n"
        "Rules:\n"
        "- You may describe an example long/short idea, including entry range, stop, target, "
        "  reward/risk, and notional position size using the metrics you are given.\n"
        "- Use the 'Model drivers' (feature contributions) when explaining why a setup is ranked; "
        "  ground your reasoning strictly in those drivers and the metrics provided.\n"
        "- Treat InstitutionalCoreScore (ICS) and QualityScore as high-level strength/quality gauges; "
        "  win_prob_10d is a probabilistic helper from a meta-model, never a guarantee.\n"
        "- If ICS/QualityScore/win_prob_10d point in different directions, explain the nuance briefly.\n"
        "- If volatility context is provided (IV rank, regime), incorporate it into the rationale and risks.\n"
        "- Every answer must clearly state that this is an educational example, not "
        '  personalized financial advice, and that the user is responsible for their own decisions.\n'
        "- Do NOT guarantee profits or certainty about future price moves.\n"
        "- Never reference having access to live data, order books, or private information; "
        "  you only see the metrics listed.\n"
        "- Prefer a structured answer with short sections such as: Summary, Example trade plan, "
        "  Why this setup, and Key risks.\n"
    )

    # Model drivers (from SHAP or similar) and context
    driver_text = explanation if explanation else "Not available."
    vol_context = []
    if iv_rank is not None and not pd.isna(iv_rank):
        vol_context.append(f"IV rank: {iv_rank}")
    if regime_trend or regime_vol:
        vol_context.append(f"Regime: trend={regime_trend}, vol={regime_vol}")
    vol_block = "; ".join(vol_context) if vol_context else "None provided."

    context_bits: list[str] = []
    if event_summary:
        context_bits.append(f"Events: {event_summary}")
    elif event_flags:
        context_bits.append(f"Event flags: {event_flags}")
    if fundamental_snapshot:
        context_bits.append(f"Fundamentals: {fundamental_snapshot}")
    context_block = ""
    if context_bits:
        context_block = "Context:\n" + "\n".join(f"- {line}" for line in context_bits) + "\n\n"

    user_msg = f"""
Current symbol: {symbol}

Scanner metrics:
{metrics_block}

Model drivers (SHAP-based):
{driver_text}

Volatility/Risk context:
{vol_block}

{context_block}User question:
{question}
""".strip()

    client = get_llm_client()

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=650,
        )

        final_answer = resp.choices[0].message.content.strip()
        return final_answer

    except Exception as exc:  # noqa: BLE001 show a friendly fallback
        return (
            "Quant Copilot ran into an error while talking to the AI backend. "
            "Please try again in a moment.\n\n"
            f"(Details for developer: {type(exc).__name__}: {exc})"
        )
