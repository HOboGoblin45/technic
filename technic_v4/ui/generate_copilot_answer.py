from __future__ import annotations

from typing import Any
import pandas as pd

from openai import OpenAI
from config_secrets import OPENAI_API_KEY


# --- Single shared LLM client for Copilot ------------------------------------
_llm_client: OpenAI | None = None


def get_llm_client() -> OpenAI:
    """
    Lazily create and cache a single OpenAI client for Quant Copilot.
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(api_key=OPENAI_API_KEY.strip())
    return _llm_client


# --- Main Copilot function ----------------------------------------------------


def generate_copilot_answer(question: str, row: "pd.Series | dict | None" = None) -> str:
    """
    Turn the current symbol's metrics + the user's question into a detailed,
    plain-English answer using the OpenAI API.

    The answer is allowed to outline an example trade plan, but must always
    make it clear this is educational information, not personalized advice.
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
        "- Every answer must clearly state that this is an educational example, not "
        '  personalized financial advice, and that the user is responsible for their own decisions.\n'
        "- Do NOT guarantee profits or certainty about future price moves.\n"
        "- Never reference having access to live data, order books, or private information; "
        "  you only see the metrics listed.\n"
        "- Prefer a structured answer with short sections such as: Summary, Example trade plan, "
        "  Why this setup, and Key risks.\n"
    )

    user_msg = f"""
Current symbol: {symbol}

Scanner metrics:
{metrics_block}

User question:
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

    except Exception as exc:  # noqa: BLE001 – show a friendly fallback
        return (
            "Quant Copilot ran into an error while talking to the AI backend. "
            "Please try again in a moment.\n\n"
            f"(Details for developer: {type(exc).__name__}: {exc})"
        )
