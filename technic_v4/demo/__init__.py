"""
Demo Mode Module

Provides predetermined, high-quality results for investor demonstrations.
Ensures consistent, impressive output every time.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Demo mode flag - can be set via environment variable
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# Path to demo data files
DEMO_DIR = Path(__file__).parent


def is_demo_mode() -> bool:
    """Check if demo mode is enabled."""
    return DEMO_MODE or os.getenv("DEMO_MODE", "false").lower() == "true"


def load_demo_scan_results() -> Dict[str, Any]:
    """Load predetermined scan results for demo and transform to API schema."""
    demo_file = DEMO_DIR / "demo_scan_results.json"
    with open(demo_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Transform field names to match API schema
    # API expects: ticker, rrr, delta, note, isPositive, title, meta, plan
    # Demo has: symbol, rewardRisk, change, changePct, idea, rationale
    
    # Transform results
    for result in data.get('results', []):
        if 'symbol' in result:
            result['ticker'] = result.pop('symbol')
        if 'rewardRisk' in result:
            rr = result.pop('rewardRisk')
            result['rrr'] = f"R:R {rr:.2f}"
    
    # Transform movers
    for mover in data.get('movers', []):
        if 'symbol' in mover:
            mover['ticker'] = mover.pop('symbol')
        if 'change' not in mover:
            mover['delta'] = mover.get('changePct', 0.0)
        else:
            mover['delta'] = mover.pop('change')
        if 'signal' in mover:
            mover['note'] = mover.pop('signal')
        if 'isPositive' not in mover:
            mover['isPositive'] = mover.get('delta', 0) >= 0
    
    # Transform ideas
    for idea in data.get('ideas', []):
        if 'symbol' in idea:
            idea['ticker'] = idea.pop('symbol')
        if 'idea' in idea:
            idea['title'] = idea.pop('idea')
        if 'rationale' in idea:
            # Split rationale into meta and plan
            rationale = idea.pop('rationale')
            idea['meta'] = rationale[:100] + "..." if len(rationale) > 100 else rationale
            idea['plan'] = "See full analysis for entry/exit strategy"
    
    return data


def load_demo_copilot_responses() -> Dict[str, Any]:
    """Load predetermined copilot responses for demo."""
    demo_file = DEMO_DIR / "demo_copilot_responses.json"
    with open(demo_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_demo_copilot_response(question: str) -> str:
    """
    Get a demo copilot response based on the question.
    
    Args:
        question: The user's question
        
    Returns:
        A predetermined, intelligent response
    """
    responses = load_demo_copilot_responses()
    
    # Normalize question for matching
    q_lower = question.lower().strip()
    
    # Match question to predetermined responses
    if "pltr" in q_lower or "palantir" in q_lower:
        return responses["responses"]["why_pltr"]["answer"]
    elif "crwd" in q_lower or "crowdstrike" in q_lower:
        return responses["responses"]["why_crwd"]["answer"]
    elif "coin" in q_lower or "coinbase" in q_lower or ("risk" in q_lower and "coin" in q_lower):
        return responses["responses"]["why_coin"]["answer"]
    elif "top" in q_lower and ("3" in q_lower or "three" in q_lower):
        return responses["responses"]["top_3_picks"]["answer"]
    elif "allocat" in q_lower or "portfolio" in q_lower:
        return responses["responses"]["portfolio_allocation"]["answer"]
    elif "sector" in q_lower:
        return responses["responses"]["sectors_strong"]["answer"]
    elif "risk" in q_lower and "manage" in q_lower:
        return responses["responses"]["risk_management"]["answer"]
    elif "market" in q_lower:
        return responses["responses"]["general"]["answer"]
    else:
        # Default response
        return responses["default_response"]["answer"]


def get_demo_symbol_detail(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get demo symbol detail from scan results.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        
    Returns:
        Symbol detail dict or None if not found
    """
    scan_results = load_demo_scan_results()
    
    for result in scan_results.get("results", []):
        if result.get("symbol", "").upper() == symbol.upper():
            return result
    
    return None


def enable_demo_mode():
    """Enable demo mode programmatically."""
    global DEMO_MODE
    DEMO_MODE = True
    os.environ["DEMO_MODE"] = "true"


def disable_demo_mode():
    """Disable demo mode programmatically."""
    global DEMO_MODE
    DEMO_MODE = False
    os.environ["DEMO_MODE"] = "false"


__all__ = [
    'is_demo_mode',
    'load_demo_scan_results',
    'load_demo_copilot_responses',
    'get_demo_copilot_response',
    'get_demo_symbol_detail',
    'enable_demo_mode',
    'disable_demo_mode',
]
