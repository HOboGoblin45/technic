from __future__ import annotations

import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

# Path to your universe file (same folder as this script)
UNIVERSE_FILE = Path(__file__).resolve().with_name("ticker_universe.csv")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip() or None

# -------------------------------------------------------------------
# Keyword maps (you can tune these over time)
# -------------------------------------------------------------------

SECTOR_KEYWORDS: Dict[str, Dict[str, float]] = {
    # sector: {keyword: weight}
    "Health Care": {
        "PHARMACEUTICAL": 3.0,
        "PHARMACEUTICAL PREPARATIONS": 4.0,
        "BIOLOGICAL": 3.0,
        "DIAGNOSTIC": 2.5,
        "HOSPITAL": 2.5,
        "HEALTH": 1.0,
        "MEDICAL": 2.0,
        "OUTPATIENT": 2.0,
        "NURSING": 2.0,
        "SURGICAL": 2.0,
        "DENTAL": 2.0,
        "BIOTECH": 4.0,
        "IN VITRO": 2.0,
        "IN VIVO": 2.0,
        "DRUG": 3.0,
    },
    "Financials": {
        "STATE COMMERCIAL BANKS": 5.0,
        "COMMERCIAL BANKS": 4.0,
        "BANK": 3.0,
        "INSURANCE": 3.0,
        "SECURITIES": 2.5,
        "BROKERAGE": 2.5,
        "FINANCE": 2.0,
        "INVESTMENT": 2.0,
        "CREDIT": 2.0,
        "MORTGAGE": 2.0,
        "SAVINGS": 1.5,
        "REINSURANCE": 3.0,
        "ASSET MANAGEMENT": 3.0,
        "FINANCIAL": 2.0,
    },
    "Information Technology": {
        "SEMICONDUCTOR": 4.0,
        "ELECTRONIC COMPUTERS": 4.0,
        "COMPUTER": 3.0,
        "SOFTWARE": 3.0,
        "PROGRAMMING": 2.0,
        "DATA PROCESSING": 2.0,
        "INFORMATION RETRIEVAL": 2.0,
        "INFORMATION TECHNOLOGY": 3.0,
        "ELECTRONIC": 1.5,
        "INTERNET": 2.5,
        "ONLINE": 2.0,
        "WEB PORTAL": 2.0,
        "COMMUNICATIONS EQUIPMENT": 3.0,
        "TELECOMMUNICATIONS EQUIPMENT": 3.0,
    },
    "Communication Services": {
        "BROADCASTING": 3.0,
        "CABLE": 2.0,
        "TELEVISION": 2.5,
        "RADIO": 2.0,
        "PUBLISHING": 2.0,
        "NEWS": 1.5,
        "MEDIA": 2.0,
        "ENTERTAINMENT": 3.0,
        "MOTION PICTURE": 3.0,
        "ADVERTISING": 2.0,
        "TELEPHONE": 2.0,
        "TELECOMMUNICATION": 2.0,
        "WIRELESS": 2.0,
    },
    "Consumer Discretionary": {
        "RETAIL-": 3.0,
        "RETAIL ": 2.5,
        "AUTO": 1.5,
        "RESTAURANT": 3.0,
        "MOTION PICTURE": 2.0,
        "HOTEL": 2.5,
        "GAMING": 2.5,
        "APPAREL": 2.5,
        "FURNITURE": 2.0,
        "RECREATION": 2.0,
        "LEISURE": 2.0,
        "HOME FURNISHINGS": 2.0,
        "DEPARTMENT STORES": 3.0,
        "CATALOG": 1.5,
        "MAIL ORDER": 1.5,
        "BOOK STORES": 1.5,
        "JEWELRY": 2.0,
        "SHOE": 1.5,
        "MOTOR VEHICLE": 2.0,
        "CRUISE": 2.0,
    },
    "Consumer Staples": {
        "GROCERY": 3.0,
        "FOOD": 2.0,
        "BEVERAGE": 3.0,
        "BREWERY": 3.0,
        "DISTILLER": 3.0,
        "WINERY": 3.0,
        "CIGARETTE": 3.0,
        "TOBACCO": 3.0,
        "HOUSEHOLD PRODUCTS": 2.5,
        "PERSONAL CARE": 2.5,
        "SOAP": 2.0,
        "DAIRY": 2.0,
        "MEAT": 2.0,
        "AGRICULTURAL COMMODITIES": 2.0,
    },
    "Energy": {
        "OIL": 3.0,
        "GAS": 3.0,
        "PETROLEUM": 3.0,
        "COAL": 3.0,
        "PIPELINE": 2.5,
        "DRILLING": 2.5,
        "EXPLORATION": 2.0,
        "REFINING": 2.5,
        "FOSSIL": 2.0,
        "ENERGY": 2.0,
    },
    "Materials": {
        "CHEMICAL": 3.0,
        "PLASTIC": 2.0,
        "RUBBER": 2.0,
        "GLASS": 2.0,
        "PAPER": 2.0,
        "STEEL": 3.0,
        "METAL": 2.5,
        "ALUMINUM": 3.0,
        "FOREST PRODUCTS": 2.0,
        "LUMBER": 2.0,
        "MINING": 3.0,
        "QUARRYING": 2.0,
        "GOLD": 3.0,
        "SILVER": 3.0,
        "COPPER": 3.0,
        "NONMETALLIC MINERALS": 2.5,
    },
    "Industrials": {
        "MACHINERY": 3.0,
        "AEROSPACE": 3.0,
        "DEFENSE": 2.5,
        "CONSTRUCTION": 2.5,
        "ENGINEERING": 2.5,
        "TRUCKING": 2.0,
        "RAILROAD": 2.0,
        "SHIP": 2.0,
        "AIR TRANSPORTATION": 2.5,
        "DELIVERY": 2.0,
        "INDUSTRIAL": 2.0,
        "EQUIPMENT": 2.5,
        "ELECTRIC EQUIPMENT": 2.5,
        "MANUFACTURING": 2.5,
        "PRINTING": 1.5,
        "PACKAGING": 2.0,
        "TRANSPORTATION SERVICES": 2.0,
        "LOGISTICS": 2.5,
    },
    "Real Estate": {
        "REAL ESTATE": 4.0,
        "REIT": 4.0,
        "REITS": 4.0,
        "PROPERTY MANAGEMENT": 3.0,
        "OPERATORS OF APARTMENT": 3.0,
        "MORTGAGE REIT": 4.0,
    },
    "Utilities": {
        "ELECTRIC": 3.0,
        "GAS TRANSMISSION": 3.0,
        "WATER": 2.5,
        "UTILITY": 3.0,
        "UTILITIES": 3.0,
        "POWER": 2.5,
        "SANITARY SERVICES": 2.0,
    },
}

# Example sub-industry keywords: tune as needed
SUBINDUSTRY_KEYWORDS: Dict[str, Dict[str, float]] = {
    "Biotechnology": {"BIOTECH": 4.0, "BIOLOGICAL": 2.5, "GENETIC": 2.5},
    "Pharmaceuticals": {
        "PHARMACEUTICAL": 4.0,
        "DRUG": 3.0,
        "GENERIC DRUGS": 3.0,
    },
    "Health Care Providers & Services": {
        "HOSPITAL": 3.0,
        "OUTPATIENT": 2.5,
        "NURSING": 2.5,
        "HOME HEALTH": 2.5,
        "HEALTH SERVICES": 2.5,
    },
    "Banks": {"STATE COMMERCIAL BANKS": 4.0, "COMMERCIAL BANKS": 3.5, "BANK": 3.0},
    "Asset Management & Custody": {
        "ASSET MANAGEMENT": 3.5,
        "INVESTMENT": 2.5,
        "SECURITIES": 2.5,
        "BROKERAGE": 2.5,
    },
    "Semiconductors": {"SEMICONDUCTOR": 4.0, "INTEGRATED CIRCUITS": 3.0, "CHIPS": 3.0},
    "Software": {"SOFTWARE": 4.0, "PROGRAMMING": 3.0, "SAAS": 3.0},
    "Internet & Direct Marketing Retail": {
        "INTERNET": 3.0,
        "ONLINE": 3.0,
        "WEB PORTAL": 3.0,
        "MAIL ORDER": 2.0,
        "CATALOG": 2.0,
    },
    # You can keep expanding this map.
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^A-Z0-9]+", text.upper()) if t]


def _score_keywords(text: str, keyword_weights: Dict[str, float]) -> float:
    """
    Score a text against a dict of {keyword: weight}.

    - Direct substring match: full weight * 1.0
    - Fuzzy word match (>= 0.88 similarity): full weight * 0.5
    """
    if not text:
        return 0.0

    text_up = text.upper()
    tokens = _tokenize(text_up)
    score = 0.0

    for kw, weight in keyword_weights.items():
        kw_up = kw.upper()

        # Direct substring (highest importance)
        if kw_up in text_up:
            score += weight
            continue

        # Fuzzy token-level match
        for tok in tokens:
            if not tok:
                continue
            if SequenceMatcher(None, tok, kw_up).ratio() >= 0.88:
                score += weight * 0.5
                break

    return score


def _best_match(text: str, keyword_map: Dict[str, Dict[str, float]]) -> Tuple[str, float]:
    best_label = "Unknown"
    best_score = 0.0

    for label, kw_weights in keyword_map.items():
        s = _score_keywords(text, kw_weights)
        if s > best_score:
            best_label, best_score = label, s

    return best_label, best_score


def classify_sector_and_subindustry(
    industry: Optional[str],
    sic_desc: Optional[str],
    existing_sector: Optional[str],
) -> Tuple[str, str]:
    """
    Decide which Sector and SubIndustry to assign.

    Priority:
      1) Keep existing Sector if present and non-empty.
      2) Else, use weighted/fuzzy matches vs SECTOR_KEYWORDS.
      3) SubIndustry always uses keyword scoring (unless nothing matches).
    """
    # Build combined text blob
    parts = []
    if isinstance(industry, str):
        parts.append(industry)
    if isinstance(sic_desc, str):
        parts.append(sic_desc)

    text = " ".join(parts).upper()

    # Sector: keep if already present
    if isinstance(existing_sector, str) and existing_sector.strip():
        sector = existing_sector.strip()
    else:
        sector, sector_score = _best_match(text, SECTOR_KEYWORDS)
        if sector_score == 0.0:
            sector = "Industrials"  # neutral default

    # SubIndustry from text
    sub_industry, sub_score = _best_match(text, SUBINDUSTRY_KEYWORDS)
    if sub_score == 0.0:
        sub_industry = "Other"

    return sector, sub_industry


# -------------------------------------------------------------------
# Optional Polygon enrichment
# -------------------------------------------------------------------

def _polygon_enrich_row(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to get (sector, industry) from Polygon.
    Uses v3 reference/tickers endpoint.

    Returns (sector, industry) or (None, None) on failure.
    """
    if not POLYGON_API_KEY:
        return None, None

    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    params = {"apiKey": POLYGON_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200:
            return None, None
        data = resp.json().get("results", {}) or {}
        sector = data.get("sic_description") or data.get("share_class_title") or None
        industry = data.get("industry") or None
        return sector, industry
    except Exception:
        return None, None


def maybe_enrich_with_polygon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: for rows with missing Industry/SicDescription, try to fetch
    metadata from Polygon. You can comment this out if you don't want the
    network calls / rate limits.
    """
    if not POLYGON_API_KEY:
        print("POLYGON_API_KEY not set; skipping Polygon metadata enrichment.")
        return df

    mask = df["Industry"].isna() & df["SicDescription"].isna()
    missing_symbols = df.loc[mask, "symbol"].astype(str).tolist()

    if not missing_symbols:
        print("No rows missing Industry/SicDescription; skipping Polygon calls.")
        return df

    print(f"Attempting Polygon enrichment for {len(missing_symbols)} symbols...")
    for sym in missing_symbols:
        sector, industry = _polygon_enrich_row(sym)
        if sector or industry:
            idx = df.index[df["symbol"] == sym]
            if len(idx) == 0:
                continue
            if industry:
                df.loc[idx, "Industry"] = industry
            if sector and pd.isna(df.loc[idx, "SicDescription"]).all():
                df.loc[idx, "SicDescription"] = sector
        time.sleep(0.2)  # crude rate limiting

    return df


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"ticker_universe.csv not found at: {UNIVERSE_FILE}")

    df = pd.read_csv(UNIVERSE_FILE)

    # Ensure key columns exist
    for col in ["symbol", "Sector", "Industry", "SicDescription", "SubIndustry"]:
        if col not in df.columns:
            df[col] = None

    # Optional Polygon enrichment step
    df = maybe_enrich_with_polygon(df)

    # Compute Sector + SubIndustry
    results = df.apply(
        lambda row: classify_sector_and_subindustry(
            industry=row.get("Industry"),
            sic_desc=row.get("SicDescription"),
            existing_sector=row.get("Sector"),
        ),
        axis=1,
        result_type="expand",
    )
    df["Sector"] = results[0]
    df["SubIndustry"] = results[1]

    print("Sector distribution after enrichment:")
    print(df["Sector"].value_counts(dropna=False))
    print("\nTop 20 SubIndustries:")
    print(df["SubIndustry"].value_counts().head(20))

    # Backup old file and write the new one
    backup_path = UNIVERSE_FILE.with_suffix(".csv.bak")
    if backup_path.exists():
        backup_path.unlink()  # overwrite old backup
    UNIVERSE_FILE.rename(backup_path)
    print(f"Backed up original to: {backup_path}")

    df.to_csv(UNIVERSE_FILE, index=False)
    print(f"Updated ticker_universe.csv written to: {UNIVERSE_FILE}")


if __name__ == "__main__":
    main()

