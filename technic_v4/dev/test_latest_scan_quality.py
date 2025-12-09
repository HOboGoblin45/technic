from pathlib import Path

import pandas as pd


BASE = Path("technic_v4/engine/scanner_output")
MAIN_PATH = BASE / "technic_scan_results.csv"
RUNNERS_PATH = BASE / "technic_runners.csv"


def load_df(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV not found at {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label} CSV is empty: {path}")
    return df


def main() -> None:
    main_df = load_df(MAIN_PATH, "Main results")
    print(f"Loaded main_df with {len(main_df)} rows")

    try:
        runners_df = load_df(RUNNERS_PATH, "Runners")
        print(f"Loaded runners_df with {len(runners_df)} rows")
    except Exception as exc:
        print(f"WARNING: Could not load runners_df: {exc}")
        runners_df = pd.DataFrame()

    # ---------- HARD INSTITUTIONAL CONSTRAINTS ON MAIN LIST ----------

    # 1) Price >= 5
    if "Close" in main_df.columns:
        assert (main_df["Close"] >= 5.0).all(), "Found names with Close < $5 in main_df"

    # 2) DollarVolume >= 5M
    if "DollarVolume" in main_df.columns:
        assert (main_df["DollarVolume"] >= 5_000_000).all(), "Found names with DollarVolume < $5M in main_df"

    # 3) ATR14_pct <= 0.15
    if "ATR14_pct" in main_df.columns:
        assert (main_df["ATR14_pct"] <= 0.15 + 1e-9).all(), "Found names with ATR14_pct > 0.15 in main_df"

    # 4) market_cap >= 300M (if available)
    if "market_cap" in main_df.columns:
        assert (main_df["market_cap"] >= 300_000_000).all(), "Found names with market_cap < $300M in main_df"

    # 5) IsUltraRisky must be False on main list
    if "IsUltraRisky" in main_df.columns:
        assert (~main_df["IsUltraRisky"].fillna(False)).all(), "Main list contains IsUltraRisky=True rows"

    # ---------- RUNNERS SANITY CHECKS ----------

    if not runners_df.empty and "IsUltraRisky" in runners_df.columns:
        frac_ultra = runners_df["IsUltraRisky"].fillna(False).mean()
        print(f"Runners ultra-risky fraction: {frac_ultra:.2%}")
        assert frac_ultra > 0.5, "Runners list does not mostly contain ultra-risky names"

    # ---------- INSTITUTIONAL CORE SCORE PRESENCE ----------

    if "InstitutionalCoreScore" in main_df.columns:
        print("InstitutionalCoreScore present; head:")
        print(
            main_df[["Symbol", "InstitutionalCoreScore", "TechRating", "AlphaScorePct"]]
            .sort_values("InstitutionalCoreScore", ascending=False)
            .head(10)
        )
    else:
        print("WARNING: InstitutionalCoreScore missing from main_df")

    # ---------- SECTOR-NEUTRAL ALPHA PRESENCE ----------

    if "SectorAlphaPct" in main_df.columns:
        print("SectorAlphaPct present; sample:")
        print(
            main_df[["Symbol", "Sector", "SectorAlphaPct"]]
            .sort_values("SectorAlphaPct", ascending=False)
            .head(10)
        )
    else:
        print("WARNING: SectorAlphaPct missing from main_df")

    print("All institutional quality checks passed.")


if __name__ == "__main__":
    main()
