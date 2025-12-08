import pandas as pd
from pathlib import Path

CSV_PATH = Path("technic_v4/scanner_output/technic_scan_results.csv")

def main():
    if not CSV_PATH.exists():
        print(f"No scan results found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    # Keep only the last scan's rows (e.g., last 3 symbols)
    # Adjust N if your scan returns more rows.
    N = 3
    df = df.tail(N)

    cols = ["Symbol", "TechRating", "AlphaScore", "Signal"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("Missing columns:", missing)
        return

    df = df[cols].dropna()

    df_tr = df.sort_values("TechRating", ascending=False)
    df_alpha = df.sort_values("AlphaScore", ascending=False)

    print("\nTop by TechRating (latest scan):")
    print(df_tr.to_string(index=False))

    print("\nTop by AlphaScore (latest scan):")
    print(df_alpha.to_string(index=False))

if __name__ == "__main__":
    main()
