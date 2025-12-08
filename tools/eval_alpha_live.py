import os
import json
import requests
import pandas as pd   # <<< REQUIRED

URL = "https://technic-m5vn.onrender.com/v1/scan"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "your-key"     # Replace with your TECHNIC_API_KEY
}

payload = {
    "max_symbols": 5,
    "trade_style": "Short-term swing",
    "min_tech_rating": 0.0,
    "universe": None
}

# Live API request
resp = requests.post(URL, headers=HEADERS, json=payload)
data = resp.json()["results"]

CSV_PATH = "technic_v4/scanner_output/technic_scan_results.csv"


def main():

    # -------------------------
    # 1) LIVE API RESULTS
    # -------------------------
    print("\nLIVE API RESULTS:")
    for row in data:
        print(
            f"{row['symbol']:<6}  "
            f"Tech={row['techRating']:>5}  "
            f"Alpha={row['alphaScore']:.4f}  "
            f"Signal={row['signal']}"
        )

    # -------------------------
    # 2) LOAD LAST SCAN CSV
    # -------------------------
    if not os.path.exists(CSV_PATH):
        print(f"\nNo scan results found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    # -------------------------
    # 3) Correct required columns
    # -------------------------
    REQUIRED = ["Symbol", "TechRating", "AlphaScore", "Signal"]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print("Missing columns:", missing)
        return

    # Keep last 20 rows = latest scan
    df_latest = df.tail(20)

    print("\n--------- Top 20 by TechRating ---------")
    print(df_latest.sort_values("TechRating", ascending=False)[REQUIRED].to_string(index=False))

    print("\n--------- Top 20 by AlphaScore (ML) ---------")
    print(df_latest.sort_values("AlphaScore", ascending=False)[REQUIRED].to_string(index=False))


if __name__ == "__main__":
    main()
