import json
import requests

URL = "https://technic-m5vn.onrender.com/v1/scan"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "my-dev-technic-key",
}

payload = {
    "max_symbols": 5,
    "trade_style": "Short-term swing",
    "min_tech_rating": 0.0,
    "universe": None,
}

resp = requests.post(URL, headers=HEADERS, json=payload)
raw = resp.json()

print("HTTP", resp.status_code)
print("\nLIVE API RESULTS (TechRating vs ML AlphaScore):")
for row in raw["results"]:
    print(
        f"{row['symbol']:<6}  "
        f"Tech={row['techRating']:>5}  "
        f"Alpha={row['alphaScore']:.4f}  "
        f"Signal={row['signal']}"
    )
