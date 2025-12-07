import json
import requests

URL = "https://technic-m5vn.onrender.com/v1/scan"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": "my-dev-technic-key",  # must match TECHNIC_API_KEY on Render
}

payload = {
    "max_symbols": 5,
    "trade_style": "Short-term swing",
    "min_tech_rating": 0.0,
    "universe": None,
}

print(f"POST {URL}")
resp = requests.post(URL, headers=headers, json=payload, timeout=60)

print("Status:", resp.status_code)
try:
    data = resp.json()
    print("Body:", json.dumps(data, indent=2))
except Exception as e:
    print("Non-JSON response:", resp.text, "error:", e)
