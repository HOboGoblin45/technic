from __future__ import annotations

import os
from typing import List, Optional

import requests


class TechnicClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("TECHNIC_API_KEY", "")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def scan(
        self,
        max_symbols: int = 25,
        trade_style: str = "Short-term swing",
        min_tech_rating: float = 0.0,
        universe: Optional[List[str]] = None,
    ):
        payload = {
            "max_symbols": max_symbols,
            "trade_style": trade_style,
            "min_tech_rating": min_tech_rating,
            "universe": universe,
        }
        resp = requests.post(
            f"{self.base_url}/v1/scan",
            json=payload,
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


# Example usage:
# from technic_client import TechnicClient
# client = TechnicClient("http://localhost:8502", api_key="changeme-super-secret")
# response = client.scan(max_symbols=10)
# print(response)
