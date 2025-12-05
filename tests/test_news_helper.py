import pytest

# Stub LLM dependency so technic_app imports cleanly in tests
import types, sys
sys.modules["generate_copilot_answer"] = types.SimpleNamespace(generate_copilot_answer=lambda *a, **k: "ok")
from technic_v4.ui import technic_app


def test_bulk_news_dedup(monkeypatch):
    calls = []

    def fake_fetch(sym, limit=3):
        calls.append(sym)
        return [
            {"title": "A", "url": "u1", "published": "2024-01-01 00:00 UTC", "source": "S"},
            {"title": "B", "url": "u2", "published": "2024-01-02 00:00 UTC", "source": "S"},
        ]

    monkeypatch.setattr(technic_app, "fetch_symbol_news", fake_fetch)
    items = technic_app.fetch_bulk_news(["AAPL", "MSFT", "AAPL"], per_symbol=2, max_items=3)
    # We asked for 3 max; at least one symbol should appear; dedup by URL holds
    assert 1 <= len(items) <= 3
    # Ensure dedup by URL works
    urls = [i["url"] for i in items]
    assert len(urls) == len(set(urls))
    assert "AAPL" in calls and "MSFT" in calls
