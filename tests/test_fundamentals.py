from technic_v4.data_layer import fundamentals


def test_fundamentals_roundtrip(tmp_path, monkeypatch):
    sym = "TEST"
    payload = {"pe": 10, "name": "Test Co"}
    monkeypatch.setattr(fundamentals, "FUNDAMENTALS_DIR", tmp_path)
    fundamentals.save_fundamentals(sym, payload)
    snap = fundamentals.get_fundamentals(sym)
    assert snap.get("pe") == 10
    assert snap.get("name") == "Test Co"


def test_missing_returns_empty(tmp_path, monkeypatch):
    sym = "NONE"
    monkeypatch.setattr(fundamentals, "FUNDAMENTALS_DIR", tmp_path)
    snap = fundamentals.get_fundamentals(sym)
    assert snap.raw == {}
