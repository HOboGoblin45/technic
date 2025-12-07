import pandas as pd
import pytest

from technic_v4.scanner_core import (
    ScanConfig,
    _prepare_universe,
    _resolve_lookback_days,
    run_scan,
)
from technic_v4.universe_loader import UniverseRow
from technic_v4.config.settings import get_settings


def test_resolve_lookback_days_short_term():
    # short-term styles should clamp lookback
    assert _resolve_lookback_days("swing", 200) <= 90
    assert _resolve_lookback_days("day", 200) <= 30


def test_prepare_universe_not_empty(monkeypatch):
    fake_universe = [UniverseRow(symbol="AAPL", sector="Tech", industry="Software")]

    monkeypatch.setattr(
        "technic_v4.scanner_core.load_universe",
        lambda: fake_universe,
    )

    universe = _prepare_universe(ScanConfig(), settings=get_settings())
    assert len(universe) == 1
    assert universe[0].symbol == "AAPL"


def test_run_scan_basic(monkeypatch):
    # Minimal orchestration test using monkeypatched helpers to avoid external I/O
    fake_universe = [UniverseRow(symbol="AAPL", sector="Tech", industry="Software")]

    monkeypatch.setattr(
        "technic_v4.scanner_core._prepare_universe",
        lambda config, settings=None: fake_universe,
    )

    fake_rows = pd.DataFrame(
        [
            {
                "Symbol": "AAPL",
                "TechRating": 50.0,
                "Signal": "Long",
                "AlphaScore": 0.0,
                "Entry": 100.0,
                "Stop": 95.0,
                "Target": 110.0,
            }
        ]
    )

    monkeypatch.setattr(
        "technic_v4.scanner_core._run_symbol_scans",
        lambda **kwargs: (fake_rows.copy(), {"attempted": 1, "kept": 1, "errors": 0, "rejected": 0}),
    )
    monkeypatch.setattr(
        "technic_v4.scanner_core._finalize_results",
        lambda **kwargs: (fake_rows.copy(), "ok"),
    )

    df, status = run_scan(ScanConfig(max_symbols=1))
    assert status == "ok"
    assert not df.empty
    assert "Symbol" in df.columns
    assert "TechRating" in df.columns
