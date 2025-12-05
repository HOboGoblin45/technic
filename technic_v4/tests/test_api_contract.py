import pytest
from technic_v4.api_contract import ScanRequest, OptionsRequest


def test_scan_request_bounds():
    req = ScanRequest(limit=10, offset=0, max_symbols=100, lookback_days=120)
    assert req.limit == 10
    assert req.max_symbols == 100
    assert req.lookback_days == 120

    with pytest.raises(Exception):
        ScanRequest(limit=0)  # below min

    with pytest.raises(Exception):
        ScanRequest(max_symbols=0)  # below min


def test_options_request_validation():
    ok = OptionsRequest(symbol="AAPL", direction="call", limit=3)
    assert ok.direction == "call"

    with pytest.raises(Exception):
        OptionsRequest(symbol="AAPL", direction="foo")

    with pytest.raises(Exception):
        OptionsRequest(symbol="AAPL", direction="CALL")
