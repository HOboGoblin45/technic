import numpy as np
import pandas as pd
import pytest

from technic_v4.engine import portfolio_engine


def test_optimize_portfolio_weights():
    df = pd.DataFrame(
        {
            "Symbol": ["A", "B", "C"],
            "AlphaScore": [0.1, 0.2, 0.05],
            "VolatilityEstimate": [0.2, 0.25, 0.15],
            "Sector": ["Tech", "Health", "Tech"],
        }
    )
    class DummyRisk:
        risk_aversion = 0.1
    if portfolio_engine.cp is None:
        pytest.skip("cvxpy not installed")
    opt = portfolio_engine.optimize_portfolio(df, DummyRisk())
    assert not opt.empty
    assert np.isclose(opt["Weight"].sum(), 1.0, atol=1e-3)
    assert (opt["Weight"] >= -1e-6).all()


def test_apply_portfolio_weights_equal_weight():
    df = pd.DataFrame({"Symbol": ["A", "B"]})
    out = portfolio_engine.apply_portfolio_weights(df, risk_settings=None, use_optimizer=False)
    assert np.isclose(out["Weight"].sum(), 1.0)
