import pandas as pd
import pytest

from technic_v4.engine.alpha_models import lgbm_alpha
from technic_v4.engine.alpha_models.lgbm_alpha import LGBMAlphaModel
from technic_v4.engine import explainability


def test_explain_top_symbols():
    if lgbm_alpha.lgb is None:
        pytest.skip("lightgbm not available")
    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [1.0, 0.5, 0.0]}, index=["A", "B", "C"])
    y = pd.Series([0.0, 0.1, 0.2], index=X.index)
    model = LGBMAlphaModel(n_estimators=5)
    model.fit(X, y)
    res = explainability.explain_top_symbols(model.model, X, symbols=["A", "B"], top_n=2)
    assert "A" in res
    formatted = explainability.format_explanation(res["A"])
    assert isinstance(formatted, str)
    assert formatted != ""
