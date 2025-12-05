import pandas as pd
import numpy as np
import pytest

from technic_v4.engine.alpha_models import lgbm_alpha
from technic_v4.engine.alpha_models.lgbm_alpha import LGBMAlphaModel
from technic_v4.engine import alpha_inference


def test_lgbm_alpha_fit_predict_save_load(tmp_path):
    if lgbm_alpha.lgb is None:
        pytest.skip("lightgbm not available")
    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.5, 2.0, 2.5]})
    y = pd.Series([0.0, 0.5, 1.0, 1.5])
    model = LGBMAlphaModel(n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    path = tmp_path / "model.pkl"
    model.save(str(path))
    loaded = LGBMAlphaModel.load(str(path))
    preds_loaded = loaded.predict(X)
    assert len(preds_loaded) == len(X)


def test_alpha_inference_score_alpha(tmp_path, monkeypatch):
    if lgbm_alpha.lgb is None:
        pytest.skip("lightgbm not available")
    # When model missing -> None
    monkeypatch.setattr(alpha_inference, "DEFAULT_MODEL_PATH", tmp_path / "missing.pkl")
    assert alpha_inference.score_alpha(pd.DataFrame({"f": [1, 2, 3]})) is None

    # With a saved model -> Series
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
    y = pd.Series([0.0, 0.1, 0.2])
    model = LGBMAlphaModel(n_estimators=5)
    model.fit(X, y)
    model.save(str(tmp_path / "model.pkl"))
    monkeypatch.setattr(alpha_inference, "DEFAULT_MODEL_PATH", tmp_path / "model.pkl")
    out = alpha_inference.score_alpha(X)
    assert out is not None
    assert len(out) == len(X)
