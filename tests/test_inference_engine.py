import pandas as pd
import pytest

from technic_v4.engine import inference_engine
from technic_v4.engine.alpha_models import lgbm_alpha
from technic_v4.engine.alpha_models.lgbm_alpha import LGBMAlphaModel


def test_export_lgbm_to_onnx_not_implemented(tmp_path):
    if lgbm_alpha.lgb is None:
        pytest.skip("lightgbm not available")
    model = LGBMAlphaModel(n_estimators=5)
    model.fit(pd.DataFrame({"f": [0, 1]}), pd.Series([0, 1]))
    out = tmp_path / "model.onnx"
    try:
        inference_engine.export_lgbm_to_onnx(model.model, ["f"], str(out))
    except NotImplementedError:
        pytest.skip("skl2onnx not installed; export not implemented")
    assert out.exists()


def test_onnx_predict_empty_session():
    df = pd.DataFrame({"f": [0, 1, 2]})
    out = inference_engine.onnx_predict(None, df)
    assert out.empty
