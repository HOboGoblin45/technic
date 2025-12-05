"""
ONNX inference utilities for fast alpha scoring.
"""

from __future__ import annotations

from typing import List

import pandas as pd

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    HAVE_SKL2ONNX = True
except Exception:  # pragma: no cover
    HAVE_SKL2ONNX = False

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


def export_lgbm_to_onnx(model, feature_names: List[str], out_path: str) -> None:
    """
    Export a trained LightGBM model to ONNX.
    Raises NotImplementedError if skl2onnx is missing.
    """
    if not HAVE_SKL2ONNX:
        raise NotImplementedError("skl2onnx is required for ONNX export")
    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=12)
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())


def load_onnx_session(path: str):
    """
    Load an ONNXRuntime session with CUDA if available.
    """
    if ort is None:
        return None
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(path, providers=providers)
        return sess
    except Exception:
        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            return sess
        except Exception:
            return None


def onnx_predict(session, df_features: pd.DataFrame) -> pd.Series:
    """
    Run ONNX model on feature DataFrame.
    Assumes session input name is the first input.
    """
    if session is None or df_features is None or df_features.empty:
        return pd.Series(dtype=float)
    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: df_features.astype("float32").values})[0].ravel()
    return pd.Series(pred, index=df_features.index)
