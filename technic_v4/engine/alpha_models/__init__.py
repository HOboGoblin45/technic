from technic_v4.engine.alpha_models.base import BaseAlphaModel
from technic_v4.engine.alpha_models.lgbm_alpha import LGBMAlphaModel
from technic_v4.engine.alpha_models.xgb_alpha import XGBAlphaModel


def get_alpha_model(name: str = "lgbm_v1") -> BaseAlphaModel:
    """
    Simple factory; extend with more model variants as needed.
    """
    name = (name or "").lower()
    if name in {"lgbm_v1", "lgbm", "lightgbm"}:
        return LGBMAlphaModel()
    if name in {"xgb_v1", "xgb", "xgboost"}:
        return XGBAlphaModel()
    raise ValueError(f"Unknown alpha model '{name}'")
