from technic_v4.engine.alpha_models.base import BaseAlphaModel
from technic_v4.engine.alpha_models.lgbm_alpha import LGBMAlphaModel
from technic_v4.engine.alpha_models.xgb_alpha import XGBAlphaModel
from technic_v4.engine.alpha_models.ensemble_alpha import EnsembleAlphaModel


def get_alpha_model(name: str = "lgbm_v1") -> BaseAlphaModel:
    """
    Simple factory; extend with more model variants as needed.
    """
    name = (name or "").lower()
    if name in {"lgbm_v1", "lgbm", "lightgbm"}:
        return LGBMAlphaModel()
    if name in {"xgb_v1", "xgb", "xgboost"}:
        return XGBAlphaModel()
    if name in {"ensemble", "alpha_ensemble_v1"}:
        return EnsembleAlphaModel()
    raise ValueError(f"Unknown alpha model '{name}'")
