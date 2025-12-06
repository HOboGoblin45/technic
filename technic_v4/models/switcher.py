"""Modular model swap engine to load different alpha models by name."""

from __future__ import annotations


def load_model(model_name: str):
    """Load a model instance based on a config toggle."""
    if model_name == "xgb":
        import xgboost as xgb

        return xgb.Booster()
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier()
    if model_name == "lstm":
        from technic_v4.engine.signal_lstm import SignalLSTM

        return SignalLSTM(input_size=10, hidden_size=32)
    raise ValueError(f"Unknown model_name '{model_name}'")


__all__ = ["load_model"]
