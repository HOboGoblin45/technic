from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).lower() in {"1", "true", "yes", "y"}


@dataclass
class Settings:
    # ML / model flags
    use_ml_alpha: bool = field(default=False)
    use_meta_alpha: bool = field(default=False)
    use_deep_alpha: bool = field(default=False)
    use_onnx_alpha: bool = field(default=False)
    use_tft_features: bool = field(default=False)
    use_ray: bool = field(default=False)
    use_explainability: bool = field(default=False)

    # Scan defaults
    default_universe_name: str = field(default="us_core")
    default_min_tech_rating: float = field(default=0.0)
    default_max_positions: int = field(default=25)

    # Risk presets
    default_risk_profile: str = field(default="balanced")

    # Data/cache settings
    data_cache_dir: str = field(default="data_cache")
    models_dir: str = field(default="models")

    # Model selection
    alpha_model_name: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        prefix = "TECHNIC_"
        self.use_ml_alpha = _env_bool(f"{prefix}USE_ML_ALPHA", self.use_ml_alpha)
        self.use_meta_alpha = _env_bool(f"{prefix}USE_META_ALPHA", self.use_meta_alpha)
        self.use_deep_alpha = _env_bool(f"{prefix}USE_DEEP_ALPHA", self.use_deep_alpha)
        self.use_onnx_alpha = _env_bool(f"{prefix}USE_ONNX_ALPHA", self.use_onnx_alpha)
        self.use_tft_features = _env_bool(f"{prefix}USE_TFT_FEATURES", self.use_tft_features)
        self.use_ray = _env_bool(f"{prefix}USE_RAY", self.use_ray)
        self.use_explainability = _env_bool(f"{prefix}USE_EXPLAINABILITY", self.use_explainability)

        self.default_universe_name = os.getenv(f"{prefix}DEFAULT_UNIVERSE_NAME", self.default_universe_name)
        self.default_min_tech_rating = float(
            os.getenv(f"{prefix}DEFAULT_MIN_TECH_RATING", self.default_min_tech_rating)
        )
        self.default_max_positions = int(os.getenv(f"{prefix}DEFAULT_MAX_POSITIONS", self.default_max_positions))
        self.default_risk_profile = os.getenv(f"{prefix}DEFAULT_RISK_PROFILE", self.default_risk_profile)

        self.data_cache_dir = os.getenv(f"{prefix}DATA_CACHE_DIR", self.data_cache_dir)
        self.models_dir = os.getenv(f"{prefix}MODELS_DIR", self.models_dir)
        self.alpha_model_name = os.getenv(f"{prefix}ALPHA_MODEL_NAME", self.alpha_model_name or "")
        if self.alpha_model_name == "":
            self.alpha_model_name = None


_settings = Settings()  # singleton


def get_settings() -> Settings:
    return _settings
