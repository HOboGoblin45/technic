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
    enable_shadow_mode: bool = field(default=False)

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

    # Alpha blending weight (0..1) for factor vs ML alpha
    alpha_weight: float = field(default=0.5)

    # Parallelism control for thread-pool scans
    max_workers: int = field(default=6)

    def __post_init__(self) -> None:
        prefix = "TECHNIC_"

        # Booleans
        self.use_ml_alpha = _env_bool(f"{prefix}USE_ML_ALPHA", self.use_ml_alpha)
        self.use_meta_alpha = _env_bool(f"{prefix}USE_META_ALPHA", self.use_meta_alpha)
        self.use_deep_alpha = _env_bool(f"{prefix}USE_DEEP_ALPHA", self.use_deep_alpha)
        self.use_onnx_alpha = _env_bool(f"{prefix}USE_ONNX_ALPHA", self.use_onnx_alpha)
        self.use_tft_features = _env_bool(f"{prefix}USE_TFT_FEATURES", self.use_tft_features)
        self.use_ray = _env_bool(f"{prefix}USE_RAY", self.use_ray)
        self.use_explainability = _env_bool(f"{prefix}USE_EXPLAINABILITY", self.use_explainability)
        self.enable_shadow_mode = _env_bool(f"{prefix}ENABLE_SHADOW_MODE", self.enable_shadow_mode)

        # Scan defaults
        self.default_universe_name = os.getenv(f"{prefix}DEFAULT_UNIVERSE_NAME", self.default_universe_name)
        self.default_min_tech_rating = float(
            os.getenv(f"{prefix}DEFAULT_MIN_TECH_RATING", self.default_min_tech_rating)
        )
        self.default_max_positions = int(
            os.getenv(f"{prefix}DEFAULT_MAX_POSITIONS", self.default_max_positions)
        )
        self.default_risk_profile = os.getenv(
            f"{prefix}DEFAULT_RISK_PROFILE", self.default_risk_profile
        )

        # Paths
        self.data_cache_dir = os.getenv(f"{prefix}DATA_CACHE_DIR", self.data_cache_dir)
        self.models_dir = os.getenv(f"{prefix}MODELS_DIR", self.models_dir)

        # Model selection
        self.alpha_model_name = os.getenv(
            f"{prefix}ALPHA_MODEL_NAME", self.alpha_model_name or ""
        )
        if self.alpha_model_name == "":
            self.alpha_model_name = None

        # Alpha blending weight (ML vs factor), clamp to [0, 1]
        raw_w = os.getenv(f"{prefix}ALPHA_WEIGHT")
        if raw_w is not None:
            try:
                w = float(raw_w)
            except Exception:
                w = self.alpha_weight
        else:
            w = self.alpha_weight
        self.alpha_weight = max(0.0, min(1.0, w))

        # Max workers for thread pools
        raw_workers = os.getenv(f"{prefix}MAX_WORKERS")
        if raw_workers is not None:
            try:
                self.max_workers = max(1, int(raw_workers))
            except Exception:
                # keep default on parse error
                pass


_settings = Settings()  # singleton


def get_settings() -> Settings:
    return _settings