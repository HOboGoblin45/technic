from __future__ import annotations

"""
Lightweight JSON-based model registry for technic models.
Tracks model versions, artifact paths, and validation metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

REGISTRY_PATH = Path("models/model_registry.json")


def load_registry() -> Dict[str, Any]:
    """
    Load registry JSON from disk. Returns a dict with key "models": list[dict].
    If file is missing or invalid, returns an empty registry.
    """
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"models": []}
    return {"models": []}


def save_registry(reg: Dict[str, Any]) -> None:
    """
    Persist registry to disk.
    """
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def register_model(
    model_name: str,
    version: str,
    metrics: Dict[str, Any],
    path_pickle: str,
    path_onnx: Optional[str] = None,
    feature_names: Optional[list[str]] = None,
) -> None:
    """
    Add or update a model entry in the registry.
    """
    reg = load_registry()
    models = reg.get("models", [])
    # Drop any existing entry for the same model/version
    models = [
        m
        for m in models
        if not (m.get("model_name") == model_name and m.get("version") == version)
    ]
    entry: Dict[str, Any] = {
        "model_name": model_name,
        "version": version,
        "path_pickle": path_pickle,
        "path_onnx": path_onnx,
        "metrics": metrics or {},
    }
    if feature_names:
        entry["feature_names"] = feature_names
    models.append(entry)
    reg["models"] = models
    save_registry(reg)


def get_latest_model(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Return the latest entry for a given model_name, or None if not found.
    """
    reg = load_registry()
    models = [m for m in reg.get("models", []) if m.get("model_name") == model_name]
    if not models:
        return None
    # Sort by version string descending (YYYYMMDD or similar)
    models = sorted(models, key=lambda m: str(m.get("version", "")), reverse=True)
    return models[0]
