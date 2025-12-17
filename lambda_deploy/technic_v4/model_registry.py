from __future__ import annotations

"""
Lightweight JSON-based model registry for technic models.
Tracks model versions, artifact paths, and validation metrics.
"""

import json
from datetime import datetime
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
    is_active: bool = False,
    created_at: Optional[str] = None,
) -> None:
    """
    Add or update a model entry in the registry.
    """
    reg = load_registry()
    models = reg.get("models", [])
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
        "is_active": is_active,
        "created_at": created_at or datetime.utcnow().isoformat(),
    }
    if feature_names:
        entry["feature_names"] = feature_names
    models.append(entry)
    reg["models"] = models
    save_registry(reg)


def get_model_metadata(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Return the latest metadata dict for a given model_name.
    """
    return get_latest_model(model_name)


def get_latest_model(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Return the latest entry for a given model_name, or None if not found.
    Latest is determined by created_at then version string.
    """
    reg = load_registry()
    models = [m for m in reg.get("models", []) if m.get("model_name") == model_name]
    if not models:
        return None

    def _sort_key(m: Dict[str, Any]):
        return (
            m.get("created_at") or "",
            str(m.get("version", "")),
        )

    models = sorted(models, key=_sort_key, reverse=True)
    return models[0]


def get_active_model(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Return the most recent active entry for a given model_name.
    """
    reg = load_registry()
    models = [m for m in reg.get("models", []) if m.get("model_name") == model_name and m.get("is_active")]
    if not models:
        return None
    models = sorted(models, key=lambda m: m.get("created_at") or "", reverse=True)
    return models[0]


def set_active_model(model_name: str, version: str) -> None:
    """
    Mark a specific version as active, clearing the flag on others.
    """
    reg = load_registry()
    updated = []
    for m in reg.get("models", []):
        if m.get("model_name") != model_name:
            updated.append(m)
            continue
        m = dict(m)
        m["is_active"] = m.get("version") == version
        updated.append(m)
    reg["models"] = updated
    save_registry(reg)


def _best_by_metric(model_name: str, metric_key: str = "val_ic") -> Optional[Dict[str, Any]]:
    reg = load_registry()
    models = [m for m in reg.get("models", []) if m.get("model_name") == model_name]
    if not models:
        return None
    models = [m for m in models if metric_key in (m.get("metrics") or {})]
    if not models:
        return None
    models = sorted(models, key=lambda m: m.get("metrics", {}).get(metric_key, float("-inf")), reverse=True)
    return models[0]


def load_model(model_name: str, metric_key: str = "val_ic") -> Optional[Dict[str, Any]]:
    """
    Choose the best model entry:
    - active if present
    - else best by metric_key
    - else latest
    """
    active = get_active_model(model_name)
    if active:
        return active
    best = _best_by_metric(model_name, metric_key=metric_key)
    if best:
        return best
    return get_latest_model(model_name)
