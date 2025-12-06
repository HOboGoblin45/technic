"""Data provenance tracker."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict

PROV_PATH = Path("provenance/provenance_log.json")


def record_provenance(dataset: str, transformation: str, destination: str) -> None:
    """Track dataset -> transformation -> destination."""
    PROV_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry: Dict[str, str] = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": dataset,
        "transformation": transformation,
        "destination": destination,
    }
    data = []
    if PROV_PATH.exists():
        try:
            data = json.loads(PROV_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = []
    data.append(entry)
    PROV_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = ["record_provenance", "PROV_PATH"]
