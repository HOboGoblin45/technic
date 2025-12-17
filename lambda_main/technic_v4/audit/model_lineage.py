"""Model lineage audit system."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

LINEAGE_PATH = Path("audit/model_lineage_log.json")


def fingerprint(model_info: Dict) -> str:
    payload = json.dumps(model_info, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def log_lineage(model_info: Dict) -> str:
    """Track architecture, hyperparams, data span, checksum, fingerprint."""
    LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(model_info)
    entry = {"timestamp": datetime.utcnow().isoformat(), "fingerprint": fp, **model_info}
    data = []
    if LINEAGE_PATH.exists():
        try:
            data = json.loads(LINEAGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = []
    data.append(entry)
    LINEAGE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return fp


__all__ = ["log_lineage", "fingerprint", "LINEAGE_PATH"]
