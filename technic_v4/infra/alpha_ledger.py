"""Immutable alpha ledger with hash-linked blocks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

LEDGER_PATH = Path("ledger/alpha_ledger.jsonl")


def _hash_block(block: Dict, prev_hash: str) -> str:
    payload = json.dumps({"block": block, "prev": prev_hash}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def append_block(signals: List[Dict], prev_hash: str = "") -> str:
    """Append alpha signals as a new block and return its hash."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    block = {
        "timestamp": datetime.utcnow().isoformat(),
        "signals": signals,
    }
    block_hash = _hash_block(block, prev_hash)
    entry = {"hash": block_hash, "prev_hash": prev_hash, **block}
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return block_hash


__all__ = ["append_block", "LEDGER_PATH"]
