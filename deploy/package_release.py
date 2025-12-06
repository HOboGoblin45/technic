"""Deployment packaging script."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


def package_release(version: str = "0.0.1") -> Path:
    # Placeholder: assume docker build done externally.
    meta = {
        "version": version,
        "hash": hashlib.sha256(version.encode("utf-8")).hexdigest(),
    }
    out = Path("deploy/release_metadata.json")
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    package_release()
