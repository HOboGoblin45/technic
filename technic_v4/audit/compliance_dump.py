"""Compliance & risk audit export."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


def build_compliance_bundle():
    bundle = Path("audit/compliance_bundle.zip")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w") as z:
        for path in [
            Path("audit/model_lineage_log.json"),
            Path("provenance/provenance_log.json"),
            Path("ledger/alpha_ledger.jsonl"),
            Path("config/mode_flags.yaml"),
        ]:
            if path.exists():
                z.write(path, arcname=path.name)
    return bundle


if __name__ == "__main__":
    build_compliance_bundle()
