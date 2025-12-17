"""Cloud model cache bootstrapper."""

from __future__ import annotations

import os
from pathlib import Path


def pull_models_from_s3(bucket_path: str, dest: Path = Path("models/active")):
    """Stub: pull model bundles from S3 into models/active/. Falls back if USE_CLOUD=false."""
    use_cloud = str(os.getenv("USE_CLOUD", "true")).lower() in {"1", "true", "yes"}
    if not use_cloud:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    # TODO: integrate boto3; placeholder for now.
    return True


__all__ = ["pull_models_from_s3"]
