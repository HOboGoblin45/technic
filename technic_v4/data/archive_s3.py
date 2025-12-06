"""Market data S3 archiver."""

from __future__ import annotations

import os
from pathlib import Path
import shutil


def archive_daily_to_s3(local_dir: Path, date_str: str, bucket_uri: str):
    """Compress daily raw data and upload to S3 (stub)."""
    dest = Path(f"{local_dir}/{date_str}")
    if not dest.exists():
        return False
    zip_path = shutil.make_archive(f"raw_{date_str}", "zip", root_dir=local_dir, base_dir=date_str)
    use_cloud = str(os.getenv("USE_CLOUD", "true")).lower() in {"1", "true", "yes"}
    if not use_cloud:
        return zip_path
    # TODO: integrate boto3 upload to f"s3://{bucket_uri}/raw/{date_str.replace('-', '/')}/"
    return zip_path


__all__ = ["archive_daily_to_s3"]
