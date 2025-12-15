#!/usr/bin/env python3
"""
Bootstrap meta experience from existing scan history.

This script ensures that even without training_data.parquet, Technic can
use its own scan history to provide meta-experience insights.

Usage:
    python bootstrap_meta_experience.py
"""

from pathlib import Path
import pandas as pd
from technic_v4.infra.logging import get_logger

logger = get_logger()


def bootstrap_replay_ics():
    """
    Create or update replay_ics.parquet from scan history.
    
    This provides a fallback for meta_experience when training_data.parquet
    is not available.
    """
    history_dir = Path("technic_v4/scanner_output/history")
    history_dir.mkdir(parents=True, exist_ok=True)
    
    replay_path = history_dir / "replay_ics.parquet"
    
    # Check if we have any historical scan CSVs
    scan_files = list(history_dir.glob("scan_*.csv"))
    
    if not scan_files:
        logger.info("[BOOTSTRAP] No scan history found yet - will be created after first scan")
        # Create empty placeholder with correct schema
        df = pd.DataFrame(columns=[
            'Symbol', 'TechRating', 'AlphaScore', 'Signal',
            'Entry', 'Stop', 'Target', 'Sector', 'Industry'
        ])
        df.to_parquet(replay_path, index=False)
        logger.info("[BOOTSTRAP] Created empty replay_ics.parquet placeholder")
        return
    
    # Combine all historical scans
    logger.info(f"[BOOTSTRAP] Found {len(scan_files)} historical scan files")
    
    all_scans = []
    for scan_file in scan_files:
        try:
            df = pd.read_csv(scan_file)
            # Add date from filename (scan_2025-12-15.csv)
            date_str = scan_file.stem.replace('scan_', '')
            df['as_of_date'] = pd.to_datetime(date_str)
            all_scans.append(df)
        except Exception as e:
            logger.warning(f"[BOOTSTRAP] Failed to load {scan_file}: {e}")
    
    if not all_scans:
        logger.warning("[BOOTSTRAP] No valid scan files found")
        return
    
    # Combine and save
    combined = pd.concat(all_scans, ignore_index=True)
    
    # Ensure we have the key columns
    required_cols = ['Symbol', 'TechRating']
    if not all(col in combined.columns for col in required_cols):
        logger.warning("[BOOTSTRAP] Missing required columns in scan history")
        return
    
    # Save as replay_ics.parquet
    combined.to_parquet(replay_path, index=False)
    logger.info(f"[BOOTSTRAP] Created replay_ics.parquet with {len(combined)} rows from {len(all_scans)} scans")
    logger.info(f"[BOOTSTRAP] Meta experience will now use production scan data!")


if __name__ == "__main__":
    bootstrap_replay_ics()
