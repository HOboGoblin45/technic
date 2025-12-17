"""
Meta-layer summaries: map a live setup into historical buckets and return a short
explanation of how similar setups behaved in the past.

Uses a cached parquet dataset with forward returns (e.g., data/training_data_v2.parquet)
to compute:
- Score buckets (ICS or TechRating proxy) with mean returns / win rates
- PlayStyle averages when available

Expose a single entry point:
    describe_row(row: pd.Series) -> str
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from technic_v4.infra.logging import get_logger

logger = get_logger()

DEFAULT_META_PATHS: list[Path] = [
    Path(os.getenv("TECHNIC_META_DATA_PATH", "")),
    Path("data/training_data_v2.parquet"),
    Path("data/training_data.parquet"),
    Path("technic_v4/scanner_output/history/replay_ics.parquet"),
]


@dataclass
class MetaExperience:
    score_col: str
    edges: np.ndarray
    bucket_stats: pd.DataFrame
    playstyle_stats: Optional[pd.DataFrame]
    percentiles: dict
    horizons: list[int]
    scores: np.ndarray

    @classmethod
    def from_parquet(
        cls,
        path: Path,
        horizons: Iterable[int] = (5, 10),
        n_buckets: int = 10,
    ) -> Optional["MetaExperience"]:
        if not path.exists():
            return None

        df = pd.read_parquet(path)
        score_col = None
        for cand in ("InstitutionalCoreScore", "TechRating", "tech_rating"):
            if cand in df.columns:
                score_col = cand
                break
        if score_col is None:
            return None

        label_cols: list[str] = []
        horizons_found: list[int] = []
        for h in horizons:
            col = f"fwd_ret_{h}d"
            if col in df.columns:
                label_cols.append(col)
                horizons_found.append(h)
        # Fallback: auto-detect any forward-return columns
        if not label_cols:
            label_cols = [c for c in df.columns if c.startswith("fwd_ret_") and c.endswith("d")]
            for c in label_cols:
                try:
                    horizons_found.append(int(c.replace("fwd_ret_", "").replace("d", "")))
                except Exception:
                    continue

        if not label_cols:
            return None

        use_df = df[[score_col] + label_cols].copy()
        if "PlayStyle" in df.columns:
            use_df["PlayStyle"] = df["PlayStyle"]
        use_df = use_df.dropna(subset=[score_col])
        if use_df.empty:
            return None

        # Build bucket edges from quantiles; ensure unique ascending edges
        quantiles = np.linspace(0, 1, n_buckets + 1)
        edges = np.quantile(use_df[score_col], quantiles)
        edges = np.unique(edges)
        if len(edges) < 3:
            return None
        edges[0] -= 1e-9
        edges[-1] += 1e-9

        # Assign buckets via digitize using these edges
        use_df["bucket"] = np.digitize(use_df[score_col], edges[1:-1], right=True)

        agg: dict[str, tuple[str, str] | tuple[str, callable]] = {}
        for col in label_cols:
            agg[f"mean_{col}"] = (col, "mean")
            agg[f"hit_{col}"] = (col, lambda x: float((pd.Series(x) > 0).mean()))

        bucket_stats = use_df.groupby("bucket").agg(**agg)
        bucket_stats = bucket_stats.reset_index()

        playstyle_stats = None
        if "PlayStyle" in use_df.columns:
            playstyle_stats = use_df.groupby("PlayStyle").agg(**agg).reset_index()

        percentiles = use_df[score_col].quantile([0.5, 0.75, 0.9, 0.95]).to_dict()

        return cls(
            score_col=score_col,
            edges=edges,
            bucket_stats=bucket_stats,
            playstyle_stats=playstyle_stats,
            percentiles=percentiles,
            horizons=horizons_found,
            scores=use_df[score_col].to_numpy(),
        )

    # ---------------------- Helpers ----------------------
    def bucket_for_score(self, score: float) -> Optional[int]:
        if score is None or pd.isna(score):
            return None
        try:
            idx = int(np.digitize([float(score)], self.edges[1:-1], right=True)[0])
            return idx
        except Exception:
            return None

    def percentile_rank(self, score: float) -> Optional[float]:
        if score is None or pd.isna(score) or self.scores.size == 0:
            return None
        try:
            return float((self.scores <= score).mean())
        except Exception:
            return None

    def _format_bucket_text(self, bucket: int) -> str:
        row = self.bucket_stats[self.bucket_stats["bucket"] == bucket]
        if row.empty:
            return ""
        row = row.iloc[0]
        parts: list[str] = []
        for h in sorted(self.horizons):
            m = row.get(f"mean_fwd_ret_{h}d")
            hit = row.get(f"hit_fwd_ret_{h}d")
            if pd.notna(m):
                parts.append(f"{h}d avg {m*100:.1f}%")
            if pd.notna(hit):
                parts.append(f"{h}d win {hit*100:.0f}%")
        return ", ".join(parts)

    def _format_playstyle_text(self, playstyle: str) -> str:
        if self.playstyle_stats is None or not playstyle:
            return ""
        row = self.playstyle_stats[self.playstyle_stats["PlayStyle"] == playstyle]
        if row.empty:
            return ""
        row = row.iloc[0]
        parts: list[str] = []
        for h in sorted(self.horizons):
            m = row.get(f"mean_fwd_ret_{h}d")
            hit = row.get(f"hit_fwd_ret_{h}d")
            if pd.notna(m):
                parts.append(f"{h}d avg {m*100:.1f}%")
            if pd.notna(hit):
                parts.append(f"{h}d win {hit*100:.0f}%")
        if not parts:
            return ""
        return f"{playstyle} setups historically: " + ", ".join(parts)

    def describe_row(self, row: pd.Series) -> str:
        score_val = row.get("InstitutionalCoreScore")
        if score_val is None or pd.isna(score_val):
            score_val = row.get("TechRating")
        try:
            score_val = float(score_val) if score_val is not None else None
        except Exception:
            score_val = None

        if score_val is None or pd.isna(score_val):
            return ""

        bucket = self.bucket_for_score(score_val)
        pct_rank = self.percentile_rank(score_val)
        parts: list[str] = []

        if pct_rank is not None:
            pct = (1 - pct_rank) * 100
            if pct_rank >= 0.95:
                parts.append("This setup ranks in the top 5% of prior samples.")
            elif pct_rank >= 0.9:
                parts.append("This setup sits in roughly the top 10% of prior samples.")
            elif pct_rank <= 0.25:
                parts.append("This setup is in the lower quartile versus history.")
            else:
                parts.append(f"This setup is around the top {pct:.0f}% of the historical range.")

        if bucket is not None:
            bucket_txt = self._format_bucket_text(bucket)
            if bucket_txt:
                parts.append(f"Similar-score bucket outcomes: {bucket_txt}.")

        playstyle = row.get("PlayStyle") or row.get("playstyle")
        ps_txt = self._format_playstyle_text(playstyle) if isinstance(playstyle, str) else ""
        if ps_txt:
            parts.append(ps_txt)

        return " ".join(parts)


_META_CACHE: Optional[MetaExperience] = None


def load_meta_experience() -> Optional[MetaExperience]:
    global _META_CACHE
    if _META_CACHE is not None:
        return _META_CACHE

    for path in DEFAULT_META_PATHS:
        if not path or str(path) in {"", "."}:
            continue
        path = Path(path)
        if path.is_dir():
            continue
        try:
            meta = MetaExperience.from_parquet(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[META] failed to load meta experience from %s: %s", path, exc)
            meta = None
        if meta is not None:
            _META_CACHE = meta
            logger.info("[META] loaded meta experience from %s", path)
            break
    return _META_CACHE


def describe_row(row: pd.Series) -> str:
    meta = load_meta_experience()
    if meta is None:
        return ""
    try:
        return meta.describe_row(row)
    except Exception:
        return ""


__all__ = ["load_meta_experience", "describe_row", "MetaExperience"]
