"""Ranking agreement score across ensemble models."""

from __future__ import annotations

import pandas as pd
from pathlib import Path


def compute_agreement(rank_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute agreement rate across ensemble models on top-10 stocks."""
    # rank_matrix rows: symbols; cols: model names with rank/score
    binary = rank_matrix.rank(ascending=False, method="first") <= 10
    agreement = binary.sum(axis=0) / binary.shape[0]
    return agreement.to_frame(name="agreement")


def save_consensus(rank_matrix: pd.DataFrame, path: Path = Path("logs/consensus_rank.csv")) -> None:
    agreement_df = compute_agreement(rank_matrix)
    path.parent.mkdir(parents=True, exist_ok=True)
    agreement_df.to_csv(path)


__all__ = ["compute_agreement", "save_consensus"]
