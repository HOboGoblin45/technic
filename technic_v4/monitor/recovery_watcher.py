"""Model recovery & self-healing trigger."""

from __future__ import annotations

from typing import Sequence


def check_recovery(perf_series: Sequence[float], benchmark_series: Sequence[float]) -> str:
    """Track underperformance vs rolling benchmark and suggest action."""
    if len(perf_series) < 3 or len(benchmark_series) < 3:
        return "noop"
    recent_perf = perf_series[-3:]
    recent_bench = benchmark_series[-3:]
    underperf = sum(p < b for p, b in zip(recent_perf, recent_bench)) >= 2
    if underperf:
        return "retrain"
    return "noop"


__all__ = ["check_recovery"]
