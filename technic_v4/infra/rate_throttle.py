"""Custom API rate throttler (token-bucket) with stats tracking."""

from __future__ import annotations

import time


class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.last = time.time()
        self.hits = 0
        self.throttles = 0

    def consume(self, amount: float = 1.0) -> bool:
        now = time.time()
        delta = now - self.last
        self.tokens = min(self.capacity, self.tokens + delta * self.rate)
        self.last = now
        if self.tokens >= amount:
            self.tokens -= amount
            self.hits += 1
            return True
        self.throttles += 1
        return False

    def stats(self):
        return {"hits": self.hits, "throttles": self.throttles, "tokens": self.tokens}


__all__ = ["TokenBucket"]
