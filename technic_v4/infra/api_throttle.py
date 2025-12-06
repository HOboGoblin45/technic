"""Public API throttle + abuse monitor."""

from __future__ import annotations

import time
from collections import defaultdict, deque


class ApiThrottle:
    def __init__(self, limit_per_minute: int = 20):
        self.limit = limit_per_minute
        self.calls = defaultdict(deque)
        self.blacklist = set()

    def allow(self, ip: str) -> bool:
        if ip in self.blacklist:
            return False
        now = time.time()
        window = self.calls[ip]
        window.append(now)
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) > self.limit:
            self.blacklist.add(ip)
            return False
        return True

    def stats(self, ip: str):
        return {"count": len(self.calls[ip]), "blacklisted": ip in self.blacklist}


__all__ = ["ApiThrottle"]
