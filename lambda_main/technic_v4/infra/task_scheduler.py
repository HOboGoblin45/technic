"""Distributed task scheduler for microagents (async job queue)."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict


class TaskScheduler:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register_handler(self, task_type: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        self.handlers[task_type] = handler

    async def add_job(self, job: Dict[str, Any]) -> None:
        """Inject dynamic jobs from a centralized agent controller."""
        await self.queue.put(job)

    async def run(self):
        """Consume jobs and dispatch to handlers."""
        while True:
            job = await self.queue.get()
            handler = self.handlers.get(job.get("type"))
            if handler:
                try:
                    await asyncio.to_thread(handler, job)
                except Exception:
                    pass
            self.queue.task_done()


__all__ = ["TaskScheduler"]
