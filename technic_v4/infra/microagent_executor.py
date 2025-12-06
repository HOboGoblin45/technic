"""Microagent executor harness with context-specific memory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class MicroAgent(ABC):
    def __init__(self, context: Dict[str, Any] | None = None):
        self.context = context or {}

    @abstractmethod
    def prepare(self) -> None:
        ...

    @abstractmethod
    def execute(self) -> Any:
        ...

    @abstractmethod
    def report(self) -> Dict[str, Any]:
        ...


def run_agent(agent: MicroAgent) -> Dict[str, Any]:
    agent.prepare()
    agent.execute()
    return agent.report()


__all__ = ["MicroAgent", "run_agent"]
