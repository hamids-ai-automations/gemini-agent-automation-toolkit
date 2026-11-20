"""Guards that keep agent loops from misbehaving.

* :class:`ExecutionGuard` enforces the per-run tool budget and stops the same
  call from being repeated indefinitely.
* :class:`IdempotentCache` memoises explicitly declared idempotent tool calls so
  a retried run cannot duplicate a side effect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..exceptions import RecursiveToolCallError, ToolLimitExceededError


def arguments_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    """Stable fingerprint of a tool call, used for repeat detection and approvals."""
    try:
        payload = json.dumps(arguments, sort_keys=True, default=str, separators=(",", ":"))
    except TypeError:  # pragma: no cover - defensive
        payload = repr(sorted(arguments.items(), key=lambda kv: kv[0]))
    return f"{tool_name}:{payload}"


@dataclass
class ExecutionGuard:
    """Counts tool calls and rejects runaway repetition or self recursion."""

    max_tool_calls: int = 8
    max_repeated_calls: int = 2
    total_calls: int = 0
    _counts: dict[str, int] = field(default_factory=dict)
    _stack: list[str] = field(default_factory=list)

    def check(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Validate that one more call is allowed.

        Raises:
            RecursiveToolCallError: the tool is already on the execution stack.
            ToolLimitExceededError: the run or call-repeat budget is exhausted.
        """
        if tool_name in self._stack:
            raise RecursiveToolCallError(
                f"tool '{tool_name}' tried to invoke itself recursively"
            )
        if self.total_calls >= self.max_tool_calls:
            raise ToolLimitExceededError(
                f"tool call budget of {self.max_tool_calls} exhausted for this run"
            )
        key = arguments_fingerprint(tool_name, arguments)
        if self._counts.get(key, 0) >= self.max_repeated_calls:
            raise ToolLimitExceededError(
                f"call '{key}' repeated more than {self.max_repeated_calls} times"
            )

    def record(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self.total_calls += 1
        key = arguments_fingerprint(tool_name, arguments)
        self._counts[key] = self._counts.get(key, 0) + 1

    # -- recursion stack -------------------------------------------------- #
    def push(self, tool_name: str) -> None:
        self._stack.append(tool_name)

    def pop(self, tool_name: str) -> None:
        if tool_name in self._stack:
            self._stack.remove(tool_name)


class IdempotentCache:
    """In-memory cache for tool calls declared ``idempotent=True``."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, Any]:
        key = arguments_fingerprint(tool_name, arguments)
        if key in self._store:
            return True, self._store[key]
        return False, None

    def set(self, tool_name: str, arguments: dict[str, Any], value: Any) -> None:
        self._store[arguments_fingerprint(tool_name, arguments)] = value

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


__all__ = ["ExecutionGuard", "IdempotentCache", "arguments_fingerprint"]
