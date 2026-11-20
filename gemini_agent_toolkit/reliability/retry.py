"""Retry policy: exponential backoff that only retries transient failures."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..exceptions import (
    ApprovalError,
    ConfigError,
    PermanentClientError,
    RetryExhaustedError,
    StructuredOutputError,
    ToolArgumentError,
    ToolkitError,
    TransientClientError,
)
from ..exceptions import (
    RetryExhaustedError as _RetryExhaustedError,
)

#: HTTP-ish codes that are worth retrying.
TRANSIENT_CODES: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: gRPC / SDK status strings that are worth retrying.
TRANSIENT_STATUS: frozenset[str] = frozenset(
    {
        "ABORTED",
        "DEADLINE_EXCEEDED",
        "INTERNAL",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "UNKNOWN",
    }
)

#: Errors that describe a deterministic problem in the caller's code or in the
#: approval flow. Retrying them would only waste time (and, for side-effecting
#: tools, could duplicate effects).
NON_RETRYABLE: tuple[type[BaseException], ...] = (
    ToolArgumentError,
    ApprovalError,
    StructuredOutputError,
    ConfigError,
)

_TRANSIENT_HINTS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "resource exhausted",
    "resource_exhausted",
    "429",
    "500",
    "502",
    "503",
    "504",
    "deadline exceeded",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "overloaded",
    "capacity",
)

#: Placeholder kept for readability in isinstance checks below.
_UNUSED = _RetryExhaustedError


def _int_code(exc: BaseException) -> int | None:
    for attr in ("code", "status_code", "http_status", "http_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _status_text(exc: BaseException) -> str:
    parts: list[str] = []
    for attr in ("status", "code", "reason"):
        value = getattr(exc, attr, None)
        if isinstance(value, str):
            parts.append(value.upper())
        elif value is not None and hasattr(value, "name"):
            parts.append(str(value.name).upper())
    return " ".join(parts)


def is_transient(exc: BaseException) -> bool:
    """Decide whether ``exc`` is worth retrying.

    Returns ``False`` for permanent errors (bad request, missing key, tool
    argument validation, approval problems) and ``True`` for rate limits,
    capacity errors, timeouts and network failures.
    """
    if isinstance(exc, NON_RETRYABLE):
        return False
    if isinstance(exc, PermanentClientError):
        return False
    if isinstance(exc, (TransientClientError, TimeoutError, ConnectionError)):
        return True

    code = _int_code(exc)
    if code in TRANSIENT_CODES:
        return True
    status = _status_text(exc)
    if status and any(token in status for token in TRANSIENT_STATUS):
        return True

    message = str(exc).lower()
    return any(hint in message for hint in _TRANSIENT_HINTS)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Read a server supplied retry hint if the exception carries one."""
    for attr in ("retry_delay", "retry_after", "retry_after_seconds"):
        value = getattr(exc, attr, None)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        total = getattr(value, "total_seconds", None)
        if callable(total):
            return float(total())
    headers = getattr(exc, "headers", None)
    if isinstance(headers, dict):
        for key in ("retry-after", "Retry-After"):
            if key in headers:
                try:
                    return float(headers[key])
                except (TypeError, ValueError):
                    return None
    return None


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff with jitter."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: float = 0.1
    respect_retry_after: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ConfigError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ConfigError("delays must be >= 0")

    @classmethod
    def from_config(cls, config: Any) -> RetryPolicy:
        """Build a policy from an :class:`~gemini_agent_toolkit.config.AgentConfig`."""
        return cls(
            max_attempts=max(1, int(config.max_retries)),
            base_delay=float(config.retry_base_delay),
            max_delay=float(config.retry_max_delay),
            jitter=float(getattr(config, "retry_jitter", 0.1)),
        )

    def delay_for(
        self,
        attempt: int,
        exc: BaseException | None = None,
        rng: Callable[[], float] = random.random,
    ) -> float:
        """Delay before the next attempt (``attempt`` is 1-based)."""
        if self.respect_retry_after and exc is not None:
            hinted = retry_after_seconds(exc)
            if hinted is not None:
                return max(0.0, min(hinted, self.max_delay))
        raw = self.base_delay * (2 ** (attempt - 1))
        capped = min(raw, self.max_delay)
        if self.jitter:
            capped *= 1 + self.jitter * (2 * rng() - 1)
        return max(0.0, capped)


def execute_with_retry(
    fn: Callable[[], Any],
    policy: RetryPolicy | None = None,
    *,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> Any:
    """Call ``fn`` with retries.

    The policy is only consulted for transient errors; anything else raises
    immediately so callers do not mask programming mistakes.

    Raises:
        RetryExhaustedError: when every attempt failed with a transient error.
    """
    policy = policy or RetryPolicy()
    last_error: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            if not is_transient(exc):
                raise
            last_error = exc
            if attempt >= policy.max_attempts:
                break
            delay = policy.delay_for(attempt, exc, rng=rng)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            sleep(delay)

    assert last_error is not None  # for type checkers
    raise RetryExhaustedError(policy.max_attempts, last_error) from last_error


__all__ = [
    "RetryPolicy",
    "TRANSIENT_CODES",
    "TRANSIENT_STATUS",
    "execute_with_retry",
    "is_transient",
    "retry_after_seconds",
]

# Keep a reference so linters do not flag the import used in type docs.
_ = ToolkitError
