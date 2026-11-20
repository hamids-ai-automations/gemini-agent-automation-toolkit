"""Hard timeout wrapper around blocking API calls.

The Gemini SDK is synchronous, so timeouts are enforced with a worker thread.
A timed-out call cannot be interrupted mid-flight, but the caller stops waiting
and the run fails fast with :class:`TimeoutExceededError`.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, TypeVar

from ..exceptions import TimeoutExceededError

T = TypeVar("T")


def run_with_timeout(
    fn: Callable[[], T],
    timeout: float | None,
    *,
    label: str = "operation",
) -> T:
    """Run ``fn`` and raise :class:`TimeoutExceededError` after ``timeout`` seconds.

    ``timeout`` of ``None`` or ``<= 0`` disables the guard.
    """
    if not timeout or timeout <= 0:
        return fn()

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gat-{label}")
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout)
    except FutureTimeout as exc:
        future.cancel()
        raise TimeoutExceededError(f"{label} timed out after {timeout:g}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def clamp_timeout(timeout: float | None) -> float | None:
    """Normalise a timeout value; negative or zero means 'no timeout'."""
    if timeout is None:
        return None
    return timeout if timeout > 0 else None


__all__ = ["clamp_timeout", "run_with_timeout"]


_ = Any
