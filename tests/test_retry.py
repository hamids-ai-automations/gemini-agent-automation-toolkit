"""Retry, timeout and transient/permanent classification."""

from __future__ import annotations

import pytest

from gemini_agent_toolkit import PermanentClientError, RetryPolicy, TransientClientError
from gemini_agent_toolkit.exceptions import RetryExhaustedError
from gemini_agent_toolkit.reliability.retry import execute_with_retry


def policy(**overrides: object) -> RetryPolicy:
    defaults = {"max_attempts": 2, "base_delay": 0.0, "max_delay": 0.0, "jitter": 0.0}
    return RetryPolicy(**{**defaults, **overrides})


def test_retries_transient_errors_until_success() -> None:
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientClientError("503 capacity")
        return "ok"

    assert execute_with_retry(flaky, policy=policy(max_attempts=3)) == "ok"
    assert attempts["n"] == 3


def test_permanent_errors_are_not_retried() -> None:
    attempts = {"n": 0}

    def unauthorized() -> str:
        attempts["n"] += 1
        raise PermanentClientError("401 invalid key")

    with pytest.raises(PermanentClientError):
        execute_with_retry(unauthorized, policy=policy(max_attempts=5))
    assert attempts["n"] == 1


def test_exhaustion_reports_attempts_and_last_error() -> None:
    def always_busy() -> str:
        raise TransientClientError("503")

    with pytest.raises(RetryExhaustedError) as exc:
        execute_with_retry(always_busy, policy=policy(max_attempts=2))
    assert exc.value.attempts == 2
    assert isinstance(exc.value.last_error, TransientClientError)


def test_retry_callback_receives_each_failure() -> None:
    seen: list[int] = []

    def always_busy() -> str:
        raise TransientClientError("503")

    with pytest.raises(RetryExhaustedError):
        execute_with_retry(
            always_busy,
            policy=policy(max_attempts=2),
            on_retry=lambda attempt, exc, delay: seen.append(attempt),
        )
    assert seen == [1]


def test_non_transient_exceptions_are_not_swallowed() -> None:
    def buggy() -> str:
        raise ValueError("toolkit bug")

    with pytest.raises(ValueError):
        execute_with_retry(buggy, policy=policy(max_attempts=3))
