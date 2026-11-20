"""Retry behaviour: transient failures are retried, permanent ones are not.

This example runs fully offline with a stub client, so no API key is needed.

    python examples/retry_handling.py
"""

from __future__ import annotations

from typing import Any

from gemini_agent_toolkit import (
    Agent,
    AgentConfig,
    LLMResponse,
    PermanentClientError,
    RetryExhaustedError,
    TransientClientError,
)
from gemini_agent_toolkit.models import UsageSummary


class FlakyClient:
    """Fails twice with a transient error, then succeeds."""

    def __init__(self) -> None:
        self.attempts = 0

    def generate(self, **_: Any) -> LLMResponse:
        self.attempts += 1
        if self.attempts < 3:
            raise TransientClientError(f"503 service unavailable (attempt {self.attempts})")
        return LLMResponse(text="Recovered after retries.", usage=UsageSummary(total_tokens=9))


class BrokenClient:
    """Always fails with an error that must not be retried."""

    def __init__(self) -> None:
        self.attempts = 0

    def generate(self, **_: Any) -> LLMResponse:
        self.attempts += 1
        raise PermanentClientError("401 invalid API key")


def main() -> None:
    agent = Agent(
        name="Reliable Agent",
        instructions="Answer briefly.",
        client=FlakyClient(),
        config=AgentConfig(max_retries=3, retry_base_delay=0.05, retry_max_delay=0.1),
    )
    result = agent.run("Say hello.")
    print(f"transient case -> status={result.status} output={result.output!r}")

    strict = Agent(
        name="No Retry Agent",
        instructions="Answer briefly.",
        client=BrokenClient(),
        config=AgentConfig(max_retries=3, retry_base_delay=0.0, retry_max_delay=0.0),
    )
    try:
        strict.run("Say hello.")
    except (PermanentClientError, RetryExhaustedError) as exc:
        print(f"permanent case -> no retry, raised {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
