"""Shared pytest fixtures: a scripted, offline stand-in for the Gemini client."""

from __future__ import annotations

from typing import Any, Iterable

import pytest

from gemini_agent_toolkit import Agent, AgentConfig, LLMResponse
from gemini_agent_toolkit.models import FunctionCallRequest, UsageSummary


class ScriptedClient:
    """Returns pre-baked responses (or raises pre-baked errors) in order."""

    def __init__(self, script: Iterable[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("ScriptedClient ran out of scripted responses")
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    @property
    def call_count(self) -> int:
        return len(self.calls)


def text_response(text: str, **kwargs: Any) -> LLMResponse:
    return LLMResponse(text=text, usage=UsageSummary(total_tokens=10), **kwargs)


def tool_response(*calls: tuple[str, dict[str, Any]]) -> LLMResponse:
    return LLMResponse(
        text=None,
        function_calls=[FunctionCallRequest(name=name, args=args) for name, args in calls],
        usage=UsageSummary(total_tokens=12),
    )


def build_agent(client: ScriptedClient, tools: Iterable[Any] | None = None, **kwargs: Any) -> Agent:
    config = AgentConfig(
        retry_base_delay=0.0,
        retry_max_delay=0.0,
        retry_jitter=0.0,
        **{k: v for k, v in kwargs.items() if k in {f for f in AgentConfig.__dataclass_fields__}},
    )
    agent_kwargs = {k: v for k, v in kwargs.items() if k not in {f for f in AgentConfig.__dataclass_fields__}}
    return Agent(
        name="Test Agent",
        instructions="You are a test agent.",
        tools=tools,
        config=config,
        client=client,
        **agent_kwargs,
    )


@pytest.fixture
def scripted_client() -> type[ScriptedClient]:
    return ScriptedClient
