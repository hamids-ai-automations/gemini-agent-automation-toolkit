"""Client request building and response normalisation (no network)."""

from __future__ import annotations

from typing import Any

import pytest

from gemini_agent_toolkit import GeminiClient, TransientClientError
from gemini_agent_toolkit.exceptions import ClientError, ModelResponseError


class FakeCall:
    def __init__(self, name: str, args: dict[str, Any], id: str | None = None) -> None:
        self.name = name
        self.args = args
        self.id = id


class FakeResponse:
    def __init__(
        self,
        text: str | None = "ok",
        function_calls: list[FakeCall] | None = None,
        usage: Any = None,
        model_version: str = "gemini-test",
    ) -> None:
        self.text = text
        self.function_calls = function_calls or []
        self.candidates = []
        self.usage_metadata = usage
        self.model_version = model_version


def test_build_request_config_carries_system_instruction_and_sampling() -> None:
    client = GeminiClient()
    config = client.build_request_config(
        system_instruction="be brief",
        tools=None,
        response_schema=None,
        generation_kwargs={"temperature": 0.1},
    )
    assert "be brief" in str(config.system_instruction)
    assert config.temperature == pytest.approx(0.1)


def test_build_request_config_declares_tools_and_schema() -> None:
    client = GeminiClient()
    config = client.build_request_config(
        system_instruction=None,
        tools=[{"name": "add", "description": "add", "parameters": {"type": "object", "properties": {}}}],
        response_schema={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema is not None
    assert config.tools


def test_generate_normalises_text_response() -> None:
    client = GeminiClient(generate_call=lambda **_: FakeResponse(text="hello"))
    response = client.generate(model="gemini-test", contents=[])
    assert response.text == "hello"
    assert response.function_calls == []
    assert response.model_version == "gemini-test"


def test_generate_normalises_function_calls() -> None:
    payload = FakeResponse(text=None, function_calls=[FakeCall("add", {"a": 1, "b": 2}, "call-1")])
    client = GeminiClient(generate_call=lambda **_: payload)
    response = client.generate(model="gemini-test", contents=[])
    assert [c.name for c in response.function_calls] == ["add"]
    assert response.function_calls[0].args == {"a": 1, "b": 2}
    assert response.function_calls[0].call_id == "call-1"


def test_empty_response_is_an_error() -> None:
    client = GeminiClient(generate_call=lambda **_: FakeResponse(text=None))
    with pytest.raises(ModelResponseError):
        client.generate(model="gemini-test", contents=[])


def test_timeouts_become_transient_errors() -> None:
    def boom(**_: Any) -> FakeResponse:
        raise TimeoutError("read timed out")

    client = GeminiClient(generate_call=boom)
    with pytest.raises(TransientClientError):
        client.generate(model="gemini-test", contents=[])


def test_sdk_errors_are_wrapped() -> None:
    def boom(**_: Any) -> FakeResponse:
        raise RuntimeError("sdk exploded")

    client = GeminiClient(generate_call=boom)
    with pytest.raises(ClientError):
        client.generate(model="gemini-test", contents=[])
