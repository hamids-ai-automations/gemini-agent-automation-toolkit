"""Configuration validation and secret handling."""

from __future__ import annotations

import dataclasses

import pytest

from gemini_agent_toolkit import API_KEY_ENV_VAR, DEFAULT_MODEL, AgentConfig
from gemini_agent_toolkit.exceptions import ConfigError


def test_defaults_are_sane() -> None:
    config = AgentConfig()
    assert config.model == DEFAULT_MODEL
    assert config.max_retries == 3
    assert config.timeout > 0
    assert config.approval_expires is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", ""),
        ("timeout", 0),
        ("max_retries", -1),
        ("max_tool_calls", -1),
        ("max_repeated_tool_calls", 0),
        ("temperature", 3.0),
        ("top_p", 1.5),
        ("retry_base_delay", -0.1),
    ],
)
def test_invalid_values_raise(field: str, value: object) -> None:
    with pytest.raises(ConfigError):
        AgentConfig(**{field: value})


def test_config_is_frozen() -> None:
    config = AgentConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.model = "other"  # type: ignore[misc]


def test_with_overrides_replaces_only_given_fields() -> None:
    config = AgentConfig(temperature=0.2)
    updated = config.with_overrides(model="gemini-x", timeout=None)
    assert updated.model == "gemini-x"
    assert updated.timeout == config.timeout
    assert updated.temperature == 0.2


def test_with_overrides_rejects_unknown_fields() -> None:
    with pytest.raises(ConfigError):
        AgentConfig().with_overrides(not_a_field=1)


def test_api_key_resolution_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "AIza-env-key")
    config = AgentConfig()
    assert config.has_api_key is True
    assert config.resolve_api_key() == "AIza-env-key"
    assert config.redact_values() == ("AIza-env-key",)


def test_missing_api_key_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    config = AgentConfig()
    assert config.has_api_key is False
    with pytest.raises(Exception) as exc:
        config.resolve_api_key()
    assert "GEMINI_API_KEY" in str(exc.value)


def test_explicit_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")
    assert AgentConfig(api_key="explicit").resolve_api_key() == "explicit"
