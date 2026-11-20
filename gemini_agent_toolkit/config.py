"""Configuration objects with validation and environment based secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from .exceptions import ConfigError

#: Model used when the caller does not specify one. Model names are never
#: hardcoded anywhere else in the package.
DEFAULT_MODEL = "gemini-3.8-flash"

#: Name of the environment variable holding the Gemini API key.
API_KEY_ENV_VAR = "GEMINI_API_KEY"

_VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


@dataclass(frozen=True)
class AgentConfig:
    """Immutable runtime configuration.

    Secrets are never stored in logs: :meth:`redact_values` exposes only the
    values that must be scrubbed from log records.
    """

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = 2048
    timeout: float = 60.0
    max_retries: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 8.0
    retry_jitter: float = 0.1
    max_tool_calls: int = 8
    max_tool_iterations: int = 6
    max_repeated_tool_calls: int = 2
    approval_ttl_seconds: float = 900.0
    approval_expires: bool = True
    log_level: str = "INFO"
    json_logs: bool = False
    logger_name: str = "gemini_agent_toolkit"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model or not isinstance(self.model, str):
            raise ConfigError("model must be a non-empty string")
        if self.timeout <= 0:
            raise ConfigError("timeout must be > 0")
        if self.max_retries < 0:
            raise ConfigError("max_retries must be >= 0")
        if self.retry_base_delay < 0 or self.retry_max_delay < 0:
            raise ConfigError("retry delays must be >= 0")
        if self.retry_max_delay < self.retry_base_delay:
            raise ConfigError("retry_max_delay must be >= retry_base_delay")
        if self.max_tool_calls < 0:
            raise ConfigError("max_tool_calls must be >= 0")
        if self.max_tool_iterations < 0:
            raise ConfigError("max_tool_iterations must be >= 0")
        if self.max_repeated_tool_calls < 1:
            raise ConfigError("max_repeated_tool_calls must be >= 1")
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ConfigError("temperature must be between 0.0 and 2.0")
        if self.top_p is not None and not 0.0 <= self.top_p <= 1.0:
            raise ConfigError("top_p must be between 0.0 and 1.0")
        if self.log_level.upper() not in _VALID_LOG_LEVELS:
            raise ConfigError(f"invalid log_level: {self.log_level!r}")

    # -- secrets ---------------------------------------------------------- #
    def resolve_api_key(self) -> str:
        """Return the API key from config or environment.

        Raises:
            MissingAPIKeyError: when neither source provides a key.
        """
        from .exceptions import MissingAPIKeyError

        key = self.api_key or os.environ.get(API_KEY_ENV_VAR)
        if not key:
            raise MissingAPIKeyError(
                f"No Gemini API key found. Set the {API_KEY_ENV_VAR} environment "
                "variable or pass api_key=... explicitly."
            )
        return key

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key or os.environ.get(API_KEY_ENV_VAR))

    def redact_values(self) -> tuple[str, ...]:
        """Literal strings that must never appear in logs."""
        key = self.api_key or os.environ.get(API_KEY_ENV_VAR) or ""
        return (key,) if key else ()

    # -- helpers ---------------------------------------------------------- #
    def with_overrides(self, **overrides: Any) -> AgentConfig:
        """Return a copy with only the provided (non-None) values replaced."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        if not clean:
            return self
        unknown = set(clean) - set(self.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")
        return replace(self, **clean)
