"""Structured, redacting logging for agent runs.

Only standard-library logging is used, so output can be routed anywhere
(console, file, JSON shipper, cloud agent) by configuring a handler on the
``gemini_agent_toolkit`` logger.
"""

from __future__ import annotations

import json
import logging as _logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

LOGGER_NAME = "gemini_agent_toolkit"
REDACTED = "[REDACTED]"

#: Keys whose values must never reach a log sink.
SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|apikey|secret|token|password|passwd|authorization|credential|bearer|private[_-]?key)",
    re.IGNORECASE,
)

#: Value-level scrubbing for things that look like Google API keys.
KEY_LIKE_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")

_MAX_DEPTH = 6


def redact(value: Any, *, extra_values: tuple[str, ...] = (), depth: int = 0) -> Any:
    """Return a copy of ``value`` with secrets replaced by ``[REDACTED]``.

    Handles nested mappings/sequences, scrubs key names matching
    :data:`SENSITIVE_KEY_PATTERN` and any literal in ``extra_values``.
    """
    if depth > _MAX_DEPTH:
        return "<max-depth>"

    if isinstance(value, str):
        scrubbed = value
        for literal in extra_values:
            if literal:
                scrubbed = scrubbed.replace(literal, REDACTED)
        return KEY_LIKE_PATTERN.sub(REDACTED, scrubbed)

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            # Counts are not secrets: ``total_tokens=123`` must stay readable.
            if key_text.lower().endswith("_tokens"):
                result[key_text] = redact(item, extra_values=extra_values, depth=depth + 1)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                result[key_text] = item
            elif SENSITIVE_KEY_PATTERN.search(key_text):
                result[key_text] = REDACTED
            else:
                result[key_text] = redact(
                    item, extra_values=extra_values, depth=depth + 1
                )
        return result

    if isinstance(value, (list, tuple, set)):
        return [redact(item, extra_values=extra_values, depth=depth + 1) for item in value]

    return value


@dataclass
class ExecutionLogger:
    """Emits one structured event per agent/tool lifecycle transition."""

    name: str = LOGGER_NAME
    level: str = "INFO"
    json_output: bool = False
    redact_values: tuple[str, ...] = field(default_factory=tuple)
    _logger: _logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = _logging.getLogger(self.name)
        self._logger.setLevel(getattr(_logging, str(self.level).upper(), _logging.INFO))

    # -- events ----------------------------------------------------------- #
    def event(self, event: str, /, **fields: Any) -> None:
        """Log ``event`` with redacted structured fields."""
        payload = redact(fields, extra_values=self.redact_values)
        level = _LEVELS.get(event, _logging.INFO)
        if self.json_output:
            record = {"event": event, "logger": self.name, **payload}
            self._logger.log(level, json.dumps(record, default=str, sort_keys=False))
        else:
            detail = " ".join(f"{key}={value!r}" for key, value in payload.items())
            self._logger.log(level, "%s %s", event, detail)

    def debug(self, event: str, **fields: Any) -> None:
        self.event(event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(_logging.WARNING, event, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(_logging.ERROR, event, fields)

    def _emit(self, level: int, event: str, fields: dict[str, Any]) -> None:
        payload = redact(fields, extra_values=self.redact_values)
        if self.json_output:
            self._logger.log(level, json.dumps({"event": event, "logger": self.name, **payload}, default=str))
        else:
            detail = " ".join(f"{key}={value!r}" for key, value in payload.items())
            self._logger.log(level, "%s %s", event, detail)


_LEVELS = {
    "run_started": _logging.INFO,
    "run_completed": _logging.INFO,
    "tool_call": _logging.INFO,
    "tool_result": _logging.INFO,
    "retry": _logging.WARNING,
    "approval_requested": _logging.INFO,
    "approval_decided": _logging.INFO,
    "run_failed": _logging.ERROR,
}


def configure_logging(
    *, level: str = "INFO", json_output: bool = False, handler: _logging.Handler | None = None
) -> _logging.Logger:
    """Initialise the package logger once and return it.

    Calling this twice does not duplicate handlers.
    """
    logger = _logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(_logging, level.upper(), _logging.INFO))
    if handler is not None:
        handler.setFormatter(
            _logging.Formatter('{"event": "%(message)s"}') if json_output else _logging.Formatter("%(levelname)s %(message)s")
        )
        logger.addHandler(handler)
    elif not logger.handlers:
        stream = _logging.StreamHandler()
        stream.setFormatter(_logging.Formatter("%(levelname)s %(name)s %(message)s"))
        logger.addHandler(stream)
    logger.propagate = False
    return logger


__all__ = [
    "ExecutionLogger",
    "LOGGER_NAME",
    "REDACTED",
    "SENSITIVE_KEY_PATTERN",
    "configure_logging",
    "redact",
]
