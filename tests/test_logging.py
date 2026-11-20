"""Structured logging: redaction of secrets and stable event shape."""

from __future__ import annotations

import json
import logging

from gemini_agent_toolkit.logging import REDACTED, ExecutionLogger, redact


def test_secret_fields_are_redacted() -> None:
    redacted = redact(
        {
            "api_key": "abc",
            "nested": {"authorization": "Bearer x", "ok": 1},
            "list": [{"token": "t"}],
        }
    )
    assert redacted["api_key"] == REDACTED
    assert redacted["nested"]["authorization"] == REDACTED
    assert redacted["nested"]["ok"] == 1
    assert redacted["list"][0]["token"] == REDACTED


def test_secret_looking_strings_are_masked() -> None:
    text = redact("key=AIzaSyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    assert "AIzaSy" not in text
    assert "REDACTED" in text


def test_explicit_literal_values_are_masked() -> None:
    logged: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logged.append(record.getMessage())

    handler = Capture()
    logger = ExecutionLogger(level="INFO", redact_values=("super-secret",))
    logger._logger.addHandler(handler)
    try:
        logger.event("run_started", note="used super-secret here")
    finally:
        logger._logger.removeHandler(handler)

    assert logged
    assert "super-secret" not in logged[-1]


def test_events_are_logged_with_toolkit_logger(caplog) -> None:
    logger = ExecutionLogger(level="INFO")
    with caplog.at_level(logging.INFO):
        logger.event("tool_started", tool="add", attempt=1)

    matching = [r for r in caplog.records if r.name.startswith("gemini_agent_toolkit")]
    assert matching
    assert any("tool_started" in record.getMessage() for record in matching)


def test_json_output_is_parseable(caplog) -> None:
    logger = ExecutionLogger(level="INFO", json_output=True)
    with caplog.at_level(logging.INFO):
        logger.event("run_finished", status="success", tool_calls=2)

    payloads = []
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith("{"):
            payloads.append(json.loads(message))
    assert payloads
    assert payloads[-1]["event"] == "run_finished"
    assert payloads[-1]["status"] == "success"
