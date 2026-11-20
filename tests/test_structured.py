"""Structured output: schema preparation, fence stripping, validation."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from gemini_agent_toolkit import StructuredOutput
from gemini_agent_toolkit.exceptions import StructuredOutputError
from gemini_agent_toolkit.outputs.structured import prepare_output_schema


class Verdict(BaseModel):
    status: Literal["pass", "fail"]
    score: int = Field(ge=0, le=100)
    notes: list[str] = Field(default_factory=list)


def test_schema_is_gemini_friendly() -> None:
    schema = StructuredOutput(Verdict).json_schema
    assert "title" not in schema
    assert schema["properties"]["status"]["enum"] == ["pass", "fail"]


def test_prepare_rejects_non_models() -> None:
    with pytest.raises(TypeError):
        prepare_output_schema(dict)  # type: ignore[arg-type]


def test_parses_plain_json() -> None:
    output = StructuredOutput(Verdict)
    result = output.validate_text('{"status": "pass", "score": 91}')
    assert isinstance(result, Verdict)
    assert result.notes == []


def test_strips_markdown_fences() -> None:
    output = StructuredOutput(Verdict)
    raw = '```json\n{"status": "fail", "score": 0, "notes": ["missing h1"]}\n```'
    assert output.validate_text(raw).notes == ["missing h1"]


def test_invalid_payload_raises_with_details() -> None:
    output = StructuredOutput(Verdict)
    with pytest.raises(StructuredOutputError) as exc:
        output.validate_text('{"status": "maybe", "score": 140}')
    assert exc.value.raw_text is not None
    assert exc.value.errors


def test_empty_answer_raises() -> None:
    with pytest.raises(StructuredOutputError):
        StructuredOutput(Verdict).validate_text("   ")


def test_non_json_answer_raises() -> None:
    with pytest.raises(StructuredOutputError):
        StructuredOutput(Verdict).validate_text("I cannot do that.")


def test_validate_reports_ok() -> None:
    parsed = StructuredOutput(Verdict).validate_text('{"status": "pass", "score": 10}')
    assert isinstance(parsed, Verdict)
    assert parsed.score == 10
