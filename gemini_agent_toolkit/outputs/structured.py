"""Validated structured output built on Pydantic models."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from ..exceptions import StructuredOutputError
from .validators import is_pydantic_model, parse_json_payload, sanitize_json_schema


class StructuredOutput:
    """Binds a Pydantic model to a Gemini request and validates the answer."""

    def __init__(self, schema: type[BaseModel]) -> None:
        if not is_pydantic_model(schema):
            raise TypeError(
                "output_schema must be a pydantic BaseModel subclass, "
                f"got {schema!r}"
            )
        self.schema = schema

    # -- request side ----------------------------------------------------- #
    @property
    def json_schema(self) -> dict[str, Any]:
        return sanitize_json_schema(self.schema.model_json_schema())

    @property
    def request_kwargs(self) -> dict[str, Any]:
        """Extra config passed to the client for this call."""
        return {
            "response_mime_type": "application/json",
            "response_json_schema": self.json_schema,
        }

    # -- response side ---------------------------------------------------- #
    def validate_text(self, text: str | None) -> BaseModel:
        """Parse and validate ``text`` into an instance of the schema.

        Raises:
            StructuredOutputError: on empty, non-JSON or schema-invalid answers.
        """
        if not text or not text.strip():
            raise StructuredOutputError(
                "model returned an empty answer where structured output was required",
                raw_text=text,
            )
        try:
            payload = parse_json_payload(text)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"model answer is not valid JSON: {exc.msg}",
                raw_text=text,
            ) from exc

        if not isinstance(payload, dict):
            raise StructuredOutputError(
                f"expected a JSON object for {self.schema.__name__}, got {type(payload).__name__}",
                raw_text=text,
            )

        try:
            return self.schema.model_validate(payload)
        except ValidationError as exc:
            raise StructuredOutputError(
                f"model answer does not match {self.schema.__name__}: "
                f"{exc.error_count()} validation error(s)",
                raw_text=text,
                errors=exc.errors(),
            ) from exc

    def describe_errors(self, error: StructuredOutputError) -> str:
        """Human readable list of validation failures."""
        if not error.errors:
            return str(error)
        return "\n".join(
            f"- {'.'.join(str(p) for p in item['loc']) or '<root>'}: {item['msg']}"
            for item in error.errors
        )


def prepare_output_schema(output_schema: type[BaseModel] | None) -> StructuredOutput | None:
    """Normalise the ``output_schema`` argument accepted by ``Agent.run``."""
    if output_schema is None:
        return None
    if not is_pydantic_model(output_schema):
        raise TypeError(
            "output_schema must be a pydantic BaseModel subclass, "
            f"got {output_schema!r}"
        )
    return StructuredOutput(output_schema)


__all__ = ["StructuredOutput", "prepare_output_schema"]
