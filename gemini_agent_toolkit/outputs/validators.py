"""Helpers for JSON schema preparation, parsing and repair."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def is_pydantic_model(value: Any) -> bool:
    """True when ``value`` is a Pydantic model *class*."""
    return isinstance(value, type) and issubclass(value, BaseModel)


def sanitize_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove decorative keys from a Pydantic JSON schema.

    Gemini accepts a JSON Schema subset; ``title``, ``default`` and ``examples``
    add tokens without improving the model's understanding.
    """
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"title", "default", "examples", "$schema"}:
            continue
        if isinstance(value, dict):
            cleaned[key] = sanitize_json_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                sanitize_json_schema(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def strip_markdown_fences(text: str) -> str:
    """Return the payload inside ```json fences when the model adds them."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    match = _FENCE.match(text)
    return match.group(1) if match else text.strip()


def parse_json_payload(text: str) -> Any:
    """Parse JSON, tolerating surrounding prose or code fences.

    Raises:
        json.JSONDecodeError: when no JSON value can be recovered.
    """
    candidate = strip_markdown_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as first_error:
        start = min(
            (idx for idx in (candidate.find("{"), candidate.find("[")) if idx >= 0),
            default=-1,
        )
        if start < 0:
            raise first_error
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        if end <= start:
            raise first_error
        return json.loads(candidate[start : end + 1])


__all__ = [
    "is_pydantic_model",
    "parse_json_payload",
    "sanitize_json_schema",
    "strip_markdown_fences",
]
