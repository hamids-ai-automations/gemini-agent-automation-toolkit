"""Tool declaration, validation and registration."""

from __future__ import annotations

import pytest

from gemini_agent_toolkit import ToolRegistry, tool
from gemini_agent_toolkit.exceptions import ToolArgumentError, ToolError, ToolNotFoundError
from gemini_agent_toolkit.tools.decorators import get_tool_spec


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: first addend.
        b: second addend.
    """
    return a + b


@tool(requires_approval=True, approval_reason="writes files", idempotent=True)
def write_note(path: str, content: str = "") -> str:
    """Write a note to disk.

    Args:
        path: destination path.
        content: text to write.
    """
    return f"wrote {len(content)} chars to {path}"


def test_decorator_attaches_spec_and_keeps_callable() -> None:
    spec = get_tool_spec(add)
    assert spec is not None
    assert spec.name == "add"
    assert spec.description.startswith("Add two integers")
    assert add(1, 2) == 3


def test_declaration_contains_parameters_and_descriptions() -> None:
    declaration = get_tool_spec(add).declaration()  # type: ignore[union-attr]
    assert declaration["name"] == "add"
    assert set(declaration["parameters"]["properties"]) == {"a", "b"}
    assert declaration["parameters"]["required"] == ["a", "b"]
    assert declaration["parameters"]["properties"]["a"]["description"] == "first addend."


def test_schema_has_no_decorative_keys() -> None:
    schema = get_tool_spec(add).parameters_schema()  # type: ignore[union-attr]
    assert "title" not in schema


def test_invoke_coerces_and_validates_arguments() -> None:
    spec = get_tool_spec(add)
    assert spec.invoke({"a": "2", "b": 3}) == 5


def test_unknown_argument_is_rejected() -> None:
    spec = get_tool_spec(add)
    with pytest.raises(ToolArgumentError):
        spec.invoke({"a": 1, "b": 2, "c": 3})


def test_bad_argument_type_is_rejected() -> None:
    spec = get_tool_spec(add)
    with pytest.raises(ToolArgumentError):
        spec.invoke({"a": "not-a-number", "b": 1})


def test_varargs_cannot_be_exposed() -> None:
    with pytest.raises(ToolError):

        @tool
        def variadic(*args: int) -> int:  # noqa: ANN002
            """Bad tool."""
            return len(args)


def test_invalid_tool_name_is_rejected() -> None:
    with pytest.raises(ToolError):

        @tool(name="9nope!")
        def broken() -> None:
            """Bad name."""


def test_registry_registers_decorated_function_with_metadata() -> None:
    registry = ToolRegistry([add, write_note])
    spec = registry.get("write_note")
    assert spec.requires_approval is True
    assert spec.idempotent is True
    assert spec.approval_reason == "writes files"
    assert registry.names() == ["add", "write_note"]
    assert len(registry.declarations()) == 2


def test_registry_can_override_metadata_for_plain_function() -> None:
    def ping(host: str) -> str:
        """Ping a host."""
        return host

    registry = ToolRegistry()
    registry.register(ping, requires_approval=True)
    assert registry.get("ping").requires_approval is True


def test_duplicate_registration_fails() -> None:
    registry = ToolRegistry([add])
    with pytest.raises(ToolError):
        registry.register(add)


def test_unknown_tool_lookup_fails() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError) as exc:
        registry.get("missing")
    assert "missing" in str(exc.value)


def test_registry_rejects_non_callable() -> None:
    with pytest.raises(ToolError):
        ToolRegistry(["not-a-function"])
