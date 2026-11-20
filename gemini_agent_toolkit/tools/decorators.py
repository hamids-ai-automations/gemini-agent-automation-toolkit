"""Declare Python functions as agent tools.

The decorator is intentionally small: it attaches a :class:`ToolSpec` to the
function object, so the developer keeps a normal Python callable while the
runtime gets a validated schema and approval metadata.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, get_type_hints

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from ..exceptions import ToolArgumentError, ToolError

_EMPTY = inspect.Parameter.empty
_ARG_LINE = re.compile(r"^\s*([*]{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def _parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into a summary and per-argument descriptions."""
    if not docstring:
        return "", {}
    lines = docstring.expandtabs().strip().splitlines()
    summary_parts: list[str] = []
    arg_docs: dict[str, str] = {}
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in {"args:", "arguments:", "parameters:"}:
            in_args = True
            continue
        if in_args and stripped.lower() in {"returns:", "raises:", "yields:", "examples:"}:
            in_args = False
            continue
        if in_args:
            match = _ARG_LINE.match(line)
            if match:
                arg_docs[match.group(1).lstrip("*")] = match.group(2).strip()
            continue
        if stripped:
            summary_parts.append(stripped)
    return " ".join(summary_parts).strip(), arg_docs


def build_input_model(func: Callable[..., Any], *, model_name: str) -> type[BaseModel]:
    """Create a Pydantic model from a function signature.

    Raises:
        ToolError: when the function uses varargs/kwargs, which cannot be
            described to a function-calling API.
    """
    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:  # pragma: no cover - unresolvable forward refs
        hints = {}

    _, arg_docs = _parse_docstring(func.__doc__)
    fields: dict[str, tuple[Any, Any]] = {}

    for name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ToolError(
                f"tool '{func.__name__}' uses *args/**kwargs, which cannot be "
                "described to the model; declare explicit parameters instead"
            )
        annotation = hints.get(name, param.annotation)
        if annotation is _EMPTY:
            annotation = Any
        default = ... if param.default is _EMPTY else param.default
        description = arg_docs.get(name, "")
        fields[name] = (
            annotation,
            _FieldInfo(default=default, description=description),
        )

    model = create_model(f"{model_name}_Input", __config__=ConfigDict(extra="forbid"), **fields)
    return model


def _FieldInfo(*, default: Any, description: str) -> Any:
    from pydantic import Field

    if default is ...:
        return Field(..., description=description or None)
    return Field(default, description=description or None)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop keys Gemini's JSON-schema subset does not need."""
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"title", "default", "examples", "$schema"}:
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _clean_schema(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


@dataclass
class ToolSpec:
    """Everything the runtime knows about one tool."""

    func: Callable[..., Any]
    name: str
    description: str
    input_model: type[BaseModel]
    requires_approval: bool = False
    idempotent: bool = False
    timeout: float | None = None
    approval_reason: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        requires_approval: bool = False,
        idempotent: bool = False,
        timeout: float | None = None,
        approval_reason: str | None = None,
        tags: Iterable[str] = (),
    ) -> "ToolSpec":
        if not callable(func):
            raise ToolError(f"tool target must be callable, got {type(func)!r}")
        tool_name = name or func.__name__
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}", tool_name):
            raise ToolError(
                f"invalid tool name {tool_name!r}: use letters, digits, underscores, "
                "dots or dashes (max 64 characters, must not start with a digit)"
            )
        summary, _ = _parse_docstring(func.__doc__)
        return cls(
            func=func,
            name=tool_name,
            description=description or summary or f"Call {tool_name}.",
            input_model=build_input_model(func, model_name=tool_name),
            requires_approval=requires_approval,
            idempotent=idempotent,
            timeout=timeout,
            approval_reason=approval_reason,
            tags=tuple(tags),
        )

    # -- behaviour -------------------------------------------------------- #
    def parameters_schema(self) -> dict[str, Any]:
        """JSON schema describing the arguments accepted by this tool."""
        return _clean_schema(self.input_model.model_json_schema())

    def declaration(self) -> dict[str, Any]:
        """Function declaration payload for the Gemini API."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema(),
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and coerce model-supplied arguments.

        Raises:
            ToolArgumentError: with the full validation report attached.
        """
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ToolArgumentError(
                f"arguments for tool '{self.name}' must be an object, got {type(arguments).__name__}"
            )
        try:
            model = self.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentError(
                f"invalid arguments for tool '{self.name}': {exc.error_count()} error(s): "
                + "; ".join(
                    f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
                    for err in exc.errors()
                )
            ) from exc
        return model.model_dump()

    def invoke(self, arguments: dict[str, Any]) -> Any:
        """Validate ``arguments`` then call the underlying function."""
        validated = self.validate_arguments(arguments)
        return self.func(**validated)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
    idempotent: bool = False,
    timeout: float | None = None,
    approval_reason: str | None = None,
    tags: Iterable[str] = (),
) -> Any:
    """Register a function as an agent tool.

    Works both bare (``@tool``) and configured (``@tool(requires_approval=True)``).
    The decorated object is still the original callable; metadata lives on
    ``func.__gemini_tool__`` and is picked up by :class:`ToolRegistry`.
    """
    options = {
        "name": name,
        "description": description,
        "requires_approval": requires_approval,
        "idempotent": idempotent,
        "timeout": timeout,
        "approval_reason": approval_reason,
        "tags": tags,
    }

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        spec = ToolSpec.from_function(target, **options)  # type: ignore[arg-type]
        setattr(target, "__gemini_tool__", spec)
        return target

    if func is not None:
        return decorate(func)
    return decorate


def get_tool_spec(func: Any) -> ToolSpec | None:
    """Return the attached spec, if the object was decorated with ``@tool``."""
    return getattr(func, "__gemini_tool__", None)


__all__ = ["ToolSpec", "build_input_model", "get_tool_spec", "tool"]
