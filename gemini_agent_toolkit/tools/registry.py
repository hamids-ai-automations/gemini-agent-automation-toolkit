"""Registry of tools exposed to the model."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from ..exceptions import ToolError, ToolNotFoundError
from .decorators import ToolSpec, get_tool_spec


class ToolRegistry:
    """An ordered, name-addressed collection of :class:`ToolSpec` objects."""

    def __init__(self, tools: Iterable[Any] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for item in tools or ():
            self.register(item)

    # -- registration ----------------------------------------------------- #
    def register(
        self,
        target: ToolSpec | Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        requires_approval: bool | None = None,
        idempotent: bool | None = None,
        timeout: float | None = None,
        approval_reason: str | None = None,
        tags: Iterable[str] = (),
    ) -> ToolSpec:
        """Register a decorated function, a plain callable or an explicit spec.

        Raises:
            ToolError: on duplicate names or non-callable input.
        """
        if isinstance(target, ToolSpec):
            spec = target
        else:
            if not callable(target):
                raise ToolError(f"cannot register non-callable object: {type(target)!r}")
            existing = get_tool_spec(target)
            if (
                existing is not None
                and not any(
                    v is not None
                    for v in (
                        name,
                        description,
                        requires_approval,
                        idempotent,
                        timeout,
                        approval_reason,
                    )
                )
                and not tuple(tags)
            ):
                spec = existing
            else:
                base = existing
                spec = ToolSpec.from_function(
                    target,
                    name=name or (base.name if base else None),
                    description=description or (base.description if base else None),
                    requires_approval=(
                        base.requires_approval
                        if requires_approval is None and base
                        else bool(requires_approval)
                    ),
                    idempotent=(
                        base.idempotent if idempotent is None and base else bool(idempotent)
                    ),
                    timeout=timeout if timeout is not None else (base.timeout if base else None),
                    approval_reason=approval_reason or (base.approval_reason if base else None),
                    tags=tuple(tags) or (base.tags if base else ()),
                )

        if spec.name in self._tools:
            raise ToolError(f"tool '{spec.name}' is already registered")
        self._tools[spec.name] = spec
        return spec

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # -- lookup ----------------------------------------------------------- #
    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(
                f"tool '{name}' is not registered; available tools: {sorted(self._tools) or 'none'}"
            ) from exc

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    # -- model payload ---------------------------------------------------- #
    def declarations(self) -> builtins.list[dict[str, Any]]:
        """Function declaration dicts ready to be handed to the API."""
        return [spec.declaration() for spec in self._tools.values()]


__all__ = ["ToolRegistry"]
