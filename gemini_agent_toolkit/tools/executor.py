"""Validated, time-limited execution of tool calls."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from ..exceptions import ToolArgumentError, ToolExecutionError
from ..models import FunctionCallRequest, ToolCallRecord
from ..reliability.timeout import run_with_timeout
from .registry import ToolRegistry


def _json_safe(value: Any) -> Any:
    """Coerce a tool return value into something the API accepts back."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _json_safe(dump())
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


class ToolExecutor:
    """Runs a single tool call and returns an audit record.

    The executor never raises for tool failures: it records them so the model
    can see the error and decide what to do next. Argument validation and
    unknown-tool errors are therefore returned as records too.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        default_timeout: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.registry = registry
        self.default_timeout = default_timeout
        self._clock = clock

    def execute(
        self,
        call: FunctionCallRequest,
        *,
        approval_token: str | None = None,
    ) -> tuple[ToolCallRecord, Any]:
        """Execute ``call``.

        Returns:
            ``(record, value_for_model)`` where the second item is what gets
            sent back to Gemini as the function response payload.
        """
        started = self._clock()
        record = ToolCallRecord(
            name=call.name,
            arguments=dict(call.args or {}),
            approval_token=approval_token,
        )

        try:
            spec = self.registry.get(call.name)
        except Exception as exc:  # ToolNotFoundError
            record.error = str(exc)
            return record, {"error": str(exc)}

        try:
            validated = spec.validate_arguments(call.args or {})
        except ToolArgumentError as exc:
            record.error = str(exc)
            record.duration_ms = (self._clock() - started) * 1000
            return record, {"error": str(exc)}

        timeout = spec.timeout if spec.timeout is not None else self.default_timeout
        try:
            result = run_with_timeout(
                lambda: spec.invoke(validated), timeout, label=f"tool:{spec.name}"
            )
        except ToolArgumentError as exc:  # pragma: no cover - already validated
            record.error = str(exc)
            return record, {"error": str(exc)}
        except BaseException as exc:  # noqa: BLE001 - converted into a record
            wrapped = (
                exc
                if isinstance(exc, ToolExecutionError)
                else ToolExecutionError(spec.name, exc)
            )
            record.error = str(wrapped)
            record.duration_ms = (self._clock() - started) * 1000
            return record, {"error": str(wrapped)}

        record.ok = True
        record.result = result
        record.duration_ms = (self._clock() - started) * 1000
        return record, {"result": _json_safe(result)}


__all__ = ["ToolExecutor", "make_function_response"]


def make_function_response(call: FunctionCallRequest, payload: Any) -> dict[str, Any]:
    """Build the ``function_response`` turn sent back to the model."""
    part: dict[str, Any] = {
        "function_response": {
            "name": call.name,
            "response": payload if isinstance(payload, dict) else {"result": _json_safe(payload)},
        }
    }
    if call.call_id:
        part["function_response"]["id"] = call.call_id
    return {"role": "user", "parts": [part]}
