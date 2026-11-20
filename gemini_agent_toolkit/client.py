"""Thin, testable adapter around Google's official ``google-genai`` SDK.

Design notes
------------
* The SDK is imported lazily so importing the toolkit never requires an API key
  or a network stack.
* Automatic function calling is explicitly disabled: the toolkit executes tools
  itself so that approval gates, timeouts and audit logging cannot be bypassed.
* Any object exposing ``generate(...) -> LLMResponse`` satisfies :class:`LLMClient`,
  which is how tests and alternate backends are injected.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from .config import AgentConfig
from .exceptions import (
    ClientError,
    ModelResponseError,
    PermanentClientError,
    TransientClientError,
)
from .models import FunctionCallRequest, LLMResponse, UsageSummary
from .reliability.retry import is_transient
from .reliability.timeout import run_with_timeout


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface the runtime depends on."""

    def generate(
        self,
        *,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        **generation_kwargs: Any,
    ) -> LLMResponse:  # pragma: no cover - protocol definition
        ...


def _usage_from_metadata(metadata: Any) -> UsageSummary | None:
    if metadata is None:
        return None
    return UsageSummary(
        prompt_tokens=getattr(metadata, "prompt_token_count", None),
        candidate_tokens=getattr(metadata, "candidates_token_count", None),
        thought_tokens=getattr(metadata, "thoughts_token_count", None),
        total_tokens=getattr(metadata, "total_token_count", None),
    )


def _text_of(response: Any) -> str | None:
    try:
        text = getattr(response, "text", None)
    except Exception:  # pragma: no cover - SDK raises when no text parts exist
        return None
    return text or None


def _function_calls_of(response: Any) -> tuple[list[FunctionCallRequest], Any]:
    """Extract tool calls and the raw model turn (needed to echo thinking parts)."""
    calls: list[FunctionCallRequest] = []
    raw_content = None
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        if content is not None and raw_content is None:
            raw_content = content
        for part in getattr(content, "parts", None) or []:
            call = getattr(part, "function_call", None)
            if call is None:
                continue
            calls.append(
                FunctionCallRequest(
                    name=call.name or "",
                    args=dict(call.args or {}),
                    call_id=getattr(call, "id", None),
                )
            )
    if not calls:
        # The SDK also exposes a flat ``response.function_calls`` shortcut; use it
        # when candidates are absent (or a fake/mocked response only sets it).
        for call in getattr(response, "function_calls", None) or []:
            calls.append(
                FunctionCallRequest(
                    name=getattr(call, "name", "") or "",
                    args=dict(getattr(call, "args", None) or {}),
                    call_id=getattr(call, "id", None),
                )
            )
    return calls, raw_content


class GeminiClient:
    """Synchronous Gemini client with configurable generation settings."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        client: Any | None = None,
        generate_call: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self._client = client
        self._generate_call = generate_call

    # -- internals -------------------------------------------------------- #
    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ClientError(
                    "google-genai is not installed. Run: pip install gemini-agent-toolkit"
                ) from exc
            from google.genai import types

            timeout_ms = int(self.config.timeout * 1000) if self.config.timeout else None
            http_options = types.HttpOptions(timeout=timeout_ms) if timeout_ms else None
            self._client = genai.Client(
                api_key=self.config.resolve_api_key(), http_options=http_options
            )
        return self._client

    def build_request_config(
        self,
        *,
        system_instruction: str | None,
        tools: list[dict[str, Any]] | None,
        response_schema: dict[str, Any] | None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Build a ``GenerateContentConfig`` value for the SDK."""
        from google.genai import types

        overrides = generation_kwargs or {}
        params: dict[str, Any] = {}

        instruction = system_instruction or self.config.extra.get("system_instruction")
        if instruction:
            params["system_instruction"] = instruction

        temperature = overrides.get("temperature", self.config.temperature)
        if temperature is not None:
            params["temperature"] = temperature
        top_p = overrides.get("top_p", self.config.top_p)
        if top_p is not None:
            params["top_p"] = top_p
        max_tokens = overrides.get("max_output_tokens", self.config.max_output_tokens)
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        if tools:
            params["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=decl["name"],
                            description=decl.get("description"),
                            parameters_json_schema=decl.get("parameters") or {"type": "object"},
                        )
                        for decl in tools
                    ]
                )
            ]
            # The toolkit executes tools itself (approval + audit + limits).
            try:
                params["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            except AttributeError:  # pragma: no cover - older SDK surface
                pass

        if response_schema:
            params["response_mime_type"] = "application/json"
            params["response_json_schema"] = response_schema

        return types.GenerateContentConfig(**params)

    # -- public API ------------------------------------------------------- #
    def generate(
        self,
        *,
        model: str,
        contents: list[dict[str, Any]],
        system_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        **generation_kwargs: Any,
    ) -> LLMResponse:
        """Call the model once and normalise the response.

        Raises:
            TransientClientError: retryable API/network failure.
            PermanentClientError: non-retryable failure (auth, bad request).
        """
        config = self.build_request_config(
            system_instruction=system_instruction,
            tools=tools,
            response_schema=response_schema,
            generation_kwargs=generation_kwargs,
        )

        if self._generate_call is not None:
            invoke = lambda: self._generate_call(model=model, contents=contents, config=config)
        else:
            client = self._ensure_client()
            invoke = lambda: client.models.generate_content(
                model=model, contents=contents, config=config
            )

        try:
            response = run_with_timeout(invoke, self.config.timeout, label="gemini-generate")
        except ClientError:
            raise
        except BaseException as exc:  # noqa: BLE001 - normalised below
            message = f"Gemini request failed: {type(exc).__name__}: {exc}"
            if is_transient(exc):
                raise TransientClientError(message) from exc
            raise PermanentClientError(message) from exc

        calls, raw_content = _function_calls_of(response)
        text = _text_of(response)
        if text is None and not calls:
            raise ModelResponseError(
                "model returned neither text nor tool calls "
                f"(finish reason: {getattr(response.candidates[0], 'finish_reason', None) if getattr(response, 'candidates', None) else None})"
            )

        return LLMResponse(
            text=text,
            function_calls=calls,
            usage=_usage_from_metadata(getattr(response, "usage_metadata", None)),
            model_version=getattr(response, "model_version", None),
            raw_model_content=raw_content,
        )


__all__ = ["GeminiClient", "LLMClient"]
