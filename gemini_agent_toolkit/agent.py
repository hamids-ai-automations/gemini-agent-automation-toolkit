"""The agent runtime: model conversation + validated, approved tool execution."""

from __future__ import annotations

from typing import Any, Sequence

from .approval.manager import ApprovalManager
from .approval.policies import ApprovalPolicy, DenyAllPolicy, describe_action
from .client import GeminiClient, LLMClient
from .config import AgentConfig
from .context import ConversationContext
from .exceptions import (
    ApprovalError,
    ClientError,
    RecursiveToolCallError,
    RetryExhaustedError,
    StructuredOutputError,
    ToolkitError,
    ToolArgumentError,
    ToolError,
    ToolLimitExceededError,
)
from .logging.logger import ExecutionLogger
from .models import (
    AgentResult,
    ApprovalRequest,
    ApprovalStatus,
    FunctionCallRequest,
    LLMResponse,
    RunStatus,
    ToolCallRecord,
    UsageSummary,
    new_execution_id,
    utc_now,
)
from .outputs.structured import StructuredOutput, prepare_output_schema
from .reliability.recovery import ExecutionGuard, IdempotentCache, arguments_fingerprint
from .reliability.retry import RetryPolicy, execute_with_retry
from .tools.executor import ToolExecutor, make_function_response
from .tools.registry import ToolRegistry


def _merge_usage(current: UsageSummary | None, new: UsageSummary | None) -> UsageSummary | None:
    if new is None:
        return current
    if current is None:
        return new

    def add(a: int | None, b: int | None) -> int | None:
        if a is None and b is None:
            return None
        return (a or 0) + (b or 0)

    return UsageSummary(
        prompt_tokens=add(current.prompt_tokens, new.prompt_tokens),
        candidate_tokens=add(current.candidate_tokens, new.candidate_tokens),
        thought_tokens=add(current.thought_tokens, new.thought_tokens),
        total_tokens=add(current.total_tokens, new.total_tokens),
    )


class Agent:
    """A Gemini-powered agent with tools, structured output and approvals.

    Example:
        >>> agent = Agent(name="Assistant", instructions="Be helpful.")
        >>> result = agent.run("Explain AI agents.")  # doctest: +SKIP
        >>> result.output  # doctest: +SKIP
    """

    def __init__(
        self,
        name: str,
        instructions: str = "",
        *,
        tools: Sequence[Any] | None = None,
        model: str | None = None,
        api_key: str | None = None,
        config: AgentConfig | None = None,
        client: LLMClient | None = None,
        approval_manager: ApprovalManager | None = None,
        approval_policy: ApprovalPolicy | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
        logger: ExecutionLogger | None = None,
    ) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("Agent requires a non-empty name")

        base = config or AgentConfig()
        self.config = base.with_overrides(
            model=model,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )
        self.name = name
        self.instructions = instructions

        self.registry = ToolRegistry(tools or [])
        self._client: LLMClient = client or GeminiClient(self.config)

        ttl = self.config.approval_ttl_seconds if self.config.approval_expires else None
        self.approval = approval_manager or ApprovalManager(
            policy=approval_policy or DenyAllPolicy(), default_ttl=ttl
        )
        if approval_manager is not None and approval_policy is not None:
            self.approval.policy = approval_policy

        self._executor = ToolExecutor(self.registry, default_timeout=self.config.timeout)
        self._retry_policy = RetryPolicy.from_config(self.config)
        self._idempotent = IdempotentCache()
        self._logger = logger or ExecutionLogger(
            level=self.config.log_level,
            json_output=self.config.json_logs,
            redact_values=self.config.redact_values(),
        )
        self._pending: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def tools(self) -> list[str]:
        """Names of the registered tools."""
        return self.registry.names()

    def run(
        self,
        prompt: str,
        *,
        output_schema: type | None = None,
        context: ConversationContext | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the agent on ``prompt`` and return an :class:`AgentResult`.

        Never raises for expected failures: inspect ``result.status`` and
        ``result.error`` instead. Programming errors (bad schema type, missing
        API key when a real client is used) still propagate.
        """
        structured = prepare_output_schema(output_schema)
        ctx = context if context is not None else ConversationContext()
        ctx.add_user(prompt)

        return self._drive(
            contents=ctx.to_list(),
            structured=structured,
            guard=ExecutionGuard(
                max_tool_calls=self.config.max_tool_calls,
                max_repeated_calls=self.config.max_repeated_tool_calls,
            ),
            execution_id=new_execution_id(),
            context=ctx,
            generation_kwargs=generation_kwargs,
        )

    def resume(
        self, execution_id: str | None = None, *, generation_kwargs: dict[str, Any] | None = None
    ) -> AgentResult:
        """Continue a run that stopped for approval.

        Approved calls execute now; still-pending calls keep the run pending;
        rejected calls are reported to the model as errors.
        """
        if execution_id is None:
            if not self._pending:
                raise ApprovalError("no pending run to resume")
            execution_id = next(reversed(self._pending))
        state = self._pending.pop(execution_id, None)
        if state is None:
            raise ApprovalError(f"unknown or already resumed execution: {execution_id}")

        contents = list(state["contents"])
        guard: ExecutionGuard = state["guard"]
        # Reuse the tokens handed to the human so a decision is never asked twice.
        resumed = {
            arguments_fingerprint(item.tool_name, item.arguments): item
            for item in state.get("pending_requests", [])
        }
        records, tool_turns, pending_calls, pending = self._execute_calls(
            state["calls"], guard, resumed=resumed
        )

        if pending:
            self._pending[execution_id] = {
                **state,
                "contents": contents,
                "calls": pending_calls,
                "pending_requests": pending,
            }
            return AgentResult(
                execution_id=execution_id,
                agent_name=self.name,
                status=RunStatus.PENDING_APPROVAL,
                tool_calls=records,
                pending_approvals=pending,
                model=self.config.model,
            )

        contents.extend(tool_turns)
        return self._drive(
            contents=contents,
            structured=state["structured"],
            guard=guard,
            execution_id=execution_id,
            context=state["context"],
            generation_kwargs=generation_kwargs or state.get("generation_kwargs"),
            tool_records=records,
            usage=state.get("usage"),
            started_at=state.get("started_at"),
        )

    # ------------------------------------------------------------------ #
    # Conversation driver
    # ------------------------------------------------------------------ #
    def _drive(
        self,
        *,
        contents: list[dict[str, Any]],
        structured: StructuredOutput | None,
        guard: ExecutionGuard,
        execution_id: str,
        context: ConversationContext,
        generation_kwargs: dict[str, Any] | None,
        tool_records: list[ToolCallRecord] | None = None,
        usage: UsageSummary | None = None,
        started_at: Any = None,
    ) -> AgentResult:
        records = list(tool_records or [])
        started = started_at or utc_now()
        self._logger.event(
            "run_started",
            execution_id=execution_id,
            agent=self.name,
            model=self.config.model,
            tools=self.registry.names(),
            structured_output=bool(structured),
        )

        for _iteration in range(self.config.max_tool_iterations + 1):
            try:
                response = self._call_model(contents, structured, generation_kwargs)
            except RetryExhaustedError as exc:
                return self._finish_failure(execution_id, records, started, usage, str(exc))
            except (ClientError, ToolkitError) as exc:
                return self._finish_failure(execution_id, records, started, usage, str(exc))

            usage = _merge_usage(usage, response.usage)

            if response.function_calls:
                self._append_model_turn(contents, response)
                try:
                    new_records, tool_turns, pending_calls, pending = self._execute_calls(
                        response.function_calls, guard
                    )
                except ToolLimitExceededError as exc:
                    result = self._finish(
                        execution_id, RunStatus.TOOL_LIMIT_REACHED, records, started, usage,
                        error=str(exc),
                    )
                    return result

                records.extend(new_records)
                if pending:
                    self._pending[execution_id] = {
                        "contents": contents,
                        "calls": pending_calls,
                        "pending_requests": pending,
                        "records": records,
                        "structured": structured,
                        "guard": guard,
                        "context": context,
                        "generation_kwargs": generation_kwargs,
                        "started_at": started,
                        "usage": usage,
                    }
                    self._logger.event(
                        "run_pending_approval",
                        execution_id=execution_id,
                        agent=self.name,
                        pending=[p.to_dict() for p in pending],
                    )
                    return self._finish(
                        execution_id, RunStatus.PENDING_APPROVAL, records, started, usage,
                        pending=pending,
                    )

                contents.extend(tool_turns)
                context.turns = list(contents)
                continue

            # Final answer.
            text = response.text
            if structured is not None:
                try:
                    output = structured.validate_text(text)
                except StructuredOutputError as exc:
                    self._logger.error(
                        "structured_output_invalid",
                        execution_id=execution_id,
                        error=str(exc),
                        details=structured.describe_errors(exc),
                    )
                    return self._finish(
                        execution_id, RunStatus.FAILED, records, started, usage,
                        error=str(exc), text=text,
                    )
                self._append_text_turn(contents, text)
                context.turns = list(contents)
                return self._finish(
                    execution_id, RunStatus.SUCCESS, records, started, usage,
                    output=output, text=text,
                )

            self._append_text_turn(contents, text)
            context.turns = list(contents)
            return self._finish(
                execution_id, RunStatus.SUCCESS, records, started, usage, output=text, text=text
            )

        return self._finish(
            execution_id,
            RunStatus.TOOL_LIMIT_REACHED,
            records,
            started,
            usage,
            error=(
                f"stopped after {self.config.max_tool_iterations} model turns without a "
                "final answer"
            ),
        )

    # ------------------------------------------------------------------ #
    # Model call
    # ------------------------------------------------------------------ #
    def _call_model(
        self,
        contents: list[dict[str, Any]],
        structured: StructuredOutput | None,
        generation_kwargs: dict[str, Any] | None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(generation_kwargs or {})
        if structured is not None:
            kwargs.update(structured.request_kwargs)

        def once() -> LLMResponse:
            response = self._client.generate(
                model=self.config.model,
                contents=contents,
                system_instruction=self.instructions or None,
                tools=self.registry.declarations() or None,
                response_schema=structured.json_schema if structured else None,
                **kwargs,
            )
            return response

        def on_retry(attempt: int, error: BaseException, delay: float) -> None:
            self._logger.event(
                "retry",
                attempt=attempt + 1,
                max_attempts=self._retry_policy.max_attempts,
                error=f"{type(error).__name__}: {error}",
                delay_seconds=round(delay, 3),
            )

        return execute_with_retry(once, self._retry_policy, on_retry=on_retry)

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #
    def _execute_calls(
        self,
        calls: Sequence[FunctionCallRequest],
        guard: ExecutionGuard,
        *,
        resumed: dict[str, ApprovalRequest] | None = None,
    ) -> tuple[
        list[ToolCallRecord],
        list[dict[str, Any]],
        list[FunctionCallRequest],
        list[ApprovalRequest],
    ]:
        """Execute every call that is allowed to run.

        Returns ``(records, function_response_turns, pending_calls, pending_approvals)``.
        Sensitive calls without an approval stay pending and are *not* executed.
        ``resumed`` re-uses approval tokens issued by an earlier attempt so a
        human decision is never silently replaced by a new request.
        """
        records: list[ToolCallRecord] = []
        turns: list[dict[str, Any]] = []
        pending_calls: list[FunctionCallRequest] = []
        pending: list[ApprovalRequest] = []
        resumed = resumed or {}

        for call in calls:
            try:
                spec = self.registry.get(call.name)
            except ToolError as exc:
                record = ToolCallRecord(name=call.name, arguments=dict(call.args), ok=False, error=str(exc))
                records.append(record)
                turns.append(make_function_response(call, {"error": str(exc)}))
                self._logger.warning("tool_unknown", tool=call.name, error=str(exc))
                continue

            self._logger.event(
                "tool_call",
                tool=spec.name,
                arguments=call.args,
                requires_approval=spec.requires_approval,
            )

            token: str | None = None
            if spec.requires_approval:
                existing = resumed.get(arguments_fingerprint(spec.name, call.args))
                request = (
                    existing
                    if existing is not None
                    else self.approval.request(
                        spec.name,
                        call.args,
                        reason=spec.approval_reason or f"{spec.name} is a sensitive action",
                    )
                )
                token = request.token
                self._logger.event(
                    "approval_requested",
                    tool=spec.name,
                    token=request.token,
                    status=request.status.value,
                    action=describe_action(request),
                )
                if request.status is not ApprovalStatus.APPROVED:
                    if request.status is ApprovalStatus.PENDING:
                        # Auditable: the run stopped, nothing was executed.
                        records.append(
                            ToolCallRecord(
                                name=spec.name,
                                arguments=dict(call.args),
                                ok=False,
                                error="action blocked: awaiting human approval",
                                approval_token=token,
                                blocked=True,
                                executed=False,
                            )
                        )
                        pending.append(request)
                        pending_calls.append(call)
                        continue
                    # Rejected or expired: fail closed, tell the model, move on.
                    record = ToolCallRecord(
                        name=spec.name,
                        arguments=dict(call.args),
                        ok=False,
                        error=(
                            f"action blocked: approval {request.status.value}"
                            + (f" ({request.reason})" if request.reason else "")
                        ),
                        approval_token=token,
                        blocked=True,
                    )
                    records.append(record)
                    turns.append(make_function_response(call, {"error": record.error}))
                    self._logger.warning(
                        "approval_decided",
                        tool=spec.name,
                        token=token,
                        status=request.status.value,
                        executed=False,
                    )
                    continue
                if not self.approval.is_approved(token, spec.name, call.args):
                    record = ToolCallRecord(
                        name=spec.name,
                        arguments=dict(call.args),
                        ok=False,
                        error=(
                            "action blocked: approval token is expired, already used "
                            "or bound to different arguments"
                        ),
                        approval_token=token,
                        blocked=True,
                    )
                    records.append(record)
                    turns.append(make_function_response(call, {"error": record.error}))
                    self._logger.warning(
                        "approval_decided",
                        tool=spec.name,
                        token=token,
                        status="invalid",
                        executed=False,
                    )
                    continue

            # Approved calls still respect the global budget and recursion guard.
            guard.check(spec.name, call.args)

            cached, cached_value = (False, None)
            if spec.idempotent:
                cached, cached_value = self._idempotent.get(spec.name, call.args)

            if cached:
                record = ToolCallRecord(
                    name=spec.name,
                    arguments=dict(call.args),
                    ok=True,
                    result=cached_value,
                    approval_token=token,
                    cached=True,
                )
                records.append(record)
                turns.append(make_function_response(call, {"result": cached_value}))
                continue

            guard.record(spec.name, call.args)
            guard.push(spec.name)
            try:
                record, payload = self._executor.execute(call, approval_token=token)
            finally:
                guard.pop(spec.name)

            if record.ok and spec.idempotent:
                self._idempotent.set(spec.name, call.args, record.result)
            record.executed = True

            records.append(record)
            turns.append(make_function_response(call, payload))
            self._logger.event(
                "tool_result",
                tool=spec.name,
                ok=record.ok,
                result=record.result,
                error=record.error,
                duration_ms=round(record.duration_ms, 2),
                approval_token=token,
            )
            if token:
                self.approval.consume(token)
                self._logger.event(
                    "approval_decided",
                    tool=spec.name,
                    token=token,
                    status=ApprovalStatus.APPROVED.value,
                    executed=True,
                )

        return records, turns, pending_calls, pending

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _append_model_turn(contents: list[dict[str, Any]], response: LLMResponse) -> None:
        if response.raw_model_content is not None:
            contents.append(response.raw_model_content)
            return
        contents.append(
            {
                "role": "model",
                "parts": [
                    {"function_call": {"name": call.name, "args": call.args or {}}}
                    for call in response.function_calls
                ],
            }
        )

    @staticmethod
    def _append_text_turn(contents: list[dict[str, Any]], text: str | None) -> None:
        if text:
            contents.append({"role": "model", "parts": [{"text": text}]})

    def _finish(
        self,
        execution_id: str,
        status: RunStatus,
        records: list[ToolCallRecord],
        started: Any,
        usage: UsageSummary | None,
        *,
        output: Any = None,
        text: str | None = None,
        error: str | None = None,
        pending: list[ApprovalRequest] | None = None,
    ) -> AgentResult:
        result = AgentResult(
            execution_id=execution_id,
            agent_name=self.name,
            status=status,
            output=output,
            text=text,
            tool_calls=records,
            pending_approvals=pending or [],
            usage=usage,
            error=error,
            model=self.config.model,
            started_at=started,
            finished_at=utc_now(),
        )
        level = "run_completed" if status is RunStatus.SUCCESS else "run_failed"
        self._logger.event(
            level,
            execution_id=execution_id,
            agent=self.name,
            status=status.value,
            tool_calls=len(records),
            error=error,
            usage=usage.to_dict() if usage else None,
            duration_seconds=(result.finished_at - started).total_seconds() if result.finished_at else None,
        )
        return result

    def _finish_failure(
        self,
        execution_id: str,
        records: list[ToolCallRecord],
        started: Any,
        usage: UsageSummary | None,
        error: str,
    ) -> AgentResult:
        return self._finish(
            execution_id, RunStatus.FAILED, records, started, usage, error=error
        )


__all__ = ["Agent"]
