"""Agent loop behaviour against a scripted client (no network)."""

from __future__ import annotations

import pytest

from gemini_agent_toolkit import ApprovalStatus, RunStatus, tool
from gemini_agent_toolkit.exceptions import RecursiveToolCallError, ToolLimitExceededError
from gemini_agent_toolkit.models import ApprovalRequest
from gemini_agent_toolkit.reliability.recovery import ExecutionGuard

from conftest import ScriptedClient, build_agent, text_response, tool_response


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: first factor.
        b: second factor.
    """
    return a * b


@tool
def exploding(text: str) -> str:
    """Always fails.

    Args:
        text: ignored.
    """
    raise RuntimeError("boom")


@tool(requires_approval=True)
def delete_file(path: str) -> str:
    """Delete a file.

    Args:
        path: file to delete.
    """
    return f"deleted {path}"


class PendingForever:
    """Policy that always leaves the decision to the caller."""

    name = "pending_forever"

    def decide(self, request: ApprovalRequest) -> ApprovalStatus:
        return ApprovalStatus.PENDING


def test_plain_text_run() -> None:
    client = ScriptedClient([text_response("hello there")])
    result = build_agent(client).run("say hi")
    assert result.status is RunStatus.SUCCESS
    assert result.output == "hello there"
    assert result.tool_calls == []
    assert result.usage.total_tokens >= 10


def test_tool_call_then_answer_feeds_result_back() -> None:
    client = ScriptedClient([tool_response(("multiply", {"a": 6, "b": 7})), text_response("42")])
    agent = build_agent(client, tools=[multiply])
    result = agent.run("6*7?")

    assert result.status is RunStatus.SUCCESS
    assert result.output == "42"
    assert [c.name for c in result.tool_calls] == ["multiply"]
    assert result.tool_calls[0].ok is True
    assert result.tool_calls[0].result == 42

    second_call_contents = client.calls[1]["contents"]
    assert any(
        isinstance(turn, dict)
        and "function_response" in str(turn)
        and "42" in str(turn)
        for turn in second_call_contents
    )


def test_unknown_tool_is_reported_to_model_not_raised() -> None:
    client = ScriptedClient([tool_response(("nonexistent", {})), text_response("sorry")])
    result = build_agent(client, tools=[multiply]).run("go")
    assert result.status is RunStatus.SUCCESS
    assert result.tool_calls[0].ok is False
    assert "not registered" in (result.tool_calls[0].error or "")


def test_invalid_arguments_do_not_crash_run() -> None:
    client = ScriptedClient(
        [tool_response(("multiply", {"a": "x", "b": 2})), text_response("bad args")]
    )
    result = build_agent(client, tools=[multiply]).run("go")
    assert result.status is RunStatus.SUCCESS
    assert result.tool_calls[0].ok is False


def test_tool_exception_is_captured() -> None:
    client = ScriptedClient([tool_response(("exploding", {"text": "x"})), text_response("failed")])
    result = build_agent(client, tools=[exploding]).run("go")
    assert result.tool_calls[0].ok is False
    assert "boom" in (result.tool_calls[0].error or "")
    assert result.status is RunStatus.SUCCESS


def test_tool_budget_is_enforced() -> None:
    script = [tool_response(("multiply", {"a": 1, "b": 1})) for _ in range(5)]
    client = ScriptedClient(script)
    result = build_agent(client, tools=[multiply], max_tool_calls=2).run("loop")
    assert result.status is RunStatus.TOOL_LIMIT_REACHED
    assert len(result.tool_calls) == 2


def test_sensitive_tool_pauses_the_run() -> None:
    client = ScriptedClient([tool_response(("delete_file", {"path": "a.txt"}))])
    agent = build_agent(client, tools=[delete_file], approval_policy=PendingForever())
    result = agent.run("delete a.txt")

    assert result.status is RunStatus.PENDING_APPROVAL
    assert len(result.pending_approvals) == 1
    assert result.tool_calls[0].executed is False
    assert result.output is None


def test_resume_after_approval_executes_tool() -> None:
    client = ScriptedClient(
        [tool_response(("delete_file", {"path": "a.txt"})), text_response("deleted")]
    )
    agent = build_agent(client, tools=[delete_file], approval_policy=PendingForever())
    paused = agent.run("delete a.txt")
    request = paused.pending_approvals[0]

    agent.approval.approve(request.token)
    resumed = agent.resume(paused.execution_id)

    assert resumed.status is RunStatus.SUCCESS
    assert resumed.output == "deleted"
    assert resumed.tool_calls[0].executed is True
    # The human decision was re-used instead of asking twice.
    assert client.call_count == 2


def test_rejected_approval_blocks_the_action() -> None:
    client = ScriptedClient(
        [tool_response(("delete_file", {"path": "a.txt"})), text_response("ok, skipped")]
    )
    agent = build_agent(client, tools=[delete_file], approval_policy=PendingForever())
    paused = agent.run("delete a.txt")
    agent.approval.reject(paused.pending_approvals[0].token)

    resumed = agent.resume(paused.execution_id)
    assert resumed.status is RunStatus.SUCCESS
    assert resumed.tool_calls[0].blocked is True
    assert resumed.tool_calls[0].executed is False


def test_default_policy_never_executes_sensitive_tools() -> None:
    client = ScriptedClient(
        [tool_response(("delete_file", {"path": "a.txt"})), text_response("blocked")]
    )
    result = build_agent(client, tools=[delete_file]).run("delete a.txt")
    assert result.status is RunStatus.SUCCESS
    assert result.tool_calls[0].blocked is True
    assert result.tool_calls[0].executed is False


def test_retry_recovers_from_transient_error() -> None:
    from gemini_agent_toolkit import TransientClientError

    client = ScriptedClient([TransientClientError("503"), text_response("recovered")])
    result = build_agent(client, max_retries=2).run("hi")
    assert result.status is RunStatus.SUCCESS
    assert result.output == "recovered"
    assert client.call_count == 2


def test_context_records_conversation() -> None:
    from gemini_agent_toolkit.context import ConversationContext

    client = ScriptedClient([text_response("first"), text_response("second")])
    agent = build_agent(client)
    ctx = ConversationContext()
    agent.run("q1", context=ctx)
    agent.run("q2", context=ctx)
    assert len(ctx.turns) == 4


def test_guard_budget_and_recursion() -> None:
    guard = ExecutionGuard(max_tool_calls=1, max_repeated_calls=1)
    guard.check("t", {"a": 1})
    guard.record("t", {"a": 1})
    with pytest.raises(ToolLimitExceededError):
        guard.check("t", {"a": 1})

    guard2 = ExecutionGuard()
    guard2.push("outer")
    with pytest.raises(RecursiveToolCallError):
        guard2.check("outer", {"x": 2})


def test_idempotent_tool_result_is_cached() -> None:
    calls = {"n": 0}

    @tool(idempotent=True)
    def counted(tag: str) -> int:
        """Count invocations.

        Args:
            tag: label.
        """
        calls["n"] += 1
        return calls["n"]

    client = ScriptedClient(
        [
            tool_response(("counted", {"tag": "x"})),
            tool_response(("counted", {"tag": "x"})),
            text_response("done"),
        ]
    )
    result = build_agent(client, tools=[counted]).run("go")
    assert result.tool_calls[0].cached is False
    assert result.tool_calls[0].result == 1
    assert result.tool_calls[1].cached is True
    assert calls["n"] == 1
