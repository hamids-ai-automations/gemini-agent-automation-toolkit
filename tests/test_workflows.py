"""Workflow runner: ordering, state passing, failures, approval gate."""

from __future__ import annotations

import pytest

from gemini_agent_toolkit import ApprovalManager, RunStatus
from gemini_agent_toolkit.approval.policies import AllowAllPolicy, DenyAllPolicy
from gemini_agent_toolkit.exceptions import WorkflowError
from gemini_agent_toolkit.workflows.runner import WorkflowRunner, make_workflow


def test_steps_run_in_order_and_pass_state() -> None:
    workflow = make_workflow("demo")
    workflow.add_step("fetch", lambda state: "raw")
    workflow.add_step("clean", lambda state: state["fetch"].upper())

    result = WorkflowRunner(workflow).run()

    assert result.status is RunStatus.SUCCESS
    assert [record.name for record in result.history] == ["fetch", "clean"]
    assert result.outputs == {"fetch": "raw", "clean": "RAW"}


def test_initial_state_is_available() -> None:
    workflow = make_workflow("demo")
    workflow.add_step("echo", lambda state: state["input"])
    result = WorkflowRunner(workflow).run({"input": 7})
    assert result.outputs["echo"] == 7


def test_failure_stops_the_workflow() -> None:
    workflow = make_workflow("demo")
    workflow.add_step("ok", lambda state: 1)
    workflow.add_step("boom", lambda state: 1 / 0)
    workflow.add_step("never", lambda state: 3)

    result = WorkflowRunner(workflow).run()

    assert result.status is RunStatus.FAILED
    assert [record.name for record in result.history] == ["ok", "boom"]
    assert "ZeroDivisionError" in (result.error or "")


def test_duplicate_step_names_are_rejected() -> None:
    workflow = make_workflow("demo")
    workflow.add_step("a", lambda state: 1)
    with pytest.raises(WorkflowError):
        workflow.add_step("a", lambda state: 2)


def test_empty_workflow_name_is_rejected() -> None:
    with pytest.raises(WorkflowError):
        make_workflow("")


def test_sensitive_step_pauses_without_approval_manager() -> None:
    workflow = make_workflow("deploy")
    workflow.add_step("ship", lambda state: "done", requires_approval=True)

    result = WorkflowRunner(workflow).run()

    assert result.status is RunStatus.PENDING_APPROVAL
    assert result.outputs == {}


def test_denied_step_fails_the_workflow() -> None:
    workflow = make_workflow("deploy")
    workflow.add_step("ship", lambda state: "done", requires_approval=True)

    result = WorkflowRunner(
        workflow, approval_manager=ApprovalManager(policy=DenyAllPolicy())
    ).run()

    assert result.status is RunStatus.FAILED
    assert "rejected" in (result.error or "")


def test_approved_step_runs() -> None:
    workflow = make_workflow("deploy")
    workflow.add_step("ship", lambda state: "done", requires_approval=True)

    result = WorkflowRunner(
        workflow, approval_manager=ApprovalManager(policy=AllowAllPolicy(quiet=True))
    ).run()

    assert result.status is RunStatus.SUCCESS
    assert result.outputs == {"ship": "done"}
