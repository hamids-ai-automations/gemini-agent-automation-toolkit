"""Minimal multi-step workflow runner.

A workflow is an ordered list of named steps. Each step is a plain callable that
receives a mutable ``state`` dict and returns a value; the value is stored under
the step name and becomes available to later steps. That is enough for the
reliable automations most tools need, without pulling in a heavyweight
orchestration framework.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..approval.manager import ApprovalManager
from ..exceptions import ApprovalError, StepFailedError, WorkflowError
from ..logging.logger import ExecutionLogger
from ..models import RunStatus, StepRecord, WorkflowResult, utc_now
from ..reliability.retry import RetryPolicy, execute_with_retry

StepFn = Callable[[dict[str, Any]], Any]


@dataclass
class Step:
    """One unit of work inside a workflow.

    Args:
        name: unique step name; also the key used in ``state``.
        action: called with the workflow state, returns the step output.
        requires_approval: block the workflow until this step is approved.
        retry_policy: overrides the workflow default for this step.
    """

    name: str
    action: StepFn
    requires_approval: bool = False
    retry_policy: RetryPolicy | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise WorkflowError("step name must not be empty")
        if not callable(self.action):
            raise WorkflowError(f"step '{self.name}' action must be callable")


@dataclass
class Workflow:
    """Ordered collection of steps."""

    name: str
    steps: list[Step] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise WorkflowError("workflow name must not be empty")

    def add_step(
        self,
        name: str,
        action: StepFn,
        *,
        requires_approval: bool = False,
        retry_policy: RetryPolicy | None = None,
        description: str = "",
    ) -> Workflow:
        """Append a step and return ``self`` so calls can be chained."""
        if any(step.name == name for step in self.steps):
            raise WorkflowError(f"duplicate step name: {name}")
        self.steps.append(
            Step(
                name=name,
                action=action,
                requires_approval=requires_approval,
                retry_policy=retry_policy,
                description=description,
            )
        )
        return self


class WorkflowRunner:
    """Executes a :class:`Workflow` with state passing and audit history."""

    def __init__(
        self,
        workflow: Workflow,
        *,
        approval_manager: ApprovalManager | None = None,
        retry_policy: RetryPolicy | None = None,
        logger: ExecutionLogger | None = None,
    ) -> None:
        self.workflow = workflow
        self.approval = approval_manager
        self.retry_policy = retry_policy or RetryPolicy()
        self._logger = logger or ExecutionLogger()

    def run(self, initial_state: dict[str, Any] | None = None) -> WorkflowResult:
        """Run every step in order.

        The workflow stops at the first failing step, or at the first step that
        requires approval that has not been granted.
        """
        state: dict[str, Any] = dict(initial_state or {})
        history: list[StepRecord] = []
        started = utc_now()
        outputs: dict[str, Any] = {}

        for step in self.workflow.steps:
            record = StepRecord(name=step.name, status=RunStatus.SUCCESS)
            self._logger.event("step_started", workflow=self.workflow.name, step=step.name)

            if step.requires_approval:
                request = None
                if self.approval is not None:
                    request = self.approval.request(
                        f"workflow_step:{step.name}",
                        {"step": step.name, "state_keys": sorted(state)},
                        reason=step.description or f"step '{step.name}' requires approval",
                    )
                if request is None or request.status.value == "pending":
                    record.status = RunStatus.PENDING_APPROVAL
                    record.finished_at = utc_now()
                    history.append(record)
                    pending = [request] if request is not None else []
                    return WorkflowResult(
                        workflow_name=self.workflow.name,
                        status=RunStatus.PENDING_APPROVAL,
                        outputs=outputs,
                        history=history,
                        pending_approvals=[p for p in pending if p is not None],
                        started_at=started,
                        finished_at=utc_now(),
                    )
                if request.status.value != "approved":
                    record.status = RunStatus.FAILED
                    record.error = f"step approval {request.status.value}"
                    record.finished_at = utc_now()
                    history.append(record)
                    return WorkflowResult(
                        workflow_name=self.workflow.name,
                        status=RunStatus.FAILED,
                        outputs=outputs,
                        history=history,
                        error=record.error,
                        started_at=started,
                        finished_at=utc_now(),
                    )

            policy = step.retry_policy or self.retry_policy
            current_step = step

            def _run_step(current_step: Step = current_step) -> Any:
                return current_step.action(state)

            def _on_step_retry(
                attempt: int,
                error: BaseException,
                delay: float,
                current_step: Step = current_step,
            ) -> None:
                self._logger.event(
                    "step_retry",
                    workflow=self.workflow.name,
                    step=current_step.name,
                    attempt=attempt + 1,
                    error=f"{type(error).__name__}: {error}",
                )

            try:
                value = execute_with_retry(
                    _run_step,
                    policy,
                    on_retry=_on_step_retry,
                )
            except Exception as exc:  # noqa: BLE001 - converted to a failed result
                record.status = RunStatus.FAILED
                record.error = f"{type(exc).__name__}: {exc}"
                record.finished_at = utc_now()
                history.append(record)
                self._logger.error(
                    "step_failed", workflow=self.workflow.name, step=step.name, error=record.error
                )
                return WorkflowResult(
                    workflow_name=self.workflow.name,
                    status=RunStatus.FAILED,
                    outputs=outputs,
                    history=history,
                    error=f"step '{step.name}' failed: {record.error}",
                    started_at=started,
                    finished_at=utc_now(),
                )

            state[step.name] = value
            state.setdefault("state", state)
            outputs[step.name] = value
            record.output = value
            record.finished_at = utc_now()
            history.append(record)
            self._logger.event("step_completed", workflow=self.workflow.name, step=step.name)

        return WorkflowResult(
            workflow_name=self.workflow.name,
            status=RunStatus.SUCCESS,
            outputs=outputs,
            history=history,
            started_at=started,
            finished_at=utc_now(),
        )


def make_workflow(name: str, steps: Sequence[Step] | None = None) -> Workflow:
    """Convenience constructor."""
    return Workflow(name=name, steps=list(steps or []))


__all__ = ["Step", "Workflow", "WorkflowRunner", "make_workflow"]


_ = (ApprovalError, StepFailedError)
