"""Plain data contracts shared by the runtime, tests and user code."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp (naive datetimes cause subtle bugs)."""
    return datetime.now(timezone.utc)


def new_execution_id() -> str:
    return f"run_{uuid.uuid4().hex[:16]}"


class RunStatus(str, Enum):
    """Terminal state of an agent run or workflow run."""

    SUCCESS = "success"
    PENDING_APPROVAL = "pending_approval"
    FAILED = "failed"
    TOOL_LIMIT_REACHED = "tool_limit_reached"


class ApprovalStatus(str, Enum):
    """Lifecycle of a single approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class FunctionCallRequest:
    """A tool call *proposed* by the model. Never proof of execution."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(frozen=True)
class UsageSummary:
    """Token usage reported by the API when available."""

    prompt_tokens: int | None = None
    candidate_tokens: int | None = None
    thought_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "candidate_tokens": self.candidate_tokens,
            "thought_tokens": self.thought_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    """Normalised model response used by the runtime.

    The real :class:`~gemini_agent_toolkit.client.GeminiClient` adapts the SDK
    response into this shape; tests and alternative backends provide it directly.
    """

    text: str | None = None
    function_calls: list[FunctionCallRequest] = field(default_factory=list)
    usage: UsageSummary | None = None
    model_version: str | None = None
    raw_model_content: Any | None = None


@dataclass
class ToolCallRecord:
    """Audit record for one executed (or blocked) tool call."""

    name: str
    arguments: dict[str, Any]
    ok: bool = False
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    approval_token: str | None = None
    blocked: bool = False
    executed: bool = False
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 3),
            "approval_token": self.approval_token,
            "blocked": self.blocked,
            "executed": self.executed,
            "cached": self.cached,
        }


@dataclass
class ApprovalRequest:
    """A pending human decision about a sensitive tool call."""

    token: str
    tool_name: str
    arguments: dict[str, Any]
    fingerprint: str
    reason: str = ""
    created_at: datetime = field(default_factory=utc_now)
    expires_at: datetime | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_at: datetime | None = None
    decided_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "decided_by": self.decided_by,
        }


@dataclass
class AgentResult:
    """Everything a caller needs to know about one agent run."""

    execution_id: str
    agent_name: str
    status: RunStatus
    output: Any = None
    text: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    pending_approvals: list[ApprovalRequest] = field(default_factory=list)
    usage: UsageSummary | None = None
    error: str | None = None
    model: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "output": self.output
            if not hasattr(self.output, "model_dump")
            else self.output.model_dump(),
            "text": self.text,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "pending_approvals": [a.to_dict() for a in self.pending_approvals],
            "usage": self.usage.to_dict() if self.usage else None,
            "error": self.error,
            "model": self.model,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


@dataclass
class StepRecord:
    """Audit record for one workflow step."""

    name: str
    status: RunStatus
    output: Any = None
    error: str | None = None
    attempts: int = 1
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "output": self.output
            if not hasattr(self.output, "model_dump")
            else self.output.model_dump(),
            "error": self.error,
            "attempts": self.attempts,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


@dataclass
class WorkflowResult:
    """Outcome of a workflow run, including per-step history."""

    workflow_name: str
    status: RunStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    history: list[StepRecord] = field(default_factory=list)
    error: str | None = None
    pending_approvals: list[ApprovalRequest] = field(default_factory=list)
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "outputs": {
                k: (v if not hasattr(v, "model_dump") else v.model_dump())
                for k, v in self.outputs.items()
            },
            "history": [h.to_dict() for h in self.history],
            "error": self.error,
            "pending_approvals": [a.to_dict() for a in self.pending_approvals],
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }
