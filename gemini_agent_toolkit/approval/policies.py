"""Approval policies: who is allowed to press the button, and how.

The toolkit ships a *deny-by-default* policy so that a sensitive tool can never
run just because the developer forgot to wire an approval channel. Every other
policy must be chosen explicitly.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..models import ApprovalRequest, ApprovalStatus


def describe_action(request: ApprovalRequest) -> str:
    """Human readable one-liner describing the proposed action."""
    args = (
        ", ".join(f"{key}={value!r}" for key, value in request.arguments.items()) or "no arguments"
    )
    return f"{request.tool_name}({args})"


@runtime_checkable
class ApprovalPolicy(Protocol):
    """Decides whether a proposed tool call may run."""

    def decide(self, request: ApprovalRequest) -> ApprovalStatus:
        """Return APPROVED, REJECTED or PENDING (defer to a human via API)."""
        ...  # pragma: no cover - protocol definition


class DenyAllPolicy:
    """Fail-safe default: refuses every sensitive action."""

    name = "deny_all"

    def decide(self, request: ApprovalRequest) -> ApprovalStatus:
        return ApprovalStatus.REJECTED


class AllowAllPolicy:
    """Explicitly approve everything.

    Only appropriate for local experiments and tests. It is *not* a human
    approval mechanism, so it emits a warning when instantiated.
    """

    name = "allow_all"

    def __init__(self, *, quiet: bool = False) -> None:
        if not quiet:
            warnings.warn(
                "AllowAllPolicy approves every sensitive tool call automatically. "
                "Use it only in tests or local experiments.",
                RuntimeWarning,
                stacklevel=2,
            )

    def decide(self, request: ApprovalRequest) -> ApprovalStatus:
        return ApprovalStatus.APPROVED


class PendingPolicy:
    """Leave the decision to the caller of ``agent.run``.

    The run stops with ``RunStatus.PENDING_APPROVAL`` and the developer decides
    later through :class:`~gemini_agent_toolkit.approval.manager.ApprovalManager`.
    """

    name = "pending"

    def decide(self, request: ApprovalRequest) -> ApprovalStatus:
        return ApprovalStatus.PENDING


class CallablePolicy:
    """Adapt a predicate ``(request) -> bool`` into a policy."""

    name = "callable"

    def __init__(
        self,
        predicate: Callable[[ApprovalRequest], bool | ApprovalStatus],
        *,
        name: str | None = None,
    ) -> None:
        if not callable(predicate):
            raise TypeError("predicate must be callable")
        self._predicate = predicate
        if name:
            self.name = name

    def decide(self, request: ApprovalRequest) -> ApprovalStatus:
        outcome = self._predicate(request)
        if isinstance(outcome, ApprovalStatus):
            return outcome
        return ApprovalStatus.APPROVED if outcome else ApprovalStatus.REJECTED


class TerminalPolicy:
    """Ask a human in the terminal (or any provided input/output pair)."""

    name = "terminal"

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], Any] = print,
    ) -> None:
        self._input = input_fn
        self._output = output_fn

    def decide(self, request: ApprovalRequest) -> ApprovalStatus:
        self._output(
            "\n--- Approval required ---\n"
            f"Tool: {request.tool_name}\n"
            f"Action: {describe_action(request)}\n"
            f"Reason: {request.reason or 'sensitive action'}\n"
            f"Token: {request.token}\n"
        )
        answer = self._input("Approve this action? [y/N]: ").strip().lower()
        return ApprovalStatus.APPROVED if answer in {"y", "yes"} else ApprovalStatus.REJECTED


__all__ = [
    "AllowAllPolicy",
    "ApprovalPolicy",
    "CallablePolicy",
    "DenyAllPolicy",
    "PendingPolicy",
    "TerminalPolicy",
    "describe_action",
]
