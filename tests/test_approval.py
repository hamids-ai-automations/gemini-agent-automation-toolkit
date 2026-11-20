"""Approval manager: fail-closed default, binding, expiry, single use."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gemini_agent_toolkit import ApprovalManager, ApprovalStatus
from gemini_agent_toolkit.approval.policies import (
    AllowAllPolicy,
    CallablePolicy,
    DenyAllPolicy,
    PendingPolicy,
)
from gemini_agent_toolkit.exceptions import (
    ApprovalExpiredError,
    ApprovalMismatchError,
    ApprovalRejectedError,
)


class Clock:
    """Deterministic clock: the manager works in ``datetime`` space."""

    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def test_default_policy_rejects_sensitive_actions() -> None:
    manager = ApprovalManager()
    request = manager.request("delete_file", {"path": "a.txt"})
    assert request.status is ApprovalStatus.REJECTED
    assert manager.is_approved(request.token, "delete_file", {"path": "a.txt"}) is False


def test_deny_all_policy_is_explicit() -> None:
    manager = ApprovalManager(policy=DenyAllPolicy())
    assert manager.request("send_email", {}).status is ApprovalStatus.REJECTED


def test_allow_all_policy_approves_but_is_bind_to_arguments() -> None:
    manager = ApprovalManager(policy=AllowAllPolicy(quiet=True))
    request = manager.request("send_email", {"to": "a@b.c"})
    assert request.status is ApprovalStatus.APPROVED
    assert manager.is_approved(request.token, "send_email", {"to": "a@b.c"}) is True
    assert manager.is_approved(request.token, "send_email", {"to": "evil@b.c"}) is False
    with pytest.raises(ApprovalMismatchError):
        manager.assert_matches(request.token, "send_email", {"to": "evil@b.c"})


def test_callable_policy_can_scope_by_tool() -> None:
    manager = ApprovalManager(policy=CallablePolicy(lambda req: req.tool_name == "read_log"))
    assert manager.request("read_log", {}).status is ApprovalStatus.APPROVED
    assert manager.request("delete_file", {}).status is ApprovalStatus.REJECTED


def test_pending_then_human_approval() -> None:
    clock = Clock()
    manager = ApprovalManager(default_ttl=60.0, policy=PendingPolicy(), clock=clock)
    request = manager.request("delete_file", {"path": "a.txt"})
    assert request.status is ApprovalStatus.PENDING

    approved = manager.approve(request.token)
    assert approved.status is ApprovalStatus.APPROVED
    assert manager.is_approved(request.token, "delete_file", {"path": "a.txt"}) is True


def test_approval_is_single_use() -> None:
    manager = ApprovalManager(policy=PendingPolicy())
    request = manager.request("delete_file", {"path": "a.txt"})
    manager.approve(request.token)
    manager.consume(request.token)
    assert manager.is_approved(request.token, "delete_file", {"path": "a.txt"}) is False


def test_rejection_is_terminal() -> None:
    manager = ApprovalManager(policy=PendingPolicy())
    request = manager.request("delete_file", {"path": "a.txt"})
    manager.reject(request.token)
    assert request.status is ApprovalStatus.REJECTED
    with pytest.raises(ApprovalRejectedError):
        manager.approve(request.token)


def test_expired_request_cannot_be_approved() -> None:
    clock = Clock()
    manager = ApprovalManager(default_ttl=10.0, policy=PendingPolicy(), clock=clock)
    request = manager.request("delete_file", {"path": "a.txt"})
    clock.advance(11)
    assert manager.is_approved(request.token, "delete_file", {"path": "a.txt"}) is False
    with pytest.raises(ApprovalExpiredError):
        manager.approve(request.token)


def test_expire_stale_reports_requests() -> None:
    clock = Clock()
    manager = ApprovalManager(default_ttl=5.0, policy=PendingPolicy(), clock=clock)
    request = manager.request("delete_file", {"path": "a.txt"})
    clock.advance(6)
    expired = manager.expire_stale()
    assert request.token in {item.token for item in expired}


def test_unknown_token_is_not_approved() -> None:
    manager = ApprovalManager()
    assert manager.is_approved("nope", "delete_file", {}) is False
