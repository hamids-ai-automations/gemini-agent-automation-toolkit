"""Approval request lifecycle.

Guarantees enforced here:

1. A sensitive tool never runs before a decision exists.
2. An approval is bound to the exact arguments it was issued for (fingerprint).
3. An approval token is single-use: replaying it after consumption fails.
4. Requests expire after a TTL, and expired tokens cannot be approved.
5. Without a policy the manager denies, never allows.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ..exceptions import (
    ApprovalExpiredError,
    ApprovalMismatchError,
    ApprovalRejectedError,
)
from ..models import ApprovalRequest, ApprovalStatus
from ..reliability.recovery import arguments_fingerprint
from .policies import ApprovalPolicy, DenyAllPolicy


def _utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ApprovalManager:
    """Issues, stores and validates approval requests."""

    policy: ApprovalPolicy | None = None
    default_ttl: float | None = 900.0
    clock: Callable[[], datetime] = _utc
    _requests: dict[str, ApprovalRequest] = field(default_factory=dict)
    _consumed: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.policy is None:
            # Fail safe: no policy means nothing sensitive can run.
            self.policy = DenyAllPolicy()
        if self.default_ttl is not None and self.default_ttl <= 0:
            self.default_ttl = None

    # -- creation --------------------------------------------------------- #
    def request(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        reason: str = "",
        ttl: float | None = None,
        policy: ApprovalPolicy | None = None,
    ) -> ApprovalRequest:
        """Create a request and let the policy decide immediately (if it can)."""
        now = self.clock()
        effective_ttl = ttl if ttl is not None else self.default_ttl
        request = ApprovalRequest(
            token=secrets.token_urlsafe(18),
            tool_name=tool_name,
            arguments=dict(arguments or {}),
            fingerprint=arguments_fingerprint(tool_name, arguments or {}),
            reason=reason,
            created_at=now,
            expires_at=now + timedelta(seconds=effective_ttl) if effective_ttl else None,
        )
        self._requests[request.token] = request

        decision = (policy or self.policy).decide(request)
        if decision is not ApprovalStatus.PENDING:
            self._apply(request, decision, decided_by="policy")
        return request

    # -- manual decisions ------------------------------------------------- #
    def approve(self, token: str, *, decided_by: str = "human") -> ApprovalRequest:
        """Approve a pending request.

        Raises:
            KeyError: unknown token.
            ApprovalExpiredError: the request is past its TTL.
            ApprovalRejectedError: the request was already rejected.
        """
        request = self._get(token)
        if request.status is ApprovalStatus.REJECTED:
            raise ApprovalRejectedError(f"approval {token} was already rejected")
        if self._expired(request):
            self._apply(request, ApprovalStatus.EXPIRED, decided_by=decided_by)
            raise ApprovalExpiredError(f"approval {token} expired at {request.expires_at}")
        self._apply(request, ApprovalStatus.APPROVED, decided_by=decided_by)
        return request

    def reject(self, token: str, *, decided_by: str = "human") -> ApprovalRequest:
        """Reject a pending request."""
        request = self._get(token)
        self._apply(request, ApprovalStatus.REJECTED, decided_by=decided_by)
        return request

    # -- verification ----------------------------------------------------- #
    def is_approved(
        self,
        token: str | None,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> bool:
        """True only if the token is approved, unexpired, unused and matches the call."""
        if not token:
            return False
        request = self._requests.get(token)
        if request is None or request.status is not ApprovalStatus.APPROVED:
            return False
        if token in self._consumed:
            return False
        if self._expired(request):
            self._apply(request, ApprovalStatus.EXPIRED, decided_by="system")
            return False
        if tool_name is not None and request.tool_name != tool_name:
            return False
        if arguments is not None and request.fingerprint != arguments_fingerprint(tool_name or request.tool_name, arguments):
            return False
        return True

    def assert_matches(self, token: str, tool_name: str, arguments: dict[str, Any]) -> None:
        """Raise when the token was issued for a different call."""
        request = self._get(token)
        expected = arguments_fingerprint(tool_name, arguments)
        if request.fingerprint != expected:
            raise ApprovalMismatchError(
                f"approval token {token} was issued for a different action; "
                "approvals cannot be reused for other arguments"
            )

    def consume(self, token: str) -> None:
        """Mark a token as used so it cannot be replayed."""
        if token in self._requests:
            self._consumed.add(token)

    # -- introspection ---------------------------------------------------- #
    def get(self, token: str) -> ApprovalRequest | None:
        return self._requests.get(token)

    def pending(self) -> list[ApprovalRequest]:
        self.expire_stale()
        return [r for r in self._requests.values() if r.status is ApprovalStatus.PENDING]

    def expire_stale(self) -> list[ApprovalRequest]:
        expired: list[ApprovalRequest] = []
        for request in self._requests.values():
            if request.status is ApprovalStatus.PENDING and self._expired(request):
                self._apply(request, ApprovalStatus.EXPIRED, decided_by="system")
                expired.append(request)
        return expired

    def clear(self) -> None:
        self._requests.clear()
        self._consumed.clear()

    def history(self) -> list[ApprovalRequest]:
        return list(self._requests.values())

    # -- internals -------------------------------------------------------- #
    def _get(self, token: str) -> ApprovalRequest:
        try:
            return self._requests[token]
        except KeyError as exc:
            raise KeyError(f"unknown approval token: {token}") from exc

    def _expired(self, request: ApprovalRequest) -> bool:
        return request.expires_at is not None and self.clock() >= request.expires_at

    def _apply(
        self, request: ApprovalRequest, status: ApprovalStatus, *, decided_by: str | None
    ) -> None:
        request.status = status
        request.decided_at = self.clock()
        request.decided_by = decided_by


__all__ = ["ApprovalManager"]
