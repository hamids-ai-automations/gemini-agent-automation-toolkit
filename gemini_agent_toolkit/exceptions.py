"""Exception hierarchy for the Gemini Agent Automation Toolkit.

Every error raised deliberately by the toolkit inherits from :class:`ToolkitError`
so applications can catch a single base class. Errors caused by the model, the
network or the developer's own tool code are separated so that retry logic and
human approval logic can make precise decisions.
"""

from __future__ import annotations


class ToolkitError(Exception):
    """Base class for every error raised by this toolkit."""


# --------------------------------------------------------------------------- #
# Configuration and credentials
# --------------------------------------------------------------------------- #
class ConfigError(ToolkitError):
    """Invalid or inconsistent configuration."""


class MissingAPIKeyError(ConfigError):
    """No Gemini API key was supplied by the caller or the environment."""


# --------------------------------------------------------------------------- #
# Client / model boundary
# --------------------------------------------------------------------------- #
class ClientError(ToolkitError):
    """The Gemini client could not complete the request."""


class TransientClientError(ClientError):
    """A retryable failure: rate limit, capacity, network blip or timeout."""


class PermanentClientError(ClientError):
    """A failure that will not be fixed by retrying (auth, bad request)."""


class TimeoutExceededError(TransientClientError):
    """The configured timeout elapsed before the call finished."""


class RetryExhaustedError(ClientError):
    """All retry attempts failed."""

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        super().__init__(
            f"Retry exhausted after {attempts} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error


class ModelResponseError(ClientError):
    """The model returned a response that cannot be interpreted."""


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
class ToolError(ToolkitError):
    """Base class for tool related errors."""


class ToolNotFoundError(ToolError):
    """The model requested a tool that is not registered."""


class ToolArgumentError(ToolError):
    """Arguments proposed for a tool call failed validation."""


class ToolExecutionError(ToolError):
    """A tool raised while executing."""

    def __init__(self, tool_name: str, original: BaseException) -> None:
        super().__init__(f"Tool '{tool_name}' failed: {type(original).__name__}: {original}")
        self.tool_name = tool_name
        self.original = original


class ToolLimitExceededError(ToolError):
    """The configured tool call budget for one run was exceeded."""


class RecursiveToolCallError(ToolError):
    """A tool tried to trigger itself again before returning."""


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #
class StructuredOutputError(ToolkitError):
    """The model answer did not satisfy the requested output schema."""

    def __init__(self, message: str, *, raw_text: str | None = None, errors: list | None = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.errors = errors or []


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #
class ApprovalError(ToolkitError):
    """Base class for human approval failures."""


class ApprovalRequiredError(ApprovalError):
    """A tool needing approval was reached without an usable approval."""


class ApprovalRejectedError(ApprovalError):
    """A human rejected the proposed action."""


class ApprovalExpiredError(ApprovalError):
    """The approval request expired before a decision was made."""


class ApprovalMismatchError(ApprovalError):
    """An approval token was presented for different arguments than it was issued for."""


class ApprovalUnavailableError(ApprovalError):
    """No approval mechanism is configured; the toolkit fails safe."""


# --------------------------------------------------------------------------- #
# Workflows
# --------------------------------------------------------------------------- #
class WorkflowError(ToolkitError):
    """A workflow definition or run failed."""


class StepFailedError(WorkflowError):
    """A workflow step raised an unrecoverable error."""
