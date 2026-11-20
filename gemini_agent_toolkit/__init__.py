"""Gemini Agent Automation Toolkit.

A modular toolkit for building reliable Google Gemini agents: validated tool
calling, structured outputs, retries, human approval and execution logging.

    from gemini_agent_toolkit import Agent

    agent = Agent(name="Assistant", instructions="Be concise.")
    result = agent.run("Explain how AI agents work.")
    print(result.output)
"""

from __future__ import annotations

from .agent import Agent
from .approval.manager import ApprovalManager
from .approval.policies import (
    AllowAllPolicy,
    ApprovalPolicy,
    CallablePolicy,
    DenyAllPolicy,
    PendingPolicy,
    TerminalPolicy,
)
from .client import GeminiClient, LLMClient
from .config import API_KEY_ENV_VAR, DEFAULT_MODEL, AgentConfig
from .context import ConversationContext
from .exceptions import (
    ApprovalError,
    ApprovalExpiredError,
    ApprovalMismatchError,
    ApprovalRejectedError,
    ApprovalRequiredError,
    ApprovalUnavailableError,
    ClientError,
    ConfigError,
    MissingAPIKeyError,
    ModelResponseError,
    PermanentClientError,
    RecursiveToolCallError,
    RetryExhaustedError,
    StepFailedError,
    StructuredOutputError,
    TimeoutExceededError,
    ToolArgumentError,
    ToolError,
    ToolExecutionError,
    ToolkitError,
    ToolLimitExceededError,
    ToolNotFoundError,
    TransientClientError,
    WorkflowError,
)
from .logging.logger import ExecutionLogger, configure_logging, redact
from .models import (
    AgentResult,
    ApprovalRequest,
    ApprovalStatus,
    FunctionCallRequest,
    LLMResponse,
    RunStatus,
    UsageSummary,
    WorkflowResult,
)
from .outputs.structured import StructuredOutput
from .reliability.recovery import ExecutionGuard, IdempotentCache, arguments_fingerprint
from .reliability.retry import RetryPolicy, execute_with_retry, is_transient
from .reliability.timeout import run_with_timeout
from .tools.decorators import ToolSpec, tool
from .tools.executor import ToolExecutor
from .tools.registry import ToolRegistry
from .workflows.runner import Step, Workflow, WorkflowRunner

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "API_KEY_ENV_VAR",
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AllowAllPolicy",
    "ApprovalError",
    "ApprovalExpiredError",
    "ApprovalManager",
    "ApprovalMismatchError",
    "ApprovalPolicy",
    "ApprovalRejectedError",
    "ApprovalRequest",
    "ApprovalRequiredError",
    "ApprovalStatus",
    "ApprovalUnavailableError",
    "CallablePolicy",
    "ClientError",
    "ConfigError",
    "ConversationContext",
    "DEFAULT_MODEL",
    "DenyAllPolicy",
    "ExecutionGuard",
    "ExecutionLogger",
    "FunctionCallRequest",
    "GeminiClient",
    "IdempotentCache",
    "LLMClient",
    "LLMResponse",
    "MissingAPIKeyError",
    "ModelResponseError",
    "PendingPolicy",
    "PermanentClientError",
    "RecursiveToolCallError",
    "RetryExhaustedError",
    "RetryPolicy",
    "RunStatus",
    "Step",
    "StepFailedError",
    "StructuredOutput",
    "StructuredOutputError",
    "TerminalPolicy",
    "TimeoutExceededError",
    "ToolArgumentError",
    "ToolError",
    "ToolExecutionError",
    "ToolExecutor",
    "ToolLimitExceededError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolSpec",
    "ToolkitError",
    "TransientClientError",
    "UsageSummary",
    "Workflow",
    "WorkflowError",
    "WorkflowResult",
    "WorkflowRunner",
    "arguments_fingerprint",
    "configure_logging",
    "execute_with_retry",
    "is_transient",
    "redact",
    "run_with_timeout",
    "tool",
]
