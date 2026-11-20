# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

First public release. The API is usable but may change before `1.0.0`.

### Added

- `Agent` engine: instructions, per-run prompts, in-memory `ConversationContext`, typed
  `AgentResult` with usage metadata, `resume()` for paused executions.
- `GeminiClient` built on the official `google-genai` SDK, plus an `LLMClient` protocol so tests and
  custom transports can be injected instead.
- Tool system: `@tool` decorator, `ToolRegistry`, `ToolExecutor`, schema generation from type hints,
  argument validation, per-run call/iteration limits and recursion protection.
- Structured outputs: Pydantic v2 schemas validated on the way out with `StructuredOutputError`
  carrying the offending payload.
- Reliability: `RetryPolicy` with exponential backoff and jitter, transient/permanent error
  classification, `run_with_timeout`, idempotency cache and execution guard.
- Human approval: `ApprovalManager` with `DenyAllPolicy` (default), `PendingPolicy`,
  `AllowAllPolicy`, `CallablePolicy`, `TerminalPolicy`; single-use tokens bound to an argument
  fingerprint with expiry and fail-closed behaviour.
- Workflows: `Workflow` / `Step` / `WorkflowRunner` with state passing, per-step retries, approval
  checkpoints and execution history.
- Observability: `ExecutionLogger` with plain or structured JSON output, `redact()` for secrets and
  token-count preservation.
- Documentation: README, `docs/RESEARCH.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, six runnable
  examples.
- Tooling: `pyproject.toml` packaging, pytest suite (85 tests, fully offline), GitHub Actions CI on
  Python 3.10–3.12.

[0.1.0]: https://github.com/hamids-ai-automations/gemini-agent-automation-toolkit/releases/tag/v0.1.0
