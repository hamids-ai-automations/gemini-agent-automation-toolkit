# Roadmap

Scope discipline is a feature. Each item below ships only with tests, docs and a changelog entry.

## 0.1.0 — Foundation (current)

- [x] `Agent` run/resume with typed results and usage metadata
- [x] `GeminiClient` adapter + injectable `LLMClient` protocol
- [x] `@tool` registry with schema generation, validation and execution limits
- [x] Pydantic structured outputs with validation errors
- [x] Retry policy, timeouts, idempotency and loop guards
- [x] Human approval manager with policies, fingerprints and expiry
- [x] Multi-step workflow runner with history
- [x] Structured logging with secret redaction
- [x] Packaging, CI, six runnable examples, docs set

## 0.2.0 — Ergonomics

- [ ] Streaming responses (`Agent.run_stream`) with the same result model
- [ ] Async agent (`await agent.arun(...)`) using `google-genai`'s async client
- [ ] Publish to PyPI via trusted publishing from CI
- [ ] `pytest` fixtures package so downstream apps can reuse the stub client
- [ ] Better schema support: nested models, enums, `Optional` unions documented and tested

## 0.3.0 — Production operations

- [ ] Pluggable trace exporters (OpenTelemetry spans for runs, model calls and tool calls)
- [ ] Cost estimation from usage metadata and a user-supplied price table
- [ ] Persistent approval store interface (SQLite reference implementation)
- [ ] Cancellation support (`agent.cancel(execution_id)`)

## 0.4.0 — Orchestration (only if demand exists)

- [ ] Branching and parallel steps in workflows
- [ ] Retry/approval policy inheritance from workflow to step to tool
- [ ] Multi-agent handoff with explicit, typed contracts

## 1.0.0 — Stability

- [ ] Freeze the public API, document deprecations, guarantee semantic versioning
- [ ] Adapter conformance suite (fake client replaying recorded Gemini transcripts)
- [ ] Security review of approval and redaction paths, published in `SECURITY.md`

## Explicitly out of scope

- Hosting, deployment or a managed service.
- A general-purpose multi-provider abstraction layer.
- A UI. This is a library; the applications belong to its users.
