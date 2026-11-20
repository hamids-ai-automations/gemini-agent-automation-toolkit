# Architecture

## Principles

1. **Thin SDK boundary.** One adapter (`GeminiClient`) knows about `google-genai`; everything else
   talks to the `LLMClient` protocol, which makes the whole toolkit testable without network access.
2. **Explicit failure.** Every failure mode has its own exception in `gemini_agent_toolkit.exceptions`.
   No silent fallbacks, no truthy defaults that hide a broken configuration.
3. **Safety before convenience.** The default approval policy denies, retries never duplicate side
   effects, and logs redact secrets by default.
4. **Small surface.** Frozen dataclasses for configuration and results, plain functions elsewhere.

## Modules

```
Agent (agent.py)
├── config.AgentConfig          immutable runtime settings (model, retries, limits, logging)
├── client.LLMClient            protocol: generate(request) -> LLMResponse
│   └── client.GeminiClient     google-genai adapter (only module importing the SDK)
├── tools.ToolRegistry          name -> ToolSpec (schema, approval flag, timeout, tags)
│   └── tools.ToolExecutor      validate args, enforce limits, run, capture errors
├── outputs.StructuredOutput    Pydantic schema -> Gemini response_schema, then validation
├── reliability.RetryPolicy     backoff schedule + transient classification
├── reliability.ExecutionGuard  per-run call accounting, recursion and loop protection
├── reliability.IdempotentCache argument fingerprint -> cached result
├── approval.ApprovalManager    request/approve/reject/consume tokens with policies
├── logging.ExecutionLogger     structured events + redaction
└── context.ConversationContext in-memory turn history
```

`WorkflowRunner` (`workflows/runner.py`) composes the same primitives at a higher level: it owns a
state dict, invokes `Step` callables, applies per-step retry policies and approval checkpoints, and
returns a `WorkflowResult` with a history of `StepRecord`s.

## Request lifecycle

```
Agent.run(prompt)
 1. context append (user message)  ──────────────────────────────┐
 2. build GeminiClient request: contents + tool declarations      │
 3. execute_with_retry(client.generate)   ← transient errors only │
 4. parse LLMResponse: text? function calls? schema?              │
 5a. plain text / validated structured output → AgentResult       │
 5b. function calls → for each call:                              │
       guard.allow(call)? ──no──► ToolLimitExceededError / loop break
       approval required? ──yes──► pause: store pending state,
                                   return AgentResult(PENDING_APPROVAL)
       executor.execute(call)  ← validated args, timeout, idempotency
       append function_response part                            │
     if more iterations remain → back to step 3 ────────────────┘
 6. finish: status, tool_calls, usage, error, timings → AgentResult
```

`Agent.resume(execution_id)` rehydrates the paused state, reuses the approval tokens already granted,
and continues from step 5b. Approvals are consumed, so a second resume cannot re-execute them.

## Data model

| Type | Role |
| --- | --- |
| `AgentResult` | `execution_id`, `agent_name`, `status`, `output`, `text`, `tool_calls`, `pending_approvals`, `usage`, `error`, `model`, timestamps |
| `WorkflowResult` | `workflow_name`, `status`, `outputs`, `history`, `error`, `pending_approvals`, timestamps |
| `ApprovalRequest` | `token`, `tool_name`, `arguments`, `fingerprint`, `reason`, `created_at`, `expires_at`, `status`, `decided_at`, `decided_by` |
| `LLMResponse` | normalised text, function calls, usage, finish reason |
| `UsageSummary` | prompt / candidate / total token counts when reported |
| `RunStatus` | `SUCCESS`, `PENDING_APPROVAL`, `FAILED`, `TOOL_LIMIT_REACHED` |

## Approval design

- Policies are pluggable: `DenyAllPolicy` (default), `PendingPolicy` (queue for a human),
  `AllowAllPolicy` (explicit opt-in), `CallablePolicy` (delegate to your own function),
  `TerminalPolicy` (non-interactive environments that must refuse).
- A request is created with a random token and a **fingerprint** of `(tool name, normalised
  arguments)`. Approval records the token plus fingerprint; execution requires the token to match
  *and* to be unconsumed *and* unexpired.
- `consume()` makes tokens single-use, so replaying a run cannot reuse a previous decision.
- If the manager cannot be reached or no policy is installed, the call raises
  `ApprovalUnavailableError` — the agent never silently proceeds.

## Reliability design

- `is_transient()` classifies 429/5xx/network/timeout as retryable and 400/401/403/validation as
  permanent; `execute_with_retry` honours `RetryPolicy` (attempts, base delay, cap, jitter).
- `run_with_timeout` enforces wall-clock limits for tool execution; slow tools raise
  `TimeoutExceededError` rather than blocking the run.
- Idempotency: tools marked `@tool(idempotent=True)` cache results by argument fingerprint within a
  run, so a retried or resumed step reuses the previous result instead of repeating the effect.
- `ExecutionGuard` caps total calls, model↔tool iterations, and repeated identical calls
  (`max_repeated_tool_calls`) to break runaway loops before they burn quota.

## Logging design

Events are emitted through the standard `logging` module with a stable structured payload. Redaction
is applied to both keys and values: any key containing `key`, `token`, `secret`, `password` or
`authorization` is masked, values equal to the configured API key are masked wherever they appear, and
numeric token counters (`*_tokens`) stay readable. JSON mode produces one object per line, which is
what an external collector expects.

## Deliberate non-goals for v0.1

- No persistence layer, database, or long-term memory.
- No streaming responses.
- No multi-agent handoff or graph orchestration.
- No provider abstraction beyond the `LLMClient` protocol (Gemini-first by design).

These are tracked in [ROADMAP.md](ROADMAP.md) and will only be added with tests and an honest README
update.
