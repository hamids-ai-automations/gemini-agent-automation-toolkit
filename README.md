# Gemini Agent Automation Toolkit

[![tests](https://github.com/hamids-ai-automations/gemini-agent-automation-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/hamids-ai-automations/gemini-agent-automation-toolkit/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A small, modular Python toolkit for building **reliable** Google Gemini agents: validated tool
calling, Pydantic structured outputs, retries with backoff, human approval for sensitive actions,
multi-step workflows and redacted execution logging.

It is deliberately opinionated and dependency-light — a thin, testable layer over the official
[`google-genai`](https://github.com/googleapis/python-genai) SDK instead of another framework.

- Project page: **https://marvis.web.app/projects/gemini-agent-automation-toolkit.html**
- Repository: **https://github.com/hamids-ai-automations/gemini-agent-automation-toolkit**

---

## Why this toolkit

Most "Gemini wrapper" packages stop at a single API call. Real agents need the boring parts:
tools that validate their own arguments, outputs that fail loudly instead of silently, retries that
do not duplicate side effects, a human in the loop before anything destructive happens, and logs you
can hand to a security reviewer.

| Capability | What you get |
| --- | --- |
| Agent engine | `Agent(...).run(prompt)` with instructions, context, usage metadata and typed results |
| Tools | `@tool` decorator, JSON schema generation, argument validation, recursion and call limits |
| Structured output | Pydantic schemas validated on the way out, with explicit `StructuredOutputError` |
| Reliability | Exponential backoff with jitter, transient/permanent error classification, timeouts |
| Human approval | Policies, single-use tokens bound to arguments, expiry, safe failure when unavailable |
| Workflows | Ordered steps, state passing, per-step retries, approval checkpoints, execution history |
| Observability | Structured JSON or plain logs, secrets and API keys redacted, token usage recorded |

Everything is in-memory by design; nothing is persisted unless you persist it yourself.

## Install

```bash
pip install gemini-agent-toolkit
```

From source (development):

```bash
git clone https://github.com/hamids-ai-automations/gemini-agent-automation-toolkit.git
cd gemini-agent-automation-toolkit
python -m venv .venv
.\.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
```

Python 3.10+ is required. The only runtime dependencies are `google-genai` and `pydantic` v2.

## Quick start

Set your key once (never commit it):

```bash
set GEMINI_API_KEY=your-key        # PowerShell: $env:GEMINI_API_KEY="your-key"
```

```python
from gemini_agent_toolkit import Agent

agent = Agent(
    name="Research Assistant",
    instructions="You are a helpful research assistant.",
)

result = agent.run("Explain how AI agents work in three sentences.")
print(result.output)
print(result.usage)  # token usage when the API reports it
```

## Tools

```python
from gemini_agent_toolkit import Agent, tool


@tool
def calculate_total(price: float, quantity: int) -> float:
    """Calculate a purchase total."""
    return price * quantity


agent = Agent(
    name="Shopping Assistant",
    instructions="Help users calculate totals.",
    tools=[calculate_total],
)

result = agent.run("Calculate the total for 5 items costing 20 each.")
```

The model proposes a tool call; the toolkit validates arguments against the generated schema,
executes the function, feeds the real result back to the model, and records the call. A proposal is
never reported as an executed call.

Useful decorator options:

```python
@tool(requires_approval=True, approval_reason="Deletes user data")
def delete_file(file_path: str) -> str: ...


@tool(idempotent=True, timeout=5.0, tags=("files",))
def read_file(file_path: str) -> str: ...
```

## Structured output

```python
from pydantic import BaseModel
from gemini_agent_toolkit import Agent


class ResearchResult(BaseModel):
    title: str
    summary: str
    key_points: list[str]


agent = Agent(name="Research Agent", instructions="Produce structured research summaries.")
result = agent.run("Explain the benefits of Python automation.", output_schema=ResearchResult)

print(result.output.summary)  # a validated ResearchResult instance
```

Validation failures raise `StructuredOutputError` with the offending payload attached, so you can
decide to retry, repair or surface the problem — the toolkit will not hand you a dict that only
looks like your model.

## Retries, timeouts and error handling

```python
agent = Agent(name="Reliable Agent", max_retries=3, timeout=30)

from gemini_agent_toolkit import RetryPolicy, is_transient

policy = RetryPolicy(max_attempts=4, base_delay=0.5, max_delay=8.0, jitter=0.1)
```

Only transient conditions (rate limits, 5xx, network errors, timeouts) are retried; authentication
and validation errors fail immediately. Side-effecting tools are guarded by an idempotency cache and
an execution guard so a retry cannot silently run the same call twice. Exhausted retries raise
`RetryExhaustedError`.

## Human approval

```python
from gemini_agent_toolkit import Agent, tool, PendingPolicy


@tool(requires_approval=True)
def delete_file(file_path: str) -> str:
    """Delete a file after human approval."""
    import os

    os.remove(file_path)
    return f"deleted {file_path}"


agent = Agent(
    name="File Management Agent",
    tools=[delete_file],
    approval_policy=PendingPolicy(),  # safe default is DenyAllPolicy
)

result = agent.run("Delete the old report.")
if result.status is RunStatus.PENDING_APPROVAL:
    request = result.pending_approvals[0]
    print(request.tool_name, request.arguments, request.expires_at)
    # ... human decides ...
    agent.approval.approve(request.token)  # or .reject(request.token)
    result = agent.resume(result.execution_id)
```

Guarantees:

- An approval-gated tool never executes before approval.
- Tokens are single-use and bound to a fingerprint of the arguments, so approving `a.txt` cannot
  authorise `b.txt`.
- Requests expire (`AgentConfig.approval_ttl_seconds`, default 900s) and expired tokens are rejected.
- The default policy denies everything; `AllowAllPolicy` exists but must be requested explicitly.
- If the approval mechanism is unavailable, execution fails closed with `ApprovalUnavailableError`.

## Workflows

```python
from gemini_agent_toolkit import Workflow, WorkflowRunner

workflow = (
    Workflow(name="report")
    .add_step("collect", lambda state: {"rows": 42})
    .add_step("summarise", lambda state: f"{state['collect']['rows']} rows collected")
)

result = WorkflowRunner(workflow).run()
print(result.status, result.outputs["summarise"])
print([record.name for record in result.history])
```

Steps receive the running state dict, return a value stored under the step name, and can require
approval or carry their own retry policy. A failing step stops the run with `StepFailedError`
captured on the result.

## Logging and observability

```python
from gemini_agent_toolkit import Agent, AgentConfig

agent = Agent(
    name="Audited Agent",
    config=AgentConfig(log_level="DEBUG", json_logs=True),
)
```

Records include execution id, agent name, model, timings, status, tool calls and results, retries,
approval decisions, errors and token usage. API keys, bearer tokens, passwords and other secrets are
redacted by `redact()` before anything is written; token *counts* stay readable so cost tracking
still works. Logs go through the standard `logging` module, so you can route them anywhere later.

## Configuration

`AgentConfig` is an immutable dataclass; anything can be overridden per agent or via keyword
arguments (`Agent(..., model=..., max_retries=..., timeout=...)`).

| Field | Default | Purpose |
| --- | --- | --- |
| `model` | `gemini-3.8-flash` | Generation model (never hardcoded elsewhere) |
| `temperature` / `top_p` | `None` | Optional sampling controls |
| `max_output_tokens` | `2048` | Output cap |
| `timeout` | `60.0` | Per-request timeout in seconds |
| `max_retries` | `3` | Retry attempts for transient failures |
| `max_tool_calls` | `8` | Tool calls allowed per run |
| `max_tool_iterations` | `6` | Model↔tool round trips per run |
| `max_repeated_tool_calls` | `2` | Guards against tool loops |
| `approval_ttl_seconds` | `900` | Approval token lifetime |
| `log_level` / `json_logs` | `INFO` / `False` | Logging behaviour |

The API key comes from `GEMINI_API_KEY` or an explicit `api_key=` argument — never from source.

## Project layout

```
gemini_agent_toolkit/
├── agent.py          # run / resume orchestration
├── client.py         # thin google-genai adapter + LLMClient protocol for mocking
├── config.py         # AgentConfig, defaults, env handling
├── context.py        # in-memory conversation context
├── models.py         # typed results: AgentResult, WorkflowResult, ApprovalRequest ...
├── tools/            # registry, executor, @tool decorator
├── outputs/          # Pydantic structured-output helpers and validators
├── reliability/      # retry, timeout, idempotency/recovery guards
├── approval/         # policies and approval manager
├── logging/          # execution logger + redaction
└── workflows/        # step runner
```

Design notes live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the ecosystem survey behind the
design is in [docs/RESEARCH.md](docs/RESEARCH.md); planned work is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Examples

| File | Shows |
| --- | --- |
| `examples/basic_agent.py` | Minimal agent, usage metadata |
| `examples/tool_calling_agent.py` | Registered tools and real execution |
| `examples/structured_output.py` | Pydantic schema validation |
| `examples/human_approval.py` | Pending approval → approve → resume |
| `examples/retry_handling.py` | Backoff, transient vs permanent errors |
| `examples/multi_step_workflow.py` | Multi-step workflow with history |

Examples require `GEMINI_API_KEY` (except `retry_handling.py` and `multi_step_workflow.py`, which run
entirely offline against a stub client).

## Development

```bash
pytest -q          # 85 tests, no network access required
ruff check .
mypy gemini_agent_toolkit
```

Tests never call the real API: a `StubClient` implements the same `LLMClient` protocol as
`GeminiClient`, so retries, approvals and tool loops are all exercised deterministically.

## Status and limitations

- Version `0.1.0` — the API is usable today but may still change before `1.0.0`.
- Context is in-memory only; there is no built-in persistence or vector memory.
- No streaming API yet (see the roadmap).
- Workflows are intentionally minimal: no branching, parallelism or distributed execution.
- Model availability differs per tier; `gemini-3.8-flash` is a default, not a promise.

## Contributing and security

Contributions are welcome — read [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md). Please report vulnerabilities privately as described in
[SECURITY.md](SECURITY.md) rather than in a public issue.

## License

MIT © Muhammad Hamid. See [LICENSE](LICENSE).

Built and maintained as part of the [Marvis](https://marvis.web.app/) automation portfolio.
