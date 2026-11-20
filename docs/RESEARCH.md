# Research Notes

Survey performed before writing code, and revisited while shaping the v0.1 API. Sources are the
official Gemini documentation listed at the end; anything not verified against a first-party source
is marked as an assumption and is not relied on by the implementation.

## 1. The Gemini developer surface

- **SDK.** `google-genai` is the maintained Python client (the older `google-generativeai` package is
  legacy). It exposes `client.models.generate_content(...)` with `contents`, `config`
  (`GenerateContentConfig`) and returns `response.text`, `response.candidates[0].content.parts` and
  `response.usage_metadata`.
- **Function calling.** Tools are declared as function declarations (name, description, JSON-schema
  parameters). The model returns `function_call` parts; the application executes the function and
  sends back `function_response` parts inside a new turn. Execution is therefore always the
  application's responsibility — the SDK never runs Python for you.
- **Structured output.** `response_mime_type="application/json"` plus `response_schema` (a Pydantic
  model or an OpenAPI-style schema) constrains generation; the payload still needs validation
  because models can emit truncated or `null`-filled JSON when interrupted.
- **Errors and limits.** Rate limiting surfaces as HTTP 429 with `RetryInfo` guidance, capacity
  problems as 503, invalid requests as 400, auth problems as 401/403. Free-tier model availability
  shifts over time (models get retired or capacity-restricted), which is why the model name must be
  configuration, never a literal scattered through the code.
- **Usage metadata.** `prompt_token_count`, `candidates_token_count` and `total_token_count` are
  reported per response when available — the basis for the toolkit's `UsageSummary`.

## 2. Prior art and what we borrowed

| Project | Strength | Gap this toolkit targets |
| --- | --- | --- |
| `google-genai` SDK | Correct, complete API surface | Raw transport: no agent loop, no validation, no approval |
| Google ADK | Full agent platform, multi-agent, deployment | Heavy for a script that needs one reliable agent |
| LangChain / LangGraph | Huge integration surface, graph orchestration | Abstraction weight and churn; Gemini specifics leak |
| Pydantic AI | Excellent typed agent ergonomics | Opinionated runtime; approval story is thinner |
| OpenAI Agents SDK | Clean handoffs/guardrails design | OpenAI-first; Gemini users need a port |
| Instructor | Reliable structured extraction | Extraction only, no tool execution or approvals |
| Plain scripts | Zero dependencies | Each team re-implements retries, validation, logging badly |

The design conclusion: keep the SDK boundary thin (one `GeminiClient` adapter behind an `LLMClient`
protocol), own the reliability and safety layers explicitly, and refuse to invent an orchestration
framework in v0.1.

## 3. Patterns that shaped specific decisions

**Tool validation.** Generate the JSON schema from type hints so the model sees the same contract the
executor enforces; validate *before* execution and return a structured error to the model so it can
self-correct instead of crashing the run.

**Retry safety.** Retrying a model call is cheap; retrying a tool that sent an email is not. Hence
two separate mechanisms: transient-only retries for client calls, and an idempotency cache plus
execution guard for side-effecting tools, so a resumed or retried run cannot duplicate an effect.

**Human-in-the-loop.** The pattern that survives review is: pause with a durable token, bind the
token to the arguments (fingerprint), expire it, and fail closed when no approval channel exists.
Auto-approval shims are the most common way this feature becomes theatre.

**Logging.** Structured JSON with a stable field set (`execution_id`, `agent`, `model`, `status`,
`tool`, `attempt`, `usage`) is what makes agent runs debuggable in production. Redaction must be
key-based *and* value-based (the API key string can appear inside an error message), while numeric
counters such as `total_tokens` must stay readable for cost analysis.

**Workflows.** A step list with named outputs and a history array covers the majority of real
automation jobs (fetch → transform → summarise → publish). Branching and parallelism are deliberately
out of scope for v0.1.

**Context.** In-memory conversation store with explicit trimming/reset is honest and predictable.
Claiming "memory" without persistence is a common source of user surprise, so the README states the
limitation plainly.

## 4. Assumptions we did not build on

- Exact model names and free-tier availability are treated as configuration, not constants. The
  default `gemini-3.8-flash` reflects the current tier this project was validated against; callers
  should verify with their own account.
- No claim is made about tool-calling accuracy improvements from schema descriptions beyond what the
  documentation states; the toolkit only guarantees validation and observability.
- Pricing and quota behaviour per model are not encoded anywhere.

## 5. Sources

- Gemini API docs — https://ai.google.dev/gemini-api/docs
- Quickstart — https://ai.google.dev/gemini-api/docs/quickstart
- Function calling — https://ai.google.dev/gemini-api/docs/function-calling
- Structured output — https://ai.google.dev/gemini-api/docs/structured-output
- Models — https://ai.google.dev/gemini-api/docs/models
- Google Gen AI Python SDK — https://github.com/googleapis/python-genai
- Google AI developer resources — https://ai.google.dev/
- Prior art: Google ADK, Pydantic AI, LangGraph, OpenAI Agents SDK, Instructor (public docs)
