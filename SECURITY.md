# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| `0.1.x` | Yes |

Pre-`1.0` releases receive fixes on the latest minor version only.

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

- Email: **hamid.digitalmarketer@gmail.com** with the subject `[SECURITY] gemini-agent-toolkit`
- Or use GitHub's private "Report a vulnerability" advisory flow on the repository.

Include:

- Affected version and Python version.
- A description of the issue and its impact.
- A minimal reproduction (code, configuration, log excerpts).
- Any suggested mitigation.

You can expect an acknowledgement within **3 business days** and a status update within
**10 business days**. Please give us a reasonable window to ship a fix before public disclosure; we
will credit reporters who wish to be named.

## Scope

In scope:

- Approval bypass: executing an approval-gated tool without a valid, single-use, argument-bound
  approval.
- Secret leakage in logs produced by `ExecutionLogger` / `redact()`.
- Idempotency or retry defects that can duplicate a side-effecting tool call.
- Prompt-injection paths that let model output escalate tool privileges beyond the registered tools.
- Dependency or supply-chain issues in the published package.

Out of scope:

- Vulnerabilities in Google's Gemini API or the `google-genai` SDK (report those upstream).
- Misuse by application authors, e.g. registering a destructive tool without `requires_approval`,
  or opting into `AllowAllPolicy` in a production path.
- Denial of service through unbounded prompts or outputs you configured yourself.

## Handling secrets

- The toolkit reads the API key from `GEMINI_API_KEY` (or an explicit argument) and never persists it.
- Logs redact keys, bearer tokens and passwords; token **counts** remain visible for cost tracking.
- Never commit `.env` files, captured API traffic or production logs to the repository.
- If you believe a key was exposed, rotate it in Google AI Studio immediately.

## Safe defaults

The default approval policy is `DenyAllPolicy`, so an agent constructed without an explicit policy
will not execute approval-gated tools. Failures in the approval layer are closed, not open.
