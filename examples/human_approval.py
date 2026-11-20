"""Human approval before a destructive tool runs.

Requires GEMINI_API_KEY in the environment. The approval decision here is
simulated in the terminal; in a real application it would come from a UI, a
chat command or an approval service.

    python examples/human_approval.py
"""

from __future__ import annotations

import os
import tempfile

from gemini_agent_toolkit import Agent, PendingPolicy, RunStatus, tool


@tool(requires_approval=True, approval_reason="Deleting a file is irreversible")
def delete_file(file_path: str) -> str:
    """Delete a file after human approval."""
    os.remove(file_path)
    return f"deleted {file_path}"


def main() -> None:
    handle, path = tempfile.mkstemp(prefix="approval-demo-", suffix=".txt")
    os.close(handle)

    agent = Agent(
        name="File Management Agent",
        instructions="Manage files using the available tools.",
        tools=[delete_file],
        approval_policy=PendingPolicy(),
    )

    result = agent.run(f"Delete the file at {path}")

    if result.status is not RunStatus.PENDING_APPROVAL:
        print(f"unexpected status: {result.status} / {result.error}")
        return

    request = result.pending_approvals[0]
    print(f"approval required: {request.tool_name}")
    print(f"arguments:         {request.arguments}")
    print(f"reason:            {request.reason}")
    print(f"expires at:        {request.expires_at}")

    decision = input("approve? [y/N] ").strip().lower()

    if decision == "y":
        agent.approval.approve(request.token, decided_by="cli-user")
        result = agent.resume(result.execution_id)
        print(f"status: {result.status}")
        print(f"output: {result.output}")
    else:
        agent.approval.reject(request.token, decided_by="cli-user")
        print("rejected — the tool was never executed.")

    if os.path.exists(path):
        os.remove(path)


if __name__ == "__main__":
    main()
