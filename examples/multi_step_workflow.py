"""A three-step workflow with state passing and execution history.

Runs fully offline — the steps are plain Python callables.

    python examples/multi_step_workflow.py
"""

from __future__ import annotations

from typing import Any

from gemini_agent_toolkit import Workflow, WorkflowRunner


def collect(state: dict[str, Any]) -> dict[str, int]:
    """Pretend to gather metrics from a data source."""
    return {"visitors": 1_240, "signups": 96}


def summarise(state: dict[str, Any]) -> str:
    data = state["collect"]
    rate = data["signups"] / data["visitors"] * 100
    return f"{data['signups']} signups from {data['visitors']} visitors ({rate:.1f}% conversion)"


def headline(state: dict[str, Any]) -> str:
    return f"Weekly update: {state['summarise']}"


def main() -> None:
    workflow = (
        Workflow(name="weekly-report")
        .add_step("collect", collect, description="Gather raw metrics")
        .add_step("summarise", summarise, description="Compute conversion rate")
        .add_step("headline", headline, description="Produce the final sentence")
    )

    result = WorkflowRunner(workflow).run()

    print(f"status:  {result.status}")
    print(f"output:  {result.outputs['headline']}")
    print("history:")
    for record in result.history:
        print(f"  - {record.name}: {record.status}")


if __name__ == "__main__":
    main()
