"""Register a Python function as a tool and let the model call it.

Requires GEMINI_API_KEY in the environment.

    python examples/tool_calling_agent.py
"""

from __future__ import annotations

from gemini_agent_toolkit import Agent, tool


@tool
def calculate_total(price: float, quantity: int) -> float:
    """Calculate a purchase total."""
    return price * quantity


def main() -> None:
    agent = Agent(
        name="Shopping Assistant",
        instructions="Help users calculate totals. Use the tool instead of guessing.",
        tools=[calculate_total],
    )

    result = agent.run("Calculate the total for 5 items costing 20 each.")

    print(f"status: {result.status}")
    print(f"output: {result.output}")
    print("tool calls:")
    for call in result.tool_calls:
        print(f"  - {call.get('name')}({call.get('args')}) -> {call.get('result')}")


if __name__ == "__main__":
    main()
