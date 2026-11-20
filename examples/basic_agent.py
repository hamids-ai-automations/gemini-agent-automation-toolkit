"""Minimal agent: one prompt, one answer.

Requires GEMINI_API_KEY in the environment.

    python examples/basic_agent.py
"""

from __future__ import annotations

from gemini_agent_toolkit import Agent


def main() -> None:
    agent = Agent(
        name="Research Assistant",
        instructions="You are a concise, helpful research assistant.",
    )

    result = agent.run("Explain how AI agents work in three sentences.")

    print(f"status: {result.status}")
    print(f"model:  {result.model}")
    print(f"output: {result.output}")
    print(f"usage:  {result.usage}")


if __name__ == "__main__":
    main()
