"""Request a Pydantic-validated structured output.

Requires GEMINI_API_KEY in the environment.

    python examples/structured_output.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from gemini_agent_toolkit import Agent, StructuredOutputError


class ResearchResult(BaseModel):
    title: str = Field(description="Short title of the summary")
    summary: str = Field(description="Two or three sentence summary")
    key_points: list[str] = Field(description="Between two and five key points")


def main() -> None:
    agent = Agent(
        name="Research Agent",
        instructions="Produce structured research summaries.",
    )

    try:
        result = agent.run(
            "Explain the benefits of Python automation.",
            output_schema=ResearchResult,
        )
    except StructuredOutputError as exc:
        # The model returned something that did not satisfy the schema.
        print(f"invalid output: {exc}")
        return

    parsed: ResearchResult = result.output
    print(f"title:      {parsed.title}")
    print(f"summary:    {parsed.summary}")
    for point in parsed.key_points:
        print(f"  - {point}")


if __name__ == "__main__":
    main()
