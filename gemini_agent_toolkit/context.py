"""In-memory conversation context.

Persistence is deliberately *not* claimed: everything lives in the process that
created the :class:`ConversationContext` instance. Callers that need durable
memory can serialise :meth:`ConversationContext.to_list` themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationContext:
    """Ordered conversation turns with a bounded window.

    Args:
        max_turns: maximum number of *turns* (user + model messages) retained.
            ``0`` disables the window. Trimming always drops the oldest turns
            and never splits a turn.
        system_note: optional extra note appended to the agent instructions.
    """

    turns: list[dict[str, Any]] = field(default_factory=list)
    max_turns: int = 40
    system_note: str | None = None

    def __post_init__(self) -> None:
        if self.max_turns < 0:
            raise ValueError("max_turns must be >= 0")
        self._trim()

    # -- mutation --------------------------------------------------------- #
    def add_user(self, text: str) -> None:
        self.add({"role": "user", "parts": [{"text": text}]})

    def add_model(self, content: dict[str, Any]) -> None:
        self.add(content)

    def add(self, turn: dict[str, Any]) -> None:
        """Append a raw Gemini ``Content`` dict."""
        self.turns.append(turn)
        self._trim()

    def reset(self) -> None:
        """Drop the whole history (explicit context reset)."""
        self.turns.clear()

    def _trim(self) -> None:
        if self.max_turns and len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    # -- read ------------------------------------------------------------- #
    def to_list(self) -> list[dict[str, Any]]:
        """Return a copy of the turns, safe to hand to the SDK.

        Turns may be plain dicts (toolkit-built) or SDK ``Content`` objects
        (echoed model turns); the latter are passed through unchanged.
        """
        return [dict(turn) if isinstance(turn, dict) else turn for turn in self.turns]

    def copy(self) -> ConversationContext:
        return ConversationContext(
            turns=self.to_list(), max_turns=self.max_turns, system_note=self.system_note
        )

    def __len__(self) -> int:
        return len(self.turns)
