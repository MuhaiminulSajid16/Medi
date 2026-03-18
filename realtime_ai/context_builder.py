"""
context_builder.py – Incremental conversation context builder.

Combines the last N turns from the hot cache with semantically relevant
historical utterances retrieved from long-term memory to produce a
compact context string for the reasoning engine.

The context window is refreshed every *update_interval_sec* seconds so
that Moshi always reasons over up-to-date information without processing
the full history on every token.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .memory import ConversationMemory, Utterance

logger = logging.getLogger(__name__)

# Default tuning knobs (can be overridden at construction time).
DEFAULT_RECENT_TURNS = 10
DEFAULT_SEMANTIC_TOP_K = 3
DEFAULT_UPDATE_INTERVAL_SEC = 1.5


@dataclass
class ConversationContext:
    """A snapshot of the conversation context ready for reasoning."""

    recent_turns: list[Utterance]
    relevant_history: list[Utterance]
    current_query: str
    built_at: float = field(default_factory=time.monotonic)

    def render(self) -> str:
        """
        Render a human-readable context string suitable for injection
        into an LLM prompt.
        """
        lines: list[str] = []

        if self.relevant_history:
            lines.append("=== Relevant earlier discussion ===")
            for utt in self.relevant_history:
                ts = time.strftime("%H:%M:%S", time.localtime(utt.timestamp))
                lines.append(f"[{ts}] {utt.speaker_id}: {utt.text}")
            lines.append("")

        if self.recent_turns:
            lines.append("=== Recent conversation ===")
            for utt in self.recent_turns:
                ts = time.strftime("%H:%M:%S", time.localtime(utt.timestamp))
                lines.append(f"[{ts}] {utt.speaker_id}: {utt.text}")
            lines.append("")

        if self.current_query:
            lines.append(f"=== Current query ===\n{self.current_query}")

        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ConversationContext("
            f"recent={len(self.recent_turns)}, "
            f"history={len(self.relevant_history)}, "
            f"query={self.current_query!r})"
        )


class ContextBuilder:
    """
    Builds incremental conversation context for the reasoning engine.

    Parameters
    ----------
    memory:
        :class:`~memory.ConversationMemory` instance shared with the
        rest of the pipeline.
    recent_turns:
        Number of most recent utterances to include verbatim.
    semantic_top_k:
        Number of semantically relevant historical utterances to retrieve.
    update_interval_sec:
        Minimum seconds between full context rebuilds.  Within this
        window, :meth:`build` returns a cached result to avoid redundant
        retrieval overhead.

    Usage
    -----
    ::

        builder = ContextBuilder(memory)
        ctx = builder.build(current_query="What should we do now?")
        print(ctx.render())
    """

    def __init__(
        self,
        memory: ConversationMemory,
        recent_turns: int = DEFAULT_RECENT_TURNS,
        semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
        update_interval_sec: float = DEFAULT_UPDATE_INTERVAL_SEC,
    ) -> None:
        self._memory = memory
        self._recent_turns = recent_turns
        self._semantic_top_k = semantic_top_k
        self._update_interval_sec = update_interval_sec
        self._last_built: float = 0.0
        self._cached_context: Optional[ConversationContext] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, current_query: str = "") -> ConversationContext:
        """
        Return a :class:`ConversationContext` for *current_query*.

        When the context was last rebuilt less than *update_interval_sec*
        ago and *current_query* has not changed, the cached context is
        returned directly to reduce latency.
        """
        now = time.monotonic()
        if (
            self._cached_context is not None
            and (now - self._last_built) < self._update_interval_sec
            and self._cached_context.current_query == current_query
        ):
            return self._cached_context

        recent = self._memory.get_recent(self._recent_turns)

        # Semantic retrieval is only meaningful when there is a query.
        relevant: list[Utterance] = []
        if current_query:
            semantic_hits = self._memory.search_semantic(
                current_query, top_k=self._semantic_top_k
            )
            # Exclude utterances already in the recent window to avoid
            # duplication.
            recent_ids = {u.utterance_id for u in recent}
            relevant = [u for u in semantic_hits if u.utterance_id not in recent_ids]

        ctx = ConversationContext(
            recent_turns=recent,
            relevant_history=relevant,
            current_query=current_query,
        )
        self._last_built = now
        self._cached_context = ctx
        logger.debug(
            "Context built: %d recent, %d relevant, query=%r",
            len(recent),
            len(relevant),
            current_query,
        )
        return ctx

    def invalidate(self) -> None:
        """Force a full rebuild on the next :meth:`build` call."""
        self._last_built = 0.0
        self._cached_context = None
