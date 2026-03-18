"""
reasoning.py – Trigger-based reasoning engine.

Receives :class:`~triggers.TriggerEvent` objects together with a built
:class:`~context_builder.ConversationContext` and produces a natural-
language response that:

* attributes ideas to the speakers who expressed them;
* references the trigger type in its reasoning strategy;
* is generated within the ~200–500 ms latency budget.

When a large language model (``transformers`` / ``openai``) is available
it will be used; otherwise a rule-based response generator produces a
coherent fallback reply that is still semantically grounded in the
conversation memory.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .context_builder import ConversationContext
from .memory import Utterance
from .triggers import TriggerEvent, TriggerType

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResponse:
    """The AI's generated reply."""

    text: str
    trigger_type: TriggerType
    latency_ms: float
    speaker_attributions: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text.strip())

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ReasoningResponse(type={self.trigger_type.name}, "
            f"latency={self.latency_ms:.0f} ms, "
            f"text={self.text!r})"
        )


# ------------------------------------------------------------------
# Fallback rule-based responder
# ------------------------------------------------------------------

class _RuleBasedResponder:
    """
    Produces human-readable, context-grounded responses without an LLM.

    Extracts key noun phrases from the recent conversation and weaves them
    into templated responses that mimic the desired speaker-attribution
    style.
    """

    _TEMPLATES_QUESTION = [
        "Based on our discussion, {attribution}. {summary}",
        "Looking at what was said earlier, {attribution}. {summary}",
        "To answer that, {attribution}. {summary}",
    ]

    _TEMPLATES_SILENCE = [
        "I'd like to add that {attribution}. {summary}",
        "Building on the conversation so far, {attribution}. {summary}",
        "One thing worth noting — {attribution}. {summary}",
    ]

    _TEMPLATES_INVOCATION = [
        "Sure! {attribution}. {summary}",
        "Happy to contribute — {attribution}. {summary}",
        "Here's my take: {attribution}. {summary}",
    ]

    _TEMPLATE_MAP = {
        TriggerType.QUESTION: _TEMPLATES_QUESTION,
        TriggerType.SILENCE: _TEMPLATES_SILENCE,
        TriggerType.INVOCATION: _TEMPLATES_INVOCATION,
    }

    _template_index: int = 0

    def generate(
        self,
        trigger: TriggerEvent,
        context: ConversationContext,
    ) -> tuple[str, list[str]]:
        """Return ``(response_text, speaker_attributions)``."""
        attribution, attributions = self._build_attribution(context)
        summary = self._build_summary(context)

        templates = self._TEMPLATE_MAP.get(trigger.trigger_type, self._TEMPLATES_QUESTION)
        template = templates[self._template_index % len(templates)]
        self._template_index += 1

        text = template.format(attribution=attribution, summary=summary)
        return text, attributions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_attribution(
        self, context: ConversationContext
    ) -> tuple[str, list[str]]:
        """Build a natural-language attribution string from context."""
        candidates: list[Utterance] = []

        # Prefer semantically relevant history first.
        if context.relevant_history:
            candidates = context.relevant_history[:2]
        elif context.recent_turns:
            # Fall back to the most recent substantive turn.
            candidates = [
                u for u in reversed(context.recent_turns) if u.text.strip()
            ][:2]

        if not candidates:
            return "the conversation has touched on several interesting points", []

        attributions: list[str] = []
        parts: list[str] = []
        for utt in candidates:
            phrase = self._extract_key_phrase(utt.text)
            parts.append(f"{utt.speaker_id} mentioned \"{phrase}\"")
            attributions.append(utt.speaker_id)

        return " and ".join(parts), attributions

    def _build_summary(self, context: ConversationContext) -> str:
        """Build a one-sentence summary of the recent discussion."""
        if not context.recent_turns:
            return "There are multiple perspectives worth considering."

        speakers = list({u.speaker_id for u in context.recent_turns if u.text.strip()})
        if len(speakers) == 1:
            return f"{speakers[0]} has been exploring this topic in depth."
        elif len(speakers) == 2:
            return f"Both {speakers[0]} and {speakers[1]} have contributed valuable ideas."
        else:
            others = ", ".join(speakers[:-1])
            return f"{others} and {speakers[-1]} have each offered different perspectives."

    @staticmethod
    def _extract_key_phrase(text: str) -> str:
        """Extract the most meaningful short phrase from an utterance."""
        text = text.strip().rstrip(".")
        # Take the longest clause (split on commas/conjunctions)
        parts = re.split(r"\b(and|but|or|because|however|,)\b", text, flags=re.IGNORECASE)
        meaningful = [p.strip() for p in parts if len(p.strip()) > 5]
        if meaningful:
            # Return the longest part, capped at 60 characters
            longest = max(meaningful, key=len)
            return longest[:60] + ("…" if len(longest) > 60 else "")
        return text[:60]


# ------------------------------------------------------------------
# LLM-backed responder (optional)
# ------------------------------------------------------------------

class _TransformersResponder:
    """Uses a local HuggingFace model for response generation."""

    def __init__(self, model_name: str) -> None:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]

        self._pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
        )
        self._model_name = model_name
        logger.info("Loaded HuggingFace model: %s", model_name)

    def generate(
        self,
        trigger: TriggerEvent,
        context: ConversationContext,
    ) -> tuple[str, list[str]]:
        prompt = self._build_prompt(trigger, context)
        output = self._pipe(prompt)[0]["generated_text"]
        # Strip the prompt prefix
        response = output[len(prompt):].strip()
        # Simple attribution extraction
        attributions = re.findall(r"(Speaker_\w+|Mr\.?\s+\w+)", response)
        return response, attributions

    def _build_prompt(self, trigger: TriggerEvent, context: ConversationContext) -> str:
        ctx_text = context.render()
        return (
            f"{ctx_text}\n\n"
            f"As an AI participant in this discussion, provide a helpful, "
            f"concise response that references earlier speakers by name "
            f"when relevant.\nAI response:"
        )


# ------------------------------------------------------------------
# Public facade
# ------------------------------------------------------------------

class ReasoningEngine:
    """
    Trigger-based reasoning engine.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  When ``None`` or unavailable the
        rule-based responder is used.
    passive_mode:
        When ``True`` the engine only responds to :attr:`~TriggerType.INVOCATION`
        triggers.  When ``False`` (default) it also responds to
        ``SILENCE`` and ``QUESTION`` triggers.

    Usage
    -----
    ::

        engine = ReasoningEngine()
        response = engine.respond(trigger_event, context)
        if response:
            speak(response.text)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        passive_mode: bool = False,
    ) -> None:
        self.passive_mode = passive_mode
        self._responder = self._build_responder(model_name)
        logger.info(
            "ReasoningEngine ready (passive=%s, backend=%s)",
            passive_mode,
            type(self._responder).__name__,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def respond(
        self,
        trigger: TriggerEvent,
        context: ConversationContext,
    ) -> Optional[ReasoningResponse]:
        """
        Generate a response for *trigger* given *context*.

        Returns ``None`` when the engine decides not to respond (e.g.
        passive mode + non-invocation trigger, or empty context).
        """
        if self.passive_mode and trigger.trigger_type != TriggerType.INVOCATION:
            logger.debug("Passive mode: ignoring %s trigger", trigger.trigger_type.name)
            return None

        if not context.recent_turns and not context.relevant_history:
            return ReasoningResponse(
                text="I don't have enough context to contribute yet.",
                trigger_type=trigger.trigger_type,
                latency_ms=0.0,
            )

        t0 = time.monotonic()
        text, attributions = self._responder.generate(trigger, context)
        latency_ms = (time.monotonic() - t0) * 1_000

        logger.info(
            "ReasoningEngine responded in %.0f ms (trigger=%s)",
            latency_ms,
            trigger.trigger_type.name,
        )
        return ReasoningResponse(
            text=text,
            trigger_type=trigger.trigger_type,
            latency_ms=latency_ms,
            speaker_attributions=attributions,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_responder(self, model_name: Optional[str]):
        if model_name:
            try:
                return _TransformersResponder(model_name)
            except ImportError:
                logger.warning("transformers not available; using rule-based responder")
            except Exception as exc:
                logger.warning("Could not load model '%s' (%s); using rule-based responder", model_name, exc)
        return _RuleBasedResponder()
