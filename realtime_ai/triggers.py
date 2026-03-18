"""
triggers.py – Conversation trigger detection.

Determines **when** the reasoning engine should fire by monitoring three
independent signals:

1. **Silence trigger** – a configurable gap (default 500 ms) with no
   meaningful audio activity.
2. **Question trigger** – the ASR partial or final transcript ends with a
   question-word pattern (WH-form or rising intonation marker).
3. **Invocation trigger** – the AI is directly addressed by a configurable
   wake-phrase (default ``"moshi"``).

Triggers are debounced: once a trigger fires it cannot fire again until
*cooldown_sec* seconds have elapsed.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .asr import ASRResult
from .audio_chunker import AudioFrame

logger = logging.getLogger(__name__)

# RMS threshold for considering a frame "silent".
_SILENCE_RMS = 0.01

# WH-question patterns (case-insensitive).
_QUESTION_PATTERNS = re.compile(
    r"\b(what|why|how|where|when|who|which|whom|whose|"
    r"should we|can we|do you|does|is this|are you)\b.*\??\s*$",
    re.IGNORECASE,
)

# Pattern that matches a sentence ending in "?" (for partial transcripts
# that may not yet have punctuation).
_RISING_INTONATION = re.compile(
    r"\b(right|correct|agree|think|suggest|mean|know)\s*\??\s*$",
    re.IGNORECASE,
)


class TriggerType(Enum):
    SILENCE = auto()
    QUESTION = auto()
    INVOCATION = auto()


@dataclass
class TriggerEvent:
    """Emitted by :class:`TriggerDetector` when a trigger fires."""

    trigger_type: TriggerType
    text: str           # The transcript text that caused the trigger (may be empty)
    speaker_id: str     # Who caused the trigger
    timestamp: float

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TriggerEvent(type={self.trigger_type.name}, "
            f"speaker={self.speaker_id!r}, "
            f"text={self.text!r})"
        )


class TriggerDetector:
    """
    Monitors the audio + ASR stream and fires :class:`TriggerEvent` objects
    when the reasoning engine should respond.

    Parameters
    ----------
    silence_ms:
        Consecutive silence required to fire a silence trigger (ms).
    wake_phrase:
        Lower-cased word(s) that directly invoke the AI (default: "moshi").
    cooldown_sec:
        Minimum seconds between successive trigger firings.

    Usage
    -----
    ::

        detector = TriggerDetector()

        # Feed audio frames to track silence
        event = detector.feed_frame(frame, current_speaker="Speaker_0")

        # Feed ASR results to detect questions / invocations
        event = detector.feed_asr(asr_result)
    """

    def __init__(
        self,
        silence_ms: float = 500.0,
        wake_phrase: str = "moshi",
        cooldown_sec: float = 2.0,
    ) -> None:
        self.silence_ms = silence_ms
        self.wake_phrase = wake_phrase.lower()
        self.cooldown_sec = cooldown_sec

        self._silence_start: Optional[float] = None
        self._last_trigger_time: float = 0.0
        self._last_speaker: str = "UNKNOWN"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_frame(
        self,
        frame: AudioFrame,
        current_speaker: str = "UNKNOWN",
    ) -> Optional[TriggerEvent]:
        """
        Process one audio frame.  Returns a silence :class:`TriggerEvent`
        when the configured gap has elapsed, otherwise ``None``.

        Silence duration is measured using the audio frame's own timestamp
        (which advances at the audio sample rate) so that the detector
        behaves correctly both in real-time streaming and in accelerated
        test playback.
        """
        import numpy as np  # local import to keep module-level imports minimal

        rms = float(np.sqrt(np.mean(frame.samples ** 2)))
        is_silent = rms < _SILENCE_RMS

        if is_silent:
            if self._silence_start is None:
                self._silence_start = frame.timestamp
            elapsed_ms = (frame.timestamp - self._silence_start) * 1_000
            if elapsed_ms >= self.silence_ms and self._can_trigger():
                self._record_trigger()
                self._silence_start = None
                return TriggerEvent(
                    trigger_type=TriggerType.SILENCE,
                    text="",
                    speaker_id=self._last_speaker,
                    timestamp=frame.timestamp,
                )
        else:
            self._silence_start = None
            self._last_speaker = current_speaker

        return None

    def feed_asr(self, result: ASRResult) -> Optional[TriggerEvent]:
        """
        Inspect an :class:`~asr.ASRResult` for question patterns and
        wake-phrase invocations.

        Returns a :class:`TriggerEvent` or ``None``.
        """
        if not result.text:
            return None

        text = result.text.strip()
        speaker_id = result.speaker_id or "UNKNOWN"

        # Invocation check (highest priority)
        if self.wake_phrase in text.lower() and self._can_trigger():
            self._record_trigger()
            return TriggerEvent(
                trigger_type=TriggerType.INVOCATION,
                text=text,
                speaker_id=speaker_id,
                timestamp=result.timestamp,
            )

        # Question check (final transcripts only to reduce false positives)
        if result.is_final and self._is_question(text) and self._can_trigger():
            self._record_trigger()
            return TriggerEvent(
                trigger_type=TriggerType.QUESTION,
                text=text,
                speaker_id=speaker_id,
                timestamp=result.timestamp,
            )

        return None

    def reset(self) -> None:
        """Reset internal state (silence timer, cooldown)."""
        self._silence_start = None
        self._last_trigger_time = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _can_trigger(self) -> bool:
        return (time.monotonic() - self._last_trigger_time) >= self.cooldown_sec

    def _record_trigger(self) -> None:
        self._last_trigger_time = time.monotonic()

    def _is_question(self, text: str) -> bool:
        return bool(
            _QUESTION_PATTERNS.search(text) or _RISING_INTONATION.search(text)
        )
