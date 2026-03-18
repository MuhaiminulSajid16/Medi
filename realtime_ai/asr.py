"""
asr.py – Streaming Automatic Speech Recognition with partial transcripts.

Supports two modes:
* **Vosk** (recommended for low-latency local inference) – used when the
  ``vosk`` package is installed.
* **Lightweight mock** – a rule-based fallback that produces placeholder
  partial and final transcripts.  Useful for testing the pipeline without
  installing heavy ML packages.

The public interface is the same in both modes, enabling transparent
swapping.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .audio_chunker import AudioFrame

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """A single ASR output (partial or final)."""

    text: str
    is_final: bool
    confidence: float = 1.0
    speaker_id: Optional[str] = None
    timestamp: float = field(default_factory=time.monotonic)

    def __bool__(self) -> bool:
        return bool(self.text.strip())

    def __repr__(self) -> str:  # pragma: no cover
        label = "final" if self.is_final else "partial"
        speaker = f", speaker={self.speaker_id!r}" if self.speaker_id else ""
        return f"ASRResult({label!r}, text={self.text!r}{speaker})"


class _MockASRBackend:
    """
    Lightweight mock ASR backend.

    Accumulates audio energy and emits realistic partial transcripts by
    replaying a small phrase pool.  Intended **only** for unit tests and
    development without Vosk installed.
    """

    _PHRASES = [
        "I think we should use approach A",
        "Maybe method B would work better",
        "What do you think about this idea",
        "Let me add something here",
        "Can we revisit the earlier point",
        "I agree with the previous suggestion",
        "How about combining both approaches",
        "What should we do now",
    ]
    _PARTIAL_STEPS = [0.25, 0.55, 0.80, 1.0]

    def __init__(self) -> None:
        self._phrase_index: int = 0
        self._energy_accumulator: float = 0.0
        self._energy_threshold: float = 0.5
        self._partial_step: int = 0
        self._last_phrase: Optional[str] = None

    def accept_waveform(self, samples: np.ndarray) -> None:
        energy = float(np.sqrt(np.mean(samples ** 2)))
        self._energy_accumulator += energy

    def partial_result(self) -> str:
        """Return current partial transcript as a JSON string (Vosk-compatible)."""
        if self._energy_accumulator < self._energy_threshold * 0.3:
            return json.dumps({"partial": ""})

        phrase = self._PHRASES[self._phrase_index % len(self._PHRASES)]
        step = self._PARTIAL_STEPS[self._partial_step % len(self._PARTIAL_STEPS)]
        partial_text = phrase[: max(1, int(len(phrase) * step))]
        self._partial_step = (self._partial_step + 1) % len(self._PARTIAL_STEPS)
        return json.dumps({"partial": partial_text})

    def result(self) -> str:
        """Return the final transcript as a JSON string (Vosk-compatible)."""
        if self._energy_accumulator < self._energy_threshold:
            return json.dumps({"text": ""})

        phrase = self._PHRASES[self._phrase_index % len(self._PHRASES)]
        self._phrase_index += 1
        self._energy_accumulator = 0.0
        self._partial_step = 0
        self._last_phrase = phrase
        return json.dumps({"text": phrase, "result": [{"conf": 0.95, "word": w} for w in phrase.split()]})

    def reset(self) -> None:
        self._energy_accumulator = 0.0
        self._partial_step = 0


class StreamingASR:
    """
    Streaming ASR engine that processes :class:`~audio_chunker.AudioFrame`
    objects and emits partial + final :class:`ASRResult` objects.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz (must match the audio stream).
    model_path:
        Path to a Vosk model directory.  When ``None`` the lightweight
        mock backend is used automatically.
    language:
        BCP-47 language code (informational; passed to the backend).

    Usage
    -----
    ::

        asr = StreamingASR(sample_rate=16_000)
        for frame in frames:
            partial = asr.feed(frame)
            if partial:
                print("Partial:", partial.text)

        final = asr.finalize()
        if final:
            print("Final:", final.text)
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        model_path: Optional[str] = None,
        language: str = "en-US",
    ) -> None:
        self.sample_rate = sample_rate
        self.language = language

        self._recognizer = self._build_recognizer(model_path, sample_rate)
        self._last_partial: str = ""
        logger.info("StreamingASR initialised (backend=%s)", type(self._recognizer).__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, frame: AudioFrame, speaker_id: Optional[str] = None) -> Optional[ASRResult]:
        """
        Feed one audio frame to the recogniser.

        Returns a :class:`ASRResult` with *is_final=False* when the partial
        transcript has changed, or ``None`` when there is nothing new to
        report.
        """
        pcm_int16 = (frame.samples * 32768).astype(np.int16)
        self._recognizer.accept_waveform(pcm_int16)

        raw = json.loads(self._recognizer.partial_result())
        partial_text: str = raw.get("partial", "").strip()

        if partial_text and partial_text != self._last_partial:
            self._last_partial = partial_text
            return ASRResult(
                text=partial_text,
                is_final=False,
                confidence=0.7,
                speaker_id=speaker_id,
                timestamp=frame.timestamp,
            )
        return None

    def finalize(self, speaker_id: Optional[str] = None) -> Optional[ASRResult]:
        """
        Flush the recogniser and return the final transcript for the
        current utterance, or ``None`` if the utterance was silent.
        """
        raw = json.loads(self._recognizer.result())
        text: str = raw.get("text", "").strip()
        self._last_partial = ""

        if not text:
            return None

        words = raw.get("result", [])
        confidence = (
            float(np.mean([w.get("conf", 1.0) for w in words])) if words else 1.0
        )

        return ASRResult(
            text=text,
            is_final=True,
            confidence=confidence,
            speaker_id=speaker_id,
        )

    def reset(self) -> None:
        """Reset internal recogniser state."""
        self._recognizer.reset()
        self._last_partial = ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_recognizer(self, model_path: Optional[str], sample_rate: int):
        """Return a Vosk recogniser or the lightweight mock backend."""
        if model_path is not None:
            try:
                import vosk  # type: ignore[import]

                model = vosk.Model(model_path)
                recognizer = vosk.KaldiRecognizer(model, sample_rate)
                recognizer.SetWords(True)
                logger.info("Using Vosk ASR backend (model=%s)", model_path)
                return recognizer
            except ImportError:
                logger.warning("vosk package not installed; falling back to mock ASR")
            except Exception as exc:
                logger.warning("Failed to load Vosk model (%s); using mock ASR", exc)

        return _MockASRBackend()
