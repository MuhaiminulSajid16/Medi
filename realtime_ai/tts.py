"""
tts.py – Low-latency text-to-speech engine.

Supports two backends:
* **pyttsx3** (offline, cross-platform) – used when available.
* **gTTS** (Google TTS, requires internet) – secondary option.
* **MockTTS** – a no-op fallback that logs the text and returns an empty
  audio array.  Useful for testing and headless environments.

The public :class:`TTSEngine` interface always returns a
:class:`TTSResult` containing either a waveform (float32 numpy array) or
a path to a synthesised audio file, enabling callers to stream or save the
audio as needed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Synthesised speech output."""

    text: str
    audio_data: Optional[np.ndarray]  # float32 PCM samples, or None
    sample_rate: int
    audio_path: Optional[str]         # path to saved WAV file, if any
    latency_ms: float

    def has_audio(self) -> bool:
        return self.audio_data is not None and len(self.audio_data) > 0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TTSResult(text={self.text!r}, "
            f"has_audio={self.has_audio()}, "
            f"latency={self.latency_ms:.0f} ms)"
        )


# ------------------------------------------------------------------
# Backend implementations
# ------------------------------------------------------------------

class _MockTTSBackend:
    """
    No-op backend for testing.  Returns a short sine-wave burst instead
    of real speech to allow audio-pipeline tests to proceed end-to-end.
    """

    SAMPLE_RATE = 16_000

    def synthesise(self, text: str) -> TTSResult:
        t0 = time.monotonic()
        duration_sec = max(0.5, len(text.split()) * 0.25)
        t_arr = np.linspace(0, duration_sec, int(self.SAMPLE_RATE * duration_sec), endpoint=False)
        audio = (0.1 * np.sin(2 * np.pi * 440 * t_arr)).astype(np.float32)
        latency_ms = (time.monotonic() - t0) * 1_000
        logger.debug("MockTTS synthesised %d chars (%.0f ms)", len(text), latency_ms)
        return TTSResult(
            text=text,
            audio_data=audio,
            sample_rate=self.SAMPLE_RATE,
            audio_path=None,
            latency_ms=latency_ms,
        )


class _Pyttsx3Backend:
    """Offline TTS using pyttsx3."""

    SAMPLE_RATE = 22_050

    def __init__(self) -> None:
        import pyttsx3  # type: ignore[import]

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 160)
        logger.info("Using pyttsx3 TTS backend")

    def synthesise(self, text: str) -> TTSResult:
        import tempfile
        import os

        t0 = time.monotonic()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name

        self._engine.save_to_file(text, path)
        self._engine.runAndWait()

        audio = self._load_wav(path)
        latency_ms = (time.monotonic() - t0) * 1_000
        return TTSResult(
            text=text,
            audio_data=audio,
            sample_rate=self.SAMPLE_RATE,
            audio_path=path,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _load_wav(path: str) -> Optional[np.ndarray]:
        try:
            import wave

            with wave.open(path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return audio
        except Exception as exc:
            logger.warning("Could not load TTS WAV: %s", exc)
            return None


class _GTTSBackend:
    """Google Text-to-Speech backend (requires internet access)."""

    SAMPLE_RATE = 24_000

    def __init__(self) -> None:
        from gtts import gTTS as _gTTS  # type: ignore[import]

        self._gTTS = _gTTS
        logger.info("Using gTTS backend")

    def synthesise(self, text: str) -> TTSResult:
        import tempfile

        t0 = time.monotonic()
        tts = self._gTTS(text=text, lang="en", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
        tts.save(path)
        latency_ms = (time.monotonic() - t0) * 1_000
        return TTSResult(
            text=text,
            audio_data=None,
            sample_rate=self.SAMPLE_RATE,
            audio_path=path,
            latency_ms=latency_ms,
        )


# ------------------------------------------------------------------
# Public facade
# ------------------------------------------------------------------

class TTSEngine:
    """
    Low-latency TTS engine with automatic backend selection.

    Preferred backend order: pyttsx3 → gTTS → MockTTS.

    Parameters
    ----------
    backend:
        Force a specific backend: ``"pyttsx3"``, ``"gtts"``, or
        ``"mock"``.  When ``None`` (default) the best available backend
        is selected automatically.

    Usage
    -----
    ::

        tts = TTSEngine()
        result = tts.speak("Earlier Mr Q suggested approach A.")
        if result.has_audio():
            play(result.audio_data, result.sample_rate)
    """

    def __init__(self, backend: Optional[str] = None) -> None:
        self._backend = self._build_backend(backend)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> TTSResult:
        """Synthesise *text* and return a :class:`TTSResult`."""
        if not text.strip():
            return TTSResult(
                text="",
                audio_data=np.array([], dtype=np.float32),
                sample_rate=16_000,
                audio_path=None,
                latency_ms=0.0,
            )
        return self._backend.synthesise(text)

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_backend(self, preference: Optional[str]):
        if preference == "mock":
            return _MockTTSBackend()
        if preference == "gtts":
            return self._try_gtts() or _MockTTSBackend()
        if preference == "pyttsx3":
            return self._try_pyttsx3() or _MockTTSBackend()

        # Auto-select
        backend = self._try_pyttsx3() or self._try_gtts()
        if backend is None:
            logger.info("No TTS backend available; using MockTTS")
            backend = _MockTTSBackend()
        return backend

    @staticmethod
    def _try_pyttsx3():
        try:
            return _Pyttsx3Backend()
        except Exception as exc:
            logger.debug("pyttsx3 unavailable: %s", exc)
            return None

    @staticmethod
    def _try_gtts():
        try:
            return _GTTSBackend()
        except Exception as exc:
            logger.debug("gTTS unavailable: %s", exc)
            return None
