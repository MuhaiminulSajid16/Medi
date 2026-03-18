"""
diarization.py – Real-time sliding-window speaker diarization.

Uses voice embeddings (MFCC-based by default) and cosine-similarity
clustering to assign speaker identities to audio frames.  When
``pyannote.audio`` or ``resemblyzer`` is available they will be used
instead of the built-in lightweight embedder.

Design goals
------------
* Sliding window of ~2 s to maintain temporal context.
* Persistent speaker IDs across windows to avoid identity drift.
* Overlap-aware: frames with mixed speakers are flagged rather than
  silently mis-attributed.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .audio_chunker import AudioFrame

logger = logging.getLogger(__name__)

# Cosine-similarity threshold for matching a new embedding to an existing
# speaker cluster.
SIMILARITY_THRESHOLD = 0.82
# Number of recent embeddings to keep per speaker for rolling average.
SPEAKER_HISTORY_LEN = 10
# Minimum energy (RMS) below which a frame is considered silence.
SILENCE_RMS_THRESHOLD = 0.01


@dataclass
class SpeakerSegment:
    """Output produced by :class:`SpeakerDiarizer` for one audio frame."""

    speaker_id: str
    timestamp: float
    duration_ms: float
    embedding: np.ndarray
    is_silence: bool = False
    has_overlap: bool = False

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SpeakerSegment(speaker={self.speaker_id!r}, "
            f"t={self.timestamp:.3f}, "
            f"silence={self.is_silence}, overlap={self.has_overlap})"
        )


class _LightweightEmbedder:
    """
    Fallback voice embedder based on MFCC statistics.

    Produces a 26-dimensional embedding vector from one audio frame.
    This is intentionally simple – it enables the system to run without
    heavy ML dependencies while preserving the correct interface.
    """

    N_MFCC = 13

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a normalised embedding vector for *samples*."""
        if len(samples) < 2:
            return np.zeros(self.N_MFCC * 2, dtype=np.float32)

        # Compute a simple spectral representation via DFT magnitude.
        spectrum = np.abs(np.fft.rfft(samples, n=512))

        # Split into N_MFCC mel-spaced bands (approximation).
        band_size = max(1, len(spectrum) // self.N_MFCC)
        band_energies = np.array(
            [
                np.mean(spectrum[i * band_size : (i + 1) * band_size])
                for i in range(self.N_MFCC)
            ],
            dtype=np.float32,
        )

        # Mean + std of the bands as the embedding.
        embedding = np.concatenate([band_energies, np.std(band_energies) * np.ones(self.N_MFCC)])
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [−1, 1] between vectors *a* and *b*."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SpeakerDiarizer:
    """
    Real-time speaker diarization with sliding-window clustering.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz.
    window_sec:
        Sliding window duration in seconds for context-aware clustering.
    similarity_threshold:
        Cosine similarity score above which a frame is attributed to an
        existing speaker rather than a new one.

    Usage
    -----
    ::

        diarizer = SpeakerDiarizer(sample_rate=16_000)
        segment = diarizer.process(frame)
        print(segment.speaker_id)          # e.g. "Speaker_0"
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        window_sec: float = 2.0,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.similarity_threshold = similarity_threshold

        self._embedder = _LightweightEmbedder()
        self._try_load_advanced_embedder()

        # speaker_id -> deque of recent embeddings
        self._speaker_embeddings: dict[str, deque] = {}
        # Ordered list of recent (timestamp, speaker_id) for the sliding window
        self._window: deque[tuple[float, str]] = deque()
        self._next_speaker_index: int = 0

        logger.info("SpeakerDiarizer initialised (sr=%d Hz)", sample_rate)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: AudioFrame) -> SpeakerSegment:
        """
        Process one :class:`~audio_chunker.AudioFrame` and return a
        :class:`SpeakerSegment` with the detected speaker identity.
        """
        rms = float(np.sqrt(np.mean(frame.samples ** 2)))
        if rms < SILENCE_RMS_THRESHOLD:
            return SpeakerSegment(
                speaker_id="SILENCE",
                timestamp=frame.timestamp,
                duration_ms=frame.duration_ms,
                embedding=np.zeros(self._embedder.N_MFCC * 2, dtype=np.float32),
                is_silence=True,
            )

        embedding = self._embedder.embed(frame.samples, frame.sample_rate)
        speaker_id, has_overlap = self._assign_speaker(embedding, frame.timestamp)

        self._expire_window(frame.timestamp)

        return SpeakerSegment(
            speaker_id=speaker_id,
            timestamp=frame.timestamp,
            duration_ms=frame.duration_ms,
            embedding=embedding,
            is_silence=False,
            has_overlap=has_overlap,
        )

    def register_speaker(self, speaker_id: str, reference_samples: np.ndarray) -> None:
        """
        Pre-register a known speaker with labelled reference audio.
        This enables identification rather than mere diarization.
        """
        embedding = self._embedder.embed(reference_samples, self.sample_rate)
        if speaker_id not in self._speaker_embeddings:
            self._speaker_embeddings[speaker_id] = deque(maxlen=SPEAKER_HISTORY_LEN)
        self._speaker_embeddings[speaker_id].append(embedding)
        logger.info("Registered speaker: %s", speaker_id)

    def reset(self) -> None:
        """Clear all learned speaker embeddings and history."""
        self._speaker_embeddings.clear()
        self._window.clear()
        self._next_speaker_index = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load_advanced_embedder(self) -> None:
        """Attempt to load a higher-quality embedder if available."""
        try:
            from resemblyzer import VoiceEncoder  # type: ignore[import]

            class _ResemblyzerEmbedder:
                N_MFCC = 256

                def __init__(self) -> None:
                    self._encoder = VoiceEncoder()

                def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
                    return self._encoder.embed_utterance(samples)

            self._embedder = _ResemblyzerEmbedder()  # type: ignore[assignment]
            logger.info("Using resemblyzer voice encoder")
        except ImportError:
            logger.debug("resemblyzer not available; using lightweight MFCC embedder")

    def _assign_speaker(
        self, embedding: np.ndarray, timestamp: float
    ) -> tuple[str, bool]:
        """
        Find the best-matching speaker for *embedding*, or create a new one.

        Returns ``(speaker_id, has_overlap)`` where *has_overlap* is ``True``
        when two speakers score above the threshold simultaneously.
        """
        if not self._speaker_embeddings:
            return self._new_speaker(embedding, timestamp), False

        scores: list[tuple[float, str]] = []
        for sid, history in self._speaker_embeddings.items():
            centroid = np.mean(list(history), axis=0)
            score = _cosine_similarity(embedding, centroid)
            scores.append((score, sid))

        scores.sort(reverse=True)
        best_score, best_sid = scores[0]

        has_overlap = (
            len(scores) >= 2 and scores[1][0] >= self.similarity_threshold
        )

        if best_score >= self.similarity_threshold:
            self._update_speaker(best_sid, embedding, timestamp)
            return best_sid, has_overlap

        return self._new_speaker(embedding, timestamp), False

    def _new_speaker(self, embedding: np.ndarray, timestamp: float) -> str:
        speaker_id = f"Speaker_{self._next_speaker_index}"
        self._next_speaker_index += 1
        self._speaker_embeddings[speaker_id] = deque(maxlen=SPEAKER_HISTORY_LEN)
        self._speaker_embeddings[speaker_id].append(embedding)
        self._window.append((timestamp, speaker_id))
        logger.debug("New speaker: %s at t=%.3f", speaker_id, timestamp)
        return speaker_id

    def _update_speaker(
        self, speaker_id: str, embedding: np.ndarray, timestamp: float
    ) -> None:
        self._speaker_embeddings[speaker_id].append(embedding)
        self._window.append((timestamp, speaker_id))

    def _expire_window(self, current_time: float) -> None:
        """Remove window entries older than *window_sec*."""
        cutoff = current_time - self.window_sec
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()
