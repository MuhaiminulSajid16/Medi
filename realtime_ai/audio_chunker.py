"""
audio_chunker.py – Streaming audio frame chunker.

Splits a continuous audio byte-stream (PCM 16-bit, mono) into fixed-size
frames of 20–100 ms and exposes them as an async generator.  The default
frame size is 40 ms which gives a good balance between latency and
processing overhead.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator

import numpy as np

logger = logging.getLogger(__name__)

# Default audio parameters (can be overridden at construction time).
DEFAULT_SAMPLE_RATE: int = 16_000  # Hz
DEFAULT_FRAME_MS: int = 40         # milliseconds
MIN_FRAME_MS: int = 20
MAX_FRAME_MS: int = 100


class AudioFrame:
    """Single audio frame produced by :class:`AudioChunker`."""

    __slots__ = ("samples", "sample_rate", "timestamp", "frame_index")

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: int,
        timestamp: float,
        frame_index: int,
    ) -> None:
        self.samples = samples          # float32 in [-1, 1]
        self.sample_rate = sample_rate
        self.timestamp = timestamp      # wall-clock time of frame start
        self.frame_index = frame_index

    @property
    def duration_ms(self) -> float:
        return len(self.samples) / self.sample_rate * 1_000

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AudioFrame(index={self.frame_index}, "
            f"duration={self.duration_ms:.1f} ms, "
            f"t={self.timestamp:.3f})"
        )


class AudioChunker:
    """
    Splits a raw PCM byte-stream into fixed-size :class:`AudioFrame` objects.

    Parameters
    ----------
    sample_rate:
        Audio sampling rate in Hz (default 16 000).
    frame_ms:
        Target frame duration in milliseconds.  Must be in [20, 100].
    bytes_per_sample:
        Bytes per PCM sample (2 for 16-bit, default).

    Usage
    -----
    ::

        chunker = AudioChunker(sample_rate=16_000, frame_ms=40)

        # Feed raw PCM bytes incrementally:
        for raw_bytes in audio_source:
            for frame in chunker.feed(raw_bytes):
                process(frame)

        # Flush any remaining samples:
        for frame in chunker.flush():
            process(frame)
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        bytes_per_sample: int = 2,
    ) -> None:
        if not (MIN_FRAME_MS <= frame_ms <= MAX_FRAME_MS):
            raise ValueError(
                f"frame_ms must be between {MIN_FRAME_MS} and {MAX_FRAME_MS}, "
                f"got {frame_ms}"
            )
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.bytes_per_sample = bytes_per_sample

        self._frame_size: int = int(sample_rate * frame_ms / 1_000)
        self._buffer: list[np.ndarray] = []
        self._buffered_samples: int = 0
        self._frame_index: int = 0
        self._start_time: float = time.monotonic()

        logger.debug(
            "AudioChunker initialised: sr=%d Hz, frame=%d ms (%d samples)",
            sample_rate,
            frame_ms,
            self._frame_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, raw_bytes: bytes) -> list[AudioFrame]:
        """
        Ingest raw PCM bytes and return any complete frames ready for
        processing.  Partial frames are held in an internal buffer.
        """
        if not raw_bytes:
            return []

        samples = self._bytes_to_float32(raw_bytes)
        self._buffer.append(samples)
        self._buffered_samples += len(samples)

        frames: list[AudioFrame] = []
        while self._buffered_samples >= self._frame_size:
            frames.append(self._pop_frame())

        return frames

    def flush(self) -> list[AudioFrame]:
        """
        Return a final (possibly shorter) frame containing any remaining
        buffered samples.  Call this when the audio stream ends.
        """
        if self._buffered_samples == 0:
            return []
        return [self._pop_frame(partial=True)]

    def reset(self) -> None:
        """Clear internal buffer and reset counters."""
        self._buffer = []
        self._buffered_samples = 0
        self._frame_index = 0
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Async generator interface
    # ------------------------------------------------------------------

    async def stream(
        self,
        audio_queue: asyncio.Queue,
    ) -> AsyncGenerator[AudioFrame, None]:
        """
        Async generator that reads raw PCM byte-chunks from *audio_queue*
        and yields :class:`AudioFrame` objects.

        The sentinel value ``None`` signals end-of-stream.
        """
        while True:
            chunk: bytes | None = await audio_queue.get()
            if chunk is None:
                for frame in self.flush():
                    yield frame
                break
            for frame in self.feed(chunk):
                yield frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bytes_to_float32(self, raw_bytes: bytes) -> np.ndarray:
        """Convert 16-bit PCM bytes to float32 samples normalised to [-1, 1]."""
        int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        return int16.astype(np.float32) / 32768.0

    def _pop_frame(self, partial: bool = False) -> AudioFrame:
        """Build one :class:`AudioFrame` from the internal buffer."""
        # Flatten the buffer into a single array for efficient slicing.
        all_samples = np.concatenate(self._buffer)
        size = min(self._frame_size, len(all_samples)) if partial else self._frame_size

        frame_samples = all_samples[:size]
        remainder = all_samples[size:]

        # Replace buffer with leftover samples.
        self._buffer = [remainder] if len(remainder) else []
        self._buffered_samples = len(remainder)

        elapsed = self._start_time + self._frame_index * self.frame_ms / 1_000
        frame = AudioFrame(
            samples=frame_samples,
            sample_rate=self.sample_rate,
            timestamp=elapsed,
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return frame
