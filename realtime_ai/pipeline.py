"""
pipeline.py – Async orchestration pipeline.

Wires together all components into a single real-time discussion
pipeline:

    AudioChunker → SpeakerDiarizer → StreamingASR
        → ConversationMemory → ContextBuilder
        → TriggerDetector → ReasoningEngine → TTSEngine

The pipeline runs as a set of coroutines connected by ``asyncio.Queue``
objects so each stage can operate at its own cadence without blocking
the others.

Public entry points
-------------------
* :class:`RealtimePipeline` – high-level façade.
* :meth:`RealtimePipeline.start` – begin processing from an async audio
  generator.
* :meth:`RealtimePipeline.feed_text` – inject a pre-transcribed turn
  directly (useful for testing or when audio is unavailable).
* :meth:`RealtimePipeline.get_response` – await the next AI response.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import numpy as np

from .audio_chunker import AudioChunker, AudioFrame
from .asr import ASRResult, StreamingASR
from .context_builder import ContextBuilder
from .diarization import SpeakerDiarizer, SpeakerSegment
from .memory import ConversationMemory
from .reasoning import ReasoningEngine, ReasoningResponse
from .tts import TTSEngine, TTSResult
from .triggers import TriggerDetector, TriggerEvent

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Tunable parameters for the full pipeline."""

    # Audio
    sample_rate: int = 16_000
    frame_ms: int = 40

    # Diarization
    diarization_window_sec: float = 2.0
    similarity_threshold: float = 0.82

    # ASR
    asr_model_path: Optional[str] = None

    # Memory
    hot_cache_size: int = 100
    redis_url: Optional[str] = None

    # Context
    recent_turns: int = 10
    semantic_top_k: int = 3
    context_update_interval_sec: float = 1.5

    # Triggers
    silence_ms: float = 500.0
    wake_phrase: str = "moshi"
    trigger_cooldown_sec: float = 2.0

    # Reasoning
    llm_model_name: Optional[str] = None
    passive_mode: bool = False

    # TTS
    tts_backend: Optional[str] = None  # None = auto-select


@dataclass
class PipelineEvent:
    """Emitted by the pipeline for each significant event."""

    event_type: str  # "asr_partial", "asr_final", "trigger", "response", "tts"
    speaker_id: str
    text: str
    timestamp: float = field(default_factory=time.time)
    response: Optional[ReasoningResponse] = None
    tts_result: Optional[TTSResult] = None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PipelineEvent(type={self.event_type!r}, "
            f"speaker={self.speaker_id!r}, "
            f"text={self.text!r})"
        )


class RealtimePipeline:
    """
    End-to-end real-time multi-speaker discussion pipeline.

    Parameters
    ----------
    config:
        :class:`PipelineConfig` instance.  Defaults are suitable for a
        development / testing environment.

    Usage
    -----
    ::

        pipeline = RealtimePipeline()
        pipeline.start_background()

        # Push raw PCM bytes into the pipeline:
        pipeline.push_audio(pcm_bytes)

        # Or inject text turns directly:
        pipeline.feed_text("Speaker_0", "We should use approach A.")

        # Retrieve events (non-blocking):
        event = pipeline.poll_event()
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._build_components()
        self._event_queue: asyncio.Queue[PipelineEvent] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._running: bool = False
        logger.info("RealtimePipeline initialised")

    # ------------------------------------------------------------------
    # Component construction
    # ------------------------------------------------------------------

    def _build_components(self) -> None:
        cfg = self.config
        self.chunker = AudioChunker(
            sample_rate=cfg.sample_rate,
            frame_ms=cfg.frame_ms,
        )
        self.diarizer = SpeakerDiarizer(
            sample_rate=cfg.sample_rate,
            window_sec=cfg.diarization_window_sec,
            similarity_threshold=cfg.similarity_threshold,
        )
        self.asr = StreamingASR(
            sample_rate=cfg.sample_rate,
            model_path=cfg.asr_model_path,
        )
        self.memory = ConversationMemory(
            hot_cache_size=cfg.hot_cache_size,
            redis_url=cfg.redis_url,
        )
        self.context_builder = ContextBuilder(
            memory=self.memory,
            recent_turns=cfg.recent_turns,
            semantic_top_k=cfg.semantic_top_k,
            update_interval_sec=cfg.context_update_interval_sec,
        )
        self.trigger_detector = TriggerDetector(
            silence_ms=cfg.silence_ms,
            wake_phrase=cfg.wake_phrase,
            cooldown_sec=cfg.trigger_cooldown_sec,
        )
        self.reasoning_engine = ReasoningEngine(
            model_name=cfg.llm_model_name,
            passive_mode=cfg.passive_mode,
        )
        self.tts_engine = TTSEngine(backend=cfg.tts_backend)

    # ------------------------------------------------------------------
    # Public API – synchronous helpers
    # ------------------------------------------------------------------

    def push_audio(self, raw_bytes: bytes) -> None:
        """Enqueue raw PCM bytes for asynchronous processing."""
        self._audio_queue.put_nowait(raw_bytes)

    def stop_audio(self) -> None:
        """Signal end-of-stream to the audio processing loop."""
        self._audio_queue.put_nowait(None)

    def feed_text(self, speaker_id: str, text: str) -> Optional[PipelineEvent]:
        """
        Inject a pre-transcribed utterance directly.

        This bypasses audio → diarization → ASR and stores the utterance
        in memory, then checks triggers and optionally generates a
        response.  Returns the :class:`PipelineEvent` if a response was
        produced, otherwise ``None``.
        """
        self.memory.store(speaker_id, text)
        self.context_builder.invalidate()

        asr_result = ASRResult(
            text=text,
            is_final=True,
            confidence=1.0,
            speaker_id=speaker_id,
        )
        trigger = self.trigger_detector.feed_asr(asr_result)
        if trigger is None:
            return None

        return self.handle_trigger(trigger)

    def poll_event(self) -> Optional[PipelineEvent]:
        """Return the next pipeline event without blocking, or ``None``."""
        try:
            return self._event_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_conversation_history(self, n: int = 20) -> list[dict]:
        """Return the last *n* utterances as plain dicts."""
        return [u.to_dict() for u in self.memory.get_recent(n)]

    def reset(self) -> None:
        """Reset all components (clears memory and speaker models)."""
        self.chunker.reset()
        self.diarizer.reset()
        self.asr.reset()
        self.memory.clear()
        self.context_builder.invalidate()
        self.trigger_detector.reset()
        logger.info("Pipeline reset")

    # ------------------------------------------------------------------
    # Public API – async entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        audio_source: AsyncGenerator[Optional[bytes], None],
    ) -> None:
        """
        Main async loop.

        Reads PCM chunks from *audio_source* (sentinel ``None`` = EOS),
        processes them through the full pipeline, and deposits
        :class:`PipelineEvent` objects into the internal event queue.
        """
        self._running = True
        current_speaker: str = "UNKNOWN"

        async for chunk in audio_source:
            if not self._running:
                break
            if chunk is None:
                frames = self.chunker.flush()
            else:
                frames = self.chunker.feed(chunk)

            for frame in frames:
                await self._process_frame(frame, current_speaker)

        self._running = False

    async def stop(self) -> None:
        """Gracefully stop the pipeline."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    async def _process_frame(
        self, frame: AudioFrame, current_speaker: str
    ) -> str:
        """Process one audio frame through diarization → ASR → trigger."""
        # 1. Diarize
        segment: SpeakerSegment = self.diarizer.process(frame)
        current_speaker = segment.speaker_id

        if segment.is_silence:
            trigger = self.trigger_detector.feed_frame(frame, current_speaker)
            if trigger:
                event = self.handle_trigger(trigger)
                if event:
                    await self._event_queue.put(event)
            return current_speaker

        # 2. ASR
        partial: Optional[ASRResult] = self.asr.feed(frame, speaker_id=current_speaker)
        if partial:
            await self._event_queue.put(
                PipelineEvent(
                    event_type="asr_partial",
                    speaker_id=current_speaker,
                    text=partial.text,
                )
            )
            trigger = self.trigger_detector.feed_asr(partial)
            if trigger:
                event = self.handle_trigger(trigger)
                if event:
                    await self._event_queue.put(event)

        return current_speaker

    async def _finalise_utterance(self, speaker_id: str) -> None:
        """Called at utterance boundaries (silence / end-of-stream)."""
        final: Optional[ASRResult] = self.asr.finalize(speaker_id=speaker_id)
        if final:
            self.memory.store(speaker_id, final.text)
            self.context_builder.invalidate()
            await self._event_queue.put(
                PipelineEvent(
                    event_type="asr_final",
                    speaker_id=speaker_id,
                    text=final.text,
                )
            )
            trigger = self.trigger_detector.feed_asr(final)
            if trigger:
                event = self.handle_trigger(trigger)
                if event:
                    await self._event_queue.put(event)

    def handle_trigger(self, trigger: TriggerEvent) -> Optional[PipelineEvent]:
        """Run reasoning and TTS for a trigger; return a PipelineEvent."""
        context = self.context_builder.build(current_query=trigger.text)
        response: Optional[ReasoningResponse] = self.reasoning_engine.respond(
            trigger, context
        )
        if response is None:
            return None

        tts_result: TTSResult = self.tts_engine.speak(response.text)

        return PipelineEvent(
            event_type="response",
            speaker_id="AI",
            text=response.text,
            response=response,
            tts_result=tts_result,
        )
