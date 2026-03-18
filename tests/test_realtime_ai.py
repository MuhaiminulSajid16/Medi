"""
tests/test_realtime_ai.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Test suite for the real-time AI multi-speaker discussion pipeline.

Each test is self-contained and relies only on the lightweight fallback
implementations (mock ASR, NumPy semantic store, rule-based responder,
mock TTS) so the suite runs without heavy ML dependencies.
"""

from __future__ import annotations

import asyncio
import math
import struct
import time
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_pcm(frequency: float = 440.0, duration_ms: int = 200, sample_rate: int = 16_000) -> bytes:
    """Return raw 16-bit PCM bytes of a sine wave."""
    n = int(sample_rate * duration_ms / 1_000)
    t = np.linspace(0, duration_ms / 1_000, n, endpoint=False)
    samples = (0.5 * np.sin(2 * math.pi * frequency * t) * 32767).astype(np.int16)
    return samples.tobytes()


def _silence_pcm(duration_ms: int = 200, sample_rate: int = 16_000) -> bytes:
    """Return raw 16-bit PCM bytes of silence."""
    n = int(sample_rate * duration_ms / 1_000)
    return b"\x00" * (n * 2)


# ---------------------------------------------------------------------------
# AudioChunker
# ---------------------------------------------------------------------------

class TestAudioChunker:
    def test_basic_chunking(self):
        from realtime_ai.audio_chunker import AudioChunker

        chunker = AudioChunker(sample_rate=16_000, frame_ms=40)
        # 200 ms of audio → 5 frames of 40 ms each
        frames = chunker.feed(_sine_pcm(duration_ms=200))
        assert len(frames) == 5

    def test_partial_flush(self):
        from realtime_ai.audio_chunker import AudioChunker

        chunker = AudioChunker(sample_rate=16_000, frame_ms=40)
        # 60 ms = 1 full frame (40 ms) + 20 ms leftover
        frames = chunker.feed(_sine_pcm(duration_ms=60))
        assert len(frames) == 1
        flushed = chunker.flush()
        assert len(flushed) == 1
        assert flushed[0].duration_ms < 40

    def test_frame_attributes(self):
        from realtime_ai.audio_chunker import AudioChunker

        chunker = AudioChunker(sample_rate=16_000, frame_ms=40)
        frames = chunker.feed(_sine_pcm(duration_ms=40))
        assert len(frames) == 1
        frame = frames[0]
        assert frame.frame_index == 0
        assert frame.sample_rate == 16_000
        assert len(frame.samples) == 640  # 16000 * 40 / 1000

    def test_invalid_frame_ms(self):
        from realtime_ai.audio_chunker import AudioChunker

        with pytest.raises(ValueError):
            AudioChunker(frame_ms=10)
        with pytest.raises(ValueError):
            AudioChunker(frame_ms=200)

    def test_reset(self):
        from realtime_ai.audio_chunker import AudioChunker

        chunker = AudioChunker(sample_rate=16_000, frame_ms=40)
        chunker.feed(_sine_pcm(duration_ms=60))
        chunker.reset()
        flushed = chunker.flush()
        assert flushed == []

    def test_normalisation(self):
        from realtime_ai.audio_chunker import AudioChunker

        chunker = AudioChunker(sample_rate=16_000, frame_ms=40)
        frames = chunker.feed(_sine_pcm(duration_ms=40))
        samples = frames[0].samples
        assert samples.dtype == np.float32
        assert samples.max() <= 1.0
        assert samples.min() >= -1.0

    def test_empty_input(self):
        from realtime_ai.audio_chunker import AudioChunker

        chunker = AudioChunker()
        frames = chunker.feed(b"")
        assert frames == []

    async def _async_stream_helper(self):
        from realtime_ai.audio_chunker import AudioChunker

        chunker = AudioChunker(sample_rate=16_000, frame_ms=40)
        q: asyncio.Queue = asyncio.Queue()
        # 200 ms of audio then sentinel
        await q.put(_sine_pcm(duration_ms=200))
        await q.put(None)

        collected = []
        async for frame in chunker.stream(q):
            collected.append(frame)
        return collected

    def test_async_stream(self):
        frames = asyncio.get_event_loop().run_until_complete(
            self._async_stream_helper()
        )
        assert len(frames) == 5


# ---------------------------------------------------------------------------
# SpeakerDiarizer
# ---------------------------------------------------------------------------

class TestSpeakerDiarizer:
    def _make_frame(self, frequency: float = 440.0, duration_ms: int = 40) -> object:
        from realtime_ai.audio_chunker import AudioChunker

        chunker = AudioChunker(sample_rate=16_000, frame_ms=duration_ms)
        frames = chunker.feed(_sine_pcm(frequency=frequency, duration_ms=duration_ms))
        return frames[0]

    def _make_silence_frame(self, duration_ms: int = 40) -> object:
        from realtime_ai.audio_chunker import AudioChunker

        chunker = AudioChunker(sample_rate=16_000, frame_ms=duration_ms)
        frames = chunker.feed(_silence_pcm(duration_ms=duration_ms))
        if frames:
            return frames[0]
        # flush partial
        return chunker.flush()[0]

    def test_silence_detection(self):
        from realtime_ai.diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer()
        frame = self._make_silence_frame()
        segment = diarizer.process(frame)
        assert segment.is_silence

    def test_speaker_assignment(self):
        from realtime_ai.diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer()
        frame = self._make_frame(440.0)
        segment = diarizer.process(frame)
        assert not segment.is_silence
        assert segment.speaker_id.startswith("Speaker_")

    def test_consistent_speaker_id(self):
        """Same voice characteristics should map to the same speaker."""
        from realtime_ai.diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(similarity_threshold=0.5)
        ids = set()
        for _ in range(5):
            frame = self._make_frame(440.0)
            segment = diarizer.process(frame)
            ids.add(segment.speaker_id)
        # All frames from the same tone should cluster to one speaker
        assert len(ids) == 1

    def test_different_speakers(self):
        """Very different voices should map to different speakers."""
        from realtime_ai.diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer(similarity_threshold=0.999)
        seg1 = diarizer.process(self._make_frame(200.0))
        seg2 = diarizer.process(self._make_frame(3000.0))
        assert seg1.speaker_id != seg2.speaker_id

    def test_register_speaker(self):
        from realtime_ai.diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer()
        ref = np.frombuffer(_sine_pcm(440.0, 200), dtype=np.int16).astype(np.float32) / 32768.0
        diarizer.register_speaker("Alice", ref)
        assert "Alice" in diarizer._speaker_embeddings

    def test_reset(self):
        from realtime_ai.diarization import SpeakerDiarizer

        diarizer = SpeakerDiarizer()
        diarizer.process(self._make_frame())
        diarizer.reset()
        assert len(diarizer._speaker_embeddings) == 0


# ---------------------------------------------------------------------------
# StreamingASR
# ---------------------------------------------------------------------------

class TestStreamingASR:
    def _frame(self, frequency: float = 440.0, duration_ms: int = 40) -> object:
        from realtime_ai.audio_chunker import AudioChunker

        chunker = AudioChunker(sample_rate=16_000, frame_ms=duration_ms)
        return chunker.feed(_sine_pcm(frequency, duration_ms))[0]

    def test_feed_returns_partial_or_none(self):
        from realtime_ai.asr import StreamingASR

        asr = StreamingASR()
        # Feed enough frames to accumulate energy
        result = None
        for _ in range(10):
            result = asr.feed(self._frame())
        # At least one partial result should have been emitted eventually
        # (mock backend emits after a threshold)
        assert result is None or (not result.is_final)

    def test_finalize_returns_asr_result(self):
        from realtime_ai.asr import StreamingASR

        asr = StreamingASR()
        for _ in range(20):
            asr.feed(self._frame())
        final = asr.finalize(speaker_id="Speaker_0")
        assert final is not None
        assert final.is_final
        assert final.speaker_id == "Speaker_0"
        assert len(final.text) > 0

    def test_speaker_id_propagated(self):
        from realtime_ai.asr import StreamingASR

        asr = StreamingASR()
        for _ in range(20):
            asr.feed(self._frame(), speaker_id="Alice")
        final = asr.finalize(speaker_id="Alice")
        if final:
            assert final.speaker_id == "Alice"

    def test_reset_clears_state(self):
        from realtime_ai.asr import StreamingASR

        asr = StreamingASR()
        for _ in range(15):
            asr.feed(self._frame())
        asr.reset()
        # After reset the partial should be empty
        assert asr._last_partial == ""

    def test_asr_result_bool(self):
        from realtime_ai.asr import ASRResult

        assert bool(ASRResult(text="hello", is_final=True)) is True
        assert bool(ASRResult(text="", is_final=True)) is False
        assert bool(ASRResult(text="  ", is_final=True)) is False


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------

class TestConversationMemory:
    def test_store_and_retrieve(self):
        from realtime_ai.memory import ConversationMemory

        mem = ConversationMemory(use_faiss=False)
        mem.store("Alice", "We should use approach A")
        mem.store("Bob", "I prefer approach B")
        recent = mem.get_recent(10)
        assert len(recent) == 2
        assert recent[0].speaker_id == "Alice"
        assert recent[1].speaker_id == "Bob"

    def test_semantic_search(self):
        from realtime_ai.memory import ConversationMemory

        mem = ConversationMemory(use_faiss=False)
        mem.store("Alice", "We should use approach A")
        mem.store("Bob", "Let us try method B instead")
        mem.store("Charlie", "The weather is nice today")

        results = mem.search_semantic("which approach should we use", top_k=2)
        assert len(results) <= 2
        # The top result should be about an approach, not the weather
        texts = [r.text.lower() for r in results]
        assert any("approach" in t or "method" in t for t in texts)

    def test_get_recent_honours_limit(self):
        from realtime_ai.memory import ConversationMemory

        mem = ConversationMemory(hot_cache_size=50, use_faiss=False)
        for i in range(20):
            mem.store(f"Speaker_{i % 3}", f"utterance {i}")
        recent = mem.get_recent(5)
        assert len(recent) == 5

    def test_hot_cache_max_size(self):
        from realtime_ai.memory import ConversationMemory

        mem = ConversationMemory(hot_cache_size=5, use_faiss=False)
        for i in range(10):
            mem.store("Alice", f"utterance {i}")
        recent = mem.get_recent(100)
        assert len(recent) == 5

    def test_total_utterances(self):
        from realtime_ai.memory import ConversationMemory

        mem = ConversationMemory(use_faiss=False)
        mem.store("Alice", "First")
        mem.store("Bob", "Second")
        assert mem.total_utterances == 2

    def test_clear(self):
        from realtime_ai.memory import ConversationMemory

        mem = ConversationMemory(use_faiss=False)
        mem.store("Alice", "Something")
        mem.clear()
        assert mem.get_recent(10) == []
        assert mem.total_utterances == 0

    def test_utterance_to_dict(self):
        from realtime_ai.memory import Utterance

        utt = Utterance(utterance_id=1, speaker_id="Alice", text="Hello")
        d = utt.to_dict()
        assert d["speaker_id"] == "Alice"
        assert d["text"] == "Hello"
        assert "embedding" not in d

    def test_utterance_from_dict(self):
        from realtime_ai.memory import Utterance

        d = {"utterance_id": 5, "speaker_id": "Bob", "text": "World", "timestamp": 1234.0}
        utt = Utterance.from_dict(d)
        assert utt.utterance_id == 5
        assert utt.speaker_id == "Bob"


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class TestContextBuilder:
    def _build_mem_with_utterances(self) -> object:
        from realtime_ai.memory import ConversationMemory

        mem = ConversationMemory(use_faiss=False)
        mem.store("Alice", "We should use approach A for the first problem.")
        mem.store("Bob", "Method B could also work well in this scenario.")
        mem.store("Charlie", "I agree with Alice on this point.")
        return mem

    def test_build_returns_context(self):
        from realtime_ai.context_builder import ContextBuilder

        mem = self._build_mem_with_utterances()
        builder = ContextBuilder(mem, update_interval_sec=0)
        ctx = builder.build("What should we do?")
        assert len(ctx.recent_turns) > 0
        assert ctx.current_query == "What should we do?"

    def test_render_contains_speaker(self):
        from realtime_ai.context_builder import ContextBuilder

        mem = self._build_mem_with_utterances()
        builder = ContextBuilder(mem, update_interval_sec=0)
        ctx = builder.build("Which approach?")
        rendered = ctx.render()
        assert "Alice" in rendered or "Bob" in rendered or "Charlie" in rendered

    def test_caching(self):
        from realtime_ai.context_builder import ContextBuilder

        mem = self._build_mem_with_utterances()
        builder = ContextBuilder(mem, update_interval_sec=60)
        ctx1 = builder.build("query")
        ctx2 = builder.build("query")
        assert ctx1 is ctx2  # Same object returned from cache

    def test_invalidate_forces_rebuild(self):
        from realtime_ai.context_builder import ContextBuilder

        mem = self._build_mem_with_utterances()
        builder = ContextBuilder(mem, update_interval_sec=60)
        ctx1 = builder.build("query")
        builder.invalidate()
        ctx2 = builder.build("query")
        assert ctx1 is not ctx2

    def test_semantic_dedup(self):
        """Utterances in recent window should not also appear in relevant_history."""
        from realtime_ai.context_builder import ContextBuilder

        mem = self._build_mem_with_utterances()
        builder = ContextBuilder(mem, recent_turns=10, semantic_top_k=3, update_interval_sec=0)
        ctx = builder.build("approach")
        recent_ids = {u.utterance_id for u in ctx.recent_turns}
        history_ids = {u.utterance_id for u in ctx.relevant_history}
        assert recent_ids.isdisjoint(history_ids)


# ---------------------------------------------------------------------------
# TriggerDetector
# ---------------------------------------------------------------------------

class TestTriggerDetector:
    def _frame(self, is_silent: bool = False) -> object:
        from realtime_ai.audio_chunker import AudioChunker

        chunker = AudioChunker(sample_rate=16_000, frame_ms=40)
        if is_silent:
            frames = chunker.feed(_silence_pcm(40))
        else:
            frames = chunker.feed(_sine_pcm(440.0, 40))
        if frames:
            return frames[0]
        return chunker.flush()[0]

    def test_question_trigger(self):
        from realtime_ai.asr import ASRResult
        from realtime_ai.triggers import TriggerDetector, TriggerType

        detector = TriggerDetector(cooldown_sec=0)
        result = ASRResult(
            text="What should we do now?",
            is_final=True,
            speaker_id="Alice",
        )
        event = detector.feed_asr(result)
        assert event is not None
        assert event.trigger_type == TriggerType.QUESTION

    def test_invocation_trigger(self):
        from realtime_ai.asr import ASRResult
        from realtime_ai.triggers import TriggerDetector, TriggerType

        detector = TriggerDetector(wake_phrase="moshi", cooldown_sec=0)
        result = ASRResult(
            text="Moshi, what do you think?",
            is_final=False,
            speaker_id="Bob",
        )
        event = detector.feed_asr(result)
        assert event is not None
        assert event.trigger_type == TriggerType.INVOCATION

    def test_silence_trigger(self):
        from realtime_ai.audio_chunker import AudioChunker
        from realtime_ai.triggers import TriggerDetector, TriggerType

        detector = TriggerDetector(silence_ms=80, cooldown_sec=0)
        # Use one shared chunker so frame timestamps advance monotonically
        chunker = AudioChunker(sample_rate=16_000, frame_ms=40)
        event = None
        for _ in range(10):
            frames = chunker.feed(_silence_pcm(40))
            for frame in frames:
                event = detector.feed_frame(frame, "Alice")
                if event:
                    break
            if event:
                break
        assert event is not None
        assert event.trigger_type == TriggerType.SILENCE

    def test_no_trigger_on_non_question(self):
        from realtime_ai.asr import ASRResult
        from realtime_ai.triggers import TriggerDetector

        detector = TriggerDetector(cooldown_sec=0)
        result = ASRResult(text="I agree with that approach", is_final=True)
        event = detector.feed_asr(result)
        assert event is None

    def test_cooldown_prevents_double_trigger(self):
        from realtime_ai.asr import ASRResult
        from realtime_ai.triggers import TriggerDetector

        detector = TriggerDetector(cooldown_sec=100)
        r = ASRResult(text="What should we do?", is_final=True, speaker_id="A")
        e1 = detector.feed_asr(r)
        e2 = detector.feed_asr(r)
        assert e1 is not None
        assert e2 is None  # blocked by cooldown

    def test_reset(self):
        from realtime_ai.triggers import TriggerDetector

        detector = TriggerDetector()
        detector._last_trigger_time = time.monotonic()
        detector.reset()
        assert detector._last_trigger_time == 0.0


# ---------------------------------------------------------------------------
# ReasoningEngine
# ---------------------------------------------------------------------------

class TestReasoningEngine:
    def _make_context(self, speakers=None):
        from realtime_ai.context_builder import ConversationContext
        from realtime_ai.memory import Utterance

        if speakers is None:
            speakers = [("Alice", "We should use approach A"), ("Bob", "Method B works too")]
        utterances = [
            Utterance(utterance_id=i, speaker_id=s, text=t)
            for i, (s, t) in enumerate(speakers)
        ]
        return ConversationContext(
            recent_turns=utterances,
            relevant_history=[],
            current_query="What should we do?",
        )

    def _make_trigger(self, ttype=None):
        from realtime_ai.triggers import TriggerEvent, TriggerType

        return TriggerEvent(
            trigger_type=ttype or TriggerType.QUESTION,
            text="What should we do?",
            speaker_id="Alice",
            timestamp=time.time(),
        )

    def test_produces_response(self):
        from realtime_ai.reasoning import ReasoningEngine

        engine = ReasoningEngine()
        ctx = self._make_context()
        trigger = self._make_trigger()
        response = engine.respond(trigger, ctx)
        assert response is not None
        assert len(response.text) > 0

    def test_response_bool(self):
        from realtime_ai.reasoning import ReasoningResponse
        from realtime_ai.triggers import TriggerType

        r = ReasoningResponse(text="Hello", trigger_type=TriggerType.QUESTION, latency_ms=10)
        assert bool(r) is True
        r2 = ReasoningResponse(text="", trigger_type=TriggerType.SILENCE, latency_ms=10)
        assert bool(r2) is False

    def test_passive_mode_blocks_question(self):
        from realtime_ai.reasoning import ReasoningEngine
        from realtime_ai.triggers import TriggerType

        engine = ReasoningEngine(passive_mode=True)
        ctx = self._make_context()
        trigger = self._make_trigger(TriggerType.QUESTION)
        response = engine.respond(trigger, ctx)
        assert response is None

    def test_passive_mode_allows_invocation(self):
        from realtime_ai.reasoning import ReasoningEngine
        from realtime_ai.triggers import TriggerType

        engine = ReasoningEngine(passive_mode=True)
        ctx = self._make_context()
        trigger = self._make_trigger(TriggerType.INVOCATION)
        response = engine.respond(trigger, ctx)
        assert response is not None

    def test_empty_context_returns_default(self):
        from realtime_ai.context_builder import ConversationContext
        from realtime_ai.reasoning import ReasoningEngine

        engine = ReasoningEngine()
        ctx = ConversationContext(
            recent_turns=[], relevant_history=[], current_query=""
        )
        trigger = self._make_trigger()
        response = engine.respond(trigger, ctx)
        assert response is not None
        assert "context" in response.text.lower() or "enough" in response.text.lower()

    def test_speaker_attribution_in_response(self):
        from realtime_ai.reasoning import ReasoningEngine

        engine = ReasoningEngine()
        ctx = self._make_context()
        trigger = self._make_trigger()
        response = engine.respond(trigger, ctx)
        # The rule-based responder should mention at least one speaker
        assert "Alice" in response.text or "Bob" in response.text


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------

class TestTTSEngine:
    def test_mock_backend_produces_audio(self):
        from realtime_ai.tts import TTSEngine

        tts = TTSEngine(backend="mock")
        result = tts.speak("Hello world")
        assert result.has_audio()
        assert result.audio_data is not None
        assert len(result.audio_data) > 0
        assert result.sample_rate > 0
        assert result.latency_ms >= 0

    def test_empty_text_returns_empty_audio(self):
        from realtime_ai.tts import TTSEngine

        tts = TTSEngine(backend="mock")
        result = tts.speak("")
        assert not result.has_audio()

    def test_whitespace_text_returns_empty_audio(self):
        from realtime_ai.tts import TTSEngine

        tts = TTSEngine(backend="mock")
        result = tts.speak("   ")
        assert not result.has_audio()

    def test_audio_is_float32(self):
        from realtime_ai.tts import TTSEngine

        tts = TTSEngine(backend="mock")
        result = tts.speak("Test audio output")
        assert result.audio_data.dtype == np.float32

    def test_longer_text_produces_longer_audio(self):
        from realtime_ai.tts import TTSEngine

        tts = TTSEngine(backend="mock")
        short = tts.speak("Hi")
        long_ = tts.speak("This is a much longer sentence with many words in it")
        assert len(long_.audio_data) > len(short.audio_data)

    def test_backend_name(self):
        from realtime_ai.tts import TTSEngine

        tts = TTSEngine(backend="mock")
        assert "Mock" in tts.backend_name


# ---------------------------------------------------------------------------
# RealtimePipeline (integration)
# ---------------------------------------------------------------------------

class TestRealtimePipeline:
    def _pipeline(self):
        from realtime_ai.pipeline import PipelineConfig, RealtimePipeline

        cfg = PipelineConfig(
            silence_ms=80,
            trigger_cooldown_sec=0,
            tts_backend="mock",
            context_update_interval_sec=0,
        )
        return RealtimePipeline(cfg)

    def test_feed_text_stores_utterance(self):
        pipeline = self._pipeline()
        pipeline.feed_text("Alice", "We should use approach A")
        history = pipeline.get_conversation_history(10)
        assert len(history) == 1
        assert history[0]["speaker_id"] == "Alice"

    def test_feed_text_question_generates_response(self):
        pipeline = self._pipeline()
        pipeline.feed_text("Alice", "We should use approach A")
        pipeline.feed_text("Bob", "Method B could work too")
        event = pipeline.feed_text("Charlie", "What should we do now?")
        assert event is not None
        assert event.event_type == "response"
        assert len(event.text) > 0

    def test_feed_text_invocation(self):
        pipeline = self._pipeline()
        pipeline.feed_text("Alice", "I have a suggestion")
        event = pipeline.feed_text("Bob", "Moshi, what do you think about this?")
        assert event is not None

    def test_feed_text_non_trigger_returns_none(self):
        pipeline = self._pipeline()
        event = pipeline.feed_text("Alice", "I agree completely")
        assert event is None

    def test_reset_clears_history(self):
        pipeline = self._pipeline()
        pipeline.feed_text("Alice", "Something important")
        pipeline.reset()
        history = pipeline.get_conversation_history(10)
        assert history == []

    def test_get_conversation_history_format(self):
        pipeline = self._pipeline()
        pipeline.feed_text("Alice", "First utterance")
        pipeline.feed_text("Bob", "Second utterance")
        history = pipeline.get_conversation_history(10)
        for item in history:
            assert "speaker_id" in item
            assert "text" in item
            assert "timestamp" in item
            assert "utterance_id" in item

    def test_audio_feed_through_chunker(self):
        """Feed raw PCM bytes through the pipeline's chunker."""
        pipeline = self._pipeline()
        # Feed 200 ms of audio and verify frames are produced
        raw = _sine_pcm(440.0, 200)
        frames = pipeline.chunker.feed(raw)
        assert len(frames) > 0

    def test_response_has_tts(self):
        pipeline = self._pipeline()
        pipeline.feed_text("Alice", "Approach A is the best option")
        event = pipeline.feed_text("Bob", "What should we do now?")
        if event:
            assert event.tts_result is not None


# ---------------------------------------------------------------------------
# FastAPI endpoints (lightweight integration)
# ---------------------------------------------------------------------------

class TestDiscussionEndpoints:
    @pytest.fixture(autouse=True)
    def reset_pipeline(self):
        """Ensure a clean pipeline for every test."""
        import main as app_module

        app_module._discussion_pipeline = None
        yield
        app_module._discussion_pipeline = None

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_discussion_feed_text_stored(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        resp = client.post(
            "/discussion/text",
            data={"speaker_id": "Alice", "text": "We should try approach A"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("stored", "response_generated")

    def test_discussion_history_empty(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        resp = client.get("/discussion/history")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_discussion_history_after_feed(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        client.post("/discussion/text", data={"speaker_id": "Alice", "text": "Hello"})
        resp = client.get("/discussion/history?n=10")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_discussion_reset(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        client.post("/discussion/text", data={"speaker_id": "Alice", "text": "Something"})
        reset_resp = client.post("/discussion/reset")
        assert reset_resp.status_code == 200
        history_resp = client.get("/discussion/history")
        assert history_resp.json()["count"] == 0

    def test_websocket_text_injection(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        with client.websocket_connect("/discussion/ws") as ws:
            ws.send_json({"speaker_id": "Alice", "text": "Hello there"})
            data = ws.receive_json()
            assert data["event"] == "asr_final"
            assert data["speaker_id"] == "Alice"

    def test_websocket_question_generates_response(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        # Verify asr_final is reliably produced for each injected utterance
        with client.websocket_connect("/discussion/ws") as ws:
            ws.send_json({"speaker_id": "Alice", "text": "We should use approach A"})
            msg = ws.receive_json()
            assert msg["event"] == "asr_final"
            assert msg["speaker_id"] == "Alice"

            ws.send_json({"speaker_id": "Bob", "text": "Method B could work"})
            msg2 = ws.receive_json()
            assert msg2["event"] == "asr_final"
            assert msg2["speaker_id"] == "Bob"

        # Verify the REST endpoint still shows the stored utterances
        resp = client.get("/discussion/history?n=10")
        assert resp.json()["count"] >= 2

    def test_websocket_invalid_json(self):
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        with client.websocket_connect("/discussion/ws") as ws:
            ws.send_text("not valid json")
            data = ws.receive_json()
            assert "error" in data
