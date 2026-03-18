"""
realtime_ai – Real-time AI system for multi-speaker group discussions.

Pipeline:
  AudioChunker → Diarization → ASR → Memory → ContextBuilder →
  TriggerDetector → ReasoningEngine → TTS
"""

from .audio_chunker import AudioChunker
from .diarization import SpeakerDiarizer
from .asr import StreamingASR
from .memory import ConversationMemory
from .context_builder import ContextBuilder
from .triggers import TriggerDetector
from .reasoning import ReasoningEngine
from .tts import TTSEngine
from .pipeline import RealtimePipeline

__all__ = [
    "AudioChunker",
    "SpeakerDiarizer",
    "StreamingASR",
    "ConversationMemory",
    "ContextBuilder",
    "TriggerDetector",
    "ReasoningEngine",
    "TTSEngine",
    "RealtimePipeline",
]
