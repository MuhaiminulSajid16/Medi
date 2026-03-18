"""
memory.py – Two-tier conversational memory.

* **Short-term (hot) cache** – stores the last N speaker-tagged utterances
  in a Redis-backed (or in-memory fallback) ring buffer.  Optimised for
  recent-context retrieval within O(1) time.

* **Long-term semantic store** – stores sentence embeddings in a FAISS
  index (or a NumPy cosine-similarity fallback) for semantic nearest-
  neighbour search over the full conversation history.

Both tiers share the same :class:`Utterance` data model and expose a
unified :class:`ConversationMemory` interface.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

_EMBEDDING_DIM = 128  # dimension used by the fallback embedder


@dataclass
class Utterance:
    """A single speaker turn stored in memory."""

    utterance_id: int
    speaker_id: str
    text: str
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "utterance_id": self.utterance_id,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Utterance":
        return cls(
            utterance_id=d["utterance_id"],
            speaker_id=d["speaker_id"],
            text=d["text"],
            timestamp=d.get("timestamp", time.time()),
        )


# ------------------------------------------------------------------
# Text embedder
# ------------------------------------------------------------------

class _TFIDFEmbedder:
    """
    Lightweight TF-IDF inspired bag-of-words embedder (no ML deps).

    Produces a *_EMBEDDING_DIM*-dimensional float32 vector based on
    character bigram hashing.  Sufficient for approximate semantic
    similarity in unit tests and systems without sentence-transformers.
    """

    def embed(self, text: str) -> np.ndarray:
        text = text.lower()
        vec = np.zeros(_EMBEDDING_DIM, dtype=np.float32)
        # Character bigrams hashed into fixed-size vector
        for i in range(len(text) - 1):
            bigram = text[i : i + 2]
            bucket = hash(bigram) % _EMBEDDING_DIM
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class _SentenceTransformerEmbedder:
    """Wrapper around sentence-transformers (optional heavy dependency)."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)


def _build_embedder():
    try:
        embedder = _SentenceTransformerEmbedder()
        logger.info("Using sentence-transformers embedder")
        return embedder
    except ImportError:
        logger.debug("sentence-transformers not available; using TF-IDF embedder")
        return _TFIDFEmbedder()


# ------------------------------------------------------------------
# Short-term cache
# ------------------------------------------------------------------

class _InMemoryCache:
    """Ring-buffer short-term cache (Redis fallback)."""

    def __init__(self, maxlen: int) -> None:
        self._store: deque[Utterance] = deque(maxlen=maxlen)

    def put(self, utterance: Utterance) -> None:
        self._store.append(utterance)

    def get_recent(self, n: int) -> list[Utterance]:
        items = list(self._store)
        return items[-n:] if n < len(items) else items

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class _RedisCache:
    """Redis-backed short-term cache."""

    _KEY_PREFIX = "rtai:utterance:"
    _INDEX_KEY = "rtai:utterance_index"

    def __init__(self, redis_client, maxlen: int) -> None:
        self._r = redis_client
        self._maxlen = maxlen

    def put(self, utterance: Utterance) -> None:
        key = f"{self._KEY_PREFIX}{utterance.utterance_id}"
        self._r.set(key, json.dumps(utterance.to_dict()), ex=3600)
        self._r.rpush(self._INDEX_KEY, utterance.utterance_id)
        # Trim to maxlen
        excess = self._r.llen(self._INDEX_KEY) - self._maxlen
        if excess > 0:
            for _ in range(excess):
                old_id = self._r.lpop(self._INDEX_KEY)
                if old_id:
                    self._r.delete(f"{self._KEY_PREFIX}{old_id.decode()}")

    def get_recent(self, n: int) -> list[Utterance]:
        ids = self._r.lrange(self._INDEX_KEY, -n, -1)
        results = []
        for uid in ids:
            raw = self._r.get(f"{self._KEY_PREFIX}{uid.decode()}")
            if raw:
                results.append(Utterance.from_dict(json.loads(raw)))
        return results

    def clear(self) -> None:
        ids = self._r.lrange(self._INDEX_KEY, 0, -1)
        for uid in ids:
            self._r.delete(f"{self._KEY_PREFIX}{uid.decode()}")
        self._r.delete(self._INDEX_KEY)


# ------------------------------------------------------------------
# Long-term semantic store
# ------------------------------------------------------------------

class _NumpySemanticStore:
    """Cosine-similarity semantic store (FAISS fallback)."""

    def __init__(self) -> None:
        self._embeddings: list[np.ndarray] = []
        self._utterances: list[Utterance] = []

    def add(self, utterance: Utterance) -> None:
        if utterance.embedding is not None:
            self._embeddings.append(utterance.embedding)
            self._utterances.append(utterance)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[Utterance]:
        if not self._embeddings:
            return []
        matrix = np.stack(self._embeddings, axis=0)
        scores = matrix @ query_embedding
        indices = np.argsort(scores)[::-1][:top_k]
        return [self._utterances[i] for i in indices]

    def clear(self) -> None:
        self._embeddings.clear()
        self._utterances.clear()

    def __len__(self) -> int:
        return len(self._utterances)


class _FAISSSemanticStore:
    """FAISS-backed semantic store for scalable nearest-neighbour search."""

    def __init__(self, dim: int) -> None:
        import faiss  # type: ignore[import]

        self._index = faiss.IndexFlatIP(dim)  # inner-product (cosine on L2-normalised vecs)
        self._utterances: list[Utterance] = []

    def add(self, utterance: Utterance) -> None:
        if utterance.embedding is not None:
            vec = utterance.embedding.reshape(1, -1).astype(np.float32)
            self._index.add(vec)
            self._utterances.append(utterance)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[Utterance]:
        if len(self._utterances) == 0:
            return []
        vec = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, len(self._utterances))
        _, indices = self._index.search(vec, k)
        return [self._utterances[i] for i in indices[0] if i >= 0]

    def clear(self) -> None:
        dim = self._index.d
        import faiss  # type: ignore[import]

        self._index = faiss.IndexFlatIP(dim)
        self._utterances.clear()

    def __len__(self) -> int:
        return len(self._utterances)


# ------------------------------------------------------------------
# Public facade
# ------------------------------------------------------------------

class ConversationMemory:
    """
    Two-tier conversational memory.

    Parameters
    ----------
    hot_cache_size:
        Maximum number of recent utterances kept in the short-term cache.
    redis_url:
        Optional Redis connection URL (e.g. ``"redis://localhost:6379"``).
        When ``None`` or when Redis is unavailable an in-memory ring
        buffer is used instead.
    use_faiss:
        Prefer FAISS for long-term storage when the package is available.

    Usage
    -----
    ::

        mem = ConversationMemory()
        mem.store("Speaker_0", "We should try approach A.")
        recent = mem.get_recent(10)
        relevant = mem.search_semantic("which approach?", top_k=3)
    """

    def __init__(
        self,
        hot_cache_size: int = 100,
        redis_url: Optional[str] = None,
        use_faiss: bool = True,
    ) -> None:
        self._embedder = _build_embedder()
        self._hot_cache = self._build_hot_cache(redis_url, hot_cache_size)
        self._semantic_store = self._build_semantic_store(use_faiss)
        self._next_id: int = 0
        logger.info(
            "ConversationMemory ready (cache=%s, semantic=%s)",
            type(self._hot_cache).__name__,
            type(self._semantic_store).__name__,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, speaker_id: str, text: str) -> Utterance:
        """
        Store a new speaker utterance in both memory tiers.

        Returns the created :class:`Utterance` (including its embedding).
        """
        embedding = self._embedder.embed(text)
        utterance = Utterance(
            utterance_id=self._next_id,
            speaker_id=speaker_id,
            text=text,
            embedding=embedding,
        )
        self._next_id += 1
        self._hot_cache.put(utterance)
        self._semantic_store.add(utterance)
        return utterance

    def get_recent(self, n: int = 10) -> list[Utterance]:
        """Return up to *n* most recent utterances from the hot cache."""
        return self._hot_cache.get_recent(n)

    def search_semantic(self, query: str, top_k: int = 3) -> list[Utterance]:
        """
        Return the *top_k* utterances most semantically similar to *query*
        from long-term memory.
        """
        query_embedding = self._embedder.embed(query)
        return self._semantic_store.search(query_embedding, top_k)

    def clear(self) -> None:
        """Wipe all stored utterances from both tiers."""
        self._hot_cache.clear()
        self._semantic_store.clear()
        self._next_id = 0

    @property
    def total_utterances(self) -> int:
        return self._next_id

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_hot_cache(self, redis_url: Optional[str], maxlen: int):
        if redis_url:
            try:
                import redis  # type: ignore[import]

                client = redis.from_url(redis_url, decode_responses=False, socket_timeout=2)
                client.ping()
                logger.info("Connected to Redis at %s", redis_url)
                return _RedisCache(client, maxlen)
            except Exception as exc:
                logger.warning("Redis unavailable (%s); using in-memory cache", exc)
        return _InMemoryCache(maxlen)

    def _build_semantic_store(self, use_faiss: bool):
        if use_faiss:
            try:
                dim = getattr(self._embedder, "_model", None)
                if dim is not None:
                    # sentence-transformers: get actual dim
                    sample = self._embedder.embed("hello")
                    actual_dim = len(sample)
                else:
                    actual_dim = _EMBEDDING_DIM
                store = _FAISSSemanticStore(actual_dim)
                logger.info("Using FAISS semantic store (dim=%d)", actual_dim)
                return store
            except ImportError:
                logger.debug("faiss not available; using NumPy semantic store")
        return _NumpySemanticStore()
