from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from papez.storage.base import CacheProvider


class LocalEmbeddingProvider:
    """Local embedding provider using sentence-transformers (all-MiniLM-L6-v2).

    No API key required. Model is lazy-loaded on first embed() call.
    """

    DIMENSION = 384

    def __init__(self) -> None:
        self._model: Any = None

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    # Local MiniLM cosine similarities cluster much lower than OpenAI's
    # text-embedding-3-small for genuine matches (empirically ~0.2-0.4 vs
    # ~0.5+), so recall/core-injection thresholds must consult this instead
    # of assuming an OpenAI-tuned floor.
    @property
    def recommended_min_similarity(self) -> float:
        return 0.2

    @property
    def recommended_core_min_similarity(self) -> float:
        return 0.2

    # Auto-links create permanent graph structure. MiniLM is noisy above the
    # genuine-match band (~0.2-0.4): field reports show noise pairs at ~0.44,
    # so the floor sits ABOVE the band top — locally, only near-duplicate
    # content auto-links, which is the conservative right answer for
    # permanent structure.
    @property
    def recommended_autolink_min_similarity(self) -> float:
        return 0.45

    async def embed(self, text: str) -> list[float]:
        self._load_model()
        vec: list[float] = self._model.encode(text, normalize_embeddings=True).tolist()
        return vec

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load_model()
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


class OpenAIEmbeddingProvider:
    MODEL = "text-embedding-3-small"
    DIMENSION = 1536

    def __init__(self, api_key: str, cache: CacheProvider | None = None):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key)
        self._cache = cache

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    # text-embedding-3-small produces well-separated cosine similarities for
    # genuine matches; these are the thresholds the LoCoMo benchmark was
    # tuned against.
    @property
    def recommended_min_similarity(self) -> float:
        return 0.5

    @property
    def recommended_core_min_similarity(self) -> float:
        return 0.45

    # An auto-link creates permanent structure, so its floor sits ABOVE the
    # transient recall floor (0.5): 0.6 means "clearly the same topic".
    @property
    def recommended_autolink_min_similarity(self) -> float:
        return 0.6

    def _cache_key(self, text: str) -> str:
        return f"embed:{hashlib.sha256(text.encode()).hexdigest()}"

    async def embed(self, text: str) -> list[float]:
        if self._cache:
            import json
            cached = await self._cache.get(self._cache_key(text))
            if cached:
                result: list[float] = json.loads(cached)
                return result

        if len(text) > 8000:
            text = text[:8000]
        response = await self._client.embeddings.create(input=[text], model=self.MODEL)
        vec: list[float] = response.data[0].embedding

        if self._cache:
            import json
            await self._cache.set(self._cache_key(text), json.dumps(vec), ttl_seconds=86400)

        return vec

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(input=texts, model=self.MODEL)
        return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
