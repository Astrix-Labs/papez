"""Simplified provider singleton for standalone papez use.

Supports only the in-memory backend with optional local/OpenAI embeddings.
For production backends (Postgres, FalkorDB, etc.), use genesys-server.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from papez.mcp.tools import MCPToolHandler
from papez.storage.base import CacheProvider, EmbeddingProvider, EventBusProvider, GraphStorageProvider
from papez.storage.memory import InMemoryCacheProvider, InMemoryEventBusProvider, InMemoryGraphProvider

load_dotenv()


@dataclass
class Providers:
    graph: GraphStorageProvider
    cache: CacheProvider
    embeddings: EmbeddingProvider | None
    llm: object | None
    event_bus: EventBusProvider | None
    tools: MCPToolHandler
    user_id: str


_instance: Providers | None = None


def _make_embedder() -> EmbeddingProvider | None:
    """Create embedding provider based on GENESYS_EMBEDDER env var.

    Falls back gracefully when optional dependencies or credentials are
    missing: openai -> local (if sentence-transformers is installed) ->
    None (keyword-only recall). Never constructs an OpenAI client without a
    usable API key — the client itself doesn't validate the key, so an
    empty/missing key would otherwise surface as an auth error on the first
    embed() call instead of failing gracefully here.
    """
    embedder = os.getenv("GENESYS_EMBEDDER", "openai")

    if embedder == "local":
        try:
            from papez.retrieval.embedding import LocalEmbeddingProvider
            return LocalEmbeddingProvider()
        except ImportError:
            return None

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            from papez.retrieval.embedding import OpenAIEmbeddingProvider
            return OpenAIEmbeddingProvider(api_key=api_key)
        except ImportError:
            logging.getLogger("papez").warning(
                "OPENAI_API_KEY is set but the `openai` package is not installed; "
                "falling back to the local embedder if available. Install it with: pip install 'papez[openai]'"
            )

    try:
        from papez.retrieval.embedding import LocalEmbeddingProvider
        return LocalEmbeddingProvider()
    except ImportError:
        return None


def get_providers() -> Providers:
    """Return the shared provider singleton, creating it on first call."""
    global _instance
    if _instance is not None:
        return _instance

    user_id = os.getenv("GENESYS_USER_ID", "default_user")

    default_persist = str(Path(__file__).parent.parent.parent / "data" / "memories.json")
    graph = InMemoryGraphProvider(persist_path=os.getenv("GENESYS_PERSIST_PATH", default_persist))
    cache = InMemoryCacheProvider()
    embeddings = _make_embedder()

    llm_provider = None
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        try:
            from papez.engine.llm_provider import AnthropicLLMProvider
            llm_provider = AnthropicLLMProvider(api_key=anthropic_key)
        except ImportError:
            logging.getLogger("papez").warning(
                "ANTHROPIC_API_KEY is set but the `anthropic` package is not installed; "
                "LLM enrichment is off. Install it with: pip install 'papez[anthropic]'"
            )

    event_bus = InMemoryEventBusProvider()

    if llm_provider:
        from papez.engine.background import register_handlers
        register_handlers(event_bus, graph, llm_provider, embeddings)

    tools = MCPToolHandler(
        graph=graph,
        embeddings=embeddings,
        cache=cache,
        event_bus=event_bus,
        llm=llm_provider,
    )

    _instance = Providers(
        graph=graph,
        cache=cache,
        embeddings=embeddings,
        llm=llm_provider,
        event_bus=event_bus,
        tools=tools,
        user_id=user_id,
    )
    return _instance
