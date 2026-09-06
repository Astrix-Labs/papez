"""Background handlers for LLM-powered memory processing.

All heavy operations (entity extraction, causal inference, contradiction
detection, consolidation) run here as fire-and-forget tasks dispatched
via the event bus.  The synchronous hot path in tools.py stays fast.
"""
from __future__ import annotations

import logging
import uuid
from contextlib import nullcontext
from typing import Any

from papez.models.edge import MemoryEdge
from papez.storage.base import (
    EmbeddingProvider,
    EventBusProvider,
    GraphStorageProvider,
    LLMProvider,
)

logger = logging.getLogger(__name__)


async def on_memory_created(
    payload: dict[str, Any],
    graph: GraphStorageProvider,
    llm: LLMProvider,
    embeddings: EmbeddingProvider | None,
) -> None:
    """Run all LLM enrichment after a memory is stored."""
    node_id = payload["node_id"]
    content = payload.get("content_full", "")
    if not content:
        return

    node = await graph.get_node(node_id)
    if node is None:
        return

    entities: list[str] = []
    save_ctx = graph.defer_saves() if hasattr(graph, "defer_saves") else nullcontext()
    with save_ctx:
        # 1. Entity extraction
        try:
            entities = await llm.extract_entities(content)
            if entities:
                await graph.update_node(node_id, {"entity_refs": entities})
                logger.info("Extracted %d entities for %s", len(entities), node_id)
        except Exception:
            logger.warning("Entity extraction failed for %s", node_id, exc_info=True)

        # 2. Category classification — only fill when the caller didn't set one,
        #    so an explicit memory_store(category=...) is never overwritten.
        if node.category is None:
            try:
                category = await llm.classify_category(content)
                if category:
                    await graph.update_node(node_id, {"category": category})
                    logger.info("Classified %s as '%s'", node_id, category)
            except Exception:
                logger.warning("Category classification failed for %s", node_id, exc_info=True)

        # 3. Causal edge inference
        try:
            if node.embedding:
                similar = await graph.vector_search(node.embedding, k=6)
                existing = [
                    (str(n.id), n.content_full or n.content_summary)
                    for n, _ in similar
                    if str(n.id) != node_id
                ]
                if existing:
                    causal_edges = await llm.infer_causal_edges(content, existing)
                    for target_id, edge_type, confidence, reason in causal_edges:
                        already = await graph.edge_exists(node_id, target_id, edge_type)
                        if not already:
                            edge = MemoryEdge(
                                source_id=node.id,
                                target_id=uuid.UUID(target_id),
                                type=edge_type,
                                weight=round(confidence, 4),
                                reason=reason,
                                created_by="llm_causal_inference",
                            )
                            await graph.create_edge(edge)
                            logger.info(
                                "Created %s edge %s -> %s (%.2f)",
                                edge_type.value, node_id, target_id, confidence,
                            )
        except Exception:
            logger.warning("Causal inference failed for %s", node_id, exc_info=True)

        # 4. Contradiction detection (requires an embedder to compare content)
        if embeddings is not None:
            try:
                from papez.engine.contradiction import detect_contradictions

                contradictions = await detect_contradictions(node, graph, embeddings, llm)
                if contradictions:
                    logger.info(
                        "Found %d contradictions for %s", len(contradictions), node_id,
                    )
            except Exception:
                logger.warning("Contradiction detection failed for %s", node_id, exc_info=True)

            # 5. Consolidation check for each extracted entity
            try:
                from papez.engine.consolidation import check_and_consolidate

                for entity in entities:
                    result = await check_and_consolidate(entity, graph, llm, embeddings)
                    if result:
                        logger.info(
                            "Consolidated memories for entity '%s' -> %s", entity, result,
                        )
            except Exception:
                logger.warning("Consolidation check failed for %s", node_id, exc_info=True)


def register_handlers(
    event_bus: EventBusProvider,
    graph: GraphStorageProvider,
    llm: LLMProvider,
    embeddings: EmbeddingProvider | None,
) -> None:
    """Wire background handlers to event bus channels."""

    async def _handle_created(payload: dict[str, Any]) -> None:
        try:
            await on_memory_created(payload, graph, llm, embeddings)
        except Exception:
            logger.error("Background handler failed", exc_info=True)

    if hasattr(event_bus, "_subscribers"):
        event_bus._subscribers["memory.created"].append(_handle_created)
    else:
        import asyncio
        asyncio.run(event_bus.subscribe("memory.created", _handle_created))
