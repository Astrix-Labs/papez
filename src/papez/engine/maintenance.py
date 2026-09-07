"""The maintenance pass: rescore, transition, then prune.

This is the loop that makes forgetting real. ``sweep_for_forgetting`` only
decides; something has to call it on a cadence. The hosted service runs the
same sequence every ten minutes per user; the stdio server runs
``maintenance_loop`` so a local install behaves the same way.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from papez.engine import config
from papez.engine.forgetting import sweep_for_forgetting
from papez.engine.scoring import calculate_decay_score
from papez.models.enums import MemoryStatus
from papez.storage.base import EmbeddingProvider, GraphStorageProvider

logger = logging.getLogger("papez.maintenance")


async def recalculate_decay(graph: GraphStorageProvider, embeddings: EmbeddingProvider | None) -> int:
    """Recompute ``decay_score`` for every non-core node. Returns the count updated."""
    stats = await graph.get_stats()
    max_cw = stats.get("max_causal_weight", 1) or 1
    updated = 0
    for status in (MemoryStatus.ACTIVE, MemoryStatus.EPISODIC, MemoryStatus.SEMANTIC):
        for node in await graph.get_nodes_by_status(status, limit=500):
            if node.status == MemoryStatus.CORE:
                continue
            score = await calculate_decay_score(node, None, None, graph, embeddings, max_cw)  # type: ignore[arg-type]
            await graph.update_node(str(node.id), {"decay_score": score})
            updated += 1
    return updated


async def run_maintenance(
    graph: GraphStorageProvider,
    embeddings: EmbeddingProvider | None,
    llm: Any | None = None,
    *,
    recalculate: bool = True,
) -> dict[str, Any]:
    """One pass: rescore (optional), evaluate status transitions (only with an
    LLM provider, as transitions need it), then prune. Returns what happened."""
    result: dict[str, Any] = {"rescored": 0, "transitions": 0, "pruned": []}
    if recalculate:
        result["rescored"] = await recalculate_decay(graph, embeddings)
    if llm is not None and embeddings is not None:
        from papez.engine.transitions import evaluate_transitions

        result["transitions"] = len(await evaluate_transitions(graph, embeddings, llm))
    result["pruned"] = await sweep_for_forgetting(graph)
    return result


async def maintenance_loop(
    graph: GraphStorageProvider,
    embeddings: EmbeddingProvider | None,
    llm: Any | None = None,
    interval_s: float | None = None,
) -> None:
    """Run ``run_maintenance`` forever on a cadence. Errors are logged, never fatal."""
    interval = config.MAINTENANCE_INTERVAL_S if interval_s is None else interval_s
    if interval <= 0:
        return
    while True:
        await asyncio.sleep(interval)
        try:
            r = await run_maintenance(graph, embeddings, llm)
            if r["pruned"]:
                logger.info("maintenance: rescored %d, transitions %d, pruned %d", r["rescored"], r["transitions"], len(r["pruned"]))
        except Exception:
            logger.exception("maintenance pass failed")
