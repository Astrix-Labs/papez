"""End-to-end: the maintenance pass actually forgets, under exactly the
documented conditions, against the real in-memory provider (no mocks).

Rescoring gives every orphan a decay score of zero (its connectivity factor is
zero), so what protects a memory is structure (edges), a pin, core status,
org visibility, or recent use. Each survives here; the one idle, unlinked,
unpinned memory does not."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from papez.context import current_user_id
from papez.engine.maintenance import run_maintenance
from papez.models.edge import MemoryEdge
from papez.models.enums import EdgeType, MemoryStatus, Visibility
from papez.models.node import MemoryNode
from papez.storage.memory import InMemoryGraphProvider


class FakeEmbedder:
    dimension = 16

    async def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        vec = [(((h >> (i * 2)) & 3) - 1.5) / 1.5 for i in range(self.dimension)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.fixture()
def _user_ctx():
    token = current_user_id.set("maint-user")
    yield
    current_user_id.reset(token)


def _node(summary: str, *, idle_days: float = 0, **kw) -> MemoryNode:
    when = datetime.now(timezone.utc) - timedelta(days=idle_days)
    return MemoryNode(content=summary, content_summary=summary, created_at=when, last_accessed_at=when, **kw)


@pytest.mark.asyncio
async def test_idle_orphan_is_pruned_and_everything_else_survives(_user_ctx):
    graph = InMemoryGraphProvider()
    await graph.initialize("maint-user")

    fresh = _node("stored a minute ago, no links yet")
    idle = _node("unlinked and untouched for 60 days", idle_days=60)
    pinned = _node("idle but pinned", idle_days=60, pinned=True)
    core = _node("idle but core", idle_days=60, status=MemoryStatus.CORE)
    org = _node("idle but shared with the org", idle_days=60, visibility=Visibility.ORG)
    a = _node("idle, but causally linked", idle_days=60)
    b = _node("the memory it caused", idle_days=60)
    for n in (fresh, idle, pinned, core, org, a, b):
        await graph.create_node(n)
    await graph.create_edge(MemoryEdge(source_id=a.id, target_id=b.id, type=EdgeType.CAUSED_BY))

    result = await run_maintenance(graph, FakeEmbedder(), llm=None)

    assert result["pruned"] == [str(idle.id)]
    assert await graph.get_node(str(idle.id)) is None
    for kept in (fresh, pinned, core, org, a, b):
        assert await graph.get_node(str(kept.id)) is not None, kept.content_summary


@pytest.mark.asyncio
async def test_recall_resets_the_idle_clock(_user_ctx):
    graph = InMemoryGraphProvider()
    await graph.initialize("maint-user")
    old = _node("stored long ago", idle_days=90)
    await graph.create_node(old)
    # A recall a week ago counts as use; the memory is not idle.
    await graph.update_node(str(old.id), {"last_accessed_at": datetime.now(timezone.utc) - timedelta(days=7)})

    result = await run_maintenance(graph, FakeEmbedder(), llm=None)

    assert result["pruned"] == []
    assert await graph.get_node(str(old.id)) is not None


@pytest.mark.asyncio
async def test_loop_is_off_when_interval_is_zero():
    from papez.engine.maintenance import maintenance_loop

    # Returns immediately instead of sleeping forever.
    await maintenance_loop(InMemoryGraphProvider(), None, None, interval_s=0)
