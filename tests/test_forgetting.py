"""Tests for active forgetting — safety-critical conjunctive criteria."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from papez.engine.forgetting import sweep_for_forgetting
from papez.models.enums import MemoryStatus, Visibility
from papez.models.node import MemoryNode


def _make_orphan_node(**kwargs) -> MemoryNode:
    long_ago = datetime.now(timezone.utc) - timedelta(days=90)
    defaults = {
        "content_summary": "orphan",
        "decay_score": 0.0,
        "pinned": False,
        "status": MemoryStatus.ACTIVE,
        # Idle by default so the structural criteria are what each test exercises.
        "created_at": long_ago,
        "last_accessed_at": long_ago,
    }
    defaults.update(kwargs)
    return MemoryNode(**defaults)


class TestForgetting:
    @pytest.mark.asyncio
    async def test_prune_meets_all_criteria(self):
        """Node meeting all 4 criteria should be pruned."""
        node = _make_orphan_node(decay_score=0.0, pinned=False, status=MemoryStatus.DORMANT)
        graph = AsyncMock()
        graph.get_orphans = AsyncMock(return_value=[node])
        graph.delete_node = AsyncMock()

        pruned = await sweep_for_forgetting(graph)
        assert len(pruned) == 1
        graph.delete_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_prune_if_has_edges(self):
        """Node with edges won't appear in orphans, so won't be pruned."""
        graph = AsyncMock()
        graph.get_orphans = AsyncMock(return_value=[])  # no orphans
        graph.delete_node = AsyncMock()

        pruned = await sweep_for_forgetting(graph)
        assert len(pruned) == 0
        graph.delete_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prune_if_pinned(self):
        """Pinned orphan with decay_score=0 must NOT be pruned."""
        node = _make_orphan_node(decay_score=0.0, pinned=True)
        graph = AsyncMock()
        graph.get_orphans = AsyncMock(return_value=[node])
        graph.delete_node = AsyncMock()

        pruned = await sweep_for_forgetting(graph)
        assert len(pruned) == 0
        graph.delete_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prune_if_core(self):
        """Core orphan with decay_score=0 must NOT be pruned."""
        node = _make_orphan_node(decay_score=0.0, status=MemoryStatus.CORE)
        graph = AsyncMock()
        graph.get_orphans = AsyncMock(return_value=[node])
        graph.delete_node = AsyncMock()

        pruned = await sweep_for_forgetting(graph)
        assert len(pruned) == 0
        graph.delete_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prune_if_high_decay_score(self):
        """Orphan with decay_score > 0.01 must NOT be pruned."""
        node = _make_orphan_node(decay_score=0.5)
        graph = AsyncMock()
        graph.get_orphans = AsyncMock(return_value=[node])
        graph.delete_node = AsyncMock()

        pruned = await sweep_for_forgetting(graph)
        assert len(pruned) == 0
        graph.delete_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_orphans_mixed(self):
        """Only orphans meeting ALL criteria are pruned."""
        pruneable = _make_orphan_node(decay_score=0.0, status=MemoryStatus.DORMANT)
        pinned = _make_orphan_node(decay_score=0.0, pinned=True)
        high_score = _make_orphan_node(decay_score=0.5)

        graph = AsyncMock()
        graph.get_orphans = AsyncMock(return_value=[pruneable, pinned, high_score])
        graph.delete_node = AsyncMock()

        pruned = await sweep_for_forgetting(graph)
        assert len(pruned) == 1
        graph.delete_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_prune_if_recently_used(self):
        """A fresh orphan scores zero (connectivity is zero) but must survive:
        forgetting needs FORGETTING_MIN_IDLE_DAYS of silence, not just a low score."""
        graph = AsyncMock()
        now = datetime.now(timezone.utc)
        graph.get_orphans.return_value = [_make_orphan_node(created_at=now, last_accessed_at=now)]
        pruned = await sweep_for_forgetting(graph)
        assert pruned == []
        graph.delete_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_reactivation_counts_as_use(self):
        graph = AsyncMock()
        now = datetime.now(timezone.utc)
        node = _make_orphan_node(reactivation_timestamps=[now - timedelta(days=2)])
        graph.get_orphans.return_value = [node]
        pruned = await sweep_for_forgetting(graph)
        assert pruned == []

    @pytest.mark.asyncio
    async def test_no_prune_if_org_visible(self):
        graph = AsyncMock()
        graph.get_orphans.return_value = [_make_orphan_node(visibility=Visibility.ORG)]
        pruned = await sweep_for_forgetting(graph)
        assert pruned == []
        graph.delete_node.assert_not_called()
