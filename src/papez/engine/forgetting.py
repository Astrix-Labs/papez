"""Active forgetting: prune memories meeting ALL conjunctive criteria."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from papez.engine import config
from papez.models.enums import MemoryStatus, Visibility
from papez.models.node import MemoryNode
from papez.storage.base import GraphStorageProvider


def _last_use(node: MemoryNode) -> datetime:
    """The most recent moment the memory was touched: stored, recalled, or reactivated."""
    candidates = [node.created_at, node.last_accessed_at, *node.reactivation_timestamps]
    latest = max(c for c in candidates if c is not None)
    return latest if latest.tzinfo else latest.replace(tzinfo=timezone.utc)


def is_idle(node: MemoryNode, now: datetime | None = None) -> bool:
    """True once the memory has gone FORGETTING_MIN_IDLE_DAYS without any use."""
    now = now or datetime.now(timezone.utc)
    return now - _last_use(node) >= timedelta(days=config.FORGETTING_MIN_IDLE_DAYS)


async def sweep_for_forgetting(graph: GraphStorageProvider) -> list[str]:
    """
    Prune memories meeting ALL criteria:
    1. decay_score < FORGETTING_THRESHOLD
    2. is_orphan (zero supportive edges)
    3. NOT pinned
    4. status != CORE
    5. visibility != ORG (org nodes are exempt from per-user forgetting)
    6. idle: not stored, recalled or reactivated for FORGETTING_MIN_IDLE_DAYS

    Criterion 6 is what makes "irrelevant" real. An orphan's connectivity
    factor is zero, so its decay score is zero as soon as it is rescored;
    without an idle period every unlinked memory would be pruned on the next
    maintenance pass, minutes after it was stored.

    Returns list of pruned node IDs.
    """
    pruned: list[str] = []
    now = datetime.now(timezone.utc)
    orphans = await graph.get_orphans()

    for node in orphans:
        if node.visibility == Visibility.ORG:
            continue
        if (
            node.decay_score < config.FORGETTING_THRESHOLD
            and not node.pinned
            and node.status != MemoryStatus.CORE
            and is_idle(node, now)
        ):
            node_id = str(node.id)
            await graph.delete_node(node_id)
            pruned.append(node_id)

    return pruned
