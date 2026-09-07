"""In-memory storage providers for running without Docker/Redis/FalkorDB."""
from __future__ import annotations

import asyncio
import json as _json
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from papez.context import current_org_ids, current_user_id
from papez.models.edge import MemoryEdge
from papez.models.enums import CAUSAL_EDGE_TYPES, SUPPORTIVE_EDGE_TYPES, EdgeType, MemoryStatus, Visibility
from papez.models.node import MemoryNode


def _uid() -> str:
    uid = current_user_id.get(None)
    if uid is None:
        raise RuntimeError("No user context — current_user_id not set")
    return uid


def _stem(word: str) -> str:
    """Minimal suffix-stripping stemmer (no deps)."""
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("tion") and len(word) > 5:
        return word[:-4]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokenize(text: str) -> list[str]:
    """Split text into stemmed tokens, stripping punctuation."""
    import re
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if len(w) > 2]


class InMemoryCacheProvider:
    """CacheProvider backed by a plain dict."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._data


class InMemoryGraphProvider:
    """GraphStorageProvider backed by plain dicts with per-user isolation.

    If ``persist_path`` is set, state is saved to / loaded from that JSON file
    so data survives across process restarts (e.g. separate ``claude -p`` calls).

    Performance: edges are indexed by source_id and target_id for O(degree)
    lookups instead of O(total_edges) scans. Disk writes are batched within
    ``defer_saves()`` contexts.
    """

    def __init__(self, persist_path: str | None = None):
        self._user_nodes: dict[str, dict[str, MemoryNode]] = {}
        self._user_edges: dict[str, list[MemoryEdge]] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        self._node_locks: dict[str, asyncio.Lock] = {}
        # Edge indexes: {user_id: {node_id: [edges]}}
        self._idx_by_source: dict[str, dict[str, list[MemoryEdge]]] = {}
        self._idx_by_target: dict[str, dict[str, list[MemoryEdge]]] = {}
        # Save batching: defer disk writes until outermost defer_saves() exits
        self._save_depth = 0
        self._dirty = False

    @property
    def nodes(self) -> dict[str, MemoryNode]:
        """Return nodes for the current user."""
        uid = _uid()
        if uid not in self._user_nodes:
            self._user_nodes[uid] = {}
        return self._user_nodes[uid]

    @property
    def edges(self) -> list[MemoryEdge]:
        """Return edges for the current user."""
        uid = _uid()
        if uid not in self._user_edges:
            self._user_edges[uid] = []
        return self._user_edges[uid]

    @edges.setter
    def edges(self, value: list[MemoryEdge]) -> None:
        uid = _uid()
        self._user_edges[uid] = value
        self._idx_by_source.pop(uid, None)
        self._idx_by_target.pop(uid, None)
        for edge in value:
            self._index_edge(uid, edge)

    # -- Edge index maintenance -------------------------------------------------

    def _index_edge(self, uid: str, edge: MemoryEdge) -> None:
        src, tgt = str(edge.source_id), str(edge.target_id)
        self._idx_by_source.setdefault(uid, {}).setdefault(src, []).append(edge)
        self._idx_by_target.setdefault(uid, {}).setdefault(tgt, []).append(edge)

    def _unindex_edge(self, uid: str, edge: MemoryEdge) -> None:
        eid = str(edge.id)
        src, tgt = str(edge.source_id), str(edge.target_id)
        src_list = self._idx_by_source.get(uid, {}).get(src)
        if src_list is not None:
            self._idx_by_source[uid][src] = [e for e in src_list if str(e.id) != eid]
        tgt_list = self._idx_by_target.get(uid, {}).get(tgt)
        if tgt_list is not None:
            self._idx_by_target[uid][tgt] = [e for e in tgt_list if str(e.id) != eid]

    def _rebuild_indexes(self) -> None:
        self._idx_by_source.clear()
        self._idx_by_target.clear()
        for uid, edge_list in self._user_edges.items():
            for edge in edge_list:
                self._index_edge(uid, edge)

    def _edges_touching(self, node_id: str, uid: str | None = None) -> list[MemoryEdge]:
        """All edges from a single user touching a node (by source or target)."""
        if uid is None:
            uid = _uid()
        seen: set[str] = set()
        result: list[MemoryEdge] = []
        for e in self._idx_by_source.get(uid, {}).get(node_id, []):
            eid = str(e.id)
            if eid not in seen:
                result.append(e)
                seen.add(eid)
        for e in self._idx_by_target.get(uid, {}).get(node_id, []):
            eid = str(e.id)
            if eid not in seen:
                result.append(e)
                seen.add(eid)
        return result

    # -- Save batching ----------------------------------------------------------

    @contextmanager
    def defer_saves(self) -> Generator[None, None, None]:
        """Batch disk writes. All mutations inside this context share one write."""
        self._save_depth += 1
        try:
            yield
        finally:
            self._save_depth -= 1
            if self._save_depth == 0 and self._dirty:
                self._do_save()
                self._dirty = False

    def _save(self) -> None:
        if self._save_depth > 0:
            self._dirty = True
            return
        self._do_save()

    def _do_save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for uid, nodes in self._user_nodes.items():
            data[uid] = {
                "nodes": {nid: n.model_dump(mode="json") for nid, n in nodes.items()},
                "edges": [e.model_dump(mode="json") for e in self._user_edges.get(uid, [])],
            }
        self._persist_path.write_text(_json.dumps(data))

    # -- Org visibility ---------------------------------------------------------

    def _get_org_ids(self, org_ids: list[str] | None = None) -> list[str]:
        if org_ids is not None:
            return org_ids
        return current_org_ids.get([])

    def _visible_nodes(self, org_ids: list[str] | None = None) -> dict[str, MemoryNode]:
        """Return all nodes visible to the current user: own nodes + org nodes."""
        result = dict(self.nodes)
        oids = self._get_org_ids(org_ids)
        if oids:
            oid_set = set(oids)
            for uid, user_nodes in self._user_nodes.items():
                if uid == _uid():
                    continue
                for nid, node in user_nodes.items():
                    if node.visibility == Visibility.ORG and node.org_id in oid_set:
                        result[nid] = node
        return result

    def _is_node_visible(self, node_id: str, org_ids: list[str] | None = None) -> bool:
        """Check if a node is visible to the current user."""
        node = self.nodes.get(node_id)
        if node:
            return True
        oids = self._get_org_ids(org_ids)
        if oids:
            for user_nodes in self._user_nodes.values():
                n = user_nodes.get(node_id)
                if n and n.visibility == Visibility.ORG and n.org_id in oids:
                    return True
        return False

    def _get_any_node(self, node_id: str) -> MemoryNode | None:
        """Get a node from any user's store (for cross-user org lookups)."""
        for user_nodes in self._user_nodes.values():
            if node_id in user_nodes:
                return user_nodes[node_id]
        return None

    def _all_visible_edges(self, org_ids: list[str] | None = None) -> list[MemoryEdge]:
        """Return edges where both endpoints are visible to the current user."""
        visible = self._visible_nodes(org_ids)
        visible_ids = set(visible.keys())
        result: list[MemoryEdge] = []
        seen_edge_ids: set[str] = set()
        for edge_list in self._user_edges.values():
            for e in edge_list:
                eid = str(e.id)
                if eid in seen_edge_ids:
                    continue
                if str(e.source_id) in visible_ids and str(e.target_id) in visible_ids:
                    result.append(e)
                    seen_edge_ids.add(eid)
        return result

    def _visible_edges_for_node(self, node_id: str, visible_ids: set[str] | None = None, org_ids: list[str] | None = None) -> list[MemoryEdge]:
        """Get visible edges touching a node using indexes instead of full scan."""
        uid = _uid()
        if visible_ids is None:
            visible_ids = set(self._visible_nodes(org_ids).keys())
        result: list[MemoryEdge] = []
        seen: set[str] = set()

        for user_uid in self._user_edges:
            is_own = user_uid == uid
            for e in self._idx_by_source.get(user_uid, {}).get(node_id, []):
                eid = str(e.id)
                if eid in seen:
                    continue
                if is_own or str(e.target_id) in visible_ids:
                    result.append(e)
                    seen.add(eid)
            for e in self._idx_by_target.get(user_uid, {}).get(node_id, []):
                eid = str(e.id)
                if eid in seen:
                    continue
                if is_own or str(e.source_id) in visible_ids:
                    result.append(e)
                    seen.add(eid)
        return result

    # -- Persistence ------------------------------------------------------------

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        raw = _json.loads(self._persist_path.read_text())
        # Support both old format (flat) and new format (per-user)
        if "nodes" in raw and "edges" in raw:
            # Old flat format — migrate to default user
            uid = _uid()
            self._user_nodes[uid] = {nid: MemoryNode(**v) for nid, v in raw.get("nodes", {}).items()}
            self._user_edges[uid] = [MemoryEdge(**e) for e in raw.get("edges", [])]
        else:
            # New per-user format
            for uid, user_data in raw.items():
                self._user_nodes[uid] = {nid: MemoryNode(**v) for nid, v in user_data.get("nodes", {}).items()}
                self._user_edges[uid] = [MemoryEdge(**e) for e in user_data.get("edges", [])]
        self._rebuild_indexes()

    async def initialize(self, user_id: str) -> None:
        self._load()

    async def destroy(self, user_id: str) -> None:
        self._user_nodes.pop(user_id, None)
        self._user_edges.pop(user_id, None)
        self._idx_by_source.pop(user_id, None)
        self._idx_by_target.pop(user_id, None)
        self._save()

    async def erase_user(self, user_id: str, keep_promoted_nodes: bool = True) -> dict[str, int]:
        import re
        _PII_RE = re.compile(
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|"
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"
        )

        user_nodes = self._user_nodes.get(user_id, {})
        nodes_deleted = 0
        nodes_anonymized = 0

        # Partition nodes into deletable vs. promoted-org (anonymize)
        promoted_ids: set[str] = set()
        delete_ids: set[str] = set()
        for nid, node in user_nodes.items():
            if keep_promoted_nodes and node.visibility == Visibility.ORG and node.org_id:
                promoted_ids.add(nid)
            else:
                delete_ids.add(nid)

        # Anonymize promoted nodes
        for nid in promoted_ids:
            node = user_nodes[nid]
            node.original_user_id = "erased_user"
            nodes_anonymized += 1

        # Scrub PII from edge reasons connected to promoted nodes
        edges_scrubbed = 0
        for edge_list in self._user_edges.values():
            for e in edge_list:
                src, tgt = str(e.source_id), str(e.target_id)
                if (src in promoted_ids or tgt in promoted_ids) and e.reason and _PII_RE.search(e.reason):
                    e.reason = _PII_RE.sub("[erased]", e.reason)
                    edges_scrubbed += 1

        # Count and remove edges for deleted nodes
        edges_deleted = 0
        all_delete_ids = delete_ids
        user_edges = self._user_edges.get(user_id, [])
        # Count edges being removed from user's own list (only for deleted nodes)
        edges_deleted += sum(
            1 for e in user_edges
            if str(e.source_id) in all_delete_ids or str(e.target_id) in all_delete_ids
        )

        # Remove edges in other users' lists that reference deleted nodes
        for other_uid, edge_list in list(self._user_edges.items()):
            if other_uid == user_id:
                continue
            before = len(edge_list)
            self._user_edges[other_uid] = [
                e for e in edge_list
                if str(e.source_id) not in all_delete_ids and str(e.target_id) not in all_delete_ids
            ]
            edges_deleted += before - len(self._user_edges[other_uid])

        # Delete the deletable nodes
        for nid in delete_ids:
            del user_nodes[nid]
        nodes_deleted = len(delete_ids)

        # Clean up user's edge list: remove edges referencing deleted nodes
        if user_id in self._user_edges:
            self._user_edges[user_id] = [
                e for e in self._user_edges[user_id]
                if str(e.source_id) not in all_delete_ids and str(e.target_id) not in all_delete_ids
            ]

        # If no promoted nodes remain, clean up user entirely
        if not promoted_ids:
            self._user_nodes.pop(user_id, None)
            self._user_edges.pop(user_id, None)

        self._node_locks = {k: v for k, v in self._node_locks.items() if k not in delete_ids}
        self._rebuild_indexes()
        self._save()
        return {
            "nodes_deleted": nodes_deleted,
            "nodes_anonymized": nodes_anonymized,
            "edges_deleted": edges_deleted,
            "edges_scrubbed": edges_scrubbed,
        }

    # -- Node CRUD --------------------------------------------------------------

    async def create_node(self, node: MemoryNode) -> str:
        nid = str(node.id)
        self.nodes[nid] = node
        self._save()
        return nid

    async def get_node(self, node_id: str) -> MemoryNode | None:
        node = self.nodes.get(node_id)
        if node:
            return node
        oids = self._get_org_ids()
        if oids:
            for user_nodes in self._user_nodes.values():
                n = user_nodes.get(node_id)
                if n and n.visibility == Visibility.ORG and n.org_id in oids:
                    return n
        return None

    async def update_node(self, node_id: str, updates: dict[str, Any]) -> None:
        node = self.nodes.get(node_id)
        if node:
            for k, v in updates.items():
                setattr(node, k, v)
            self._save()

    async def delete_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        uid = _uid()
        removed = [e for e in self._user_edges.get(uid, [])
                    if str(e.source_id) == node_id or str(e.target_id) == node_id]
        for e in removed:
            self._unindex_edge(uid, e)
        self._user_edges[uid] = [e for e in self._user_edges.get(uid, [])
                                  if str(e.source_id) != node_id and str(e.target_id) != node_id]
        self._save()

    async def get_nodes_by_status(self, status: MemoryStatus, limit: int = 100) -> list[MemoryNode]:
        visible = self._visible_nodes()
        return [n for n in visible.values() if n.status == status][:limit]

    # -- Edge CRUD --------------------------------------------------------------

    async def create_edge(self, edge: MemoryEdge) -> str:
        uid = _uid()
        self._user_edges.setdefault(uid, []).append(edge)
        self._index_edge(uid, edge)
        self._save()
        return str(edge.id)

    async def get_edges(self, node_id: str, direction: str, edge_type: EdgeType | None = None) -> list[MemoryEdge]:
        uid = _uid()
        result: list[MemoryEdge] = []
        seen: set[str] = set()
        if direction in ("out", "both"):
            for e in self._idx_by_source.get(uid, {}).get(node_id, []):
                if edge_type is None or e.type == edge_type:
                    eid = str(e.id)
                    if eid not in seen:
                        result.append(e)
                        seen.add(eid)
        if direction in ("in", "both", "incoming"):
            for e in self._idx_by_target.get(uid, {}).get(node_id, []):
                if edge_type is None or e.type == edge_type:
                    eid = str(e.id)
                    if eid not in seen:
                        result.append(e)
                        seen.add(eid)
        return result

    async def get_all_edges(self, node_ids: list[str] | None = None) -> list[MemoryEdge]:
        uid = _uid()
        if node_ids is None:
            return list(self._user_edges.get(uid, []))
        ids = set(node_ids)
        seen: set[str] = set()
        result: list[MemoryEdge] = []
        for nid in ids:
            for e in self._edges_touching(nid, uid):
                eid = str(e.id)
                if eid not in seen:
                    result.append(e)
                    seen.add(eid)
        return result

    async def update_edge_weight(self, edge_id: str, weight: float) -> None:
        uid = _uid()
        for e in self._user_edges.get(uid, []):
            if str(e.id) == edge_id:
                e.weight = weight
                self._save()
                return

    async def validate_edge(self, edge_id: str) -> None:
        from datetime import datetime
        uid = _uid()
        for e in self._user_edges.get(uid, []):
            if str(e.id) == edge_id:
                e.last_validated_at = datetime.now(UTC)
                self._save()
                return

    async def delete_edge(self, edge_id: str) -> None:
        uid = _uid()
        edge_list = self._user_edges.get(uid, [])
        for e in edge_list:
            if str(e.id) == edge_id:
                self._unindex_edge(uid, e)
                break
        self._user_edges[uid] = [e for e in edge_list if str(e.id) != edge_id]
        self._save()

    async def edge_exists(self, source_id: str, target_id: str, edge_type: EdgeType) -> bool:
        uid = _uid()
        for e in self._idx_by_source.get(uid, {}).get(source_id, []):
            if str(e.target_id) == target_id and e.type == edge_type:
                return True
        return False

    # -- Graph queries (indexed) ------------------------------------------------

    async def traverse(self, start_id: str, depth: int, edge_types: list[EdgeType] | None = None, org_ids: list[str] | None = None) -> list[MemoryNode]:
        visible = self._visible_nodes(org_ids)
        visible_ids = set(visible.keys())
        visited: set[str] = set()
        queue = [(start_id, 0)]
        result: list[MemoryNode] = []
        while queue:
            nid, d = queue.pop(0)
            if nid in visited or d > depth:
                continue
            visited.add(nid)
            node = visible.get(nid)
            if node:
                result.append(node)
            if d < depth:
                for e in self._visible_edges_for_node(nid, visible_ids, org_ids):
                    if edge_types and e.type not in edge_types:
                        continue
                    src, tgt = str(e.source_id), str(e.target_id)
                    next_id = tgt if src == nid else src
                    if next_id not in visited and next_id in visible_ids:
                        queue.append((next_id, d + 1))
        return result

    async def get_connecting_edges(
        self, node_ids: list[str], edge_types: list[EdgeType] | None = None, org_ids: list[str] | None = None,
    ) -> list[MemoryEdge]:
        """Return visible edges of the induced subgraph among ``node_ids``.

        An edge is included only when BOTH of its endpoints are in ``node_ids``
        and — when ``edge_types`` is given — its type is in that set. This is
        the set of *connecting edges* for a traversal result: a deliberate
        superset of the BFS tree, so callers can render/reconstruct paths.

        Visibility is honored (own nodes + org-visible nodes). The induced-
        subgraph and edge-type filtering live here (storage), not in the tool
        layer, so every backend that implements this Protocol behaves the same.
        """
        id_set = set(node_ids)
        visible_ids = set(self._visible_nodes(org_ids).keys())
        result: list[MemoryEdge] = []
        seen: set[str] = set()
        for nid in id_set:
            for e in self._visible_edges_for_node(nid, visible_ids, org_ids):
                eid = str(e.id)
                if eid in seen:
                    continue
                src, tgt = str(e.source_id), str(e.target_id)
                if src not in id_set or tgt not in id_set:
                    continue
                if edge_types and e.type not in edge_types:
                    continue
                result.append(e)
                seen.add(eid)
        return result

    async def get_causal_chain(self, node_id: str, direction: str, org_ids: list[str] | None = None) -> list[MemoryNode]:
        visible = self._visible_nodes(org_ids)
        visible_ids = set(visible.keys())
        visited: set[str] = set()
        queue = [node_id]
        result: list[MemoryNode] = []
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            for e in self._visible_edges_for_node(nid, visible_ids, org_ids):
                if e.type not in CAUSAL_EDGE_TYPES:
                    continue
                src, tgt = str(e.source_id), str(e.target_id)
                next_id = None
                if direction == "upstream" and tgt == nid and src not in visited:
                    next_id = src
                elif direction == "downstream" and src == nid and tgt not in visited:
                    next_id = tgt
                if next_id and next_id in visible_ids:
                    queue.append(next_id)
                    node = visible.get(next_id)
                    if node:
                        result.append(node)
        return result

    async def get_causal_chains_batch(
        self, node_ids: list[str], direction: str, org_ids: list[str] | None = None,
    ) -> dict[str, list[MemoryNode]]:
        """BFS from all seed nodes simultaneously, sharing one visibility scan."""
        visible = self._visible_nodes(org_ids)
        visible_ids = set(visible.keys())
        result: dict[str, list[MemoryNode]] = {nid: [] for nid in node_ids}
        visited: dict[str, set[str]] = {nid: set() for nid in node_ids}

        for seed_id in node_ids:
            queue = [seed_id]
            seen = visited[seed_id]
            while queue:
                nid = queue.pop(0)
                if nid in seen:
                    continue
                seen.add(nid)
                for e in self._visible_edges_for_node(nid, visible_ids, org_ids):
                    if e.type not in CAUSAL_EDGE_TYPES:
                        continue
                    src, tgt = str(e.source_id), str(e.target_id)
                    next_id = None
                    if direction == "upstream" and tgt == nid and src not in seen:
                        next_id = src
                    elif direction == "downstream" and src == nid and tgt not in seen:
                        next_id = tgt
                    if next_id and next_id in visible_ids:
                        queue.append(next_id)
                        node = visible.get(next_id)
                        if node:
                            result[seed_id].append(node)
        return result

    async def get_causal_weight(self, node_id: str) -> int:
        count = 0
        for e in self._edges_touching(node_id):
            if e.type in CAUSAL_EDGE_TYPES:
                count += 1
        return count

    async def get_degree(self, node_id: str) -> int:
        return len(self._edges_touching(node_id))

    async def get_supportive_degree(self, node_id: str) -> int:
        count = 0
        for e in self._edges_touching(node_id):
            if e.type in SUPPORTIVE_EDGE_TYPES:
                count += 1
        return count

    async def is_orphan(self, node_id: str) -> bool:
        return (await self.get_supportive_degree(node_id)) == 0

    async def get_orphans(self) -> list[MemoryNode]:
        uid = _uid()
        supported: set[str] = set()
        for e in self._user_edges.get(uid, []):
            if e.type in SUPPORTIVE_EDGE_TYPES:
                supported.add(str(e.source_id))
                supported.add(str(e.target_id))
        return [n for nid, n in self.nodes.items() if nid not in supported]

    # -- Search -----------------------------------------------------------------

    async def vector_search(
        self, embedding: list[float], k: int = 10, status_filter: list[MemoryStatus] | None = None, org_ids: list[str] | None = None,
    ) -> list[tuple[MemoryNode, float]]:
        import numpy as np

        visible = self._visible_nodes(org_ids)
        nodes_with_emb: list[MemoryNode] = []
        for node in visible.values():
            if status_filter and node.status not in status_filter:
                continue
            if node.embedding:
                nodes_with_emb.append(node)
        if not nodes_with_emb:
            return []

        query_vec = np.asarray(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        matrix = np.array([n.embedding for n in nodes_with_emb], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        sims = (matrix @ query_vec) / (norms * query_norm)

        top_k = min(k, len(sims))
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        return [(nodes_with_emb[i], float(sims[i])) for i in top_indices]

    async def keyword_search(self, query: str, entity_refs: list[str] | None = None, k: int = 10, org_ids: list[str] | None = None) -> list[MemoryNode]:
        visible = self._visible_nodes(org_ids)
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        scored: list[tuple[MemoryNode, float]] = []
        for node in visible.values():
            content = (node.content_full or node.content_summary).lower()
            content_tokens = set(content.split())
            hits = sum(1 for t in query_terms if any(t in tok for tok in content_tokens))
            entity_hits = 0
            if node.entity_refs:
                node_entities_lower = [e.lower() for e in node.entity_refs]
                entity_hits = sum(
                    1 for t in query_terms if any(t in ent for ent in node_entities_lower)
                )
            total = hits + entity_hits
            if total > 0:
                scored.append((node, total / len(query_terms)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scored[:k]]

    # -- Misc -------------------------------------------------------------------

    async def store_embedding(self, node_id: str, embedding: list[float]) -> None:
        node = self.nodes.get(node_id)
        if node:
            node.embedding = embedding
            self._save()

    async def promote_to_org(self, node_id: str, org_id: str) -> None:
        uid = _uid()
        node = self.nodes.get(node_id)
        if not node:
            return
        node.visibility = Visibility.ORG
        node.org_id = org_id
        node.original_user_id = uid
        self._save()

    def _get_node_lock(self, node_id: str) -> asyncio.Lock:
        if node_id not in self._node_locks:
            self._node_locks[node_id] = asyncio.Lock()
        return self._node_locks[node_id]

    async def atomic_reactivation_update(
        self, node_id: str, timestamp: datetime, stability_delta: float,
    ) -> None:
        lock = self._get_node_lock(node_id)
        async with lock:
            node = self.nodes.get(node_id)
            if not node:
                return
            node.reactivation_count += 1
            node.reactivation_timestamps.append(timestamp)
            node.stability += stability_delta
            node.last_reactivated_at = timestamp
            self._save()

    async def get_stats(self) -> dict[str, Any]:
        uid = _uid()
        edges = self._user_edges.get(uid, [])
        causal_counts: dict[str, int] = defaultdict(int)
        for e in edges:
            if e.type in CAUSAL_EDGE_TYPES:
                causal_counts[str(e.source_id)] += 1
                causal_counts[str(e.target_id)] += 1
        max_cw = max(causal_counts.values()) if causal_counts else 0
        return {"nodes": len(self.nodes), "edges": len(edges), "max_causal_weight": max_cw}


class InMemoryEventBusProvider:
    """EventBusProvider that dispatches handlers as background tasks."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Any]] = defaultdict(list)

    async def publish(self, channel: str, payload: dict[str, Any]) -> None:
        for handler in self._subscribers.get(channel, []):
            asyncio.create_task(self._safe_call(handler, payload))

    async def subscribe(self, channel: str, handler: Any) -> None:
        self._subscribers[channel].append(handler)

    @staticmethod
    async def _safe_call(handler: Any, payload: dict[str, Any]) -> None:
        import logging
        try:
            await handler(payload)
        except Exception:
            logging.getLogger(__name__).exception("Event handler failed")
