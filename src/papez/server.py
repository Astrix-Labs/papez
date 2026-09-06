"""Papez MCP Server — stdio transport for Claude Desktop."""
from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from papez.context import current_user_id
from papez.providers import get_providers

STDIO_LOCAL_USER = "stdio_local_user"

providers = get_providers()
tools = providers.tools

app = Server("papez")

# Tool name → (method, required_args, optional_args_with_defaults)
_TOOL_DISPATCH: dict[str, tuple[Any, ...]] = {
    "memory_store": (tools.memory_store, ["content"], {"source_session": "", "related_to": None, "related": None, "category": None, "visibility": "private", "org_id": None}),
    "memory_amend": (tools.memory_amend, ["node_id", "content"], {"reason": None}),
    "memory_recall": (tools.memory_recall, ["query"], {"k": 10, "max_results": None, "verbosity": "full"}),
    "memory_search": (tools.memory_search, ["query"], {"filters": None, "k": 10}),
    "memory_traverse": (tools.memory_traverse, ["node_id"], {"depth": 2, "edge_types": None}),
    "memory_explain": (tools.memory_explain, ["node_id"], {}),
    "pin_memory": (tools.pin_memory, ["node_id"], {}),
    "unpin_memory": (tools.unpin_memory, ["node_id"], {}),
    "list_core_memories": (tools.list_core_memories, [], {"category": None}),
    "delete_memory": (tools.delete_memory, ["node_id"], {}),
    "memory_stats": (tools.memory_stats, [], {}),
    "set_core_preferences": (tools.set_core_preferences, [], {"auto": None, "approval": None, "excluded": None}),
    "promote_to_org": (tools.promote_to_org, ["node_id", "org_id"], {"action": "keep_private", "dry_run": False}),
}

_TOOL_SCHEMAS = [
    Tool(name="memory_store", description="Store a new memory in the causal memory graph. Use `related` for writer-specified typed edges (each {id, type}); `related_to` is legacy and always creates caused_by edges. May return `possible_conflicts` — heuristic hints, not verified contradictions.", inputSchema={
        "type": "object", "required": ["content"],
        "properties": {
            "content": {"type": "string"},
            "source_session": {"type": "string", "default": ""},
            "related_to": {"type": "array", "items": {"type": "string"}, "description": "Legacy: ids of nodes to link via caused_by. Prefer `related`."},
            "related": {"type": "array", "description": "Typed explicit edges. Direction: new_node --type--> target.", "items": {
                "type": "object", "required": ["id", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["caused_by", "supports", "contradicts", "supersedes", "derived_from", "related_to", "temporal_sequence"]},
                },
            }},
            "category": {"type": "string", "description": "Free-form classification (suggested: professional, educational, family, location)."},
            "visibility": {"type": "string", "enum": ["private", "org"], "default": "private"},
            "org_id": {"type": "string", "description": "Required when visibility is 'org'. Must be an org the caller belongs to."},
        },
    }),
    Tool(name="memory_amend", description="Record a correction: creates a new memory that supersedes an existing one. The old memory is kept (decayed in recall results), not deleted.", inputSchema={
        "type": "object", "required": ["node_id", "content"],
        "properties": {
            "node_id": {"type": "string"},
            "content": {"type": "string"},
            "reason": {"type": "string"},
        },
    }),
    Tool(name="memory_recall", description="Recall memories using hybrid search (vector + keyword + graph spreading activation).", inputSchema={
        "type": "object", "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 10},
            "max_results": {"type": "integer"},
            "verbosity": {"type": "string", "enum": ["concise", "full"], "default": "full", "description": "concise = id/summary/status/score/activation/is_core only, no causal chains."},
        },
    }),
    Tool(name="memory_search", description="Filtered vector search by status, category, date, or entity. Pass an EMPTY query to enumerate by recency instead (no vector search, no embedder needed): with since/active_since this answers 'what's new/changed since <ts>' without knowing what to query for.", inputSchema={
        "type": "object", "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "Search text. Empty string switches to enumeration mode: nodes listed by last_reactivated_at descending, honoring the same filters."},
            "filters": {"type": "object", "description": "Supported keys: status (list), category, entity, since (ISO date — created_at >= since; tz-naive treated as UTC), active_since (ISO date — last_reactivated_at >= active_since; tz-naive treated as UTC). With a non-empty query, results are vector-seeded and k-limited; use an empty query for enumeration."},
            "k": {"type": "integer", "default": 10},
        },
    }),
    Tool(name="memory_traverse", description="Traverse the memory graph from a starting node. Returns reachable nodes AND the edges of the induced subgraph among them (source/target/type/weight/created_by) — a superset of the BFS tree, so paths can be reconstructed.", inputSchema={
        "type": "object", "required": ["node_id"],
        "properties": {
            "node_id": {"type": "string"},
            "depth": {"type": "integer", "default": 2},
            "edge_types": {"type": "array", "items": {"type": "string"}},
        },
    }),
    Tool(name="memory_explain", description="Explain a memory's score breakdown.", inputSchema={
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }),
    Tool(name="pin_memory", description="Pin a memory to core status.", inputSchema={
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }),
    Tool(name="unpin_memory", description="Unpin a memory and re-evaluate core eligibility.", inputSchema={
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }),
    Tool(name="list_core_memories", description="List all core memories, optionally filtered by category.", inputSchema={
        "type": "object",
        "properties": {"category": {"type": "string"}},
    }),
    Tool(name="delete_memory", description="Permanently delete a memory node and all its edges.", inputSchema={
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }),
    Tool(name="memory_stats", description="Get graph statistics.", inputSchema={
        "type": "object", "properties": {},
    }),
    Tool(name="set_core_preferences", description="Configure core memory category preferences.", inputSchema={
        "type": "object",
        "properties": {
            "auto": {"type": "array", "items": {"type": "string"}},
            "approval": {"type": "array", "items": {"type": "string"}},
            "excluded": {"type": "array", "items": {"type": "string"}},
        },
    }),
    Tool(name="promote_to_org", description="Promote a private memory to org visibility. Caller must own the node and belong to the target org.", inputSchema={
        "type": "object", "required": ["node_id", "org_id"],
        "properties": {
            "node_id": {"type": "string"},
            "org_id": {"type": "string"},
            "action": {"type": "string", "enum": ["keep_private", "promote_all", "delete_links"], "default": "keep_private"},
            "dry_run": {"type": "boolean", "default": False},
        },
    }),
]


@app.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
async def list_tools() -> list[Tool]:
    return _TOOL_SCHEMAS


# Tools with no side effects worth worrying about (beyond reactivation
# bookkeeping) — safe for clients to retry on failure. Writes are NOT in this
# set: a failure may have landed after a partial write, so clients should
# reconcile via memory_recall instead of blind-retrying (see README
# "Reliability & retries").
_RETRYABLE_TOOLS = {
    "memory_recall", "memory_search", "memory_traverse", "memory_explain",
    "memory_stats", "list_core_memories",
}

_logger = logging.getLogger(__name__)


def _error_content(message: str, retryable: bool) -> list[TextContent]:
    payload = {"error": message, "retryable": retryable}
    return [TextContent(type="text", text=json.dumps(payload, indent=2))]


@app.call_tool()  # type: ignore[untyped-decorator]
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Dispatch a tool call, degrading gracefully instead of hard-failing.

    A memory call is not worth crashing an agent turn over: any tool
    exception is caught and returned as a structured
    ``{"error": ..., "retryable": bool}`` payload rather than propagating as
    a protocol-level MCP failure. ``retryable`` mirrors the README guidance —
    true only for read tools; failed writes should be reconciled, not
    blind-retried.
    """
    if name not in _TOOL_DISPATCH:
        return _error_content(f"Unknown tool: {name}", retryable=False)

    method, required, optional = _TOOL_DISPATCH[name]
    missing = [k for k in required if k not in arguments]
    if missing:
        return _error_content(
            f"missing required argument(s) for {name}: {', '.join(missing)}",
            retryable=False,
        )
    kwargs = {k: arguments[k] for k in required}
    for k, default in optional.items():
        kwargs[k] = arguments.get(k, default)

    try:
        result = await method(**kwargs)
    except PermissionError as exc:
        return _error_content(str(exc), retryable=False)
    except Exception as exc:
        _logger.exception("Tool %s failed", name)
        return _error_content(
            f"{type(exc).__name__}: {exc}", retryable=name in _RETRYABLE_TOOLS
        )
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main() -> None:
    current_user_id.set(STDIO_LOCAL_USER)
    graph = providers.graph
    await graph.initialize(STDIO_LOCAL_USER)

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
