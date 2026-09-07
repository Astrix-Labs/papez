"""Papez MCP Server: stdio transport for Claude Desktop.

Runs on both the mcp 1.x and 2.x Python SDKs. The two differ in how a
low-level ``Server`` learns about tools: 1.x exposes the ``@server.list_tools()``
/ ``@server.call_tool()`` decorators, 2.x replaced them with
``Server.add_request_handler(method, params_type, handler)`` (or ``on_list_tools``
/ ``on_call_tool`` constructor arguments). ``_register_tool_handlers`` picks the
API by feature detection at import time; the tool list, schemas, dispatch and
error payloads below are shared by both paths.
"""
from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from papez.context import current_user_id
from papez.providers import get_providers

STDIO_LOCAL_USER = "stdio_local_user"

SERVER_NAME = "papez"
SERVER_WEBSITE_URL = "https://github.com/Astrix-Labs/papez"
# serverInfo.icons, as ``mcp.types.Icon`` keyword arguments (``src`` must be an
# https URL or data URI). Empty until a hosted icon asset exists; anything
# listed here is advertised on every SDK whose ``Server`` accepts ``icons``.
SERVER_ICONS: list[dict[str, Any]] = []

providers = get_providers()
tools = providers.tools


def _package_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("papez")
    except PackageNotFoundError:
        return None


def _build_server() -> Any:
    """Construct the low-level ``Server`` with the serverInfo fields this SDK accepts.

    ``version``, ``website_url`` and ``icons`` were added to ``Server.__init__``
    at different points in the 1.x line and all exist in 2.x. Passing only the
    keywords the installed signature declares keeps the floor at mcp 1.0.
    """
    accepted = inspect.signature(Server.__init__).parameters
    kwargs: dict[str, Any] = {}
    version = _package_version()
    if version and "version" in accepted:
        kwargs["version"] = version
    if "website_url" in accepted:
        kwargs["website_url"] = SERVER_WEBSITE_URL
    if SERVER_ICONS and "icons" in accepted:
        from mcp.types import Icon

        kwargs["icons"] = [Icon(**icon) for icon in SERVER_ICONS]
    return Server(SERVER_NAME, **kwargs)


app = _build_server()

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

# Tool definitions as plain dicts (MCP wire shape, camelCase keys). mcp 1.x
# names its model fields camelCase; 2.x names them snake_case with camelCase
# aliases. ``Tool.model_validate`` accepts the wire shape on both, and the
# dicts stay the single source of truth for argument validation.
_TOOL_DEFS: list[dict[str, Any]] = [
    {"name": "memory_store", "description": "Store a new memory in the causal memory graph. Use `related` for writer-specified typed edges (each {id, type}); `related_to` is legacy and always creates caused_by edges. May return `possible_conflicts` — heuristic hints, not verified contradictions.", "inputSchema": {
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
    }},
    {"name": "memory_amend", "description": "Record a correction: creates a new memory that supersedes an existing one. The old memory is kept (decayed in recall results), not deleted.", "inputSchema": {
        "type": "object", "required": ["node_id", "content"],
        "properties": {
            "node_id": {"type": "string"},
            "content": {"type": "string"},
            "reason": {"type": "string"},
        },
    }},
    {"name": "memory_recall", "description": "Recall memories using hybrid search (vector + keyword + graph spreading activation).", "inputSchema": {
        "type": "object", "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 10},
            "max_results": {"type": "integer"},
            "verbosity": {"type": "string", "enum": ["concise", "full"], "default": "full", "description": "concise = id/summary/status/score/activation/is_core only, no causal chains."},
        },
    }},
    {"name": "memory_search", "description": "Filtered vector search by status, category, date, or entity. Pass an EMPTY query to enumerate by recency instead (no vector search, no embedder needed): with since/active_since this answers 'what's new/changed since <ts>' without knowing what to query for.", "inputSchema": {
        "type": "object", "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "Search text. Empty string switches to enumeration mode: nodes listed by last_reactivated_at descending, honoring the same filters."},
            "filters": {"type": "object", "description": "Supported keys: status (list), category, entity, since (ISO date — created_at >= since; tz-naive treated as UTC), active_since (ISO date — last_reactivated_at >= active_since; tz-naive treated as UTC). With a non-empty query, results are vector-seeded and k-limited; use an empty query for enumeration."},
            "k": {"type": "integer", "default": 10},
        },
    }},
    {"name": "memory_traverse", "description": "Traverse the memory graph from a starting node. Returns reachable nodes AND the edges of the induced subgraph among them (source/target/type/weight/created_by) — a superset of the BFS tree, so paths can be reconstructed.", "inputSchema": {
        "type": "object", "required": ["node_id"],
        "properties": {
            "node_id": {"type": "string"},
            "depth": {"type": "integer", "default": 2},
            "edge_types": {"type": "array", "items": {"type": "string"}},
        },
    }},
    {"name": "memory_explain", "description": "Explain a memory's score breakdown.", "inputSchema": {
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }},
    {"name": "pin_memory", "description": "Pin a memory to core status.", "inputSchema": {
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }},
    {"name": "unpin_memory", "description": "Unpin a memory and re-evaluate core eligibility.", "inputSchema": {
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }},
    {"name": "list_core_memories", "description": "List all core memories, optionally filtered by category.", "inputSchema": {
        "type": "object",
        "properties": {"category": {"type": "string"}},
    }},
    {"name": "delete_memory", "description": "Permanently delete a memory node and all its edges.", "inputSchema": {
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }},
    {"name": "memory_stats", "description": "Get graph statistics.", "inputSchema": {
        "type": "object", "properties": {},
    }},
    {"name": "set_core_preferences", "description": "Configure core memory category preferences.", "inputSchema": {
        "type": "object",
        "properties": {
            "auto": {"type": "array", "items": {"type": "string"}},
            "approval": {"type": "array", "items": {"type": "string"}},
            "excluded": {"type": "array", "items": {"type": "string"}},
        },
    }},
    {"name": "promote_to_org", "description": "Promote a private memory to org visibility. Caller must own the node and belong to the target org.", "inputSchema": {
        "type": "object", "required": ["node_id", "org_id"],
        "properties": {
            "node_id": {"type": "string"},
            "org_id": {"type": "string"},
            "action": {"type": "string", "enum": ["keep_private", "promote_all", "delete_links"], "default": "keep_private"},
            "dry_run": {"type": "boolean", "default": False},
        },
    }},
]

_TOOL_SCHEMAS: list[Tool] = [Tool.model_validate(definition) for definition in _TOOL_DEFS]


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


# --- SDK compatibility layer -------------------------------------------------
#
# mcp 1.x's ``@call_tool()`` decorator does two things around the handler
# before the result reaches the wire: it validates ``arguments`` against the
# tool's ``inputSchema`` (returning an ``isError`` result whose text is
# ``"Input validation error: <message>"``) and it converts any exception into
# an ``isError`` result carrying ``str(exc)``. The 2.x request-handler API does
# neither, so the 2.x adapter below replicates both, and a client sees the
# same bytes whichever SDK the server was installed with.


def _tool_input_schema(name: str) -> dict[str, Any] | None:
    for definition in _TOOL_DEFS:
        if definition["name"] == name:
            schema: dict[str, Any] = definition["inputSchema"]
            return schema
    return None


def _validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Return the 1.x-style validation error text, or None when the arguments pass."""
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema is a hard dependency of mcp
        return None
    try:
        jsonschema.validate(instance=arguments, schema=schema)
    except jsonschema.ValidationError as exc:
        return f"Input validation error: {exc.message}"
    return None


def _register_tool_handlers(server: Any) -> str:
    """Attach ``list_tools``/``call_tool`` using whichever API the installed SDK has.

    Returns the mechanism used (``"decorators"`` for 1.x,
    ``"request_handlers"`` for 2.x) so tests and diagnostics can report it.
    """
    if hasattr(server, "list_tools") and hasattr(server, "call_tool"):
        server.list_tools()(list_tools)
        server.call_tool()(call_tool)
        return "decorators"

    if hasattr(server, "add_request_handler"):
        from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams

        def _sdk_error_result(message: str) -> Any:
            # Wire-shape keys: the field is ``isError`` on 1.x and ``is_error`` on 2.x.
            return CallToolResult.model_validate({"content": [{"type": "text", "text": message}], "isError": True})

        async def _on_list_tools(ctx: Any, params: Any) -> Any:
            return ListToolsResult(tools=await list_tools())

        async def _on_call_tool(ctx: Any, params: Any) -> Any:
            name = params.name
            arguments = params.arguments or {}
            try:
                schema = _tool_input_schema(name)
                if schema is not None:
                    problem = _validate_arguments(arguments, schema)
                    if problem is not None:
                        return _sdk_error_result(problem)
                content = await call_tool(name, arguments)
            except Exception as exc:
                _logger.exception("Tool %s failed outside dispatch", name)
                return _sdk_error_result(str(exc))
            return CallToolResult(content=list(content))

        server.add_request_handler("tools/list", PaginatedRequestParams, _on_list_tools)
        server.add_request_handler("tools/call", CallToolRequestParams, _on_call_tool)
        return "request_handlers"

    raise RuntimeError(
        "Unsupported mcp SDK: Server has neither list_tools/call_tool decorators (mcp 1.x) "
        "nor add_request_handler (mcp 2.x)"
    )


SDK_TOOL_API = _register_tool_handlers(app)


async def main() -> None:
    current_user_id.set(STDIO_LOCAL_USER)
    graph = providers.graph
    await graph.initialize(STDIO_LOCAL_USER)

    # Forgetting only happens if something runs the sweep. Rescore, transition
    # and prune on a cadence (GENESYS_MAINTENANCE_INTERVAL_S, default 600s).
    import asyncio

    from papez.engine.maintenance import maintenance_loop

    maintenance = asyncio.create_task(maintenance_loop(graph, providers.embeddings, providers.llm))
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    finally:
        maintenance.cancel()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
