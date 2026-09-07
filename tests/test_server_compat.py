"""The stdio server registers its tools correctly on both mcp 1.x and 2.x.

``papez.server`` feature-detects the SDK's handler-registration API at import.
These tests exercise the registered handlers the way the SDK's runner does
(not the plain ``list_tools``/``call_tool`` functions other tests call), so a
regression in either adapter shows up regardless of which SDK is installed.
The last test drives a real ``python -m papez`` subprocess over stdio.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from typing import Any

import mcp.types as t
import pytest
from mcp.server import Server

import papez.server as srv

# ``website_url`` and ``inputSchema`` validation in the call_tool decorator both
# arrived partway through the 1.x line; the floor (mcp 1.0) has neither.
_SERVER_ACCEPTS_WEBSITE_URL = "website_url" in inspect.signature(Server.__init__).parameters
_SDK_VALIDATES_INPUT = srv.SDK_TOOL_API == "request_handlers" or (
    hasattr(Server, "call_tool") and "validate_input" in inspect.signature(Server.call_tool).parameters
)
requires_input_validation = pytest.mark.skipif(
    not _SDK_VALIDATES_INPUT, reason="this mcp release's call_tool decorator does not validate inputSchema"
)

EXPECTED_TOOLS = {
    "memory_store", "memory_amend", "memory_recall", "memory_search", "memory_traverse",
    "memory_explain", "pin_memory", "unpin_memory", "list_core_memories", "delete_memory",
    "memory_stats", "set_core_preferences", "promote_to_org",
}


def _wire(model: Any) -> dict[str, Any]:
    """Model as MCP wire JSON (camelCase). Attribute names differ between mcp 1.x and 2.x."""
    dumped: dict[str, Any] = model.model_dump(by_alias=True, exclude_none=True)
    return dumped


async def _list_tools_via_sdk() -> Any:
    if srv.SDK_TOOL_API == "decorators":
        handler = srv.app.request_handlers[t.ListToolsRequest]
        return (await handler(t.ListToolsRequest(method="tools/list"))).root
    entry = srv.app.get_request_handler("tools/list")
    assert entry is not None
    return await entry.handler(None, t.PaginatedRequestParams())


async def _call_tool_via_sdk(name: str, arguments: dict[str, Any]) -> Any:
    params = t.CallToolRequestParams(name=name, arguments=arguments)
    if srv.SDK_TOOL_API == "decorators":
        handler = srv.app.request_handlers[t.CallToolRequest]
        return (await handler(t.CallToolRequest(method="tools/call", params=params))).root
    entry = srv.app.get_request_handler("tools/call")
    assert entry is not None
    return await entry.handler(None, params)


class TestRegistration:
    def test_sdk_api_is_feature_detected(self):
        legacy = hasattr(Server, "list_tools") and hasattr(Server, "call_tool")
        assert srv.SDK_TOOL_API == ("decorators" if legacy else "request_handlers")

    def test_server_identity(self):
        assert srv.app.name == "papez"
        opts = srv.app.create_initialization_options()
        assert opts.server_name == "papez"
        assert opts.capabilities.tools is not None
        if _SERVER_ACCEPTS_WEBSITE_URL:
            assert srv.app.website_url == srv.SERVER_WEBSITE_URL
            assert opts.website_url == srv.SERVER_WEBSITE_URL

    async def test_tools_list_exposes_all_13_tools_with_their_schemas(self):
        result = await _list_tools_via_sdk()
        assert isinstance(result, t.ListToolsResult)
        by_name = {tool.name: tool for tool in result.tools}
        assert set(by_name) == EXPECTED_TOOLS
        assert len(result.tools) == 13
        for declared in srv._TOOL_DEFS:
            served = _wire(by_name[declared["name"]])
            assert served["inputSchema"] == declared["inputSchema"]
            assert served["description"] == declared["description"]


class TestCallToolParity:
    @requires_input_validation
    async def test_missing_required_argument_is_an_sdk_validation_error(self):
        result = await _call_tool_via_sdk("memory_explain", {})
        assert isinstance(result, t.CallToolResult)
        assert _wire(result)["isError"] is True
        assert result.content[0].text == "Input validation error: 'node_id' is a required property"

    @requires_input_validation
    async def test_wrong_argument_type_is_an_sdk_validation_error(self):
        result = await _call_tool_via_sdk("memory_recall", {"query": "x", "k": "ten"})
        assert _wire(result)["isError"] is True
        assert result.content[0].text.startswith("Input validation error: ")

    async def test_unknown_tool_is_a_papez_error_payload(self):
        result = await _call_tool_via_sdk("not_a_tool", {})
        assert _wire(result).get("isError", False) is False
        payload = json.loads(result.content[0].text)
        assert payload == {"error": "Unknown tool: not_a_tool", "retryable": False}

    async def test_tool_exception_becomes_error_payload_not_protocol_error(self, monkeypatch):
        async def boom(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setitem(srv._TOOL_DISPATCH, "memory_stats", (boom, [], {}))
        result = await _call_tool_via_sdk("memory_stats", {})
        assert _wire(result).get("isError", False) is False
        payload = json.loads(result.content[0].text)
        assert payload == {"error": "RuntimeError: boom", "retryable": True}

    async def test_successful_call_returns_json_text(self):
        result = await _call_tool_via_sdk("memory_stats", {})
        assert _wire(result).get("isError", False) is False
        assert result.content[0].type == "text"
        assert isinstance(json.loads(result.content[0].text), dict)


class TestStdioRoundTrip:
    async def test_python_m_papez_serves_tools_over_stdio(self, tmp_path):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", str(tmp_path)),
            "GENESYS_PERSIST_PATH": str(tmp_path / "memories.json"),
            "GENESYS_MAINTENANCE_INTERVAL_S": "0",
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
        }
        params = StdioServerParameters(command=sys.executable, args=["-m", "papez"], env=env, cwd=str(tmp_path))

        async def scenario() -> None:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                info = _wire(await session.initialize())["serverInfo"]
                assert info["name"] == "papez"
                if "websiteUrl" in info:  # only SDKs whose Server accepts website_url advertise it
                    assert info["websiteUrl"] == srv.SERVER_WEBSITE_URL

                listed = await session.list_tools()
                assert {tool.name for tool in listed.tools} == EXPECTED_TOOLS

                ok = await session.call_tool("memory_stats", {})
                assert _wire(ok).get("isError", False) is False
                assert isinstance(json.loads(ok.content[0].text), dict)

                if _SDK_VALIDATES_INPUT:
                    bad = await session.call_tool("memory_explain", {})
                    assert _wire(bad)["isError"] is True
                    assert bad.content[0].text == "Input validation error: 'node_id' is a required property"

                unknown = await session.call_tool("not_a_tool", {})
                assert _wire(unknown).get("isError", False) is False
                assert json.loads(unknown.content[0].text) == {"error": "Unknown tool: not_a_tool", "retryable": False}

        await asyncio.wait_for(scenario(), timeout=60)
