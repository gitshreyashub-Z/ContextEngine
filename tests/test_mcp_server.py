"""End-to-end test of the MCP server over stdio, using the official MCP client.

Spawns `python -m app.mcp.server` as a subprocess (exactly how Claude Desktop or any
other MCP client would), lists its tools and calls some of them.
Requires the seeded database.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

ROOT = Path(__file__).parent.parent
pytestmark = pytest.mark.db


async def _with_session(fn):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server"],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


def _run(fn):
    return asyncio.run(_with_session(fn))


def _text(result) -> dict:
    return json.loads(result.content[0].text)


def test_lists_seven_tools():
    async def fn(session):
        return await session.list_tools()

    names = {t.name for t in _run(fn).tools}
    assert names == {"search_assets", "get_definition", "get_trust_signal", "get_schema",
                     "get_lineage", "execute_sql", "list_metrics"}


def test_tool_calls_round_trip():
    async def fn(session):
        definition = await session.call_tool("get_definition", {"metric_name": "net_revenue"})
        sql = await session.call_tool("execute_sql", {
            "query": "SELECT SUM(net_amount) AS v FROM business.orders WHERE status = 'completed'"})
        blocked = await session.call_tool("execute_sql", {"query": "DROP TABLE business.orders"})
        return definition, sql, blocked

    definition, sql, blocked = _run(fn)
    assert _text(definition)["status"] == "certified"
    assert float(_text(sql)["rows"][0]["v"]) > 0
    assert "error" in _text(blocked)


def test_unknown_tool_returns_error():
    async def fn(session):
        return await session.call_tool("does_not_exist", {})

    assert "error" in _text(_run(fn))
