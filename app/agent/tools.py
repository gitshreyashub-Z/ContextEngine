"""LangChain tool wrappers around ContextTools.

The same seven tools are exposed over MCP in app/mcp/server.py; both wrap the shared
ContextTools implementation, so agents and MCP clients see identical behaviour.
Each tool returns a JSON string so the LLM can read errors and recover.
"""

import json

from langchain_core.tools import tool

from app.mcp.tools import ContextTools

_tools = ContextTools()


def _json(result) -> str:
    return json.dumps(result, default=str)


@tool
def search_assets(query: str) -> str:
    """Search data assets (tables, columns, metrics) by name or description.

    Use this first to discover which metrics and tables are relevant to the question.

    Args:
        query: Search term, e.g. "revenue", "customer", "orders"
    """
    return _json(_tools.search_assets(query))


@tool
def get_definition(metric_name: str) -> str:
    """Get the business definition of a metric: calculation, source table/column,
    required filter, owner, status and usage notes. Check this before writing SQL for a metric.

    Args:
        metric_name: Metric name, e.g. as returned by search_assets or list_metrics
    """
    return _json(_tools.get_definition(metric_name))


@tool
def get_trust_signal(asset_name: str) -> str:
    """Get the trust status of an asset or metric: certified, deprecated, experimental or stale,
    with the reason and who certified it.

    Args:
        asset_name: Name of the asset or metric
    """
    return _json(_tools.get_trust_signal(asset_name))


@tool
def get_schema(table_name: str) -> str:
    """Get a table's columns with descriptions and business context (warnings, meaning).

    Args:
        table_name: One of customers, orders, products, payments
    """
    return _json(_tools.get_schema(table_name))


@tool
def get_lineage(asset_name: str) -> str:
    """Get lineage: how tables and metrics relate (foreign keys, metric sources).

    Args:
        asset_name: Table or metric name
    """
    return _json(_tools.get_lineage(asset_name))


@tool
def execute_sql(query: str) -> str:
    """Execute a read-only SQL SELECT query against the business database.

    Only SELECT queries are allowed. Use the 'business.' schema prefix for all tables
    (e.g., business.orders, business.customers). Results are limited to 100 rows.

    Args:
        query: SQL SELECT query to execute
    """
    return _json(_tools.execute_sql(query))


@tool
def list_metrics() -> str:
    """List all metric definitions with their status (certified/deprecated/experimental/draft) and owner."""
    return _json(_tools.list_metrics())


SQL_ONLY_TOOLS = [execute_sql]
ALL_CONTEXT_TOOLS = [search_assets, get_definition, get_trust_signal, get_schema, get_lineage, execute_sql, list_metrics]
