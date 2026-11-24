"""ContextEngine agent - uses structured context to answer questions reliably.

This agent runs a deterministic workflow (a fixed pipeline, not an LLM tool-calling loop):
1. Search for relevant assets
2. Pick the best metric (explicit mention first, then rank by trust signal)
3. Get the metric's business definition
4. Check its trust signal
5. Build and execute SQL from the definition (only for data questions)
6. Use the LLM once, to phrase the final answer from the collected context

The LLM does not decide tool ordering or write SQL here; the context layer does.
"""

import json
import logging
import re
import time
from datetime import date, timedelta
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import get_llm
from app.mcp.tools import ContextTools

logger = logging.getLogger(__name__)

_tools = ContextTools()

# Higher is more trustworthy. Used to rank candidate metrics.
TRUST_RANK = {"certified": 3, "experimental": 2, "stale": 1, "deprecated": 0}

# Metric statuses for which no SQL should be run (no valid calculation exists).
NON_EXECUTABLE_STATUSES = {"draft"}

# Common abbreviations / phrasings that refer to a specific metric.
METRIC_ALIASES = {
    "mrr": "monthly_recurring_revenue",
    "recurring revenue": "monthly_recurring_revenue",
    "aov": "average_order_value",
    "completion rate": "order_completion_rate",
}

# Rate metrics: numerator condition, evaluated within the definition's filter_condition
# (which acts as the denominator population).
RATE_NUMERATORS = {
    "refund_rate": "status = 'refunded'",
    "order_completion_rate": "status = 'completed'",
}

# Which date column to use for time filters, per source table.
DATE_COLUMNS = {
    "business.orders": "order_date",
    "business.payments": "paid_at",
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _month_range(year: int, month: int, column: str) -> str:
    """Half-open date range covering one calendar month (handles December rollover)."""
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return f"{column} >= '{start.isoformat()}' AND {column} < '{end.isoformat()}'"


class ContextEngineAgent:
    """Context-aware agent with deterministic tool workflow.

    Forces: search → pick metric → definition → trust → SQL → answer
    """

    def __init__(self):
        self.name = "contexthub"

    def ask(self, question: str) -> dict[str, Any]:
        """Ask the ContextEngine agent a question."""
        start_time = time.time()
        tool_calls_log = []
        sources = []

        try:
            # Step 1: Search for relevant assets
            search_query = self._extract_topic(question)
            search_result = _tools.search_assets(search_query)
            tool_calls_log.append({"tool": "search_assets", "args": {"query": search_query}, "result_preview": json.dumps(search_result, default=str)[:500]})

            # Step 2: Pick the best metric (uses trust signals, not description text)
            metric_name = self._pick_best_metric(question, search_result)

            # Step 3: Get its definition
            definition = None
            if metric_name:
                definition = _tools.get_definition(metric_name)
                tool_calls_log.append({"tool": "get_definition", "args": {"metric_name": metric_name}, "result_preview": json.dumps(definition, default=str)[:500]})
                if "error" in definition:
                    definition = None
                else:
                    sources.append({
                        "type": "definition",
                        "metric": definition["metric_name"],
                        "status": definition.get("status", "unknown"),
                        "source": f"{definition.get('source_table')}.{definition.get('source_column')}",
                    })

            # Step 4: Check trust signal
            trust = None
            if metric_name:
                trust = _tools.get_trust_signal(metric_name)
                tool_calls_log.append({"tool": "get_trust_signal", "args": {"asset_name": metric_name}, "result_preview": json.dumps(trust, default=str)[:500]})
                if "trust_level" in trust:
                    sources.append({"type": "trust_signal", "asset": metric_name, "level": trust["trust_level"]})
                else:
                    trust = None

            # Step 5: Execute SQL if this is a data question and the metric is computable
            sql_result = None
            if (
                definition
                and definition.get("status") not in NON_EXECUTABLE_STATUSES
                and self._needs_sql(question)
            ):
                sql_query = self._build_sql(question, definition)
                if sql_query:
                    sql_result = _tools.execute_sql(sql_query)
                    tool_calls_log.append({"tool": "execute_sql", "args": {"query": sql_query}, "result_preview": json.dumps(sql_result, default=str)[:500]})

            # Step 6: Use LLM to format the final answer
            answer = self._format_answer(question, search_result, definition, trust, sql_result)

            elapsed = time.time() - start_time
            return {
                "agent": self.name,
                "question": question,
                "answer": answer,
                "tool_calls": tool_calls_log,
                "sources": sources,
                "latency_seconds": round(elapsed, 2),
                "num_tool_calls": len(tool_calls_log),
                "success": True,
                "error": None,
            }

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[ContextEngine] Error: {e}")
            return {
                "agent": self.name,
                "question": question,
                "answer": f"Error: {str(e)}",
                "tool_calls": tool_calls_log,
                "sources": sources,
                "latency_seconds": round(elapsed, 2),
                "num_tool_calls": len(tool_calls_log),
                "success": False,
                "error": str(e),
            }

    def _extract_topic(self, question: str) -> str:
        """Extract the search topic from the question (simple keyword extraction)."""
        keywords = ["revenue", "customer", "order", "product", "payment", "active", "refund", "aov", "mrr"]
        q_lower = question.lower()
        for kw in keywords:
            if kw in q_lower:
                return kw
        # Fallback: use first few meaningful words
        return question.split("?")[0].strip()[:50]

    def _find_explicit_metric(self, question: str) -> str | None:
        """Return a metric the question names directly (e.g. 'gross revenue', 'MRR')."""
        q_lower = question.lower()
        for alias, metric in METRIC_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", q_lower):
                return metric

        metrics = _tools.list_metrics().get("metrics", [])
        # Longest names first so 'order_completion_rate' wins over shorter overlaps
        for m in sorted(metrics, key=lambda m: len(m["metric_name"]), reverse=True):
            name = m["metric_name"]
            if name in q_lower or name.replace("_", " ") in q_lower:
                return name
        return None

    def _pick_best_metric(self, question: str, search_result: dict) -> str | None:
        """Pick the metric to use.

        1. If the question names a metric explicitly, use it (so we can warn if it is
           deprecated/experimental instead of silently swapping it).
        2. Otherwise rank the metric assets found by search: trust level first
           (from the structured trust_signals table), then word overlap with the question.
        """
        explicit = self._find_explicit_metric(question)
        if explicit:
            return explicit

        candidates = [r for r in search_result.get("results", []) if r.get("type") == "metric"]
        if not candidates:
            return None

        q_lower = question.lower()

        def score(candidate: dict) -> tuple[int, int]:
            trust = _tools.get_trust_signal(candidate["name"])
            trust_rank = TRUST_RANK.get(trust.get("trust_level"), -1)
            overlap = sum(1 for word in candidate["name"].split("_") if word in q_lower)
            return trust_rank, overlap

        return max(candidates, key=score)["name"]

    def _needs_sql(self, question: str) -> bool:
        """Determine if the question needs a SQL query (asks for data/numbers)."""
        data_words = ["what was", "what is our", "how many", "how much", "calculate", "total", "count",
                      "average", "sum", "revenue in", "customers in", "orders in", "rate"]
        q_lower = question.lower()
        return any(w in q_lower for w in data_words)

    def _build_sql(self, question: str, definition: dict, today: date | None = None) -> str | None:
        """Build SQL based on the metric definition and question context."""
        if not definition:
            return None

        source_table = definition.get("source_table", "")
        source_column = definition.get("source_column", "")
        filter_condition = definition.get("filter_condition") or ""
        calculation = definition.get("calculation") or ""
        metric_name = definition.get("metric_name", "")

        if not source_table or not source_column:
            return None

        conditions = [filter_condition] if filter_condition else []

        # Date filters only make sense for tables that have a date column
        date_column = DATE_COLUMNS.get(source_table)
        if date_column:
            date_filter = self._extract_date_filter(question, column=date_column, today=today)
            if date_filter:
                conditions.append(date_filter)

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""

        if metric_name in RATE_NUMERATORS:
            numerator = RATE_NUMERATORS[metric_name]
            return (
                f"SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE {numerator}) / NULLIF(COUNT(*), 0), 2) "
                f"AS {metric_name}_pct FROM {source_table}{where}"
            )
        if "count" in metric_name.lower():
            if filter_condition:
                return f"SELECT COUNT(DISTINCT {source_column}) AS {metric_name} FROM {source_table}{where}"
            return f"SELECT COUNT(*) AS {metric_name} FROM {source_table}{where}"
        if "average" in metric_name.lower() or "avg" in calculation.lower():
            return f"SELECT ROUND(AVG({source_column}), 2) AS {metric_name} FROM {source_table}{where}"
        # Sum-based metric (revenue, etc.)
        return f"SELECT SUM({source_column}) AS {metric_name} FROM {source_table}{where}"

    def _extract_date_filter(self, question: str, column: str = "order_date", today: date | None = None) -> str:
        """Extract a date filter from question text.

        Supports: 'last month' / 'previous month', 'this month', '<month> <year>',
        'in <month>' (most recent such month), 'in <year>'.
        """
        today = today or date.today()
        q_lower = question.lower()

        if "last month" in q_lower or "previous month" in q_lower:
            last_month = today.replace(day=1) - timedelta(days=1)
            return _month_range(last_month.year, last_month.month, column)

        if "this month" in q_lower:
            return _month_range(today.year, today.month, column)

        # "<month> <year>", e.g. "July 2026"
        for month_name, month_num in MONTHS.items():
            match = re.search(rf"\b{month_name}\b\s+(\d{{4}})\b", q_lower)
            if match:
                return _month_range(int(match.group(1)), month_num, column)

        # "in <month>" without a year → most recent occurrence of that month
        for month_name, month_num in MONTHS.items():
            if re.search(rf"\bin {month_name}\b", q_lower):
                year = today.year if month_num <= today.month else today.year - 1
                return _month_range(year, month_num, column)

        # "in <year>"
        year_match = re.search(r"\bin (\d{4})\b", q_lower)
        if year_match:
            year = int(year_match.group(1))
            return f"{column} >= '{year}-01-01' AND {column} < '{year + 1}-01-01'"

        return ""

    def _format_answer(self, question: str, search_result: dict, definition: dict | None, trust: dict | None, sql_result: dict | None) -> str:
        """Use LLM to format a clean final answer from all collected context."""
        context_parts = []

        if definition:
            context_parts.append(f"Metric: {definition.get('metric_name')} ({definition.get('status', 'unknown')})")
            context_parts.append(f"Definition: {definition.get('definition', '')}")
            context_parts.append(f"Source: {definition.get('source_table')}.{definition.get('source_column')}")
            if definition.get("filter_condition"):
                context_parts.append(f"Filter: {definition['filter_condition']}")
            if definition.get("notes"):
                context_parts.append(f"Notes: {definition['notes']}")

        if trust:
            context_parts.append(f"Trust Level: {trust.get('trust_level', 'unknown')}")
            context_parts.append(f"Trust Reason: {trust.get('reason', '')}")
            context_parts.append(f"Certified by: {trust.get('certified_by') or 'N/A'}")

        if sql_result and "rows" in sql_result:
            context_parts.append(f"SQL Result: {json.dumps(sql_result['rows'], default=str)}")
            context_parts.append(f"Query: {sql_result.get('query', '')}")
        elif sql_result and "error" in sql_result:
            context_parts.append(f"SQL Error: {sql_result['error']}")

        if not context_parts:
            # No metric found - list what was found
            assets = search_result.get("results", [])
            if assets:
                context_parts.append("Available assets found:")
                for a in assets[:5]:
                    context_parts.append(f"  - {a['name']} ({a['type']}): {a.get('description', '')[:100]}")

        context_str = "\n".join(context_parts)

        llm = get_llm(temperature=0.0)
        prompt = f"""Based on this context, provide a SHORT and clear answer to the user's question.

Question: {question}

Context:
{context_str}

Rules:
- If SQL result has a number, state it clearly (format large numbers with commas)
- Mention the metric used and its certification status
- If a metric is deprecated, experimental, stale or draft, warn the user
- Only use information from the context; if the context does not answer the question, say so
- Keep answer to 3-5 sentences maximum
- Do NOT use tool calls or JSON in your response"""

        try:
            response = llm.invoke([
                SystemMessage(content="You are a concise data analyst. Give short, factual answers with numbers."),
                HumanMessage(content=prompt),
            ])
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception:
            # Fallback: format without LLM
            if sql_result and sql_result.get("rows"):
                value = list(sql_result["rows"][0].values())[0]
                value_str = f"{value:,.2f}" if isinstance(value, (int, float)) or hasattr(value, "as_tuple") else str(value)
                metric = definition.get("metric_name", "unknown") if definition else "unknown"
                status = trust.get("trust_level", "unknown") if trust else "unknown"
                source = f"{definition.get('source_table', '')}.{definition.get('source_column', '')}" if definition else ""
                if value is None:
                    value_str = "no matching data"
                return f"Result: {value_str}\n\nMetric: {metric} (status: {status})\nSource: {source}"
            return f"Context found but could not generate answer: {context_str[:200]}"
