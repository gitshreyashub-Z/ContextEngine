"""ContextEngine pipeline agent - uses structured context to answer questions reliably.

This agent runs a deterministic workflow (a fixed pipeline, not an LLM tool-calling loop):
1. Search for relevant assets
2. Pick the best metric (explicit mention first, then rank by trust signal)
3. Get the metric's business definition
4. Check its trust signal
5. Gather supporting context when the question needs it (schema, lineage, metric catalog)
6. Build and execute SQL from the definition (only for data questions)
7. Use the LLM once, to phrase the final answer from the collected context

The LLM does not decide tool ordering or write SQL here; the context layer does.
For the LLM-driven variant that uses the same tools, see app/agent/contexthub_react.py.
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

# Business synonyms → the vocabulary used in the context layer.
TOPIC_SYNONYMS = {
    "sales": "revenue",
    "turnover": "revenue",
    "income": "revenue",
    "earnings": "revenue",
    "clients": "customers",
    "client": "customer",
    "buyers": "customers",
    "purchases": "orders",
}

# Search topics in priority order: metric words before dimension words, so that
# "revenue by product" searches revenue, not products.
TOPIC_KEYWORDS = ["revenue", "refund", "aov", "mrr", "active", "payment", "order", "product", "customer"]

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

# Words in a question → table name
TABLE_WORDS = {"order": "orders", "customer": "customers", "product": "products", "payment": "payments"}

SEGMENTS = {
    "enterprise": "enterprise",
    "mid-market": "mid_market",
    "mid market": "mid_market",
    "mid_market": "mid_market",
    "midmarket": "mid_market",
    "smb": "smb",
    "small business": "smb",
    "startup": "startup",
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

LINEAGE_WORDS = ("relate", "relationship", "lineage", "linked", "join", "connect", "depend", "upstream", "downstream")
SCHEMA_WORDS = ("column", "represent", "difference between", "which table", "field", "what does")
GOVERNANCE_WORDS = ("deprecated", "experimental", "certified", "which metric", "should i use", "reliable", "trust")


def _month_range(year: int, month: int, column: str) -> str:
    """Half-open date range covering one calendar month (handles December rollover)."""
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return f"{column} >= '{start.isoformat()}' AND {column} < '{end.isoformat()}'"


def _normalize(question: str) -> str:
    """Lowercase and map business synonyms onto context-layer vocabulary."""
    q = question.lower()
    for word, replacement in TOPIC_SYNONYMS.items():
        q = re.sub(rf"\b{word}\b", replacement, q)
    return q


def _log(log: list, tool: str, args: dict, result: Any) -> None:
    log.append({"tool": tool, "args": args, "result_preview": json.dumps(result, default=str)[:500]})


class ContextEngineAgent:
    """Context-aware agent with deterministic tool workflow.

    Forces: search → pick metric → definition → trust → (schema/lineage/catalog) → SQL → answer
    """

    def __init__(self):
        self.name = "contexthub"

    def ask(self, question: str) -> dict[str, Any]:
        """Ask the ContextEngine agent a question."""
        start_time = time.time()
        tool_calls_log: list[dict] = []
        sources: list[dict] = []

        try:
            # Step 1: Search for relevant assets
            search_query = self._extract_topic(question)
            search_result = _tools.search_assets(search_query)
            _log(tool_calls_log, "search_assets", {"query": search_query}, search_result)

            # Step 2: Pick the best metric (uses trust signals, not description text)
            metric_name = self._pick_best_metric(question, search_result)

            # Step 3: Get its definition
            definition = None
            if metric_name:
                definition = _tools.get_definition(metric_name)
                _log(tool_calls_log, "get_definition", {"metric_name": metric_name}, definition)
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
                _log(tool_calls_log, "get_trust_signal", {"asset_name": metric_name}, trust)
                if "trust_level" in trust:
                    sources.append({"type": "trust_signal", "asset": metric_name, "level": trust["trust_level"]})
                else:
                    trust = None

            # Step 5: Supporting context (schema / lineage / metric catalog)
            extra = self._gather_extra_context(question, metric_name, tool_calls_log)

            # Step 6: Execute SQL if this is a data question and the metric is computable
            sql_result = None
            if (
                definition
                and definition.get("status") not in NON_EXECUTABLE_STATUSES
                and self._needs_sql(question)
            ):
                sql_query = self._build_sql(question, definition)
                if sql_query:
                    sql_result = _tools.execute_sql(sql_query)
                    _log(tool_calls_log, "execute_sql", {"query": sql_query}, sql_result)

            # Step 7: Use LLM to format the final answer
            answer = self._format_answer(question, search_result, definition, trust, sql_result, extra)

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

    # ------------------------------------------------------------------ topic & metric

    def _extract_topic(self, question: str) -> str:
        """Extract the search topic from the question (keyword extraction with synonyms)."""
        q = _normalize(question)
        for kw in TOPIC_KEYWORDS:
            if kw in q:
                return kw
        # Fallback: use first few meaningful words
        return question.split("?")[0].strip()[:50]

    def _find_explicit_metric(self, question: str) -> str | None:
        """Return a metric the question names directly (e.g. 'gross revenue', 'MRR')."""
        q_lower = _normalize(question)
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
           Candidates sharing no words with the question are rejected: answering
           "how many orders were cancelled?" with net revenue would be a confident
           wrong answer, which is exactly the failure this project is about.
        """
        explicit = self._find_explicit_metric(question)
        if explicit:
            return explicit

        q = _normalize(question)

        def overlap(name: str) -> int:
            return sum(1 for word in name.split("_") if re.search(rf"\b{re.escape(word)}", q))

        candidates = [
            r for r in search_result.get("results", [])
            if r.get("type") == "metric" and overlap(r["name"]) > 0
        ]
        if not candidates:
            return None

        def score(candidate: dict) -> tuple[int, int]:
            trust = _tools.get_trust_signal(candidate["name"])
            return TRUST_RANK.get(trust.get("trust_level"), -1), overlap(candidate["name"])

        return max(candidates, key=score)["name"]

    # ------------------------------------------------------------------ supporting context

    def _mentioned_tables(self, question: str) -> list[str]:
        q = _normalize(question)
        found = []
        for word, table in TABLE_WORDS.items():
            if re.search(rf"\b{word}", q) and table not in found:
                found.append(table)
        return found

    def _gather_extra_context(self, question: str, metric_name: str | None, log: list) -> dict[str, Any]:
        """Call schema / lineage / catalog tools when the question is about them."""
        q = question.lower()
        extra: dict[str, Any] = {}
        tables = self._mentioned_tables(question)

        if any(w in q for w in SCHEMA_WORDS):
            snake_tokens = set(re.findall(r"\b[a-z]+_[a-z_]+\b", q))
            schema_tables = tables or ["orders", "customers", "products", "payments"]
            for table in schema_tables:
                result = _tools.get_schema(table)
                if "error" in result:
                    continue
                columns = {c["column"] for c in result["columns"]}
                # Without an explicit table, keep only tables that own a mentioned column
                if not tables and not (snake_tokens & columns):
                    continue
                _log(log, "get_schema", {"table_name": table}, result)
                extra.setdefault("schema", []).append(result)

        if any(w in q for w in LINEAGE_WORDS):
            targets = tables[:2] or ([metric_name] if metric_name else [])
            seen = set()
            for target in targets:
                result = _tools.get_lineage(target)
                _log(log, "get_lineage", {"asset_name": target}, result)
                for rel in result.get("relationships", []):
                    key = (rel["source"], rel["target"])
                    if key not in seen:
                        seen.add(key)
                        extra.setdefault("lineage", []).append(rel)

        if any(w in q for w in GOVERNANCE_WORDS):
            result = _tools.list_metrics()
            _log(log, "list_metrics", {}, result)
            extra["metrics"] = result.get("metrics", [])

        return extra

    # ------------------------------------------------------------------ SQL

    def _needs_sql(self, question: str) -> bool:
        """Determine if the question needs a SQL query (asks for data/numbers)."""
        data_words = ["what was", "what is our", "what are our", "how many", "how much", "calculate", "count",
                      "average", "sum", "revenue in", "customers in", "orders in", "rate", "top ", "compare",
                      "most", "highest", "lowest", "trend"]
        q_lower = question.lower()
        if re.search(r"\btotal\b(?!_)", q_lower):  # "total revenue", but not the column "total_amount"
            return True
        return any(w in q_lower for w in data_words)

    def _extract_segment(self, question: str) -> str | None:
        q = question.lower()
        for phrase, segment in SEGMENTS.items():
            if re.search(rf"\b{re.escape(phrase)}s?\b", q):
                return segment
        return None

    def _extract_months(self, question: str, today: date | None = None) -> list[tuple[int, int]]:
        """All (year, month) pairs mentioned, e.g. 'June and July 2026' → [(2026, 6), (2026, 7)]."""
        today = today or date.today()
        q = question.lower()
        year_match = re.search(r"\b(20\d{2})\b", q)
        found: list[tuple[int, int]] = []
        for name, num in MONTHS.items():
            for m in re.finditer(rf"\b{name}\b", q):
                # 'may' is usually a verb unless clearly used as a month
                if name == "may" and not re.search(r"(\b(in|and|vs|between|to|of)\s+may\b)|(\bmay\s+\d{4})", q):
                    continue
                if year_match:
                    year = int(year_match.group(1))
                else:
                    year = today.year if num <= today.month else today.year - 1
                if (year, num) not in found:
                    found.append((year, num))
        return sorted(found)

    def _build_sql(self, question: str, definition: dict, today: date | None = None) -> str | None:
        """Build SQL from the metric definition plus filters/groupings parsed from the question.

        Supports: date ranges, multi-month comparisons (grouped by month), monthly trends,
        customer-segment filters and group-bys, product-category group-bys and top-N products.
        """
        if not definition:
            return None

        source_table = definition.get("source_table", "")
        source_column = definition.get("source_column", "")
        filter_condition = definition.get("filter_condition") or ""
        calculation = definition.get("calculation") or ""
        metric_name = definition.get("metric_name", "")

        if not source_table or not source_column:
            return None

        q = _normalize(question)
        is_orders = source_table == "business.orders"
        is_customers = source_table == "business.customers"

        conditions = [filter_condition] if filter_condition else []
        joins: list[str] = []
        dims: list[tuple[str, str]] = []  # (expression, alias)
        order_by: str | None = None
        limit: int | None = None

        # --- Time
        date_column = DATE_COLUMNS.get(source_table)
        if date_column:
            months = self._extract_months(question, today=today)
            if len(months) >= 2:
                ranges = " OR ".join(f"({_month_range(y, m, date_column)})" for y, m in months)
                conditions.append(f"({ranges})")
                dims.append((f"TO_CHAR({date_column}, 'YYYY-MM')", "month"))
                order_by = "month"
            else:
                date_filter = self._extract_date_filter(question, column=date_column, today=today)
                if date_filter:
                    conditions.append(date_filter)
                if re.search(r"\b(by month|monthly|per month|trend)\b", q):
                    dims.append((f"TO_CHAR({date_column}, 'YYYY-MM')", "month"))
                    order_by = "month"

        # --- Customer segment (filter or group-by)
        if is_orders or is_customers:
            segment_column = "c.segment" if is_orders else "segment"
            group_by_segment = re.search(r"\b(by|per|each|which) segments?\b|\bsegments\b", q)
            segment = self._extract_segment(question)
            if group_by_segment or segment:
                if is_orders:
                    joins.append("JOIN business.customers c ON c.id = o.customer_id")
                if group_by_segment:
                    dims.append((segment_column, "segment"))
                else:
                    conditions.append(f"{segment_column} = '{segment}'")

        # --- Product category / top products
        if is_orders:
            top = re.search(r"\btop (\d+) products?\b", q)
            if re.search(r"\bcategor(y|ies)\b", q):
                joins.append("JOIN business.products p ON p.id = o.product_id")
                dims.append(("p.category", "category"))
            elif top or re.search(r"\b(by|per|each|which) products?\b", q):
                joins.append("JOIN business.products p ON p.id = o.product_id")
                dims.append(("p.name", "product"))
                limit = int(top.group(1)) if top else 10

        # --- Aggregate
        if metric_name in RATE_NUMERATORS:
            alias = f"{metric_name}_pct"
            agg = f"ROUND(100.0 * COUNT(*) FILTER (WHERE {RATE_NUMERATORS[metric_name]}) / NULLIF(COUNT(*), 0), 2)"
        elif "count" in metric_name.lower():
            alias = metric_name
            agg = f"COUNT(DISTINCT {source_column})" if filter_condition else "COUNT(*)"
        elif "average" in metric_name.lower() or "avg" in calculation.lower():
            alias = metric_name
            agg = f"ROUND(AVG({source_column}), 2)"
        else:
            alias = metric_name
            agg = f"SUM({source_column})"

        select = ", ".join([f"{expr} AS {name}" for expr, name in dims] + [f"{agg} AS {alias}"])
        from_clause = f"{source_table} o" if joins else source_table
        sql = f"SELECT {select} FROM {from_clause}"
        if joins:
            sql += " " + " ".join(joins)
        if conditions:
            sql += f" WHERE {' AND '.join(conditions)}"
        if dims:
            sql += " GROUP BY " + ", ".join(expr for expr, _ in dims)
            sql += f" ORDER BY {order_by or alias + ' DESC'}"
        if limit:
            sql += f" LIMIT {limit}"
        return sql

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

    # ------------------------------------------------------------------ answer

    def _build_context(self, search_result: dict, definition: dict | None, trust: dict | None,
                       sql_result: dict | None, extra: dict[str, Any] | None) -> str:
        context_parts = []
        extra = extra or {}

        if definition:
            context_parts.append(f"Metric: {definition.get('metric_name')} ({definition.get('status', 'unknown')})")
            context_parts.append(f"Definition: {definition.get('definition', '')}")
            context_parts.append(f"Calculation: {definition.get('calculation', '')}")
            context_parts.append(f"Source: {definition.get('source_table')}.{definition.get('source_column')}")
            if definition.get("filter_condition"):
                context_parts.append(f"Filter: {definition['filter_condition']}")
            context_parts.append(f"Owner: {definition.get('owner', 'unknown')}")
            if definition.get("notes"):
                context_parts.append(f"Notes: {definition['notes']}")

        if trust:
            context_parts.append(f"Trust Level: {trust.get('trust_level', 'unknown')}")
            context_parts.append(f"Trust Reason: {trust.get('reason', '')}")
            context_parts.append(f"Certified by: {trust.get('certified_by') or 'N/A'}")

        for table in extra.get("schema", []):
            context_parts.append(f"Schema of {table['table']}:")
            for col in table["columns"]:
                context_parts.append(f"  - {col['column']} ({col['data_type']}): {col['description']}. {col.get('business_context') or ''}")

        if extra.get("lineage"):
            context_parts.append("Lineage:")
            for rel in extra["lineage"]:
                context_parts.append(f"  - {rel['source']} -> {rel['target']} ({rel['relationship_type']}): {rel.get('description') or ''}")

        if extra.get("metrics"):
            context_parts.append("All metrics (name: status, owner):")
            for m in extra["metrics"]:
                context_parts.append(f"  - {m['metric_name']}: {m['status']}, {m['owner']}")

        if sql_result and "rows" in sql_result:
            context_parts.append(f"SQL Result: {json.dumps(sql_result['rows'], default=str)}")
            context_parts.append(f"Query: {sql_result.get('query', '')}")
        elif sql_result and "error" in sql_result:
            context_parts.append(f"SQL Error: {sql_result['error']}")

        if not context_parts:
            context_parts.append("No certified metric matches this question.")
            assets = search_result.get("results", [])
            if assets:
                context_parts.append("Related assets found:")
                for a in assets[:5]:
                    context_parts.append(f"  - {a['name']} ({a['type']}): {a.get('description', '')[:100]}")

        return "\n".join(context_parts)

    def _format_answer(self, question: str, search_result: dict, definition: dict | None, trust: dict | None,
                       sql_result: dict | None, extra: dict[str, Any] | None = None) -> str:
        """Use LLM to format a clean final answer from all collected context."""
        context_str = self._build_context(search_result, definition, trust, sql_result, extra)

        prompt = f"""Based on this context, provide a SHORT and clear answer to the user's question.

Question: {question}

Context:
{context_str}

Rules:
- If SQL result has numbers, state them clearly (format large numbers with commas)
- Mention the metric used and its certification status
- If a metric is deprecated, experimental, stale or draft, warn the user
- Only use information from the context; if the context does not answer the question, say so
- Keep answer to 3-5 sentences maximum
- Do NOT use tool calls or JSON in your response"""

        try:
            llm = get_llm(temperature=0.0)
            response = llm.invoke([
                SystemMessage(content="You are a concise data analyst. Give short, factual answers with numbers."),
                HumanMessage(content=prompt),
            ])
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception as e:
            # Fallback without an LLM (e.g. no API key configured): return the raw context
            logger.warning(f"[ContextEngine] LLM formatting failed, using fallback: {e}")
            return self._fallback_answer(definition, trust, sql_result, context_str)

    @staticmethod
    def _fallback_answer(definition: dict | None, trust: dict | None, sql_result: dict | None, context_str: str) -> str:
        def fmt(value: Any) -> str:
            if value is None:
                return "no matching data"
            if isinstance(value, int) and not isinstance(value, bool):
                return f"{value:,}"
            if isinstance(value, float) or hasattr(value, "as_tuple"):
                return f"{value:,.2f}"
            return str(value)

        if sql_result and sql_result.get("rows"):
            rows = sql_result["rows"]
            if len(rows) == 1 and len(rows[0]) == 1:
                result = fmt(next(iter(rows[0].values())))
            else:
                result = "\n".join(", ".join(f"{k}: {fmt(v)}" for k, v in row.items()) for row in rows)
            metric = definition.get("metric_name", "unknown") if definition else "unknown"
            status = trust.get("trust_level", "unknown") if trust else "unknown"
            source = f"{definition.get('source_table', '')}.{definition.get('source_column', '')}" if definition else ""
            warning = ""
            if status in ("deprecated", "experimental", "stale"):
                warning = f"\nWARNING: {metric} is {status}. {trust.get('reason', '')}"
            notes = f"\nNote: {definition['notes']}" if definition and definition.get("notes") else ""
            return f"Result: {result}\n\nMetric: {metric} (status: {status})\nSource: {source}{warning}{notes}"
        return context_str
